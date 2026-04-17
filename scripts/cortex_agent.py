"""
cortex_agent.py — NeighbourWise AI
═══════════════════════════════════
Structured + Unstructured data agent (SQL + RAG chatbot).
Renamed from structured_unstructured_agent/app.py.

TWO MODES:
  Standalone Streamlit:  streamlit run cortex_agent.py
  Via Router:            router_agent.py calls handle_data_query() directly
                         using the same SQL + RAG + synthesize + validate pipeline.
                         This file is NOT imported by the router — the router
                         reimplements the pipeline using shared utilities.
                         This file is the Streamlit UI entry point only.

Pipeline (3 LLM calls max):
  Call 1: SQL generation (Mistral)
  Call 2: Synthesis     (Mistral) — combines SQL + RAG into rich answer
  Call 3: Validator     (Claude)  — ONLY if programmatic checks fail
"""

import streamlit as st
import json
import os
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
from pathlib import Path

# ── Universal Validator (replaces neighbourwise_validator) ────────────────────
from universal_validator import validate_and_improve

load_dotenv()
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(page_title="NeighbourWise AI", page_icon="🏘️", layout="wide")

SNOWFLAKE_CONFIG = {
    "account":   os.environ.get("SNOWFLAKE_ACCOUNT", ""),
    "user":      os.environ.get("SNOWFLAKE_USER", ""),
    "password":  os.environ.get("SNOWFLAKE_PASSWORD", ""),
    "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "NEIGHBOURWISE_AI"),
    "database":  os.environ.get("SNOWFLAKE_DATABASE", "NEIGHBOURWISE_DOMAINS"),
    "role":      os.environ.get("SNOWFLAKE_ROLE", "TRAINING_ROLE"),
}

LLM_MODEL = "mistral-large2"


@st.cache_resource
def get_connection():
    return snowflake.connector.connect(
        **SNOWFLAKE_CONFIG,
        insecure_mode=True,
        network_timeout=120,
        login_timeout=60,
    )


def run_sql(query):
    conn = get_connection()
    cur  = conn.cursor()
    try:
        cur.execute(query)
        if cur.description:
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        return []
    except Exception as e:
        return {"error": str(e)}
    finally:
        cur.close()


def cortex_complete(prompt):
    conn = get_connection()
    cur  = conn.cursor()
    try:
        safe = prompt.replace("'", "''")[:12000]
        cur.execute(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{LLM_MODEL}', '{safe}')")
        return cur.fetchone()[0]
    except Exception as e:
        return f"Error: {e}"
    finally:
        cur.close()


# ═══════════════════════════════════════════════════════════════
# AGENT 1: SQL AGENT (Mistral — 1 LLM call)
# ═══════════════════════════════════════════════════════════════

def sql_agent(question):
    prompt = f"""You are a SQL expert for NeighbourWise Boston neighborhood database.

RULES:
1. ALWAYS use NEIGHBOURWISE_DOMAINS.MARTS.<table>
2. NEIGHBORHOOD_NAME is ALWAYS UPPERCASE (e.g. 'DORCHESTER', 'BACK BAY')
3. For cross-domain comparisons, JOIN tables via LOCATION_ID
4. Output ONLY SQL. No text, no markdown, no backticks.

TABLES:
NEIGHBOURWISE_DOMAINS.MARTS.MRT_NEIGHBORHOOD_HEALTHCARE (NEIGHBORHOOD_NAME, TOTAL_FACILITIES, HOSPITAL_COUNT, CLINIC_COUNT, TOTAL_BED_CAPACITY, HEALTHCARE_GRADE, HEALTHCARE_SCORE, LOCATION_ID)
NEIGHBOURWISE_DOMAINS.MARTS.MRT_NEIGHBORHOOD_SAFETY (NEIGHBORHOOD_NAME, TOTAL_INCIDENTS, VIOLENT_CRIME_COUNT, PCT_VIOLENT, SAFETY_GRADE, SAFETY_SCORE, MOST_COMMON_OFFENSE, LOCATION_ID)
NEIGHBOURWISE_DOMAINS.MARTS.MRT_NEIGHBORHOOD_HOUSING (NEIGHBORHOOD_NAME, TOTAL_PROPERTIES, AVG_ASSESSED_VALUE, AVG_ESTIMATED_RENT, AVG_PRICE_PER_SQFT, HOUSING_GRADE, HOUSING_SCORE, LOCATION_ID)
NEIGHBOURWISE_DOMAINS.MARTS.MRT_NEIGHBORHOOD_MBTA (NEIGHBORHOOD_NAME, TOTAL_STOPS, HAS_RAPID_TRANSIT, TRANSIT_GRADE, TRANSIT_SCORE, LOCATION_ID)
NEIGHBOURWISE_DOMAINS.MARTS.MRT_NEIGHBORHOOD_RESTAURANTS (NEIGHBORHOOD_NAME, TOTAL_RESTAURANTS, AVG_RATING, CUISINE_DIVERSITY, RESTAURANT_GRADE, RESTAURANT_SCORE, LOCATION_ID)
NEIGHBOURWISE_DOMAINS.MARTS.MRT_NEIGHBORHOOD_SCHOOLS (NEIGHBORHOOD_NAME, TOTAL_SCHOOLS, PUBLIC_SCHOOL_COUNT, SCHOOL_GRADE, SCHOOL_SCORE, LOCATION_ID)
NEIGHBOURWISE_DOMAINS.MARTS.MRT_NEIGHBORHOOD_BLUEBIKES (NEIGHBORHOOD_NAME, TOTAL_STATIONS, TOTAL_DOCKS, BIKESHARE_GRADE, BIKESHARE_SCORE, LOCATION_ID)
NEIGHBOURWISE_DOMAINS.MARTS.MRT_NEIGHBORHOOD_GROCERY_STORES (NEIGHBORHOOD_NAME, TOTAL_STORES, SUPERMARKET_COUNT, GROCERY_GRADE, GROCERY_SCORE, LOCATION_ID)
NEIGHBOURWISE_DOMAINS.MARTS.MRT_NEIGHBORHOOD_UNIVERSITIES (NEIGHBORHOOD_NAME, TOTAL_UNIVERSITIES, EDUCATION_GRADE, EDUCATION_SCORE, UNIVERSITY_NAMES, LOCATION_ID)
NEIGHBOURWISE_DOMAINS.MARTS.MRT_BOSTON_HEALTHCARE (FACILITY_NAME, FACILITY_TYPE, NEIGHBORHOOD_NAME, BED_COUNT, IS_HOSPITAL, IS_CLINIC)
NEIGHBOURWISE_DOMAINS.MARTS.MRT_BOSTON_RESTAURANTS (RESTAURANT_NAME, CUISINE_CATEGORY, NEIGHBORHOOD_NAME, RATING, REVIEW_COUNT, PRICE_LABEL)
NEIGHBOURWISE_DOMAINS.MARTS.MRT_BOSTON_UNIVERSITIES (COLLEGE_NAME, INSTITUTION_TYPE, NEIGHBORHOOD_NAME, HAS_CAMPUS_HOUSING, LARGEST_PROGRAM)
NEIGHBOURWISE_DOMAINS.MARTS.MRT_BOSTON_CRIME (OFFENSE_DESCRIPTION, NEIGHBORHOOD_NAME, CRIME_SEVERITY_LABEL, IS_VIOLENT_CRIME)
NEIGHBOURWISE_DOMAINS.MARTS.MRT_BOSTON_HOUSING (BUILDING_TYPE, NEIGHBORHOOD_NAME, TOTAL_ASSESSED_VALUE, ESTIMATED_RENT, BEDROOMS)
NEIGHBOURWISE_DOMAINS.MARTS.MRT_BOSTON_MBTA_STOPS (STOP_NAME, NEIGHBORHOOD_NAME, SERVES_HEAVY_RAIL, IS_WHEELCHAIR_ACCESSIBLE)
NEIGHBOURWISE_DOMAINS.MARTS.MRT_BOSTON_SCHOOLS (SCHOOL_NAME, SCHOOL_TYPE_DESC, NEIGHBORHOOD_NAME, IS_PUBLIC)
NEIGHBOURWISE_DOMAINS.MARTS.MRT_BOSTON_BLUEBIKE_STATIONS (STATION_NAME, NEIGHBORHOOD_NAME, TOTAL_DOCKS, CAPACITY_TIER)
NEIGHBOURWISE_DOMAINS.MARTS.MRT_BOSTON_GROCERY_STORES (STORE_NAME, STORE_TYPE, NEIGHBORHOOD_NAME)

EXAMPLE JOIN:
SELECT h.NEIGHBORHOOD_NAME, h.HEALTHCARE_GRADE, s.SAFETY_GRADE, ho.HOUSING_GRADE
FROM NEIGHBOURWISE_DOMAINS.MARTS.MRT_NEIGHBORHOOD_HEALTHCARE h
JOIN NEIGHBOURWISE_DOMAINS.MARTS.MRT_NEIGHBORHOOD_SAFETY s ON h.LOCATION_ID = s.LOCATION_ID
JOIN NEIGHBOURWISE_DOMAINS.MARTS.MRT_NEIGHBORHOOD_HOUSING ho ON h.LOCATION_ID = ho.LOCATION_ID
WHERE h.NEIGHBORHOOD_NAME IN ('BACK BAY', 'ROXBURY')

Question: {question}
SQL:"""

    sql_text = cortex_complete(prompt).strip().replace("```sql", "").replace("```", "").strip()
    if not sql_text or sql_text.startswith("Error"):
        return {"sql": None, "results": None, "error": sql_text}
    results = run_sql(sql_text)
    if isinstance(results, dict) and "error" in results:
        return {"sql": sql_text, "results": None, "error": results["error"]}
    return {"sql": sql_text, "results": results}


# ═══════════════════════════════════════════════════════════════
# AGENT 2: RAG AGENT (no LLM call — pure vector search)
# ═══════════════════════════════════════════════════════════════

def rag_agent(question, domain_filter=None):
    conn = get_connection()
    cur  = conn.cursor()
    try:
        safe_q        = question.replace("'", "''")[:2000]
        domain_clause = (
            f"AND domain = '{domain_filter}'"
            if domain_filter and domain_filter != "ALL"
            else ""
        )
        sql = f"""
            SELECT chunk_text, domain, source_file,
                VECTOR_COSINE_SIMILARITY(chunk_embedding,
                    SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', 'query: {safe_q}')
                ) AS similarity
            FROM NEIGHBOURWISE_DOMAINS.RAW_UNSTRUCTURED.RAW_DOMAIN_CHUNKS
            WHERE 1=1 {domain_clause}
            ORDER BY similarity DESC LIMIT 3
        """
        cur.execute(sql)
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        return {"chunks": rows}
    except Exception as e:
        return {"chunks": [], "error": str(e)}
    finally:
        cur.close()


# ═══════════════════════════════════════════════════════════════
# SYNTHESIZER (Mistral — 1 LLM call)
# ═══════════════════════════════════════════════════════════════

def synthesize_answer(question, sql_data, rag_data):
    parts = []
    if sql_data and isinstance(sql_data.get("results"), list) and sql_data["results"]:
        parts.append(
            f"DATABASE RESULTS:\n"
            f"{json.dumps(sql_data['results'][:15], indent=2, default=str)}"
        )
    if rag_data and rag_data.get("chunks"):
        chunks = "\n\n".join([
            f"[{c.get('DOMAIN', c.get('domain', '?'))}] "
            f"{c.get('CHUNK_TEXT', c.get('chunk_text', ''))[:600]}"
            for c in rag_data["chunks"][:3]
        ])
        parts.append(f"DOCUMENT INSIGHTS:\n{chunks}")

    if not parts:
        return (
            "I couldn't find relevant information. "
            "Try being more specific about a Boston neighborhood or domain."
        )

    context = "\n\n".join(parts)
    prompt  = f"""You are NeighbourWise AI, an expert Boston neighborhood analyst. Write a detailed, insightful answer.

FORMAT — your answer MUST have these THREE sections:

### Summary
Write 3-4 conversational sentences that tell a story. Lead with the most interesting finding.
Convert UPPERCASE neighborhood names to Title Case (DORCHESTER → Dorchester).
Weave numbers naturally: "Dorchester has 5 hospitals" not "HOSPITAL_COUNT: 5".
If comparing neighborhoods, highlight the biggest contrast.
If the data reveals a disparity or trend, call it out.

### Key Data
Create a clean markdown table with the most relevant columns.
- Convert booleans: TRUE → Yes, FALSE → No
- Format numbers: 1234 → 1,234 and 45.678 → 45.7
- No IDs, timestamps, or load dates
- Include ALL rows from the data (up to 15)
- Use Title Case for neighborhood names in the table

### Insights
Write 2-3 analytical insights that go DEEPER than just restating the numbers:
- What patterns or disparities does the data reveal?
- How does this neighborhood compare to Boston overall?
- What does this mean for residents, students, or businesses?
- Are there any surprising findings or red flags?
Be specific — reference actual neighborhoods and numbers from the data.

DATA:
{context}

QUESTION: {question}

Write your detailed answer now:"""

    return cortex_complete(prompt)


# ═══════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

def ask_neighbourwise(question, domain_filter=None):
    """
    Full pipeline (3 LLM calls max):
    1. SQL agent    (1 Mistral call)
    2. RAG agent    (0 LLM calls — vector search only)
    3. Synthesize   (1 Mistral call)
    4. Validate     (0 or 1 Claude call — only if checks fail)
    """
    sql_data     = sql_agent(question)
    rag_data     = rag_agent(question, domain_filter=domain_filter)
    draft_answer = synthesize_answer(question, sql_data, rag_data)

    # ── FIX: pass conn (not cur) to validate_and_improve ─────────────────────
    # UniversalValidator needs a full connection to call Cortex internally.
    # get_connection() is cached by @st.cache_resource so no new connection
    # is opened — it reuses the existing one.
    conn      = get_connection()
    validated = validate_and_improve(conn, question, draft_answer, sql_data, rag_data)

    return {
        "answer":     validated["answer"],
        "sql_data":   sql_data,
        "rag_data":   rag_data,
        "validation": validated["feedback"],
        "improved":   validated["improved"],
    }


# ═══════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════

def main():
    with st.sidebar:
        st.title("🏘️ NeighbourWise AI")
        st.caption("Boston Neighborhood Intelligence")
        st.divider()

        domain_filter = st.selectbox(
            "Filter by domain",
            ["ALL", "HEALTHCARE", "RESTAURANTS", "UNIVERSITIES", "CRIME",
             "HOUSING", "TRANSIT", "BLUEBIKES", "GROCERY", "SCHOOLS", "GENERAL"],
        )
        st.divider()

        st.markdown("**Example questions:**")
        for q in [
            "How many hospitals are in Dorchester?",
            "Which neighborhood is the safest?",
            "Tell me about healthcare in Roxbury",
            "Compare Back Bay and Roxbury across all domains",
            "What are the mental health challenges in Boston?",
            "Best rated restaurants in Fenway",
            "Which neighborhoods have no subway access?",
            "What is Boston doing about food deserts?",
            "Average rent in each neighborhood",
            "Universities in Fenway with campus housing",
        ]:
            if st.button(q, key=f"ex_{q}", use_container_width=True):
                st.session_state.user_question = q

        st.divider()
        st.caption("Agents: SQL (Mistral) + RAG (e5) + Validator (Claude)")
        st.caption("20 mart tables + ~1,800 document chunks")

    st.header("Ask anything about Boston neighborhoods")
    st.caption("SQL + RAG → Mistral synthesizes → Claude validates → best answer")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Render chat history ───────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sql"):
                with st.expander("📊 SQL Query"):
                    st.code(msg["sql"], language="sql")
            if msg.get("results"):
                with st.expander("📋 Data Results"):
                    st.dataframe(
                        pd.DataFrame(msg["results"][:50]),
                        use_container_width=True,
                    )
            if msg.get("chunks"):
                with st.expander("📄 Document Sources"):
                    for i, c in enumerate(msg["chunks"][:3]):
                        d = c.get("DOMAIN", c.get("domain", "?"))
                        s = c.get("SOURCE_FILE", c.get("source_file", "?"))
                        t = c.get("CHUNK_TEXT", c.get("chunk_text", ""))[:300]
                        st.markdown(f"**[{d}]** {s}")
                        st.caption(t + "...")
                        if i < len(msg["chunks"]) - 1:
                            st.divider()
            if msg.get("improved") is not None:
                if msg["improved"]:
                    with st.expander("🔍 Validator (Claude): Improved"):
                        if msg.get("validation"):
                            for n, d in msg["validation"]["checks"].items():
                                st.markdown(f"{d['status']} **{n}**")
                                for issue in d.get("issues", []):
                                    st.caption(f"  → {issue[:150]}")
                else:
                    with st.expander("✅ Validator (Claude): Passed"):
                        st.caption("No issues found.")

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask about any Boston neighborhood...")

    if "user_question" in st.session_state:
        user_input = st.session_state.user_question
        del st.session_state.user_question

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("🔄 Querying data + generating answer..."):
                result = ask_neighbourwise(user_input, domain_filter=domain_filter)

            st.markdown(result["answer"])

            tag = (
                "🔍 Improved by Claude"
                if result.get("improved")
                else "✅ Passed validation"
            )
            st.caption(f"📊📄 SQL + RAG → {tag}")

            sql_query   = None
            sql_results = None
            if result.get("sql_data") and result["sql_data"].get("sql"):
                sql_query   = result["sql_data"]["sql"]
                sql_results = result["sql_data"].get("results")
                with st.expander("📊 SQL Query"):
                    st.code(sql_query, language="sql")
                if sql_results and isinstance(sql_results, list):
                    with st.expander("📋 Data Results"):
                        st.dataframe(
                            pd.DataFrame(sql_results[:50]),
                            use_container_width=True,
                        )

            rag_chunks = None
            if result.get("rag_data") and result["rag_data"].get("chunks"):
                rag_chunks = result["rag_data"]["chunks"]
                with st.expander("📄 Document Sources"):
                    for i, c in enumerate(rag_chunks[:3]):
                        d = c.get("DOMAIN", c.get("domain", "?"))
                        s = c.get("SOURCE_FILE", c.get("source_file", "?"))
                        t = c.get("CHUNK_TEXT", c.get("chunk_text", ""))[:300]
                        st.markdown(f"**[{d}]** {s}")
                        st.caption(t + "...")
                        if i < len(rag_chunks) - 1:
                            st.divider()

            if result.get("improved") is not None:
                if result["improved"]:
                    with st.expander("🔍 Validator (Claude): Improved"):
                        if result.get("validation"):
                            for n, d in result["validation"]["checks"].items():
                                st.markdown(f"{d['status']} **{n}**")
                                for issue in d.get("issues", []):
                                    st.caption(f"  → {issue[:150]}")
                else:
                    with st.expander("✅ Validator (Claude): Passed"):
                        st.caption("No issues found.")

        st.session_state.messages.append({
            "role":       "assistant",
            "content":    result["answer"],
            "sql":        sql_query,
            "results":    sql_results,
            "chunks":     rag_chunks,
            "validation": result.get("validation"),
            "improved":   result.get("improved"),
        })


if __name__ == "__main__":
    main()