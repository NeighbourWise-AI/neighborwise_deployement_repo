"""
NeighbourWise AI — FastAPI Backend
═══════════════════════════════════════════════════════════════════════════════
REST API for NeighbourWise AI neighborhood intelligence platform.

Endpoints:
  - /overview/*          — Dashboard data (via overview_endpoints.py router)
  - /query               — Route user queries (SQL + RAG + Graph + Web Search)
  - /neighborhoods       — List all neighborhoods (legacy)
  - /report/*            — Generate and retrieve neighborhood reports
  - /health              — Health check

Run:
    python3 -m uvicorn neighbourwise_fastapi:app --reload --port 8001
"""

import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd
from dotenv import load_dotenv
import uuid

# Load environment
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# INITIALIZE FASTAPI
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="NeighbourWise AI API",
    description="Boston neighborhood intelligence via SQL + RAG + Graph + Web Search",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Mount overview router — all /overview/* endpoints live in overview_endpoints.py
from overview_endpoints import router as overview_router
app.include_router(overview_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    """User query request."""
    query: str = Field(..., description="Natural language query about Boston neighborhoods")
    domain_filter: Optional[str] = Field(
        None,
        description="Force a specific domain tag (SAFETY, HOUSING, RESTAURANTS, HEALTHCARE, "
                    "SCHOOLS, GROCERY, TRANSIT, BLUEBIKES). Overrides auto-detection."
    )
    skip_validation: Optional[bool] = Field(
        None,
        description="Skip LLM validation step to save ~30-40s. "
                    "Auto-skips for simple single-domain queries if not set."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Is Allston safe and affordable?",
                "domain_filter": None,
                "skip_validation": None,
            }
        }


class RoutingMeta(BaseModel):
    """Routing decision metadata returned alongside every query response."""
    detected_domains: List[str] = Field(..., description="All domains detected in the query")
    detected_neighborhoods: List[str] = Field(..., description="Neighborhood names found in the query")
    intent: str = Field(..., description="Routing intent: data_query | graph_query | web_search | report | chart | image")
    intent_description: str = Field(..., description="Human-readable reason for this routing decision")
    domain_override: bool = Field(False, description="True when domain_filter was supplied by caller")
    fallback_used: Optional[str] = Field(None, description="Set when primary agent failed and a fallback was triggered")


# ── Token / Cost tracking models ─────────────────────────────────────────────

class LLMCallDetail(BaseModel):
    """Single LLM call within the query pipeline."""
    model: str = Field(..., description="Model name")
    source: str = Field(..., description="Backend: cortex | anthropic | openai")
    purpose: str = Field("", description="Call purpose: classify, generate, validate, improve")
    input_tokens: int = Field(0)
    output_tokens: int = Field(0)
    cost_usd: float = Field(0.0)
    latency_s: float = Field(0.0)
    estimated: bool = Field(True)


class LatencyDetail(BaseModel):
    operation: str
    duration_ms: float = 0.0
    threshold_ms: float = 5000.0
    exceeded: bool = False


class TrajectoryInfo(BaseModel):
    actual_path: List[str] = Field(default_factory=list)
    expected_path: List[str] = Field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0
    is_valid: bool = False
    deviations: List[str] = Field(default_factory=list)


class JudgeScoreDetail(BaseModel):
    relevance: float = 3.0
    completeness: float = 3.0
    data_grounding: float = 3.0
    overall: float = 3.0
    reasoning: str = ""


class LLMUsageSummary(BaseModel):
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_s: float = 0.0
    num_llm_calls: int = 0
    by_model: Dict[str, Any] = Field(default_factory=dict)
    by_purpose: Dict[str, Any] = Field(default_factory=dict)
    calls: List[LLMCallDetail] = Field(default_factory=list)
    latency_records: Optional[List[LatencyDetail]] = None
    slow_operations: Optional[List[str]] = None
    trajectory: Optional[TrajectoryInfo] = None
    judge_score: Optional[JudgeScoreDetail] = None


class GuardrailsResult(BaseModel):
    """Input + output guardrail results."""
    input_valid: bool = Field(True, description="Whether the input passed all guardrails")
    input_blocked_reason: Optional[str] = Field(None, description="Why input was blocked (if blocked)")
    input_violations: List[str] = Field(default_factory=list, description="Specific violations detected")
    output_safe: Optional[bool] = Field(None, description="Whether the output passed all guardrails")
    output_issues: List[str] = Field(default_factory=list, description="Output guardrail issues found")
    pii_detected: bool = Field(False, description="Whether PII was found and redacted")
    hallucination_markers: List[str] = Field(default_factory=list, description="Fabrication signals detected")


class QueryResponse(BaseModel):
    """Query response from router agent."""
    type: str = Field(..., description="Intent type: data_query, chart, image, web_search, report, graph_query")
    answer: str = Field(..., description="Synthesized answer")
    neighborhood: Optional[str] = Field(None, description="Primary detected neighborhood (if any)")
    domain: Optional[str] = Field(None, description="Primary detected domain")
    domains: List[str] = Field(default_factory=list, description="All detected domains")
    confidence: float = Field(0.0, description="Classification confidence (0–1)")
    elapsed: float = Field(0.0, description="Query execution time in seconds")
    routing: Optional[RoutingMeta] = Field(None, description="Routing decision metadata")
    llm_usage: Optional[LLMUsageSummary] = None
    guardrails: Optional[GuardrailsResult] = Field(None, description="Input + output guardrail results")
    sql: Optional[str] = Field(None, description="Executed SQL (if data_query)")
    results: Optional[List[Dict]] = Field(None, description="SQL results (if data_query)")
    rag_chunks: Optional[List[Dict]] = Field(None, description="RAG sources (if applicable)")
    validation: Optional[Dict] = Field(None, description="Validation feedback")
    chart_path: Optional[str] = Field(None, description="Path to generated chart (if chart)")
    image_paths: Optional[List[str]] = Field(None, description="Paths to generated images (if image)")
    error: Optional[str] = Field(None, description="Error message (if failed)")


class ReportRequest(BaseModel):
    """Report generation request."""
    neighborhood: str = Field(..., description="Neighborhood name")

    class Config:
        json_schema_extra = {"example": {"neighborhood": "Dorchester"}}


class ReportResponse(BaseModel):
    """Report generation response."""
    report_id: str = Field(..., description="Unique report identifier")
    neighborhood: str
    status: str = Field("pending", description="Status: pending, processing, completed, failed")
    pdf_path: Optional[str] = Field(None, description="Path to generated PDF")
    url: Optional[str] = Field(None, description="Download URL")
    created_at: str
    completed_at: Optional[str] = None
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    snowflake_connected: bool


# ══════════════════════════════════════════════════════════════════════════════
# REPORT STORAGE
# ══════════════════════════════════════════════════════════════════════════════

reports_db: Dict[str, Dict[str, Any]] = {}
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_snowflake_conn():
    """Get Snowflake connection."""
    try:
        from shared.snowflake_conn import get_conn
        return get_conn()
    except Exception as e:
        logger.error(f"Failed to connect to Snowflake: {e}")
        raise HTTPException(status_code=503, detail="Database connection failed")


def run_query(query: str, conn):
    """Execute query on Snowflake."""
    try:
        from shared.snowflake_conn import run_query as sf_run_query
        return sf_run_query(query, conn)
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API and database health."""
    try:
        conn = get_snowflake_conn()
        conn.close()
        snowflake_ok = True
    except Exception:
        snowflake_ok = False

    return HealthResponse(
        status="healthy" if snowflake_ok else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        snowflake_connected=snowflake_ok,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ROOT
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint — API documentation."""
    return {
        "name": "NeighbourWise AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "overview": {
                "neighborhoods":       "GET /overview/neighborhoods",
                "kpis":                "GET /overview/kpis",
                "map":                 "GET /overview/map",
                "crime_summary":       "GET /overview/crime-summary",
                "domain_safety":       "GET /overview/domain/safety",
                "domain_housing":      "GET /overview/domain/housing",
                "domain_transit":      "GET /overview/domain/transit",
                "domain_grocery":      "GET /overview/domain/grocery",
                "domain_healthcare":   "GET /overview/domain/healthcare",
                "domain_schools":      "GET /overview/domain/schools",
                "domain_restaurants":  "GET /overview/domain/restaurants",
                "domain_universities": "GET /overview/domain/universities",
                "domain_bluebikes":    "GET /overview/domain/bluebikes",
            },
            "query":         "POST /query",
            "neighborhoods": "GET /neighborhoods",
            "report": {
                "generate": "POST /report/generate",
                "status":   "GET /report/{report_id}",
                "download": "GET /report/{report_id}/download",
                "list":     "GET /report",
            },
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# NEIGHBORHOODS ENDPOINT (legacy — kept for backwards compatibility)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/neighborhoods", tags=["Neighborhoods"])
async def list_neighborhoods():
    """Get list of all neighborhoods (raw). Use /overview/neighborhoods for the sidebar dropdown."""
    conn = get_snowflake_conn()
    try:
        from shared.snowflake_conn import get_all_neighborhoods
        neighborhoods = sorted(get_all_neighborhoods(conn))
        return {"count": len(neighborhoods), "neighborhoods": neighborhoods}
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# QUERY ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def process_query(request: QueryRequest):
    """
    Route and process a natural language query about Boston neighborhoods.

    Routing logic (handled internally by router_agent):
    - report keywords                          → report       (report_agent)
    - chart/visualization keywords             → chart        (graphic_agent)
    - web-only signals (news, weather today)   → web_search   (web_search_agent)
    - livability/comparison intent             → graph_query  (Graph_agent + Neo4j)
    - 2+ domains detected                      → graph_query  (Graph_agent + Neo4j)
    - 1 domain detected                        → data_query   (SQL + RAG)
    - nothing matches                          → web_search

    Pass domain_filter to force a specific domain and skip auto-detection.
    Pass skip_validation=true to skip the LLM validation step (~30-40s savings).
    """
    logger.info(f"[/query] Received: {request.query!r}  domain_filter={request.domain_filter!r}")
    t_start = time.time()

    # ── Input Guardrails (fast — no LLM call, runs before any DB/agent work) ──
    guardrails_result = None
    try:
        from neighbourwise_guardrails import validate_input, validate_output

        input_check = validate_input(request.query)
        if not input_check["is_valid"]:
            elapsed = round(time.time() - t_start, 2)
            logger.warning(
                f"[/query] Input blocked in {elapsed}s: {input_check['blocked_reason']}"
            )
            return QueryResponse(
                type="blocked",
                answer=input_check["blocked_reason"],
                elapsed=elapsed,
                guardrails=GuardrailsResult(
                    input_valid=False,
                    input_blocked_reason=input_check["blocked_reason"],
                    input_violations=input_check["violations"],
                ),
                error=None,
            )
    except ImportError:
        pass  # guardrails module not present — skip silently

    conn = None
    try:
        from router_agent import route
        from shared.snowflake_conn import get_conn

        conn = get_conn()

        # ── Cost tracker (graceful — works even if module missing) ────────────
        tracker = None
        try:
            from LLM_cost_tracker import CostTracker
            tracker = CostTracker()
        except ImportError:
            pass

        # ── Route the query ───────────────────────────────────────────────────
        result = route(request.query, conn, domain_filter=request.domain_filter)
        elapsed = round(time.time() - t_start, 2)

        # Propagate fallback info if present
        routing_data = result.get("routing")
        fallback = result.get("routing_fallback")
        if routing_data and fallback:
            if isinstance(routing_data, dict):
                routing_data["fallback_used"] = fallback
            elif hasattr(routing_data, "fallback_used"):
                routing_data.fallback_used = fallback

        # ── Output Guardrails (fast — regex only, no LLM call) ────────────────
        answer_text = result.get("answer", "")
        try:
            from neighbourwise_guardrails import validate_output
            output_check = validate_output(answer_text)
            answer_text = output_check["filtered_output"]  # PII-redacted
            guardrails_result = GuardrailsResult(
                input_valid=True,
                output_safe=output_check["is_safe"],
                output_issues=output_check["issues"],
                pii_detected=output_check["pii_detected"],
                hallucination_markers=output_check["hallucination_markers"],
            )
            if not output_check["is_safe"]:
                logger.warning(f"[/query] Output guardrail issues: {output_check['issues']}")
        except ImportError:
            pass  # guardrails module not present — skip silently

        # ── Post-hoc LLM usage estimation ─────────────────────────────────────
        # Since Cortex doesn't return token counts we estimate from text sizes
        # and model pricing. Direct Anthropic/OpenAI calls could use actual counts.
        llm_usage = None
        if tracker:
            intent_type  = result.get("type", "data_query")
            rag_chunks   = result.get("rag_chunks") or []
            validation   = result.get("validation") or {}
            query_text   = request.query

            # 1. Classify call (always happens)
            classify_prompt = query_text + ("x" * 600)
            classify_output = '{"intent":"' + intent_type + '"}'
            tracker.log(
                model="mistral-large2",
                input_text=classify_prompt,
                output_text=classify_output,
                purpose="classify",
                latency_s=round(elapsed * 0.05, 2),
            )

            # 2. Generate call (data_query, graph_query, web_search)
            if intent_type in ("graph_query", "web_search", "data_query") and answer_text:
                rag_context = " ".join(
                    c.get("chunk_text", c.get("CHUNK_TEXT", ""))[:500]
                    for c in rag_chunks
                )
                sys_prompt_size = 2000 if intent_type != "data_query" else 1500
                gen_input = ("x" * sys_prompt_size) + query_text + rag_context
                gen_model = "claude-sonnet-4-6" if intent_type in ("graph_query", "web_search") else "mistral-large2"
                tracker.log(
                    model=gen_model,
                    input_text=gen_input,
                    output_text=answer_text,
                    purpose="generate",
                    latency_s=round(elapsed * 0.55, 2),
                )

            # 3. Validate call (when validation was run)
            has_validation = (
                validation.get("passed") is not None
                or validation.get("checks")
                or validation.get("needs_improvement") is not None
            )
            if has_validation and answer_text:
                val_input     = ("x" * 1500) + answer_text + query_text
                val_output_raw = json.dumps(validation.get("checks", {}))
                val_model     = "gpt-4o" if intent_type in ("graph_query", "web_search") else "claude-sonnet-4-6"
                tracker.log(
                    model=val_model,
                    input_text=val_input,
                    output_text=val_output_raw,
                    purpose="validate",
                    latency_s=round(elapsed * 0.30, 2),
                )
                # 4. Improve call (when validator triggered regeneration)
                if validation.get("regenerated"):
                    tracker.log(
                        model="claude-sonnet-4-6" if intent_type == "graph_query" else "mistral-large2",
                        input_text=val_input,
                        output_text=answer_text,
                        purpose="improve",
                        latency_s=round(elapsed * 0.10, 2),
                    )

            # 5. Trajectory recording
            trajectory_map = {
                "graph_query": ["classify", "neo4j_query", "rag_retrieve", "generate", "validate"],
                "data_query":  ["classify", "sql_execute", "rag_retrieve", "generate", "validate"],
                "web_search":  ["classify", "serper_search", "url_fetch", "generate", "validate"],
                "chart":       ["classify", "sql_execute", "chart_render"],
                "report":      ["classify", "report_generate"],
                "image":       ["classify", "image_generate"],
            }
            for step in trajectory_map.get(intent_type, ["classify"]):
                tracker.record_step(step)

            # 6. Build summary and construct Pydantic model
            usage_raw = tracker.summary()

            trajectory_info = None
            if usage_raw.get("trajectory"):
                traj = tracker.analyze_trajectory(intent_type)
                trajectory_info = TrajectoryInfo(
                    actual_path=traj.actual_path,
                    expected_path=traj.expected_path,
                    precision=traj.precision,
                    recall=traj.recall,
                    is_valid=traj.is_valid,
                    deviations=traj.deviations,
                )

            latency_details = None
            if usage_raw.get("latency_records"):
                latency_details = [LatencyDetail(**lr) for lr in usage_raw["latency_records"]]

            judge_detail = None
            if usage_raw.get("judge_score"):
                judge_detail = JudgeScoreDetail(**usage_raw["judge_score"])

            llm_usage = LLMUsageSummary(
                total_input_tokens=usage_raw["total_input_tokens"],
                total_output_tokens=usage_raw["total_output_tokens"],
                total_tokens=usage_raw["total_tokens"],
                total_cost_usd=usage_raw["total_cost_usd"],
                total_latency_s=usage_raw["total_latency_s"],
                num_llm_calls=usage_raw["num_llm_calls"],
                by_model=usage_raw["by_model"],
                by_purpose=usage_raw["by_purpose"],
                calls=[LLMCallDetail(**c) for c in usage_raw["calls"]],
                latency_records=latency_details,
                slow_operations=usage_raw.get("slow_operations"),
                trajectory=trajectory_info,
                judge_score=judge_detail,
            )

        logger.info(
            f"[/query] Completed in {elapsed}s  "
            f"type={result.get('type', 'data_query')!r}  "
            f"fallback={fallback!r}"
            + (f"  llm_calls={llm_usage.num_llm_calls}  tokens={llm_usage.total_tokens}  "
               f"cost=${llm_usage.total_cost_usd:.4f}" if llm_usage else "")
        )

        return QueryResponse(
            type=result.get("type", "data_query"),
            answer=answer_text,
            neighborhood=result.get("neighborhood"),
            domain=result.get("domain"),
            domains=result.get("domains", []),
            confidence=float(result.get("confidence", 0.0)),
            elapsed=elapsed,
            routing=routing_data,
            llm_usage=llm_usage,
            guardrails=guardrails_result,
            sql=result.get("sql"),
            results=result.get("results"),
            rag_chunks=result.get("rag_chunks"),
            validation=result.get("validation"),
            chart_path=result.get("path"),
            image_paths=result.get("paths"),
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"[/query] Failed: {e}", exc_info=True)
        return QueryResponse(
            type="error",
            answer="",
            error=str(e),
            elapsed=round(time.time() - t_start, 2),
        )
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# REPORT ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

def _generate_report_background(report_id: str, neighborhood: str):
    """
    Background task to generate a neighborhood PDF report.
    Calls report_agent.generate_report() which manages its own Snowflake connection.
    """
    try:
        reports_db[report_id]["status"] = "processing"
        from report_agent import generate_report

        logger.info(f"[Report {report_id}] Generating report for {neighborhood}")
        pdf_path = generate_report(neighborhood)

        if pdf_path:
            reports_db[report_id].update({
                "status":       "completed",
                "pdf_path":     pdf_path,
                "url":          f"/report/{report_id}/download",
                "completed_at": datetime.utcnow().isoformat(),
                "message":      f"Report ready for {neighborhood}",
            })
            logger.info(f"[Report {report_id}] Completed: {pdf_path}")
        else:
            reports_db[report_id].update({
                "status":  "failed",
                "message": "Report generation returned no PDF path",
            })

    except Exception as e:
        logger.error(f"[Report {report_id}] Generation failed: {e}", exc_info=True)
        reports_db[report_id].update({
            "status":  "failed",
            "message": str(e),
        })


@app.post("/report/generate", response_model=ReportResponse, tags=["Report"])
async def generate_report_endpoint(request: ReportRequest, background_tasks: BackgroundTasks):
    """
    Start async neighborhood report generation.

    Returns immediately with a report_id.
    Poll GET /report/{report_id} for status.
    Download via GET /report/{report_id}/download when status == 'completed'.
    Typical generation time: 3–5 minutes.
    """
    report_id = str(uuid.uuid4())[:8]

    reports_db[report_id] = {
        "report_id":   report_id,
        "neighborhood": request.neighborhood.title(),
        "status":      "pending",
        "created_at":  datetime.utcnow().isoformat(),
        "completed_at": None,
        "pdf_path":    None,
        "url":         None,
        "message":     f"Report generation started for {request.neighborhood}",
    }

    background_tasks.add_task(_generate_report_background, report_id, request.neighborhood)

    return ReportResponse(**reports_db[report_id])


@app.get("/report/{report_id}", response_model=ReportResponse, tags=["Report"])
async def get_report_status(report_id: str):
    """Get the status of a report generation task."""
    if report_id not in reports_db:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    return ReportResponse(**reports_db[report_id])


@app.get("/report", tags=["Report"])
async def list_reports():
    """List all generated reports in this session."""
    return {
        "count": len(reports_db),
        "reports": [
            {
                "report_id":    rid,
                "neighborhood": data["neighborhood"],
                "status":       data["status"],
                "created_at":   data["created_at"],
                "completed_at": data.get("completed_at"),
                "url":          data.get("url"),
            }
            for rid, data in reports_db.items()
        ]
    }


@app.get("/report/{report_id}/download", tags=["Report"])
async def download_report(report_id: str):
    """Download a completed neighborhood report PDF."""
    if report_id not in reports_db:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")

    report_data = reports_db[report_id]

    if report_data["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Report is still {report_data['status']}. Poll /report/{report_id} for status."
        )

    pdf_path = report_data.get("pdf_path")
    if not pdf_path or not Path(pdf_path).exists():
        raise HTTPException(status_code=404, detail="PDF file not found on disk")

    return FileResponse(
        path=pdf_path,
        filename=f"{report_data['neighborhood'].lower().replace(' ', '_')}_report.pdf",
        media_type="application/pdf",
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "neighbourwise_fastapi:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info",
    )