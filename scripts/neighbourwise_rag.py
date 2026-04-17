#!/usr/bin/env python3
"""
neighbourwise_rag.py — NeighbourWise AI RAG Pipeline
═════════════════════════════════════════════════════
Unified pipeline for downloading, chunking, embedding, and searching
unstructured documents across all domains (crime, grocery, healthcare, etc.)

All chunks are stored in a single Snowflake table:
    NEIGHBOURWISE_DOMAINS.RAW_UNSTRUCTURED.RAW_DOMAIN_CHUNKS

Each chunk is tagged with a DOMAIN column so multiple domains coexist
in one table and can be searched individually or cross-domain.

COMMANDS:
─────────
  download   Download a PDF from a URL and extract text
  load       Chunk extracted text, embed via Snowflake Cortex, load to Snowflake
  search     Semantic search over loaded chunks (single query or interactive)

EXAMPLES:
─────────
  # Download a PDF
  python3 neighbourwise_rag.py download \\
      --url "https://example.com/report.pdf" \\
      --domain crime \\
      --outdir ./rag_docs

  # Load all .txt files from a domain folder into Snowflake
  python3 neighbourwise_rag.py load \\
      --input ./rag_docs/crime/ \\
      --domain crime

  # Search within a specific domain
  python3 neighbourwise_rag.py search \\
      --domain crime \\
      --query "surveillance cameras Dorchester"

  # Search across ALL domains
  python3 neighbourwise_rag.py search \\
      --domain all \\
      --query "what programs exist in Roxbury"

  # Interactive search mode
  python3 neighbourwise_rag.py search --domain crime

VALID DOMAINS:
──────────────
  crime, grocery, healthcare, housing, schools,
  transit, restaurants, universities, bluebikes
  (or any custom domain name — just be consistent)

SETUP:
──────
  1. pip install snowflake-connector-python pdfplumber requests python-dotenv
  2. Ensure .env file has: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD,
     SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_ROLE

NOTES:
──────
  - Embedding model: e5-base-v2 (requires "passage:" prefix at ingest, "query:" at search)
  - Dedup: load checks which source files are already loaded per domain — safe to re-run
  - Chunking: 1000 chars with 200 overlap, page-boundary aware for PDFs
  - Search: hybrid 65% vector cosine similarity + 35% keyword boost
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import List, Optional

import requests
import snowflake.connector
from dotenv import load_dotenv

load_dotenv()


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_DATABASE      = "NEIGHBOURWISE_DOMAINS"
DEFAULT_SCHEMA        = "RAW_UNSTRUCTURED"
DEFAULT_TABLE         = "RAW_DOMAIN_CHUNKS"
DEFAULT_EMBED_MODEL   = "e5-base-v2"
DEFAULT_CHUNK_SIZE    = 1000    # characters per chunk
DEFAULT_CHUNK_OVERLAP = 200     # overlap between consecutive chunks
DEFAULT_EMBED_BATCH   = 50      # chunks per Cortex embedding call
DEFAULT_INSERT_BATCH  = 200     # rows per Snowflake insert batch
DEFAULT_MIN_CHARS     = 150     # discard chunks shorter than this
DEFAULT_TOP_K         = 5       # search results to return

# E5 model prefix convention (required for correct retrieval calibration)
E5_MODELS         = {"e5-base-v2", "e5-large-v2"}
E5_PASSAGE_PREFIX = "passage: "   # prepended to chunks at ingest
E5_QUERY_PREFIX   = "query: "     # prepended to queries at search

# Fully qualified table name
FQN = f"{DEFAULT_DATABASE}.{DEFAULT_SCHEMA}.{DEFAULT_TABLE}"


# ═════════════════════════════════════════════════════════════════════════════
# SNOWFLAKE CONNECTION
# ═════════════════════════════════════════════════════════════════════════════

def sf_connect():
    """
    Connect to Snowflake using environment variables from .env file.
    Required env vars: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD,
                       SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE
    """
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ.get("SNOWFLAKE_DATABASE", DEFAULT_DATABASE),
        schema=DEFAULT_SCHEMA,
        role=os.environ.get("SNOWFLAKE_ROLE"),
        insecure_mode=True,
        network_timeout=120,
        login_timeout=60,
    )


def ensure_schema(cur):
    """Create RAW_UNSTRUCTURED schema if it doesn't exist."""
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {DEFAULT_DATABASE}.{DEFAULT_SCHEMA};")
    cur.execute(f"USE SCHEMA {DEFAULT_DATABASE}.{DEFAULT_SCHEMA};")
    print(f"  Schema {DEFAULT_DATABASE}.{DEFAULT_SCHEMA} ready.")


def ensure_table(cur):
    """Create the unified RAW_DOMAIN_CHUNKS table if it doesn't exist."""
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {FQN} (
            chunk_id        NUMBER AUTOINCREMENT PRIMARY KEY,
            domain          VARCHAR,
            source_file     VARCHAR,
            chunk_index     NUMBER,
            chunk_text      VARCHAR,
            chunk_embedding VECTOR(FLOAT, 768),
            loaded_at       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        );
    """)
    print(f"  Table {FQN} ready.")


# ═════════════════════════════════════════════════════════════════════════════
# DOWNLOAD — Fetch PDF, extract text, save locally
# ═════════════════════════════════════════════════════════════════════════════

def download_pdf(url: str, outdir: Path) -> Path:
    """Download a PDF from a URL to the output directory."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Derive clean filename from URL
    filename = url.split("/")[-1]
    if not filename.endswith(".pdf"):
        filename += ".pdf"
    filename = requests.utils.unquote(filename)
    filename = re.sub(r'[^\w\-.]', '_', filename)

    pdf_path = outdir / filename
    print(f"  Downloading: {url}")

    resp = requests.get(url, timeout=60, stream=True)
    resp.raise_for_status()

    with open(pdf_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    size_kb = pdf_path.stat().st_size / 1024
    print(f"  Saved: {pdf_path} ({size_kb:.1f} KB)")
    return pdf_path


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF using pdfplumber. Adds page markers."""
    import pdfplumber

    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages_text.append(f"--- Page {i+1} ---\n{text}")

    full_text = "\n\n".join(pages_text)
    print(f"  Extracted {len(pages_text)} pages, {len(full_text)} characters")
    return full_text


def save_extracted_text(text: str, pdf_path: Path) -> Path:
    """Save extracted text as .txt file alongside the PDF."""
    txt_path = pdf_path.with_suffix('.txt')
    txt_path.write_text(text, encoding='utf-8')
    print(f"  Text saved: {txt_path}")
    return txt_path


def cmd_download(args):
    """CLI handler for the 'download' command."""
    outdir = Path(args.outdir) / args.domain
    print(f"\n{'='*60}")
    print(f"  DOWNLOAD — domain: {args.domain}")
    print(f"{'='*60}")

    pdf_path = download_pdf(args.url, outdir)
    text = extract_text_from_pdf(pdf_path)
    save_extracted_text(text, pdf_path)

    print(f"\n  ✓ Done. Text ready for loading at: {outdir}")
    print(f"  Next step: python3 neighbourwise_rag.py load "
          f"--input {outdir} --domain {args.domain}")


# ═════════════════════════════════════════════════════════════════════════════
# TEXT CHUNKING
# ═════════════════════════════════════════════════════════════════════════════

def _char_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Character-window chunking with whitespace-aware splitting."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Try to break at a whitespace boundary
            brk = end
            while brk > start and not text[brk].isspace():
                brk -= 1
            if brk > start:
                end = brk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if overlap < (end - start) else end
    return chunks


def chunk_text(text: str, chunk_size: int, overlap: int,
               filename: str = "") -> List[str]:
    """
    Hybrid chunker: splits on page boundaries (--- Page N ---) for PDF text,
    then sub-chunks large pages with character windowing.
    Falls back to pure character windowing for non-PDF text.
    """
    PAGE_RE = re.compile(r"(?=--- Page \d+ ---)")

    sections = PAGE_RE.split(text)
    if len(sections) <= 1:
        return _char_chunk(text, chunk_size, overlap)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            lines = section.split('\n', 1)
            header = lines[0].strip() if lines[0].startswith('---') else ""
            for j, sub in enumerate(_char_chunk(section, chunk_size, overlap)):
                if j > 0 and header and not sub.startswith('---'):
                    chunks.append(f"{header} (cont.)\n{sub}")
                else:
                    chunks.append(sub)

    return chunks


# ═════════════════════════════════════════════════════════════════════════════
# EMBEDDING via Snowflake Cortex
# ═════════════════════════════════════════════════════════════════════════════

def _add_passage_prefix(text: str, model: str) -> str:
    """Prepend 'passage: ' for e5 models (required for correct calibration)."""
    return (E5_PASSAGE_PREFIX + text) if model in E5_MODELS else text


def embed_batch(cur, texts: List[str], model: str) -> List[list]:
    """Embed a batch of texts in one Snowflake round-trip using UNION ALL."""
    if not texts:
        return []

    parts = []
    for i, text in enumerate(texts):
        safe = _add_passage_prefix(text, model).replace("'", "''")[:2000]
        parts.append(
            f"SELECT {i} AS idx, "
            f"SNOWFLAKE.CORTEX.EMBED_TEXT_768('{model}', '{safe}') AS vec"
        )

    sql = "\nUNION ALL\n".join(parts) + "\nORDER BY idx"
    cur.execute(sql)

    embeddings = []
    for _, vec in cur.fetchall():
        if isinstance(vec, str):
            vec = json.loads(vec)
        embeddings.append(vec)
    return embeddings


def embed_all_chunks(cur, chunks: List[str], model: str,
                     batch_size: int) -> List[list]:
    """Embed all chunks with progress bar and retry logic (3 retries per batch)."""
    total = len(chunks)
    embeddings = []
    t0 = time.time()

    for b_start in range(0, total, batch_size):
        batch = chunks[b_start:b_start + batch_size]

        for attempt in range(1, 4):
            try:
                batch_embs = embed_batch(cur, batch, model)
                break
            except Exception as exc:
                if attempt == 3:
                    raise
                wait = 2 ** attempt
                print(f"\n    Embed retry {attempt}: {exc}")
                time.sleep(wait)

        embeddings.extend(batch_embs)

        # Progress bar
        done = len(embeddings)
        pct = done / total * 100
        elapsed = time.time() - t0
        eta = (elapsed / done * (total - done)) if done > 0 else 0
        sys.stdout.write(
            f"\r  Embedding: {done}/{total} ({pct:.1f}%) ETA {eta:.0f}s   "
        )
        sys.stdout.flush()

    print()
    return embeddings


# ═════════════════════════════════════════════════════════════════════════════
# INSERT to Snowflake
# Uses temp table workaround because Snowflake doesn't allow VECTOR casting
# inside VALUES clauses — we stage as JSON VARCHAR then cast in SELECT.
# ═════════════════════════════════════════════════════════════════════════════

def insert_chunks(cur, conn, domain: str, source_file: str,
                  chunks: List[str], embeddings: List[list],
                  batch_size: int) -> None:
    """Insert chunks with domain tag into the unified table."""
    tmp = f"{DEFAULT_DATABASE}.{DEFAULT_SCHEMA}.TMP_CHUNK_STAGE"
    total = len(chunks)
    done = 0

    cur.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE {tmp} (
            domain         VARCHAR,
            source_file    VARCHAR,
            chunk_index    NUMBER,
            chunk_text     VARCHAR,
            embedding_json VARCHAR
        );
    """)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        rows = [
            (domain.upper(), source_file, start + i, chunk, json.dumps(emb))
            for i, (chunk, emb) in enumerate(
                zip(chunks[start:end], embeddings[start:end])
            )
        ]

        # Stage into temp table (plain VARCHAR)
        cur.executemany(
            f"INSERT INTO {tmp} "
            f"(domain, source_file, chunk_index, chunk_text, embedding_json) "
            f"VALUES (%s, %s, %s, %s, %s)",
            rows,
        )

        # Cast VECTOR in SELECT and insert into final table
        cur.execute(f"""
            INSERT INTO {FQN}
                (domain, source_file, chunk_index, chunk_text, chunk_embedding)
            SELECT
                domain, source_file, chunk_index, chunk_text,
                PARSE_JSON(embedding_json)::VECTOR(FLOAT, 768)
            FROM {tmp};
        """)

        cur.execute(f"TRUNCATE TABLE {tmp};")
        conn.commit()
        done += len(rows)
        sys.stdout.write(f"\r  Inserting: {done}/{total} rows   ")
        sys.stdout.flush()

    cur.execute(f"DROP TABLE IF EXISTS {tmp};")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# LOAD — Chunk, embed, and load text files to Snowflake
# ═════════════════════════════════════════════════════════════════════════════

def cmd_load(args):
    """CLI handler for the 'load' command."""
    input_dir = Path(args.input)
    domain = args.domain.upper()
    model = args.embed_model

    print(f"\n{'='*60}")
    print(f"  LOAD — domain: {domain}")
    print(f"  Source: {input_dir}")
    print(f"  Model: {model}")
    print(f"  Chunk size: {args.chunk_size}, Overlap: {args.chunk_overlap}")
    print(f"  Table: {DEFAULT_TABLE} (unified)")
    print(f"{'='*60}")

    # Find all .txt files in the input directory
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"\n  No .txt files found in {input_dir}.")
        print(f"  Run 'download' first to fetch and extract a PDF.")
        return

    print(f"  Found {len(txt_files)} text file(s)")

    conn = sf_connect()
    cur = conn.cursor()

    try:
        ensure_schema(cur)
        ensure_table(cur)

        # ── Dedup: check which source files are already loaded for this domain
        try:
            cur.execute(
                f"SELECT DISTINCT source_file FROM {FQN} "
                f"WHERE domain = '{domain}';"
            )
            already_loaded = {row[0] for row in cur.fetchall()}
        except Exception:
            already_loaded = set()

        if already_loaded:
            print(f"  Already loaded for {domain}: {len(already_loaded)} file(s)")

        total_chunks = 0
        skipped = 0

        for txt_path in txt_files:
            if txt_path.name in already_loaded:
                print(f"\n  ── {txt_path.name}  ⏭ SKIP (already loaded)")
                skipped += 1
                continue

            print(f"\n  ── {txt_path.name}")
            text = txt_path.read_text(encoding='utf-8')

            chunks = [
                c for c in chunk_text(
                    text, args.chunk_size, args.chunk_overlap, txt_path.name
                )
                if len(c) >= args.min_chars
            ]
            print(f"  {len(chunks)} chunks (min {args.min_chars} chars)")

            if not chunks:
                print("  Skipping — no valid chunks")
                continue

            print(f"  Embedding ({model}, batch={args.embed_batch})...")
            embeddings = embed_all_chunks(cur, chunks, model, args.embed_batch)

            print(f"  Loading to Snowflake...")
            insert_chunks(
                cur, conn, domain, txt_path.name,
                chunks, embeddings, args.insert_batch
            )

            total_chunks += len(chunks)
            print(f"  ✓ {len(chunks)} chunks loaded")

        # Final summary
        cur.execute(
            f"SELECT COUNT(*) FROM {FQN} WHERE domain = '{domain}';"
        )
        domain_rows = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM {FQN};")
        total_rows = cur.fetchone()[0]

        print(f"\n{'='*60}")
        print(f"  ✅ Load complete")
        print(f"  Table: {FQN}")
        print(f"  New chunks this run: {total_chunks}")
        print(f"  Skipped (already loaded): {skipped}")
        print(f"  Domain '{domain}' total: {domain_rows} chunks")
        print(f"  All domains total: {total_rows} chunks")
        print(f"{'='*60}")

    finally:
        cur.close()
        conn.close()


# ═════════════════════════════════════════════════════════════════════════════
# SEARCH — Semantic search over loaded chunks
# ═════════════════════════════════════════════════════════════════════════════

def embed_query(cur, query: str, model: str) -> list:
    """Embed a query string. Prepends 'query: ' for e5 models."""
    q = (E5_QUERY_PREFIX + query) if model in E5_MODELS else query
    safe = q.replace("'", "''")[:2000]
    cur.execute(
        f"SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768('{model}', '{safe}');"
    )
    vec = cur.fetchone()[0]
    if isinstance(vec, str):
        vec = json.loads(vec)
    return vec


def _keyword_terms(query: str) -> list:
    """Extract meaningful words (4+ chars, no stopwords) for keyword boosting."""
    stopwords = {
        "what", "who", "when", "where", "how", "which", "does", "did",
        "the", "and", "for", "are", "was", "were", "from", "with", "that",
        "this", "have", "has", "had", "tell", "about", "give", "list", "find",
        "boston", "cambridge", "somerville", "neighborhood", "area", "city",
    }
    words = re.findall(r"[a-zA-Z]{4,}", query.lower())
    return [w for w in words if w not in stopwords]


def search_chunks(cur, domain: str, query_vector: list, top_k: int,
                  raw_query: str = "",
                  source_filter: Optional[str] = None) -> list:
    """
    Hybrid search: 65% vector cosine similarity + 35% keyword match.
    Use domain='all' to search across all domains.
    """
    vec_json = json.dumps(query_vector)

    # Domain filter: 'all' searches everything
    domain_clause = (
        f"AND domain = '{domain.upper()}'"
        if domain.lower() != 'all' else ""
    )
    source_clause = (
        f"AND source_file ILIKE '%{source_filter}%'"
        if source_filter else ""
    )

    # Keyword boost terms
    terms = _keyword_terms(raw_query) if raw_query else []
    n_terms = len(terms) if terms else 1
    kw_parts = (
        " + ".join(
            [f"IFF(LOWER(chunk_text) ILIKE '%{t}%', 1, 0)" for t in terms]
        )
        if terms else "0"
    )

    sql = f"""
        WITH base AS (
            SELECT
                chunk_id, domain, source_file, chunk_index, chunk_text,
                VECTOR_COSINE_SIMILARITY(
                    chunk_embedding,
                    PARSE_JSON('{vec_json}')::VECTOR(FLOAT, 768)
                ) AS vec_score,
                ({kw_parts}) AS kw_hits,
                ({kw_parts}) / {n_terms}.0 AS kw_score
            FROM {FQN}
            WHERE 1=1 {domain_clause} {source_clause}
        )
        SELECT
            chunk_id, domain, source_file, chunk_index, chunk_text,
            vec_score, kw_score,
            ROUND(kw_hits) AS keyword_matches,
            (vec_score * 0.65 + kw_score * 0.35) AS similarity
        FROM base
        ORDER BY similarity DESC
        LIMIT {top_k};
    """
    cur.execute(sql)
    columns = [col[0].lower() for col in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def print_results(results: list, query: str, top_k: int) -> None:
    """Pretty-print search results."""
    print(f"\n{'═'*60}")
    print(f"  Query : {query}")
    print(f"  Found : {len(results)} result(s)")
    print(f"{'═'*60}\n")

    if not results:
        print("  No matching chunks found.")
        return

    for rank, r in enumerate(results, 1):
        sim = float(r["similarity"])
        vec_s = float(r.get("vec_score", sim))
        kw_s = float(r.get("kw_score", 0))
        kw_hits = int(float(r.get("keyword_matches", 0)))
        domain = r.get("domain", "?")
        text = r["chunk_text"]

        preview = textwrap.fill(
            text[:400] + ("…" if len(text) > 400 else ""),
            width=68,
            initial_indent="  ",
            subsequent_indent="  ",
        )

        print(f"  #{rank}  hybrid: {sim:.4f}  "
              f"(vec={vec_s:.4f}  kw={kw_s:.2f}  hits={kw_hits})")
        print(f"  Domain: {domain}  "
              f"Source: {r['source_file']}  chunk: {r['chunk_index']}")
        print(f"  {'─'*56}")
        print(preview)
        print()


def cmd_search(args):
    """CLI handler for the 'search' command."""
    domain = args.domain
    model = args.embed_model

    conn = sf_connect()
    cur = conn.cursor()

    try:
        if args.query:
            # ── Single query mode ───────────────────────────────────────
            print(f"\n  Embedding query ({model})...", end="", flush=True)
            vec = embed_query(cur, args.query, model)
            print(" done.")

            results = search_chunks(
                cur, domain, vec, args.top_k,
                raw_query=args.query,
                source_filter=args.source,
            )
            print_results(results, args.query, args.top_k)

        else:
            # ── Interactive REPL mode ───────────────────────────────────
            domain_label = (
                domain.upper() if domain.lower() != 'all' else 'ALL DOMAINS'
            )
            print(f"\n{'═'*60}")
            print(f"  NeighbourWise RAG Search — {domain_label}")
            print(f"  Model: {model}  Top-K: {args.top_k}")
            print(f"  Type 'exit' to quit")
            print(f"{'═'*60}\n")

            while True:
                try:
                    query = input("  🔍 Query: ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\n  Goodbye.")
                    break

                if not query or query.lower() in ("exit", "quit", "q"):
                    print("  Goodbye.")
                    break

                try:
                    print("  Embedding...", end="", flush=True)
                    vec = embed_query(cur, query, model)
                    print(" done.")

                    results = search_chunks(
                        cur, domain, vec, args.top_k,
                        raw_query=query,
                        source_filter=args.source,
                    )
                    print_results(results, query, args.top_k)
                except Exception as exc:
                    print(f"\n  ⚠ Error: {exc}\n")

    finally:
        cur.close()
        conn.close()


# ═════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NeighbourWise RAG Pipeline — download, load, search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python3 neighbourwise_rag.py download --url "https://..." --domain crime --outdir ./rag_docs
          python3 neighbourwise_rag.py load --input ./rag_docs/crime/ --domain crime
          python3 neighbourwise_rag.py search --domain crime --query "surveillance cameras"
          python3 neighbourwise_rag.py search --domain all --query "programs in Roxbury"
          python3 neighbourwise_rag.py search --domain crime  (interactive mode)
        """),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── download ──────────────────────────────────────────────────────────
    dl = subparsers.add_parser(
        "download", help="Download PDF and extract text"
    )
    dl.add_argument("--url", required=True,
                    help="URL of the PDF to download")
    dl.add_argument("--outdir", default="./rag_docs",
                    help="Base output directory (default: ./rag_docs)")
    dl.add_argument("--domain", required=True,
                    help="Domain name (crime, grocery, healthcare, etc.)")

    # ── load ──────────────────────────────────────────────────────────────
    ld = subparsers.add_parser(
        "load", help="Chunk, embed, and load to Snowflake"
    )
    ld.add_argument("--input", required=True,
                    help="Directory containing .txt files from download step")
    ld.add_argument("--domain", required=True,
                    help="Domain tag stored with each chunk (e.g. crime, grocery)")
    ld.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                    help=f"Embedding model (default: {DEFAULT_EMBED_MODEL})")
    ld.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                    help=f"Characters per chunk (default: {DEFAULT_CHUNK_SIZE})")
    ld.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                    help=f"Overlap between chunks (default: {DEFAULT_CHUNK_OVERLAP})")
    ld.add_argument("--embed-batch", type=int, default=DEFAULT_EMBED_BATCH,
                    help=f"Chunks per embedding call (default: {DEFAULT_EMBED_BATCH})")
    ld.add_argument("--insert-batch", type=int, default=DEFAULT_INSERT_BATCH,
                    help=f"Rows per insert batch (default: {DEFAULT_INSERT_BATCH})")
    ld.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS,
                    help=f"Min chunk length to keep (default: {DEFAULT_MIN_CHARS})")

    # ── search ────────────────────────────────────────────────────────────
    sr = subparsers.add_parser(
        "search", help="Semantic search over loaded chunks"
    )
    sr.add_argument("--query", default=None,
                    help="Query string (omit for interactive REPL)")
    sr.add_argument("--domain", required=True,
                    help="Domain to search (or 'all' for cross-domain)")
    sr.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                    help=f"Number of results (default: {DEFAULT_TOP_K})")
    sr.add_argument("--source", default=None,
                    help="Filter to specific source file (substring match)")
    sr.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                    help=f"Embedding model (default: {DEFAULT_EMBED_MODEL})")

    args = parser.parse_args()

    if args.command == "download":
        cmd_download(args)
    elif args.command == "load":
        cmd_load(args)
    elif args.command == "search":
        cmd_search(args)


if __name__ == "__main__":
    main()