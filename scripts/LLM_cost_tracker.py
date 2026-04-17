"""
llm_cost_tracker.py — NeighbourWise AI
═══════════════════════════════════════════════════════════════════════════════
Token counting, LLM cost-of-thought (CoT) calculator, latency monitoring,
trajectory analysis, and LLM-as-Judge evaluation.

Adapted patterns from Lab 7 (Evaluation & Observation) where applicable:
  ✓ Context-manager latency tracking  (track_latency pattern)
  ✓ Trajectory analysis               (precision/recall of execution path)
  ✓ LLM-as-Judge scoring              (multi-dimension quality evaluation)
  ✗ TruLens / Snowflake Observability  (not applicable — multi-backend arch)

Tracks every LLM call across all three backends:
  - Snowflake Cortex  (mistral-large2, claude-sonnet-4-6)
  - Anthropic direct   (claude-sonnet-4-6)
  - OpenAI direct      (gpt-4o)

Since Snowflake Cortex does not return token usage in its response,
we estimate tokens using the ~4 chars/token heuristic (calibrated for
English text with technical content). Direct API calls return actual
token counts which we use when available.

Usage:
    from llm_cost_tracker import CostTracker

    tracker = CostTracker()

    # Context-manager style (recommended — auto-logs latency + tokens):
    with tracker.track("generate", model="claude-sonnet-4-6") as t:
        t.set_input(prompt)
        resp = anthropic_client.messages.create(...)
        t.set_response(resp, source="anthropic")
    # → automatically logged with actual tokens from response.usage

    # Manual style (for Cortex calls without response objects):
    tracker.log(model="mistral-large2", input_text=prompt, output_text=response)

    # Latency tracking for non-LLM operations:
    with tracker.track_latency("neo4j_query", threshold_ms=3000) as lat:
        results = neo4j_driver.execute(cypher)
    # lat.record.exceeded == True if > 3000ms

    # Trajectory analysis:
    tracker.record_step("classify")
    tracker.record_step("neo4j_query")
    trajectory = tracker.analyze_trajectory(intent="graph_query")

    # LLM-as-Judge (optional — uses an extra LLM call):
    judge_result = tracker.judge_response(query, answer, conn)

    # Full summary:
    summary = tracker.summary()
"""

import time
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


# ══════════════════════════════════════════════════════════════════════════════
# PRICING TABLE (USD per 1K tokens)
# ══════════════════════════════════════════════════════════════════════════════

MODEL_PRICING = {
    # model_name: (input_cost_per_1k, output_cost_per_1k)
    "mistral-large2":       (0.002,  0.006),
    "mistral-large":        (0.002,  0.006),
    "claude-sonnet-4-6":    (0.003,  0.015),
    "claude-3-5-sonnet":    (0.003,  0.015),
    "gpt-4o":               (0.0025, 0.010),
    "gpt-4o-mini":          (0.00015,0.0006),
    "e5-base-v2":           (0.0001, 0.0),
}

DEFAULT_PRICING = (0.003, 0.010)


# ══════════════════════════════════════════════════════════════════════════════
# TOKEN ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    """Estimate token count (~4 chars/token for English + technical text)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LLMCall:
    """Single LLM invocation record."""
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    source: str              # "cortex" | "anthropic" | "openai"
    estimated: bool          # True if tokens were estimated (Cortex)
    purpose: str = ""        # "classify", "generate", "validate", "improve", "rag_embed", "judge"
    latency_s: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class LatencyRecord:
    """Latency measurement for any pipeline operation (not just LLM calls).

    Adapted from Lab 7's track_latency context manager pattern.
    Covers DB queries, Neo4j calls, Serper fetches, RAG retrieval, etc.
    """
    operation: str
    duration_ms: float = 0.0
    threshold_ms: float = 5000.0
    exceeded: bool = False


@dataclass
class TrajectoryStep:
    """Single step in the execution path."""
    node_name: str
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    tokens_used: int = 0


@dataclass
class TrajectoryAnalysis:
    """Result of comparing actual vs expected execution trajectory.

    Adapted from Lab 7's analyze_trajectory with precision/recall scoring.
    """
    actual_path: List[str]
    expected_path: List[str]
    precision: float = 0.0    # fraction of actual steps that were expected
    recall: float = 0.0       # fraction of expected steps that were executed
    is_valid: bool = False
    deviations: List[str] = field(default_factory=list)


@dataclass
class JudgeScore:
    """Multi-dimension quality score from LLM-as-Judge.

    Adapted from Lab 7's judge evaluation. Dimensions are tailored for
    NeighbourWise: factual accuracy replaces "safety" since our validator
    already handles hallucination checks — the judge focuses on answer quality.
    """
    relevance: float = 3.0        # 0-5: Does the answer address the query?
    completeness: float = 3.0     # 0-5: Are all queried domains covered?
    data_grounding: float = 3.0   # 0-5: Does it cite specific numbers/scores?
    overall: float = 3.0          # 0-5: Holistic quality
    reasoning: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# EXPECTED TRAJECTORIES PER INTENT
# ══════════════════════════════════════════════════════════════════════════════
# Adapted from Lab 7's EXPECTED_TRAJECTORY_NORMAL / EXPECTED_TRAJECTORY_BLOCKED.
# We define expected paths per routing intent instead of just normal/blocked.

EXPECTED_TRAJECTORIES: Dict[str, List[str]] = {
    "data_query": [
        "classify",
        "sql_execute",
        "rag_retrieve",
        "generate",
        "validate",
    ],
    "graph_query": [
        "classify",
        "neo4j_query",
        "rag_retrieve",
        "generate",
        "validate",
    ],
    "web_search": [
        "classify",
        "serper_search",
        "url_fetch",
        "generate",
        "validate",
    ],
    "chart": [
        "classify",
        "sql_execute",
        "chart_render",
    ],
    "report": [
        "classify",
        "report_generate",
    ],
    "image": [
        "classify",
        "image_generate",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT MANAGER FOR LLM CALL TRACKING
# ══════════════════════════════════════════════════════════════════════════════

class _TrackedCall:
    """Mutable handle yielded by CostTracker.track() context manager.

    The caller sets input/output text or provides the API response object,
    and the context manager auto-logs everything on exit.
    """

    def __init__(self, tracker: "CostTracker", purpose: str, model: str):
        self._tracker = tracker
        self.purpose = purpose
        self.model = model
        self.input_text: str = ""
        self.output_text: str = ""
        self._response = None
        self._source: Optional[str] = None
        self._start = time.time()

    def set_input(self, text: str):
        """Set the input/prompt text (for Cortex calls without response objects)."""
        self.input_text = text

    def set_response(self, response, source: str = "auto"):
        """Capture API response object for automatic token extraction.

        Args:
            response: Anthropic Message or OpenAI ChatCompletion object.
            source: "anthropic", "openai", or "auto" (inferred from model name).
        """
        self._response = response
        self._source = source if source != "auto" else None

    def set_output(self, text: str):
        """Set the output/completion text (for Cortex calls)."""
        self.output_text = text


class _LatencyContext:
    """Mutable handle yielded by CostTracker.track_latency() context manager.

    Adapted from Lab 7's track_latency pattern. Measures wall-clock time
    for any pipeline operation — DB queries, API calls, file I/O, etc.
    """

    def __init__(self, operation: str, threshold_ms: float):
        self.record = LatencyRecord(
            operation=operation,
            threshold_ms=threshold_ms,
        )
        self._start = time.perf_counter()

    def finalize(self):
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self.record.duration_ms = round(elapsed_ms, 2)
        self.record.exceeded = elapsed_ms > self.record.threshold_ms


# ══════════════════════════════════════════════════════════════════════════════
# LLM-AS-JUDGE PROMPT
# ══════════════════════════════════════════════════════════════════════════════
# Adapted from Lab 7's JUDGE_PROMPT but tailored for neighborhood intelligence.
# Lab 7 uses relevance/helpfulness/safety/overall for customer support.
# We use relevance/completeness/data_grounding/overall for neighborhood queries.

_JUDGE_SYSTEM_PROMPT = """You are a strict quality evaluator for a neighborhood intelligence AI.
You MUST find flaws in every response. A perfect score of 5.0 is virtually impossible.
Your job is to differentiate quality — never give the same score across all dimensions.

Score each dimension independently on a 0-5 scale:

1. **Relevance (0-5)**: Does the response address ONLY the queried neighborhood(s) and domain(s)?
   - 4 max if the response includes generic filler or unnecessary preamble.
   - 3 max if it discusses neighborhoods or domains not asked about.
   - 2 max if it fails to identify the correct neighborhood.

2. **Completeness (0-5)**: Does it cover ALL queried domains with specific data?
   - 4 max if any queried domain is mentioned but lacks specific metrics.
   - 3 max if any queried domain is entirely missing from the response.
   - Deduct 1 point for each domain mentioned without a score or grade.

3. **Data Grounding (0-5)**: Does it cite concrete numbers, scores, and grades?
   - 4 max if it uses vague language ("relatively safe", "somewhat affordable").
   - 3 max if it states scores without grades or vice versa.
   - 2 max if no numeric data is cited at all.
   - Deduct 1 point for each claim that appears fabricated or unverifiable.

4. **Overall (0-5)**: Holistic quality. MUST differ from at least one other score.
   - 4 max if the response exceeds 500 words for a simple single-domain query.
   - 3 max if the tone is generic and doesn't feel neighborhood-specific.

IMPORTANT: Your scores MUST vary across dimensions.

Respond ONLY with JSON (no markdown, no extra text):
{
    "relevance": 0.0,
    "completeness": 0.0,
    "data_grounding": 0.0,
    "overall": 0.0,
    "reasoning": "list each deduction and why"
}"""


# ══════════════════════════════════════════════════════════════════════════════
# COST TRACKER (main class)
# ══════════════════════════════════════════════════════════════════════════════

class CostTracker:
    """
    Accumulates LLM call metrics, latency records, trajectory steps,
    and optional judge scores across a single query lifecycle.

    Thread-safe for read-only summary; log() should be called from a
    single thread. Each FastAPI request gets its own tracker instance.
    """

    def __init__(self):
        self.calls: List[LLMCall] = []
        self.latency_records: List[LatencyRecord] = []
        self.trajectory_steps: List[TrajectoryStep] = []
        self.judge_score: Optional[JudgeScore] = None

    # ── Context-manager LLM call tracking (recommended) ───────────────────

    @contextmanager
    def track(self, purpose: str, model: str = "unknown"):
        """Context manager that times an LLM call and auto-logs on exit.

        Usage:
            with tracker.track("generate", model="claude-sonnet-4-6") as t:
                t.set_input(prompt)
                resp = client.messages.create(model=..., messages=[...])
                t.set_response(resp, source="anthropic")
            # → auto-logged with actual tokens + latency

            # For Cortex (no response object):
            with tracker.track("classify", model="mistral-large2") as t:
                t.set_input(prompt)
                raw = cortex_complete(prompt, conn)
                t.set_output(raw)
        """
        handle = _TrackedCall(self, purpose, model)
        try:
            yield handle
        finally:
            latency = time.time() - handle._start

            if handle._response is not None:
                # Auto-extract tokens from API response
                source = handle._source
                if source is None:
                    source = "openai" if "gpt" in handle.model.lower() else "anthropic"

                if source == "anthropic":
                    self.log_anthropic_response(
                        model=handle.model,
                        response=handle._response,
                        purpose=handle.purpose,
                        latency_s=latency,
                    )
                elif source == "openai":
                    self.log_openai_response(
                        model=handle.model,
                        response=handle._response,
                        purpose=handle.purpose,
                        latency_s=latency,
                    )
                else:
                    self.log(
                        model=handle.model,
                        input_text=handle.input_text,
                        output_text=handle.output_text,
                        purpose=handle.purpose,
                        latency_s=latency,
                    )
            else:
                # Cortex path — estimate from text
                self.log(
                    model=handle.model,
                    input_text=handle.input_text,
                    output_text=handle.output_text,
                    purpose=handle.purpose,
                    latency_s=latency,
                )

    # ── Context-manager latency tracking (non-LLM operations) ─────────────

    @contextmanager
    def track_latency(self, operation: str, threshold_ms: float = 5000.0):
        """Context manager for timing non-LLM operations.

        Adapted from Lab 7's track_latency pattern.

        Usage:
            with tracker.track_latency("neo4j_query", threshold_ms=3000) as lat:
                results = neo4j_driver.execute(cypher)
            print(lat.record.duration_ms)  # e.g. 1234.56
            print(lat.record.exceeded)      # True if > 3000ms
        """
        ctx = _LatencyContext(operation, threshold_ms)
        try:
            yield ctx
        finally:
            ctx.finalize()
            self.latency_records.append(ctx.record)

    # ── Trajectory tracking ───────────────────────────────────────────────

    def record_step(self, node_name: str, tokens: int = 0):
        """Record a trajectory step (pipeline node that was executed).

        Call this at entry of each pipeline stage:
            tracker.record_step("classify")
            tracker.record_step("neo4j_query")
            tracker.record_step("generate")
        """
        self.trajectory_steps.append(TrajectoryStep(
            node_name=node_name,
            tokens_used=tokens,
        ))

    def analyze_trajectory(self, intent: str) -> TrajectoryAnalysis:
        """Compare actual execution path against expected trajectory for this intent.

        Adapted from Lab 7's analyze_trajectory with precision/recall scoring.

        Args:
            intent: The routing intent (data_query, graph_query, web_search, etc.)

        Returns:
            TrajectoryAnalysis with precision, recall, validity, and deviations.
        """
        actual_path = [s.node_name for s in self.trajectory_steps]
        expected_path = EXPECTED_TRAJECTORIES.get(intent, [])

        actual_set = set(actual_path)
        expected_set = set(expected_path)

        # Precision: what fraction of actual steps were expected?
        precision = (
            len(actual_set & expected_set) / len(actual_set)
            if actual_set else 0.0
        )

        # Recall: what fraction of expected steps were executed?
        recall = (
            len(actual_set & expected_set) / len(expected_set)
            if expected_set else 0.0
        )

        # Identify deviations
        deviations = []
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        if missing:
            deviations.append(f"Missing expected steps: {', '.join(sorted(missing))}")
        if extra:
            deviations.append(f"Unexpected steps: {', '.join(sorted(extra))}")

        # Check ordering
        expected_order = [s for s in expected_path if s in actual_path]
        actual_order = [s for s in actual_path if s in expected_set]
        if expected_order != actual_order and len(actual_order) > 1:
            deviations.append("Step ordering differs from expected")

        is_valid = precision >= 0.8 and recall >= 0.8 and len(deviations) <= 1

        return TrajectoryAnalysis(
            actual_path=actual_path,
            expected_path=expected_path,
            precision=round(precision, 3),
            recall=round(recall, 3),
            is_valid=is_valid,
            deviations=deviations,
        )

    # ── LLM-as-Judge evaluation ───────────────────────────────────────────

    def judge_response(
        self,
        user_query: str,
        agent_answer: str,
        conn=None,
        model: str = "claude-sonnet-4-6",
    ) -> JudgeScore:
        """Run LLM-as-Judge evaluation on the final answer.

        Adapted from Lab 7's llm_judge_evaluate. Uses a separate LLM call
        (logged as purpose="judge") to score the response on 4 dimensions.

        The judge model should differ from the generation model to avoid
        self-evaluation bias. Since your generation uses Claude for graph_query
        and Mistral for data_query, Claude is an acceptable judge for Mistral
        outputs and vice versa.

        Args:
            user_query: The original user question.
            agent_answer: The final answer string.
            conn: Snowflake connection (for Cortex calls). None = skip.
            model: Which model to use for judging.

        Returns:
            JudgeScore with per-dimension scores and reasoning.
        """
        eval_input = f"User Question: {user_query}\n\nAI Response: {agent_answer}"

        try:
            t0 = time.time()

            if conn is not None:
                # Use Snowflake Cortex
                from shared.snowflake_conn import cortex_complete
                prompt = f"{_JUDGE_SYSTEM_PROMPT}\n\n{eval_input}"
                raw = cortex_complete(prompt, conn, model=model)
                latency = time.time() - t0
                self.log(
                    model=model,
                    input_text=prompt,
                    output_text=raw,
                    purpose="judge",
                    latency_s=latency,
                )
            else:
                # Use direct Anthropic
                import anthropic
                client = anthropic.Anthropic()
                resp = client.messages.create(
                    model=model,
                    max_tokens=500,
                    system=_JUDGE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": eval_input}],
                )
                latency = time.time() - t0
                self.log_anthropic_response(
                    model=model,
                    response=resp,
                    purpose="judge",
                    latency_s=latency,
                )
                raw = resp.content[0].text.strip()

            # Parse JSON response
            content = raw.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)

            # Coerce reasoning to string (Lab 7 pattern — LLMs sometimes return lists)
            if not isinstance(data.get("reasoning"), str):
                r = data.get("reasoning", "")
                data["reasoning"] = "; ".join(str(x) for x in r) if isinstance(r, list) else str(r)

            self.judge_score = JudgeScore(**data)

        except Exception as e:
            # Graceful fallback — same as Lab 7's approach
            self.judge_score = JudgeScore(
                relevance=3.0,
                completeness=3.0,
                data_grounding=3.0,
                overall=3.0,
                reasoning=f"Judge evaluation fallback: {str(e)[:100]}",
            )

        return self.judge_score

    # ── Manual logging (Snowflake Cortex) ─────────────────────────────────

    def log(
        self,
        model: str,
        input_text: str = "",
        output_text: str = "",
        purpose: str = "",
        latency_s: float = 0.0,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> LLMCall:
        """Log an LLM call. Tokens estimated from text if not provided."""
        estimated = input_tokens is None
        in_tok = input_tokens if input_tokens is not None else estimate_tokens(input_text)
        out_tok = output_tokens if output_tokens is not None else estimate_tokens(output_text)

        pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
        cost = (in_tok * pricing[0] / 1000) + (out_tok * pricing[1] / 1000)

        source = "cortex"
        if "gpt" in model.lower():
            source = "openai"
        elif "claude" in model.lower():
            source = "anthropic" if not estimated else "cortex"

        call = LLMCall(
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=round(cost, 6),
            source=source,
            estimated=estimated,
            purpose=purpose,
            latency_s=round(latency_s, 3),
        )
        self.calls.append(call)
        return call

    # ── Auto-log from Anthropic API response ──────────────────────────────

    def log_anthropic_response(
        self, model: str, response, purpose: str = "", latency_s: float = 0.0,
    ) -> LLMCall:
        """Log from an anthropic.types.Message response (actual token counts)."""
        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "input_tokens", 0) if usage else 0
        out_tok = getattr(usage, "output_tokens", 0) if usage else 0
        return self.log(
            model=model, purpose=purpose, latency_s=latency_s,
            input_tokens=in_tok, output_tokens=out_tok,
        )

    # ── Auto-log from OpenAI API response ─────────────────────────────────

    def log_openai_response(
        self, model: str, response, purpose: str = "", latency_s: float = 0.0,
    ) -> LLMCall:
        """Log from an openai ChatCompletion response (actual token counts)."""
        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        return self.log(
            model=model, purpose=purpose, latency_s=latency_s,
            input_tokens=in_tok, output_tokens=out_tok,
        )

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Full cost/token/latency/trajectory/judge summary for the query lifecycle."""
        total_in = sum(c.input_tokens for c in self.calls)
        total_out = sum(c.output_tokens for c in self.calls)
        total_cost = sum(c.cost_usd for c in self.calls)
        total_latency = sum(c.latency_s for c in self.calls)

        # Aggregate by model
        by_model: Dict[str, Dict[str, Any]] = {}
        for c in self.calls:
            if c.model not in by_model:
                by_model[c.model] = {
                    "calls": 0, "input_tokens": 0, "output_tokens": 0,
                    "cost_usd": 0.0, "estimated": False,
                }
            m = by_model[c.model]
            m["calls"] += 1
            m["input_tokens"] += c.input_tokens
            m["output_tokens"] += c.output_tokens
            m["cost_usd"] = round(m["cost_usd"] + c.cost_usd, 6)
            if c.estimated:
                m["estimated"] = True

        # Aggregate by purpose
        by_purpose: Dict[str, Dict[str, Any]] = {}
        for c in self.calls:
            purpose_key = c.purpose or "unknown"
            if purpose_key not in by_purpose:
                by_purpose[purpose_key] = {"calls": 0, "cost_usd": 0.0, "tokens": 0}
            p = by_purpose[purpose_key]
            p["calls"] += 1
            p["cost_usd"] = round(p["cost_usd"] + c.cost_usd, 6)
            p["tokens"] += c.input_tokens + c.output_tokens

        result = {
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "total_cost_usd": round(total_cost, 6),
            "total_latency_s": round(total_latency, 2),
            "num_llm_calls": len(self.calls),
            "by_model": by_model,
            "by_purpose": by_purpose,
            "calls": [
                {
                    "model": c.model,
                    "source": c.source,
                    "purpose": c.purpose,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "cost_usd": c.cost_usd,
                    "latency_s": c.latency_s,
                    "estimated": c.estimated,
                }
                for c in self.calls
            ],
        }

        # Latency records (non-LLM operations)
        if self.latency_records:
            result["latency_records"] = [
                {
                    "operation": r.operation,
                    "duration_ms": r.duration_ms,
                    "threshold_ms": r.threshold_ms,
                    "exceeded": r.exceeded,
                }
                for r in self.latency_records
            ]
            slow_ops = [r for r in self.latency_records if r.exceeded]
            if slow_ops:
                result["slow_operations"] = [
                    f"{r.operation}: {r.duration_ms:.0f}ms (threshold: {r.threshold_ms:.0f}ms)"
                    for r in slow_ops
                ]

        # Trajectory
        if self.trajectory_steps:
            result["trajectory"] = [s.node_name for s in self.trajectory_steps]

        # Judge score
        if self.judge_score:
            js = self.judge_score
            result["judge_score"] = {
                "relevance": js.relevance,
                "completeness": js.completeness,
                "data_grounding": js.data_grounding,
                "overall": js.overall,
                "reasoning": js.reasoning,
            }

        return result

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"CostTracker({s['num_llm_calls']} calls, "
            f"{s['total_tokens']} tokens, "
            f"${s['total_cost_usd']:.4f})"
        )