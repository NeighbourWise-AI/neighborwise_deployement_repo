"""
neighbourwise_guardrails.py — NeighbourWise AI
═══════════════════════════════════════════════════════════════════════════════
Input and output guardrails adapted from Lab 7 (Evaluation & Observation).

Input guardrails (run BEFORE routing — fast, no LLM call):
  1. Sanitization      — trim, length cap, strip control chars
  2. Keyword prefilter — regex-based jailbreak + toxic pattern detection
  3. Off-topic check   — reject queries unrelated to neighborhoods/Boston

Output guardrails (run AFTER generation — fast, no LLM call):
  1. PII redaction     — email, phone, SSN, credit card
  2. Hallucination markers — fabricated scores, fake neighborhoods, invented stats
  3. Empty/error check — catch empty answers or error leakage

Design decisions vs Lab 7:
  - No LLM-based policy check (Lab 7's llm_policy_check) — adds 5-10s latency
    via Cortex and we're already slow. Regex + keyword is sufficient for our use case.
  - No brand safety check — not applicable (we're not a commercial product).
  - Added neighbourhood-specific hallucination patterns (fabricated scores, etc.).
  - Added off-topic detection tuned for Boston neighborhood queries.

Usage:
    from neighbourwise_guardrails import validate_input, validate_output

    # Before routing
    input_check = validate_input(user_query)
    if not input_check["is_valid"]:
        return {"error": input_check["blocked_reason"]}

    # After generation
    output_check = validate_output(answer_text)
    safe_answer = output_check["filtered_output"]
"""

import re
from typing import Dict, Any, List, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# INPUT GUARDRAILS
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Jailbreak Patterns (adapted from Lab 7) ──────────────────────────────

BLOCKED_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?previous\s+instructions",
    r"(?i)you\s+are\s+now\s+(?:a|an|the)\s+",
    r"(?i)pretend\s+you\s+are",
    r"(?i)act\s+as\s+(?:a|an)\s+",
    r"(?i)system\s*prompt",
    r"(?i)reveal\s+your\s+(instructions|prompt|system)",
    r"(?i)sudo\s+mode",
    r"(?i)developer\s+mode",
    r"(?i)bypass\s+(safety|content|filter)",
    r"(?i)DAN\s+mode",
    r"(?i)jailbreak",
]

# ── 2. Toxicity Patterns (adapted from Lab 7) ───────────────────────────────

TOXIC_PATTERNS = [
    r"(?i)\b(kill|murder|attack|bomb|weapon|hack|exploit)\b",
    r"(?i)\b(stupid|idiot|dumb)\b.*\b(bot|ai|assistant|system)\b",
    r"(?i)\b(fuck|shit|bitch|asshole|damn)\b.*\b(bot|ai|assistant|system|you)\b",
]

# ── 3. Off-Topic Detection (new — tailored for NeighbourWise) ───────────────
# Queries must relate to Boston/Greater Boston neighborhoods, livability,
# or the 9 domains. Reject clearly unrelated requests.

_NEIGHBOURHOOD_SIGNALS = [
    # Domain keywords
    r"(?i)\b(safe|safety|crime|incident|violent)\b",
    r"(?i)\b(hous|rent|afford|price|property|mortgage|apartment)\b",
    r"(?i)\b(restaurant|dining|food|eat|bar|cafe)\b",
    r"(?i)\b(hospital|health|medical|doctor|clinic|healthcare)\b",
    r"(?i)\b(school|education|student|university|college)\b",
    r"(?i)\b(grocery|supermarket|market|store|shopping)\b",
    r"(?i)\b(transit|mbta|subway|bus|train|commut|transport)\b",
    r"(?i)\b(bike|bluebike|cycling|bicycle)\b",
    r"(?i)\b(weather|temperature|climate|rain|snow)\b",
    # Neighbourhood / location signals
    r"(?i)\b(neighborhood|neighbourhood|area|district|suburb|town|city)\b",
    r"(?i)\b(boston|allston|brighton|dorchester|roxbury|cambridge|somerville)\b",
    r"(?i)\b(brookline|newton|quincy|chelsea|revere|malden|medford|everett)\b",
    r"(?i)\b(jamaica plain|back bay|beacon hill|south end|fenway|north end)\b",
    r"(?i)\b(south boston|east boston|charlestown|hyde park|roslindale|mattapan)\b",
    r"(?i)\b(mission hill|west roxbury|chinatown|downtown|west end|bay village)\b",
    r"(?i)\b(salem|beverly|peabody|arlington|belmont|watertown|lexington|milton)\b",
    # Livability / comparison signals
    r"(?i)\b(livab|compar|rank|score|grade|best|worst|top|recommend)\b",
    r"(?i)\b(move|relocat|live in|living in|moving to)\b",
    r"(?i)\b(report|overview|summary|analyze|analysis)\b",
]

# Hard off-topic patterns — definitely not neighbourhood queries
_OFF_TOPIC_HARD = [
    r"(?i)^(write|compose|create)\s+(a\s+)?(poem|story|essay|song|code|script)",
    r"(?i)^(help me|can you)\s+(code|program|debug|write code)",
    r"(?i)\b(python|javascript|java|html|css|sql)\s+(code|script|program|function)\b",
    r"(?i)^(what is|explain|define)\s+(quantum|relativity|photosynthesis|mitosis)",
    r"(?i)^(tell me a joke|sing|play a game|trivia)",
    r"(?i)\b(recipe|cook|bake|ingredients)\b",
    r"(?i)\b(stock|crypto|bitcoin|invest|trading)\b.*\b(price|buy|sell)\b",
]


def _is_off_topic(text: str) -> bool:
    """Return True if the query is clearly unrelated to neighborhood intelligence."""
    # Check for hard off-topic patterns first
    for pattern in _OFF_TOPIC_HARD:
        if re.search(pattern, text):
            return True

    # Check for any neighbourhood signal — if found, it's on-topic
    for pattern in _NEIGHBOURHOOD_SIGNALS:
        if re.search(pattern, text):
            return False

    # Short queries without clear signals — give benefit of the doubt
    if len(text.split()) <= 5:
        return False

    # Longer queries with zero neighbourhood signals are likely off-topic
    return True


# ── 4. Input Sanitization (adapted from Lab 7) ──────────────────────────────

MAX_INPUT_LENGTH = 2000


def _sanitize(text: str) -> str:
    """Trim, cap length, strip control characters."""
    text = text.strip()
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    if len(text) > MAX_INPUT_LENGTH:
        text = text[:MAX_INPUT_LENGTH] + "... [truncated]"
    return text


# ── 5. Combined Input Validation ─────────────────────────────────────────────

def validate_input(text: str) -> Dict[str, Any]:
    """
    Run the full input guardrail pipeline.

    Returns:
        {
            "is_valid": bool,
            "sanitized_input": str,
            "blocked_reason": str or None,
            "violations": list[str],
        }
    """
    sanitized = _sanitize(text)

    # Step 1: Jailbreak patterns
    violations = []
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, sanitized):
            violations.append(f"jailbreak: {pattern}")

    if violations:
        return {
            "is_valid": False,
            "sanitized_input": sanitized,
            "blocked_reason": "Your message was flagged as a potential prompt injection. "
                              "Please rephrase your question about Boston neighborhoods.",
            "violations": violations,
        }

    # Step 2: Toxicity patterns
    for pattern in TOXIC_PATTERNS:
        if re.search(pattern, sanitized):
            violations.append(f"toxic: {pattern}")

    if violations:
        return {
            "is_valid": False,
            "sanitized_input": sanitized,
            "blocked_reason": "Your message contains inappropriate language. "
                              "Please rephrase your question respectfully.",
            "violations": violations,
        }

    # Step 3: Off-topic check
    if _is_off_topic(sanitized):
        return {
            "is_valid": False,
            "sanitized_input": sanitized,
            "blocked_reason": "This question doesn't appear to be about Boston neighborhoods. "
                              "I can help with safety, housing, transit, restaurants, schools, "
                              "healthcare, grocery, and BlueBikes information for Greater Boston areas.",
            "violations": ["off_topic"],
        }

    return {
        "is_valid": True,
        "sanitized_input": sanitized,
        "blocked_reason": None,
        "violations": [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT GUARDRAILS
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. PII Detection & Redaction (from Lab 7) ───────────────────────────────

PII_PATTERNS = {
    "email":       r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone":       r'\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b',
    "ssn":         r'\b\d{3}-\d{2}-\d{4}\b',
    "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
}


def _redact_pii(text: str) -> Tuple[str, bool, List[str]]:
    """Detect and redact PII patterns."""
    pii_found = False
    pii_types = []
    redacted = text
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, redacted):
            pii_found = True
            pii_types.append(pii_type)
            redacted = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted)
    return redacted, pii_found, pii_types


# ── 2. Hallucination Markers (new — tailored for NeighbourWise) ──────────────
# These catch common LLM fabrication patterns specific to our domain.

HALLUCINATION_PATTERNS = [
    # Fabricated scores — X/10 or X/5 format (our scores are 0-100)
    (r"(?i)\b(\d{1,3}(?:\.\d+)?)\s*/\s*(?:10|5)\b",
     "Suspicious score format (X/10 or X/5) — our scores are 0-100"),
    # Fake neighbourhood names
    (r"(?i)\b(north allston|south brighton|east dorchester|west fenway|"
     r"upper roxbury|lower beacon|central cambridge)\b",
     "Potentially fabricated sub-neighbourhood name"),
    # Hedging language around data points
    (r"(?i)(?:I\s+think|I\s+believe|probably|perhaps|maybe|might\s+be|"
     r"could\s+be)\s+.{0,30}(?:score|grade|rank|rate|rent|price|incident)",
     "Hedging language around data — scores should be stated definitively"),
    # Invented URLs or phone numbers in the answer
    (r"(?i)(?:visit|call|contact)\s+(?:us\s+at\s+)?(?:www\.|http|1-\d{3})",
     "Fabricated contact information"),
]


def _detect_hallucination_markers(text: str) -> List[str]:
    """Detect phrases suggesting the LLM fabricated neighbourhood data."""
    found = []
    for pattern, description in HALLUCINATION_PATTERNS:
        if re.search(pattern, text):
            found.append(description)
    return found


# ── 3. Error Leakage Check ──────────────────────────────────────────────────

_ERROR_LEAK_PATTERNS = [
    r"(?i)traceback\s+\(most\s+recent",
    r"(?i)snowflake\.connector\.\w+Error",
    r"(?i)neo4j\.\w+Error",
    r"(?i)anthropic\.\w+Error",
    r"(?i)openai\.\w+Error",
    r"(?i)api[_\s]?key",
    r"(?i)password\s*[:=]",
    r"(?i)SNOWFLAKE_(?:ACCOUNT|USER|PASSWORD)",
]


def _detect_error_leakage(text: str) -> List[str]:
    """Detect stack traces or credential leakage in the output."""
    found = []
    for pattern in _ERROR_LEAK_PATTERNS:
        if re.search(pattern, text):
            found.append(f"Error/credential leakage detected: {pattern}")
    return found


# ── 4. Combined Output Validation ────────────────────────────────────────────

def validate_output(text: str) -> Dict[str, Any]:
    """
    Run the full output guardrail pipeline.

    Returns:
        {
            "is_safe": bool,
            "filtered_output": str,
            "issues": list[str],
            "pii_detected": bool,
            "pii_types": list[str],
            "hallucination_markers": list[str],
        }
    """
    if not text or not text.strip():
        return {
            "is_safe": False,
            "filtered_output": "",
            "issues": ["Empty response generated"],
            "pii_detected": False,
            "pii_types": [],
            "hallucination_markers": [],
        }

    issues = []

    # Step 1: PII redaction
    redacted, pii_found, pii_types = _redact_pii(text)
    if pii_found:
        issues.append(f"PII redacted: {', '.join(pii_types)}")

    # Step 2: Hallucination markers
    hallucination_markers = _detect_hallucination_markers(redacted)
    issues.extend(hallucination_markers)

    # Step 3: Error leakage
    error_leaks = _detect_error_leakage(redacted)
    if error_leaks:
        issues.extend(error_leaks)
        # Scrub the output if credentials are leaking
        redacted = re.sub(
            r"(?i)(SNOWFLAKE_\w+|api[_\s]?key|password)\s*[:=]\s*\S+",
            "[CREDENTIAL_REDACTED]",
            redacted,
        )

    return {
        "is_safe": len(issues) == 0,
        "filtered_output": redacted,
        "issues": issues,
        "pii_detected": pii_found,
        "pii_types": pii_types,
        "hallucination_markers": hallucination_markers,
    }