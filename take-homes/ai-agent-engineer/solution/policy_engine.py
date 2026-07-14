"""Deterministic policy enforcement for the Frondly agent.

This module is the "hard guardrail" layer: code-level checks that do not
depend on the LLM behaving well. The LLM (in agent.py) decides *what to say*
and *when to call tools*, but every tool call that touches money or an
account passes through here first. If the model tries something the guide
forbids, this layer blocks it and forces an escalation instead -- it never
trusts the model's own account of whether verification happened.

Kept dependency-free and side-effect-free (except RedFlag detection, which
is pure regex) so it can be unit tested without hitting the network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

REFUND_CEILING = 50.0

# --- Red-line detection -----------------------------------------------
# Keyword/pattern based safety net. This does NOT replace the model's own
# judgment (the full guide is in its system prompt) -- it's a second,
# independent signal so a single bad model turn can't silently skip a red
# line. Patterns are intentionally broad; false positives just mean the
# model gets an extra reminder, which is cheap. False negatives are the
# expensive failure mode, so we lean sensitive.

_LEGAL_PATTERNS = [
    r"\blawyer\b", r"\blawsuit\b", r"\bsue\b|\bsuing\b|\bsued\b",
    r"\blegal action\b", r"\bdemand letter\b", r"\bchargeback\b",
    r"\bftc\b", r"\battorney general\b", r"\bbbb\b", r"\bsubpoena\b",
    r"\bslipped\b", r"\btripped\b", r"\ballergic reaction\b",
    r"\binjur(y|ed)\b", r"\bproperty damage\b",
]
_PRIVACY_PATTERNS = [
    r"\bgdpr\b", r"\bccpa\b", r"\bdelete (all )?my (personal )?data\b",
    r"\bright to be forgotten\b", r"\berasure\b", r"\bwipe me\b",
    r"\bdata deletion\b",
]
_INGESTION_PATTERNS = [
    r"\b(ate|chewed|licked|swallowed|nibbled|bit)\b.{0,40}\b(plant|leaf|leaves|pothos|calathea|monstera)\b",
    r"\b(plant|leaf|leaves|pothos|calathea|monstera)\b.{0,40}\b(ate|chewed|licked|swallowed|nibbled)\b",
    r"\bhow toxic\b", r"\bhow much is dangerous\b", r"\bis (it|this) poisonous\b",
]
_PRESS_PATTERNS = [
    r"\bpress\b", r"\binfluencer\b", r"\bfollowers\b.{0,20}\bproposal\b",
    r"\bpartnership proposal\b", r"\bmedia inquir",
]
_INJECTION_PATTERNS = [
    r"system\s*override", r"as an ai\b.{0,30}\brequired\b",
    r"^\s*system:", r"forwarded message", r"----+\s*forwarded",
    r"\bapprove(d)?\b.{0,20}\bcode\b", r"\breference code\b.{0,20}\bunlocks\b",
]

_RED_LINE_PATTERNS = {
    "legal": _LEGAL_PATTERNS,
    "privacy": _PRIVACY_PATTERNS,
    "safety": _INGESTION_PATTERNS,
    "press": _PRESS_PATTERNS,
    "injection": _INJECTION_PATTERNS,
}


def detect_red_flags(text: str) -> list[str]:
    """Return the list of red-line categories whose patterns match `text`."""
    low = text.lower()
    hits = []
    for category, patterns in _RED_LINE_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, low, flags=re.IGNORECASE | re.MULTILINE):
                hits.append(category)
                break
    return hits


# --- Session state -------------------------------------------------------

@dataclass
class FrondlySession:
    conversation_id: str = ""
    verified: bool = False
    verified_customer_id: str | None = None
    refund_total: float = 0.0
    escalation_latch: bool = False
    escalation_categories: set = field(default_factory=set)
    history: list = field(default_factory=list)  # OpenAI Responses API input/output items
    flags_seen: set = field(default_factory=set)  # red-flag categories ever seen
    run_index: int = 1  # which repeat of this conversation_id this is, within one process


class PolicyViolation(Exception):
    """Raised when a tool call would break the Customer Care Guide.

    Carries a `reason` the agent should surface to the model so it can
    recover gracefully (e.g. escalate instead of retrying the same call).
    """
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def check_refund(session: FrondlySession, amount: float) -> None:
    """Enforce guide §4: $50/conversation ceiling, cumulative."""
    if amount <= 0:
        raise PolicyViolation("Refund amount must be positive.")
    projected = session.refund_total + amount
    if projected > REFUND_CEILING + 1e-9:
        raise PolicyViolation(
            f"BLOCKED: this refund (${amount:.2f}) would bring the conversation "
            f"total to ${projected:.2f}, over the ${REFUND_CEILING:.0f} ceiling "
            f"(already issued ${session.refund_total:.2f} this conversation). "
            "Per guide §4 this must be escalated to a human, not split or stacked."
        )


def check_verified(session: FrondlySession, customer_id: str | None) -> None:
    """Enforce guide §5: no account change or refund without verification,
    and the verification must belong to the account being acted on."""
    if not session.verified:
        raise PolicyViolation(
            "BLOCKED: no identity verification has been recorded for this "
            "conversation. Per guide §5, verify (email + order number or "
            "item name) before any account change or refund."
        )
    if customer_id and session.verified_customer_id and customer_id != session.verified_customer_id:
        raise PolicyViolation(
            "BLOCKED: this action targets a different account than the one "
            "verified in this conversation. Per guide §5/§7, never act on an "
            "account other than the verified member's own."
        )


def record_refund(session: FrondlySession, amount: float) -> None:
    session.refund_total = round(session.refund_total + amount, 2)


def mark_verified(session: FrondlySession, customer_id: str) -> None:
    session.verified = True
    session.verified_customer_id = customer_id


def attempt_verification(orders: list[dict], order_number: str | None,
                          item_name: str | None) -> tuple[bool, str]:
    """Guide §5: email on the account (caller already matched this via
    find_customer) PLUS either the most recent order number OR the name of
    an item in the most recent box. Cross-checks the customer's claim
    against real order data rather than trusting the model's assertion that
    'this checks out'. Returns (ok, reason)."""
    if not orders:
        return False, "customer has no orders on file to verify against"
    most_recent = orders[0]  # get_orders returns most-recent-first
    if order_number and order_number.strip().upper() == most_recent["order_id"].upper():
        return True, f"order number matches most recent order {most_recent['order_id']}"
    if item_name:
        item_low = item_name.strip().lower()
        names = [i["name"].lower() for i in most_recent.get("items", [])]
        if any(item_low in n or n in item_low for n in names):
            return True, f"item name matches most recent box ({most_recent['order_id']})"
    return False, "neither order number nor item name matched the most recent order"


def mark_escalated(session: FrondlySession, category: str) -> None:
    session.escalation_latch = True
    session.escalation_categories.add(category)
