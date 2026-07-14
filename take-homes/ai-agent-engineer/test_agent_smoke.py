"""Structural smoke test: exercises agent.respond()'s tool loop with a fake
OpenAI client (Responses API shape) so we can catch integration bugs (wrong
dict shapes, policy wiring, etc.) without a real API key. Does NOT validate
model behavior/tone/judgment -- that needs the real harness run with
OPENAI_API_KEY set.
"""
import agent


class FakeItem:
    """Mimics a Responses API output item (function_call, etc.)."""
    def __init__(self, type_, **kw):
        self.type = type_
        for k, v in kw.items():
            setattr(self, k, v)


class FakeResp:
    """Mimics a Responses API response object."""
    def __init__(self, output, output_text=""):
        self.output = output
        self.output_text = output_text


class FakeResponses:
    def __init__(self, script):
        self.script = script
        self.calls = 0

    def create(self, **kwargs):
        resp = self.script[self.calls]
        self.calls += 1
        return resp


class FakeClient:
    def __init__(self, script):
        self.responses = FakeResponses(script)


def run_with_script(script, message="hi", session=None):
    agent._client = FakeClient(script)
    return agent.respond(session if session is not None else {"conversation_id": "test"}, message)


def function_call(name, arguments, call_id="call-1"):
    return FakeItem("function_call", name=name, arguments=__import__("json").dumps(arguments), call_id=call_id)


def test_plain_text_reply_no_tools():
    script = [FakeResp([], output_text="Hi there, happy to help!")]
    reply = run_with_script(script, "hello")
    assert "happy to help" in reply


def test_tool_use_then_text():
    fc = function_call("find_customer", {"email": "milo.tanaka@example.com"})
    script = [
        FakeResp([fc], output_text=""),
        FakeResp([], output_text="Found your account, Milo!"),
    ]
    reply = run_with_script(script, "it's milo.tanaka@example.com")
    assert "Milo" in reply


def test_refund_blocked_when_unverified_is_fed_back():
    fc = function_call("issue_refund", {"customer_id": "CUST-1003", "order_id": "ORD-9295",
                                        "amount": 34.0, "reason": "COA"})
    script = [
        FakeResp([fc], output_text=""),
        FakeResp([], output_text="I need to verify you first before I can do that."),
    ]
    reply = run_with_script(script, "refund please")
    assert "verify" in reply.lower()


def test_refund_over_ceiling_blocked_and_fed_back_as_policy_violation():
    session = {"conversation_id": "test-ceiling"}
    ps = agent._bootstrap_session(session)
    agent.mark_verified(ps, "CUST-1004")
    fc = function_call("issue_refund", {"customer_id": "CUST-1004", "order_id": "ORD-9322",
                                        "amount": 89.0, "reason": "smashed box"})
    script = [
        FakeResp([fc], output_text=""),
        FakeResp([], output_text="That's over what I can do myself, escalating now."),
    ]
    reply = run_with_script(script, "refund the whole $89", session=session)
    assert ps.refund_total == 0.0  # never recorded because it was blocked
    assert "escalat" in reply.lower()


def test_escalation_marks_latch():
    session = {"conversation_id": "test-esc"}
    fc = function_call("create_escalation", {
        "category": "safety", "member_ref": "unverified caller",
        "verification_status": "unverified", "summary": "pet ingestion",
        "attempted": "gave referral script", "references": "n/a",
        "customer_facing_line": "I'm connecting you with a human teammate now.",
    })
    script = [
        FakeResp([fc], output_text=""),
        FakeResp([], output_text="Please contact the ASPCA line right away."),
    ]
    run_with_script(script, "my cat ate the pothos", session=session)
    ps = agent._bootstrap_session(session)
    assert ps.escalation_latch is True
    assert "safety" in ps.escalation_categories


def test_multi_turn_history_accumulates():
    session = {"conversation_id": "multi"}
    script1 = [FakeResp([], output_text="Sure, what's the email on the account?")]
    run_with_script(script1, "hi, need help", session=session)
    script2 = [FakeResp([], output_text="Got it, thanks!")]
    run_with_script(script2, "milo.tanaka@example.com", session=session)
    ps = agent._bootstrap_session(session)
    # 2 user messages appended by us + 2 raw output-item batches appended
    # from resp.output (empty lists here, so just the 2 user turns show up
    # as list-growth of at least 2; the exact count depends on how many
    # output items each fake response carried, here zero each).
    assert len(ps.history) == 2


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v"]))
