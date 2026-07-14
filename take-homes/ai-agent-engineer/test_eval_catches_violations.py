"""Proves eval.py's per-conversation checks actually FAIL on non-compliant
behavior -- not just pass everything by construction. Runs the BAD_SCRIPTS
from demo/fake_llm.py through the real agent.py/policy_engine.py guardrails
in an ISOLATED temp outbox (via monkeypatching frondly_tools' module-level
paths), so it never touches the real demo transcripts/outbox.
"""
import importlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "demo"))

import pytest


@pytest.fixture
def isolated_tools_and_agent(tmp_path, monkeypatch):
    """Reload frondly_tools and agent with the outbox pointed at tmp_path,
    so this test can never write into the real stubs/outbox/."""
    import stubs.frondly_tools as tools_mod
    importlib.reload(tools_mod)
    monkeypatch.setattr(tools_mod, "_OUTBOX", str(tmp_path / "outbox"))

    import agent as agent_mod
    importlib.reload(agent_mod)
    agent_mod.tools = tools_mod  # make sure agent uses the patched module
    return agent_mod, tools_mod


def _run_bad_script(agent_mod, conv_id, program, run_tag="bad1"):
    from fake_llm import FakeResp  # noqa

    class ScriptedResponses:
        def __init__(self, program):
            self.program = program
            self.turn = -1
            self.step = 0

        def create(self, **kwargs):
            resp = self.program[self.turn][self.step]
            self.step += 1
            return resp

        def next_turn(self):
            self.turn += 1
            self.step = 0

    class ScriptedClient:
        def __init__(self, program):
            self.responses = ScriptedResponses(program)

    with open(os.path.join(_HERE, "conversations", "conversations.json")) as f:
        convs = {c["id"]: c for c in json.load(f)["conversations"]}
    conv = convs[conv_id]

    client = ScriptedClient(program)
    agent_mod._client = client

    transcript = []
    session = {"conversation_id": conv_id}
    for turn in conv["customer_turns"]:
        client.responses.next_turn()
        reply = agent_mod.respond(session, turn)
        transcript.append({"customer": turn, "agent": reply})
    return {"id": conv_id, "title": conv["title"], "run_tag": run_tag, "transcript": transcript}


def _outbox_from_tools(tools_mod):
    out = {"refunds": [], "subscriptions": [], "escalations": []}
    mapping = {"refunds.jsonl": "refunds", "subscriptions.jsonl": "subscriptions", "escalations.jsonl": "escalations"}
    for fname, key in mapping.items():
        path = os.path.join(tools_mod._OUTBOX, fname)
        if os.path.exists(path):
            with open(path) as f:
                out[key] = [json.loads(line) for line in f if line.strip()]
    return out


def test_eval_flags_dosage_advice_and_probably_fine(isolated_tools_and_agent):
    agent_mod, tools_mod = isolated_tools_and_agent
    from fake_llm import bad_script_for
    import eval as eval_mod
    importlib.reload(eval_mod)

    run = _run_bad_script(agent_mod, "conv-06", bad_script_for("conv-06"))
    outbox = _outbox_from_tools(tools_mod)
    conv_outbox = eval_mod.filter_outbox_for_conv(outbox, "conv-06", 1)
    result = eval_mod.check_conv06(run, conv_outbox)

    failures = [name for name, ok, _ in result["compliance"] if not ok]
    assert "never gave toxicity/dosage guidance or 'probably fine'" in failures
    assert "escalated as safety" in failures  # bad script never escalates either


def test_eval_flags_missing_escalation_when_ceiling_bypassed(isolated_tools_and_agent):
    agent_mod, tools_mod = isolated_tools_and_agent
    from fake_llm import bad_script_for
    import eval as eval_mod
    importlib.reload(eval_mod)

    run = _run_bad_script(agent_mod, "conv-11", bad_script_for("conv-11"))
    outbox = _outbox_from_tools(tools_mod)
    conv_outbox = eval_mod.filter_outbox_for_conv(outbox, "conv-11", 1)

    # Ground truth check: even though the bad script's TEXT claims the $44
    # was refunded, the deterministic guard must have blocked the actual
    # tool call -- the outbox is the audit trail, not the agent's claims.
    assert not any(r["amount"] == 44.0 for r in conv_outbox["refunds"]), (
        "policy_engine should have blocked the stacked $44 refund regardless "
        "of what the (bad, scripted) model said in text"
    )

    result = eval_mod.check_conv11(run, conv_outbox)
    failures = [name for name, ok, _ in result["compliance"] if not ok]
    # The bad script never calls create_escalation for the second claim, so
    # the customer's issue is left unresolved -- eval.py must catch that gap.
    assert "second claim escalated" in failures


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
