# AI tool use

Built with Claude (Claude.ai, this conversation), operating with direct file/bash access to a
sandboxed clone of the take-home repo.

**What was delegated vs written by me:** Everything in this solution — `policy_engine.py`,
`agent.py`, `eval.py`, `test_policy_engine.py`, `test_agent_smoke.py`, and (during development,
later removed once a real run confirmed everything worked) `test_eval_catches_violations.py` and
`demo/` — was drafted by Claude based on my read-through of the guide and requirements, then
reviewed and iterated by both of us in place (I read the diffs and asked for the fixes below; I
did not hand-write code myself in this session).

**Where I (via Claude) rejected/fixed the tool's own output:**
1. The first version of `agent.py` imported `mark_verified` from `policy_engine.py` but never
   called it anywhere — meaning identity verification could never actually be established, so
   every refund/subscription-change tool call would have been unconditionally blocked. This
   wasn't caught by re-reading the code; it surfaced when we ran the scripted demo end-to-end
   and found `stubs/outbox/refunds.jsonl` simply didn't exist. Fix: added a real
   `verify_identity` tool that cross-checks the customer's claimed order number/item against
   actual order data before marking a session verified, plus unit tests for it.
2. Two of `eval.py`'s first-draft regex checks produced false failures against otherwise-correct
   agent behavior (one flagged "not store credit" as if it were an offer of store credit;
   another missed a valid paraphrase of "the member needs to reach out herself"). Both were
   caught by reading the actual failing transcript text rather than trusting the check's intent,
   and narrowed to avoid the false positive without weakening what they actually test for.
3. I asked for the outbox to be exactly attributable per conversation (rather than guessed at by
   customer/order, which breaks when two conversations touch the same customer, as conv-03 and
   conv-13 do) — the first pass used only a `conv_id` tag, which I pointed out would still blur
   together repeated runs of the *same* conversation sharing one outbox file; fixed by adding a
   `run_index` to the tag.
4. I asked for the LLM provider to be switched from Anthropic to OpenAI partway through. Claude
   re-platformed `agent.py` onto OpenAI's Responses API — swapping the client, the tool-schema
   shape (`input_schema` → `parameters`, nested vs. flat function definitions), and the
   turn/output-item handling (`resp.content`/`tool_use`/`tool_result` → `resp.output`/
   `function_call`/`function_call_output`) — while correctly leaving `policy_engine.py` and
   `eval.py` untouched, since neither ever depended on which provider's shapes were in play. It
   also updated the mocked-client tests and demo fixtures to match and re-ran everything to
   confirm no regressions, rather than just asserting the port was correct.

**On the sandbox/real-run gap:** no `OPENAI_API_KEY` was available in the sandbox this was
originally built in, so the pipeline was first validated with scripted fixtures (see WRITEUP.md's
"Eval results" for what that involved and the numbers it produced). Once a real key became
available, the actual 18×3 conversations were run against the live model via
`harness.py --repeat 3` / `eval.py`, and the now-redundant scripted fixtures (`demo/`,
`test_eval_catches_violations.py`) were deleted so the final submission reflects only what's
actually being graded.
