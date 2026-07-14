# Write-up

## Design (five sentences)

The agent runs on OpenAI's `gpt-5.6-terra` via the Responses API function-calling loop, with the
full Customer Care Guide passed as `instructions` so it has the complete policy, not a summary
of it. (Built initially on Claude/Anthropic's Messages API, then re-platformed onto OpenAI's
Responses API on request — the only things that actually changed are the LLM call itself and the
tool-schema/output-item shapes; `policy_engine.py`, the guardrail wrapper, and the eval are
provider-agnostic and didn't need to change.) Every tool call that touches money or an account
passes through a small deterministic layer (`policy_engine.py`) *before* it reaches the
record-only stubs: identity verification is established by a `verify_identity` tool that
cross-checks the customer's claimed order number/item against real order data (not the model's
say-so), and refunds are checked against a cumulative-per-conversation $50 ceiling, both enforced
in code so a single bad model turn can't silently skip them. A regex-based red-flag scanner runs
on every incoming customer message as an independent second signal for the guide's red lines
(legal, privacy, safety, press, prompt-injection) and surfaces a reminder to the model rather
than replacing its judgment. I built it this way because the take-home explicitly grades the
outbox as the audit trail, not the prose — so the things that matter most (did a refund exceed
the ceiling, did an unverified change happen) needed to be enforced somewhere that doesn't
depend on the LLM getting it right every time, while tone, care advice, and judgment calls about
what counts as "resuming normal service" after a red line stay with the model, which is what it's
for.

## Tests

`test_policy_engine.py` (unit tests for the guardrail logic — refund ceiling, verification
cross-checking, red-flag detection, plus 4 regression tests pinning the exact eval-bug cases
from the debugging arc below — no network needed) and `test_agent_smoke.py` (exercises
`agent.py`'s tool-use loop against a fake OpenAI client, catching wiring bugs without a live
model) — **32/32 passing** on a clean checkout. Full output in `RUNNING.md`.

## Eval results (see `eval.py`)

`eval.py` scores policy compliance and helpfulness as separate lists of named checks per
conversation, plus a stability signature (do repeated runs make the same key decisions —
same refund/no-refund, same escalate/no-escalate). The sandbox this was originally built in had
no API key, so the pipeline was first validated with a scripted `demo/` fixture (6 representative
conversations run through the real `agent.py`/`policy_engine.py`/stub code, only the network call
canned) before a live key was available — that got 45/45 compliance checks, 18/18 helpfulness
checks, 6/6 stable, and a separate test proved `eval.py` actually fails deliberately non-compliant
scripts rather than rubber-stamping everything. Once a real `OPENAI_API_KEY` was available and
`harness.py --repeat 3` / `eval.py` were run for real against all 18 conversations, `demo/` and
its dependent test were deleted — they'd served their purpose and weren't part of the graded
solution.

**Real 18-conversation, 3-run results (final):**

```
Policy compliance:  105/105 (100.0%)
Helpfulness:        74/75 (98.7%)
Stable conversations: 18/18
```

Getting here took three rounds of scrutinizing `eval.py` itself against real transcripts, not
just re-running it and hoping — worth recounting honestly because it's the actual eval-discipline
story:

1. **First real run**: 94/105 compliance (89.5%), 68/72 helpfulness (94.4%), 16/18 stable. Root
   cause: GPT-5.6 writes curly Unicode apostrophes (`can't`) by default, and every negation check
   in `eval.py` was hardcoded to the ASCII straight apostrophe — so `"can't" not in text` never
   matched, and a correct refusal read as a violation. Fixed once at the text-extraction source.
   Also fixed conv-17's escalation-category filter being too narrow, and conv-18's check actively
   contradicting my own agent instructions (it flagged an explicitly-requested cancellation as a
   violation, when the agent's own system prompt says fulfilling separate, low-risk requests
   alongside an escalated matter is fine).
2. **Second run, after those fixes**: 104/105 compliance, 74/75 helpfulness, 17/18 stable — one
   new failure on conv-08, different phrasing than before ("Order notes are customer-provided
   content, so I ca[n't]..." didn't match my narrow negation whitelist). Rather than add yet
   another specific phrase to a growing whitelist, redesigned 5 checks (conv-08, 11, 13, 15, 16)
   to look for **affirmative confirmation** of the bad thing instead of "mentions the topic minus
   known denials."
3. **Third run**: still one failure, and it exposed a real bug in round 2's own fix — the trigger
   word `"i can"` is a literal substring of `"i can't"`, so a correct refusal ("I can't convert
   your points to cash") matched my "affirmative" pattern anyway. Replaced the whole approach with
   one shared helper, `_confirms_unnegated()`: find the topic phrase, then check whether a
   negation word appears in the ~40 characters immediately before it. This doesn't depend on
   guessing every way the model might phrase a refusal, only on whether a negation word actually
   precedes the topic — a much smaller, more general thing to get right than an ever-growing
   whitelist. Added 4 regression tests pinning the exact failure cases from rounds 2 and 3
   (including "I can't convert points to cash" and "we don't cover vet bills") so this class of
   bug can't silently reappear.
4. **Fourth run, with the final fix**: the clean 105/105 / 74/75 / 18/18 above.

I'm flagging this whole arc rather than just showing the final number because "the eval said
100%" is a much weaker claim than "the eval was wrong three times, in three different specific
ways, and each time the fix was verified against the exact transcript that broke it before being
trusted again."

**One genuine helpfulness miss remains, left as a finding rather than eval-massaged away:**

- **conv-17, run 2**: the agent didn't fix the unrelated address-abbreviation typo in one of
  three runs, after correctly declining the forwarded "VP" refund instruction in all three. This
  is the empirical face of the "does not resume normal service" judgment call discussed below —
  the instruction is inherently a judgment call about scope, and this shows the model doesn't
  apply it with 100% consistency turn to turn, which is a fair thing for a stricter production
  system to want to tighten (e.g. with a more explicit affirmative instruction, or a second pass
  that checks for unaddressed benign sub-requests before ending the turn).


## Judgment calls worth flagging

- **"Does not resume normal service"** (guide §7): I read this as *don't keep discussing/
  re-litigating the disputed matter itself*, not *refuse everything else in the conversation*
  — so in conv-08 the agent still answers "when's my next box" after refusing the injected
  refund, and in conv-17 it still fixes an address abbreviation after declining the forwarded
  "VP" refund. A stricter literal reading would block those too; I think that's worse for the
  customer with no compliance upside, but it's a real interpretation call, not an obvious one.
- Verification failures for third parties (the "husband," the roommate) get a quiet
  `create_escalation(category="verification")` for pattern-tracking, but the customer-facing
  line is a plain refusal, not "I'm connecting you to a human" — there's nothing for a human
  to *do* for that caller, and promising a handoff would reward the pressure tactic.

## What we're proud of

A few things about *how* this was built, not just what it does:

- **The outbox is treated as ground truth, never the transcript text.** `eval.py` doesn't infer
  "a refund happened" from the agent saying so — it checks `stubs/outbox/`. We proved this
  mattered during development: a scripted "bad" model that claimed in text it refunded $44 over
  the ceiling was shown to have had the *tool call itself* blocked by `policy_engine.py`
  regardless — the eval caught the gap because it never trusted the prose. (That proof lived in
  `test_eval_catches_violations.py`, removed once a real run confirmed the pipeline worked —
  see "Eval results" above.)
- **We tried to break our own eval before trusting its scores, repeatedly.** A green eval that's
  never seen a failing case is barely more trustworthy than no eval — so before the first live
  run, we wrote deliberately non-compliant scripts and confirmed `eval.py` actually failed them.
  That discipline kept paying off after the first live run too: three separate rounds of eval bugs
  surfaced against real transcripts (a Unicode-apostrophe issue, then a trigger-word whitelist that
  broke on new phrasing, then a subtler version of the same bug where the trigger word `"i can"`
  turned out to be a literal substring of its own negation, `"i can't"`). Each got root-caused
  against the exact transcript that broke it, fixed with a more general mechanism instead of a
  one-off patch, and pinned down with a regression test — see "Eval results" for the full arc.
- **A real bug only surfaced by running the thing, not by reading the code.** `agent.py` imported
  `mark_verified` but never called it — every refund would've been silently blocked forever. It
  read fine on review; it only broke visibly once we ran the pipeline end-to-end and noticed
  `refunds.jsonl` didn't exist.
- **Guardrails don't trust the model's self-report anywhere it doesn't have to.**
  `verify_identity` cross-checks the customer's claimed order/item against real order data
  instead of accepting "I've verified them" from the LLM; the refund ceiling is tracked in code
  as a running total, not asked of the model each time.
- **The provider swap (Anthropic → OpenAI) touched only the LLM-call layer.** `policy_engine.py`
  and `eval.py` — the parts that actually decide what's compliant — needed zero changes, which
  is the outcome you'd want if the guardrail logic were properly decoupled from "which model is
  talking" in the first place, rather than something we got lucky on.

## On retrieval / RAGAS (and why not here)

Considered and deliberately not built: a retrieval layer (vector DB + chunked guide) with
RAGAS-style eval (context precision/recall, faithfulness, answer relevancy). The Customer Care
Guide is one ~230-line file that fits entirely in a single system prompt call, every turn —
there's no context-window pressure and nothing worth chunking. Retrieval only pays for itself
when you *can't* fit the source material in context; here it would add real risk (an
embedding-similarity miss could silently drop a red-line clause the model needed) for zero
benefit, since full-context already guarantees the model sees every rule on every turn. RAGAS's
core metrics score *retrieval quality* specifically — with no retrieval step, there's nothing
for them to meaningfully measure. This would flip if Frondly's policy surface grew into a large,
multi-document knowledge base (say, per-product-line guides, a support-ticket history corpus,
etc.) — at that point I'd add a retrieval layer and RAGAS would be the right eval to bolt on
top of the existing compliance/helpfulness/stability checks, not a replacement for them.

## What I'd harden next

One concrete thing from the real run, not a hypothetical: conv-17's dropped address-fix is a
"benign side-request" miss in one run out of three — worth a targeted fix (e.g. an explicit
end-of-turn check: "is there an unaddressed, low-risk request in this conversation?") rather than
accepting ~99% helpfulness as the ceiling. I'd also re-run the same eval on `gpt-5.6-sol`
(flagship) to see whether the extra reasoning depth closes that gap, before locking in the cheaper
`gpt-5.6-terra` tier for production. Beyond that: outbox entries aren't tagged by conversation by
the stubs themselves, so I had to stamp a `[conv-id#run]` tag onto free-text fields to make
`eval.py` exact instead of heuristic — in a real system I'd add that to the tool contract
directly. I'd also add an LLM-judge pass for tone/tightness (the regex-based helpfulness checks
are blunt — see how many rounds of fixing *the eval itself* it took to get trustworthy numbers
above), and adversarial-test the red-flag regexes against paraphrased legal/ingestion language
they might miss.

## AI-tool disclosure

Built with Claude (this conversation) end-to-end: architecture, `policy_engine.py`, `agent.py`,
`eval.py`, and all tests were drafted by Claude and iterated in place. One clear case of
rejecting/fixing the tool's own output: the first draft of `agent.py` imported `mark_verified`
but never called it, meaning every refund would have been unconditionally blocked — caught by
actually running a scripted validation pipeline end-to-end and seeing an empty `refunds.jsonl`,
not by inspection. Fixed by adding the `verify_identity` tool with real cross-checking rather
than trusting the model's assertion that verification happened. The eval's regex checks also had
two false failures on first run (flagging "not store credit" language as if it were an offer of
store credit, and missing a valid paraphrase of the verification refusal) — both fixed after
reviewing the actual transcripts rather than assuming the checks were right. Midway through,
I asked for the LLM provider to be swapped from Anthropic to OpenAI; Claude re-platformed
`agent.py` onto the Responses API (function-tool schema, `output`/`output_text` item shapes,
`function_call_output` instead of Anthropic's `tool_result` blocks) and correctly identified
that `policy_engine.py`, the tool wrapper, and `eval.py` needed no changes since they never
touched the provider-specific shapes — then updated the mocked tests and the validation fixtures
to match the new response shape and re-ran the full suite to confirm nothing regressed. Once a
real `OPENAI_API_KEY` became available and the actual 18×3 harness run was completed, the
scripted validation fixtures (`demo/`, `test_eval_catches_violations.py`) were deleted at my
request, since they'd served their purpose and keeping temporary validation scaffolding in the
final submission isn't good hygiene.
