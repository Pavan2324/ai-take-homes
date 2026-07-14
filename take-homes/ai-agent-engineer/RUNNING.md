# Running this solution

## 0. Clean slate first (important)

Before your real run, make sure `runs/` and `stubs/outbox/` don't contain leftovers from the
`demo/` pipeline-validation fixtures (those exist because the sandbox this was built in had no
`OPENAI_API_KEY` -- see the `demo/` section below). Mixing them in isn't just clutter: `agent.py`
tags outbox entries with `[conv_id#run_index]`, where `run_index` is "the Nth fresh session for
this conversation_id in this process." A demo run and a real run are different processes, so
their tags for e.g. `conv-03`'s first run can collide (`[conv-03#1]` in both), and `eval.py`
would silently merge fake and real outbox entries under the same conversation.

```bash
rm -f runs/*.demo.r*.json
rm -f stubs/outbox/*.jsonl
```

(If you never ran `demo/run_demo.py` yourself, skip this -- it's only relevant if you're working
from the zip as shipped, which included demo artifacts for reference.)

## 1. Create the environment (uv, Python 3.11)

```bash
uv venv --python 3.11
```

Activate it:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows cmd.exe
.venv\Scripts\activate.bat
```

Install dependencies into that venv:

```bash
uv pip install -r requirements.txt
```

## 2. Set OPENAI_API_KEY

This is an environment variable, not a file in the project -- don't put it in any tracked file,
and don't commit it if you're pushing to your fork.

**Windows PowerShell**

```powershell
# current terminal session only
$env:OPENAI_API_KEY = "sk-..."

# persists across future terminals (current one won't see it until you open a new window)
setx OPENAI_API_KEY "sk-..."

# verify
echo $env:OPENAI_API_KEY
```

**Windows cmd.exe**

```cmd
:: current terminal session only
set OPENAI_API_KEY=sk-...

:: persists across future terminals (current one won't see it until you open a new window)
setx OPENAI_API_KEY "sk-..."

:: verify
echo %OPENAI_API_KEY%
```

**macOS/Linux (bash/zsh)**

```bash
export OPENAI_API_KEY=sk-...   # current session only

# persist: add the line above to ~/.bashrc or ~/.zshrc, then reload
echo 'export OPENAI_API_KEY=sk-...' >> ~/.zshrc && source ~/.zshrc
```

Whichever shell you use, set the key in the **same terminal window** you're about to run
`harness.py` from -- a key set with `setx` or added to `~/.zshrc` won't apply to an already-open
terminal until you start a new one.

## 3. Run it

```bash
python harness.py --repeat 3   # runs all 18 conversations 3x against the real agent
python eval.py                 # scores runs/ + stubs/outbox/ -> compliance/helpfulness/stability
```

If you're invoking through `uv` instead of an activated shell:

```bash
uv run python harness.py --repeat 3
uv run python eval.py
```

Model: `gpt-5.6-terra` (OpenAI's balanced cost/capability tier as of July 2026). To try the
flagship instead, change `MODEL` in `agent.py` to `gpt-5.6-sol`, or `gpt-5.5` for the prior-gen
stable option if you'd rather not be on a model that only just went GA.

## 4. Tests (no API key needed)

```bash
pytest test_policy_engine.py test_agent_smoke.py test_eval_catches_violations.py -v
```

- `test_policy_engine.py` — unit tests for the deterministic guardrails (refund ceiling,
  verification cross-checking, red-flag regex detection).
- `test_agent_smoke.py` — exercises `agent.respond()`'s tool loop with a fake OpenAI client
  (Responses API shape), catching integration bugs without a live model.
- `test_eval_catches_violations.py` — proves `eval.py` actually fails on non-compliant behavior
  (dosage-guessing on ingestion, stacking a refund over the ceiling) rather than passing
  everything.

**Status as of this write-up:** all 30 tests pass, and `harness.py --repeat 3` / `eval.py` have
been run end-to-end against the live model with all current files in place -- see WRITEUP.md's
"Eval results" section for the numbers from that run.

## 5. `demo/` — pipeline validation, not the solution

`demo/fake_llm.py` and `demo/run_demo.py` script plausible agent responses for 6 representative
conversations and run them through the *real* `agent.py`/`policy_engine.py`/stub code (only the
network call to OpenAI is canned), 3x each. This exists because the sandbox this was built in
had no `OPENAI_API_KEY`, so it's how the harness → agent → eval pipeline was validated
end-to-end without a live key, before a real run was possible. It is **not** a substitute for
the real `harness.py --repeat 3` run against the model.

Now that a real run has been completed (see step 4), it's safe to delete `demo/` entirely -- it
served its purpose and isn't part of the graded solution. If you want to regenerate the demo
artifacts anyway: `python demo/run_demo.py`, then `python eval.py` (and see step 0 about not
mixing them with a real run's outbox).
