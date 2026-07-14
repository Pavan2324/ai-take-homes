"""Ensures solution/ is importable regardless of where pytest is invoked from
(ai-agent-engineer/ root, or from inside solution/ itself). Needed because
solution/__init__.py makes it a real package (per SUBMISSION.md's preferred
layout), so pytest imports test files as `solution.test_x` and only puts the
repo root on sys.path -- not solution/ itself, which is where agent.py,
policy_engine.py etc. actually live and where they expect to be imported
from directly (bare `import agent`, not `from solution import agent`).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "solution"))
