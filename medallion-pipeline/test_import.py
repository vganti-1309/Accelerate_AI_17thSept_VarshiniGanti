"""
test_import.py
----------------
Quick smoke test: verifies every module in core/ and agents/ imports
cleanly, and that app/main.py imports without executing Streamlit UI code.
Run with: `python test_import.py`
"""

import sys
import traceback


MODULES = [
    "core.llm",
    "core.pipeline",
    "core.sttm",
    "core.hitl",
    "agents.profiler",
    "agents.sttm_generator",
    "agents.bronze_agent",
    "agents.silver_agent",
    "agents.gold_agent",
    "agents.orchestrator",
    "agents.reporter",
    "app.main",
]


def run() -> int:
    failures = []
    for mod in MODULES:
        try:
            __import__(mod)
            print(f"OK   {mod}")
        except Exception:
            failures.append(mod)
            print(f"FAIL {mod}")
            traceback.print_exc()

    if failures:
        print(f"\n{len(failures)} module(s) failed to import: {failures}")
        return 1

    print(f"\nAll {len(MODULES)} modules imported successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
