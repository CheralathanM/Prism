# PARTIAL local validation run — NOT a completed 100-example benchmark

- Commit: 4594f72 (clean tree). Harness: official FDB-v3 run_tool_benchmark.py, one fresh agent per recording.
- Stopped deliberately by the operator on 2026-09-30T14:05:51Z because of the submission deadline.
- Completed recordings: 34 of 100 (alphabetical order; ecommerce_* and the first finance_* recordings).
- Any recording without a "done" marker was interrupted and is excluded from results.
- Scoring in summary.json is the harness's evaluate_scenario_pass with use_llm=False (exact match):
  UNOFFICIAL. The official gpt-4o LLM judge was not run.
- Local-only harness workaround: CPU shim (cpu_shim/) because this machine has no CUDA GPU.
