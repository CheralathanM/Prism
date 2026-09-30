# Submission checklist (Theme 05 participant guide)

| Requirement | Status | Where |
|---|---|---|
| Code repository | ✅ | this repo (commit recorded in RESULTS.md) |
| README: architecture (one diagram), exact setup and run steps | ✅ | README.md, IMPLEMENTATION_PLAN.md, 01_master_architecture.png |
| Model provider / custom agent declaration | ✅ | README.md → "Model / provider declaration" |
| API keys documented, none included | ✅ | README.md, .env.example (placeholders only); secret scan clean |
| One-command reproduction script (install check, configure, run, evaluate) | ✅ | scripts/reproduce.sh (+ fetch_fdb.sh, fetch_piper_voice.sh) |
| Benchmark results and run logs (scores, seeds, configuration) | ⚠️ partial | RESULTS.md, results/submission/ — 34/100 recordings, unofficial exact-match scoring |
| Extension use case (end to end, clearly marked) | ✅ | fdagent/extensions/incar/, README.md → Extension, tests/test_incar_extension.py |
| Demo video (3–5 min) | ⚠️ script ready; video to be recorded by the team | docs/DEMO.md |
| Slide deck (≤ 8 slides) | ✅ Markdown/Marp source (export to PDF/PPTX with `npx @marp-team/marp-cli docs/DECK.md --pdf`) | docs/DECK.md |
| Tests | ✅ | tests/ (Windows and Linux suites) |
| No benchmark answers in source/tests | ✅ | tests/test_integrity.py |
| Third-party licenses noted | ✅ | README.md → Licenses (piper-tts is GPL-3.0-or-later) |
