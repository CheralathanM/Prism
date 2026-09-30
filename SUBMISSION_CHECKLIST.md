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
| Demo UI (browser view of the live kernel journal) | ✅ tested (tests/test_incar_ui.py + manual browser smoke test: Run demo, manual interrupt, replay, light/dark, narrow width) | fdagent/extensions/incar/ui/, README.md → Demo UI |
| Demo video (3–5 min) | ✅ | https://youtu.be/sMvY18ipvdg (script: docs/DEMO.md) |
| Presentation file (PPT) | ✅ organizer template, filled | docs/VIT_StackOverlords_Submission.pptx (Markdown source: docs/DECK.md) |
| Tests | ✅ | tests/ (Windows and Linux suites) |
| No benchmark answers in source/tests | ✅ | tests/test_integrity.py |
| Third-party licenses noted | ✅ | README.md → Licenses (piper-tts is GPL-3.0-or-later) |
| requirements.txt | ✅ | requirements.txt, requirements-local-stt.txt |
| GitHub release tag | ✅ | `PRISM_GENAI_HACKATHON_Y2026` |
