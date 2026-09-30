# Full FDB-v3 run summary (commit 4594f72, unofficial exact-match scoring)

```
{
  "recordings": 34,
  "completed": 31,
  "scored": 30,
  "exact_match_pass": 17,
  "exact_match_fail": 13,
  "categories": {
    "pass": 17,
    "harness/infrastructure": 4,
    "tool selection/arguments": 6,
    "planner": 1,
    "STT/transcription": 6
  },
  "identical_duplicate_calls": 0,
  "extra_calls": 3,
  "recordings_with_premature_dispatch": 1,
  "rollback_outcomes": {
    "corrected_only": 3
  },
  "barge_in_recordings": 11,
  "barge_in_events": 15,
  "stale_speech_dropped": 1,
  "speech_captured": 25,
  "tts_speech_failed_events": 6,
  "tts_stream_fallbacks": 0,
  "perceived_latency_s": {
    "n": 25,
    "median": 7.52,
    "mean": 10.81,
    "p90": 21.52,
    "min": 2.56,
    "max": 28.4
  },
  "planner": {
    "attempts": 138,
    "status_counts": {
      "200": 138
    },
    "retried_attempts": 1,
    "http_429": 0,
    "reasoner_failures": 0,
    "slowest_attempt_s": 19.384
  },
  "transcription": {
    "timeouts": 9,
    "failures": 1,
    "deadline_extensions": 137
  },
  "harness_issues": [
    [
      "ecommerce_02_5f4a4da1575d605c43bef871",
      "completed",
      [
        "abnormally_long_recording(host_suspend?)",
        "job_assignment_timeout_or_host_suspend",
        "no_speech_captured"
      ]
    ],
    [
      "ecommerce_04_6998abd731d2ec50d067d5bd",
      "inference_failed",
      [
        "harness_client_exit_-6"
      ]
    ],
    [
      "ecommerce_14_5ff07b5ee7a1d23e719e421e",
      "inference_failed",
      [
        "harness_client_exit_-6"
      ]
    ],
    [
      "ecommerce_15_56780a5a89319e0011644cb4",
      "inference_failed",
      [
        "harness_client_exit_-6"
      ]
    ]
  ],
  "by_difficulty": {
    "easy": "8/12",
    "medium": "4/10",
    "hard": "5/8"
  },
  "by_feature": {
    "FALSE_START": "4/4",
    "FILLER": "4/7",
    "HESITATION": "1/3",
    "PAUSE": "1/5",
    "SELF_CORRECTION": "3/3"
  }
}
```

| recording | category | passed | tool calls | perceived latency | speech | flags | reason |
|---|---|---|---|---|---|---|---|
| ecommerce_01_65e8cf8f4c7424fa062e54a3 | pass | True | 1 | 7.52 | yes |  |  |
| ecommerce_01_69a9cf80f4d7668d5c815038 | pass | True | 1 | 3.76 | yes |  |  |
| ecommerce_02_5f4a4da1575d605c43bef871 | harness/infrastructure | None | 0 | None | no | abnormally_long_recording(host_suspend?), job_assignment_timeout_or_host_suspend, no_speech_captured | agent never joined (job assignment timeout / host suspend) |
| ecommerce_04_6998abd731d2ec50d067d5bd | harness/infrastructure | None | 0 | None | no | harness_client_exit_-6 | inference_failed |
| ecommerce_05_695bd157114f0d2317f88617 | pass | True | 1 | 17.84 | yes |  |  |
| ecommerce_06_66c4f3cb14cbfc4db836bd4e | tool selection/arguments | False | 2 | 3.2 | yes |  | Unexpected tools: ['search_products'] |
| ecommerce_07_66f59c766e7e22e1f90d08f6 | pass | True | 1 | 17.2 | yes |  |  |
| ecommerce_08_5ff07b5ee7a1d23e719e421e | tool selection/arguments | False | 1 | 7.52 | yes |  | Wrong arguments for: ['search_products'] |
| ecommerce_08_61517db6a7589569521b2356 | planner | False | 0 | 14.16 | yes | tts_speech_failed | Missing tools: ['search_products'] |
| ecommerce_09_695bd157114f0d2317f88617 | pass | True | 1 | 3.04 | yes |  |  |
| ecommerce_10_6998abd731d2ec50d067d5bd | tool selection/arguments | False | 2 | 5.2 | yes |  | Wrong arguments for: ['search_products'] |
| ecommerce_11_62a885d5b6af18b3d4579e1b | pass | True | 1 | 13.28 | yes |  |  |
| ecommerce_12_6998abd731d2ec50d067d5bd | STT/transcription | False | 1 | None | no | no_speech_captured | Wrong arguments for: ['search_products'] |
| ecommerce_13_5ff07b5ee7a1d23e719e421e | STT/transcription | False | 2 | None | no | no_speech_captured | Wrong arguments for: ['add_to_cart', 'track_order'] |
| ecommerce_13_61517db6a7589569521b2356 | STT/transcription | False | 2 | None | no | job_assignment_timeout_or_host_suspend, no_speech_captured, tts_speech_failed | Missing tools: ['track_order']; Unexpected tools: ['search_products'] |
| ecommerce_14_5ff07b5ee7a1d23e719e421e | harness/infrastructure | None | 0 | None | no | harness_client_exit_-6 | inference_failed |
| ecommerce_14_61517db6a7589569521b2356 | tool selection/arguments | False | 1 | 4.64 | yes | tts_speech_failed | Wrong arguments for: ['add_to_cart'] |
| ecommerce_15_56780a5a89319e0011644cb4 | harness/infrastructure | None | 0 | None | no | harness_client_exit_-6 | inference_failed |
| ecommerce_15_66c4f3cb14cbfc4db836bd4e | pass | True | 1 | 6.08 | yes |  |  |
| ecommerce_16_695bd157114f0d2317f88617 | tool selection/arguments | False | 2 | 6.72 | yes |  | Unexpected tools: ['search_products'] |
| ecommerce_17_5f4a4da1575d605c43bef871 | pass | True | 1 | 12.4 | yes |  |  |
| ecommerce_18_62a885d5b6af18b3d4579e1b | STT/transcription | False | 3 | 17.28 | yes |  | Wrong arguments for: ['search_products', 'track_order'] |
| ecommerce_19_66f59c766e7e22e1f90d08f6 | pass | True | 2 | 2.56 | yes |  |  |
| ecommerce_20_66c4f3cb14cbfc4db836bd4e | pass | True | 2 | None | no | no_speech_captured, tts_speech_failed |  |
| ecommerce_21_65e8cf8f4c7424fa062e54a3 | tool selection/arguments | False | 3 | 7.2 | yes |  | Wrong arguments for: ['track_order'] |
| ecommerce_21_69a9cf80f4d7668d5c815038 | STT/transcription | False | 3 | 5.68 | yes |  | Wrong arguments for: ['track_order'] |
| ecommerce_23_5f4a4da1575d605c43bef871 | pass | True | 2 | 16.4 | yes | job_assignment_timeout_or_host_suspend |  |
| ecommerce_25_65e8cf8f4c7424fa062e54a3 | pass | True | 3 | 12.88 | yes | job_assignment_timeout_or_host_suspend |  |
| ecommerce_25_69a9cf80f4d7668d5c815038 | pass | True | 3 | 7.12 | yes |  |  |
| finance_01_65e8cf8f4c7424fa062e54a3 | STT/transcription | False | 0 | None | no | job_assignment_timeout_or_host_suspend, no_speech_captured | Missing tools: ['get_exchange_rate'] |
| finance_01_69a9cf80f4d7668d5c815038 | pass | True | 1 | 25.28 | yes | job_assignment_timeout_or_host_suspend |  |
| finance_02_6998abd731d2ec50d067d5bd | pass | True | 1 | 21.52 | yes | job_assignment_timeout_or_host_suspend |  |
| finance_03_5ff07b5ee7a1d23e719e421e | pass | True | 1 | 3.36 | yes | job_assignment_timeout_or_host_suspend |  |
| finance_03_61517db6a7589569521b2356 | pass | True | 1 | 28.4 | yes |  |  |
