# AI disclosure

Team **Stack Overlords** (VIT): Samsung PRISM Generative AI Hackathon 2026, Theme 05.

This project uses AI in two ways: AI models are part of the product, and an AI coding assistant was
used to build it. Both are described below.

## 1. AI models inside the product (runtime)

| Component | Model | Where it runs | Role |
|---|---|---|---|
| Voice activity detection | Silero VAD (LiveKit plugin) | Local CPU | Detects when the user starts and stops speaking |
| Speech-to-text | Whisper `base.en` (open weights, via Hugging Face `transformers`) | Local CPU, separate process | Transcribes the user's speech |
| Planner | Google Gemini `gemini-3.5-flash-lite` (Google AI Studio free tier) | Google API | Proposes which tool calls the current request needs and drafts the reply, as a JSON plan |
| Text-to-speech | Piper `en_US-lessac-medium` | Local CPU | Speaks the agent's replies |
| Demo UI (optional features) | The browser's built-in speech recognition and speech synthesis | The user's browser | Mic input and read-aloud in the demo page, only when the user turns them on |

How far the AI models are trusted:

- **The planner never executes anything.** It only proposes. A deterministic session kernel, written
  as ordinary code with no AI, decides what is executed, cancelled or spoken. Every decision is
  journaled and can be replayed.
- **What leaves the machine:** transcribed text is sent to the Gemini API for planning, and audio is
  processed locally. In the demo UI, if the microphone button is used, the browser sends the audio to
  its own speech service.
- **Optional alternative stack:** an OpenAI stack (whisper-1, gpt-4o, tts-1) exists in the code but
  is not the default and was not used for the reported results.

## 2. AI used during development

We used **Claude Code** (Anthropic's AI coding assistant) throughout development. It helped to:

- write and refactor the source code: kernel, runtime, voice agent, providers, in-car extension and
  demo UI;
- write the automated tests;
- write the documentation (README, RESULTS, demo script);
- write the benchmark run and analysis scripts;
- prepare the presentation slides.

What the team did:

- defined the architecture, requirements and constraints, including the zero-cost stack and the rule
  that no benchmark knowledge may appear in code or prompts;
- directed each development phase and reviewed the changes;
- supplied and managed the service accounts;
- ran and checked the demos, tests and benchmark runs;
- recorded the demo video.

The team takes responsibility for the submission.

## 3. Benchmark integrity

- No benchmark answers, scenario IDs or expected argument values were given to any AI model as
  hints, examples or rules. `tests/test_integrity.py` fails the build if any appear in agent code or
  tests.
- All reported results are local and unofficial. They were scored by the benchmark harness's
  exact-match fallback, not the official gpt-4o judge, and cover a partial run of 34 of 100
  recordings (see [RESULTS.md](RESULTS.md)).
