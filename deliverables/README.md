# AlPrime Document RAG Assistant: Submission Package

A document question-answering service for financial and corporate files. Users upload PDF, DOCX and TXT documents and ask questions in a chat UI or over a REST API. Every answer is built only from the uploaded text and cites the file and page it came from. When the documents do not contain the answer, the service says so instead of guessing.

The repository root (`../`) is the runnable application. This folder holds the evidence: API specifications, the verification and calibration report, and the results of a live adversarial evaluation.

---

## Results at a Glance

| Check | Result | Evidence |
| :--- | :--- | :--- |
| Offline test suite (`pytest`, no API key needed) | 86 / 86 passed | `../tests/`, [Verification report §1](VERIFICATION_AND_CALIBRATION_REPORT.md#1-offline-test-suite) |
| Live trap-question evaluation (61 questions, 13 categories) | 61 / 61 passed, graded by fixed rules | [TRAP_QUESTION_EVALUATION.md](TRAP_QUESTION_EVALUATION.md) |
| Fabricated facts in the final evaluation run | 0 | Each trap is checked for the fabrication it invites |
| Docker stack (`api` + `qdrant`) | Builds and runs healthy | [Verification report §5](VERIFICATION_AND_CALIBRATION_REPORT.md#5-live-and-container-verification) |
| Typical answer latency | Median 4.1 s, p90 7.2 s end to end (hosted LLM included) | Streaming starts before the answer is complete |

---

## Reviewing in 10 Minutes

Step-by-step setup instructions, with free key links and troubleshooting, are in the [Quick Start Guide](../QUICKSTART.md).

1. Start it. Copy `../.env.example` to `../.env` and add one LLM key of your own (Runware, Gemini or Groq; any one is enough, and Gemini and Groq have free tiers). Then run:
   ```bash
   docker compose up --build        # or: uvicorn app.main:app --port 8000
   ```
2. Open the UI at http://localhost:8000 and drag in the six files from `../sample_docs/`.
3. Ask a few trap questions and check the cited pages:
   - *"Why did revenue decline by 14.5% in Q3 2026?"* (it grew; the false premise is corrected)
   - *"Who approves a purchase of exactly KWD 5,000?"* (band edge: the Finance Director)
   - *"Did AlPrime achieve KWD 1,950,000 revenue in Q4 2026?"* (that figure is a target, not a result)
   - *"If a client needs 120 analyst hours a month, is the Fractional CFO tier enough and what would the total monthly cost be?"* (no: 1,850 + 40 × 35 = KWD 3,250, labelled as calculated)
   - *"Who is the CEO?"* (refused: not in the documents)
4. Run the offline tests: `pip install -r requirements-dev.txt && pytest -q`.
5. Re-run the evaluation yourself: `python scripts/trap_eval.py --upload` regenerates the trap report from your own run.

---

## Deliverable Map

| Deliverable | Location |
| :--- | :--- |
| Source code | `../app/` (`api/`, `core/`, `schemas/`, `services/`, `main.py`, `config.py`) |
| Quick start: setup, run, what it does and does not do | `../QUICKSTART.md` |
| Setup, architecture and design README | `../README.md` |
| Python dependencies | `../requirements.txt` (runtime), `../requirements-dev.txt` (tests), `../requirements-lock.txt` (exact verified versions) |
| Sample documents (PDF, DOCX, TXT) | `../sample_docs/`: 6 files, 3 of them written with planted traps; generator in `../scripts/generate_sample_docs.py` |
| OpenAPI 3.1 schema | `./api_docs/openapi.json` (also live at `/docs` and `/redoc`) |
| Postman v2.1 collection | `./api_docs/postman_collection.json` (mirrored at `../postman_collection.json`) |
| Request/response and cURL walkthrough | `./api_docs/API_WALKTHROUGH.md` |
| Verification, calibration and fix log | `./VERIFICATION_AND_CALIBRATION_REPORT.md` |
| Adversarial evaluation: report and raw data | `./TRAP_QUESTION_EVALUATION.md`, `./trap_eval_results.json`, harness `../scripts/trap_eval.py` |
| Bonus: streaming chat UI | `../static/`, served at `/` |
| Bonus: Docker and Compose | `../Dockerfile`, `../docker-compose.yml` |
| Bonus: multi-turn sessions, transcript export, API-key auth, per-query tracing | `../README.md` (Key Features, Observability) |
| Sample exported conversation transcript | `./Conversations-sample-alprime-transcript-a381715a.md` |
| Architecture, AI/RAG pipeline & API flowcharts (PNG + ASCII) | `./diagrams/` (`01_system_and_api_architecture.png`, `02_rag_and_guardrails_pipeline.png`, `03_llm_routing_and_api_lifecycle.png`, `ARCHITECTURE_FLOWCHARTS.txt`) |

---

## Engineering Highlights

- Five layers against hallucination:
  - An empty knowledge base is refused without calling the LLM.
  - A cosine pre-filter, then a calibrated cross-encoder gate, decide relevance.
  - Document text is XML-fenced, which blocks prompt injection.
  - A refusal sentinel lets the model decline cleanly.

  Refusal messages are built from templates, so a refusal can never introduce a fact.
- Thresholds were measured, not guessed. Scores were collected for genuine answers, close-but-wrong questions and off-topic noise, and the gates sit between them. The measurements are in the verification report.
- An evaluation that found real bugs. The first trap runs exposed five wrong refusals and one false claim about the corpus. Each has a root cause, a fix and a regression test; see the [fix log](VERIFICATION_AND_CALIBRATION_REPORT.md#4-issues-found-by-the-trap-evaluation-and-how-they-were-fixed).
- Resilient LLM layer. DeepSeek V4.1 Flash on Runware is the primary model, with automatic failover to Gemini and then to Groq/Qwen. Analytical questions get deeper reasoning. A stream that sends no answer text within 20 s fails over before any token reaches the user.
- Observable. Each query writes one JSON trace line with per-stage timings, retrieval scores, every provider attempt and token counts. No document text is logged.

## Known Limitations

These are documented in `../README.md` → *Assumptions & Limitations*. The ones a reviewer is most likely to notice:
- Arabic questions are refused safely rather than answered, because the embedding and reranking models are English-only.
- Scanned PDFs without a text layer are rejected with HTTP `422`; OCR is out of scope.
- Sessions are held in memory, so the service runs as a single process.

## Not Included in This Package

- `.env` is not included because it holds private API keys. `../.env.example` documents every setting.
- Local caches and runtime data (`.venv/`, `.pytest_cache/`, `data/`) are not included.
- The zip is built with `python scripts/package_submission.py`, which leaves out the files above and aborts if any real key value from `.env` appears in a packaged file.
