# Quick Start Guide

This guide gets the AlPrime Document RAG Assistant running on your machine in a few minutes. For architecture flowcharts, RAG calibration, and API contracts, see [README.md](README.md).

> [!IMPORTANT]
> **Docker is optional!**
> Qdrant runs natively inside Python in **embedded local disk mode** (`./data/qdrant`) and FastEmbed runs locally on CPU—so you can run the entire stack (vector DB, embeddings, reranker, API, and Web UI) with **Python alone** (`Option A`), or use **Docker Compose** (`Option B`) if you prefer containers.

---

## 1. Prerequisites

Pick **either** runtime:
- **Python 3.11 or newer** (no Docker required — runs embedded Qdrant directly in Python)
- **Or Docker Desktop** (`docker compose up --build`)

Plus **one LLM API key** in `.env`:
- Document parsing, chunking, vector storage, embeddings, and cross-encoder reranking all run locally on CPU.
- Only final answer generation calls a hosted LLM API (one key is enough; unconfigured providers are skipped automatically):

| Provider | Cost | Get a key | Setting in `.env` |
| :--- | :--- | :--- | :--- |
| Google Gemini | Free tier | https://aistudio.google.com/apikey | `GEMINI_API_KEY` |
| Groq | Free tier | https://console.groq.com/keys | `GROQ_API_KEY` |
| Runware (DeepSeek, model used in testing) | Paid, pay as you go | https://my.runware.ai | `RUNWARE_API_KEY` |
| OpenAI or any OpenAI-compatible service | Varies | Your provider | `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` |

> [!NOTE]
> **For Reviewers:** If an environment/keys file was provided with the submission email, place it in the project root and rename it to `.env` (never commit `.env` to Git).

---

## 2. Step-by-Step Local Setup & Run

### Option A: Pure Python (No Docker Required)

1. **Get the repository** (or extract the submission zip) and `cd` into it:
   ```bash
   git clone https://github.com/nishikantchandra/alprime.git
   cd alprime
   ```

2. **Create `.env`** (skip if you placed the emailed `.env` file in the root):
   - Windows (PowerShell): `copy .env.example .env`
   - macOS / Linux: `cp .env.example .env`
   Then open `.env`, paste your API key (e.g., `GEMINI_API_KEY="..."`), and leave `QDRANT_URL=""` so embedded Qdrant runs automatically.

3. **Install & launch:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     pip install -r requirements.txt
     uvicorn app.main:app --port 8000
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
     uvicorn app.main:app --port 8000
     ```

   The first startup downloads ~90 MB of local ONNX models once; subsequent starts take a few seconds.

---

### Option B: Docker Compose (Zero Python Setup)

1. Place `.env` in the project root (either the emailed `.env` or `cp .env.example .env` + your LLM key).
2. Make sure Docker Desktop / Engine is running, then run:
   ```bash
   docker compose up --build
   ```
   This starts the `api` container on `http://localhost:8000` (with ONNX models pre-cached) and a standalone `qdrant` container (`qdrant/qdrant:v1.20.0`). Stop with `Ctrl + C` or `docker compose down`.

---

## 3. Use It

| What | Where |
| :--- | :--- |
| Chat interface | http://localhost:8000 |
| Interactive API documentation (Swagger) | http://localhost:8000/docs |
| Health check, including which LLM is active | http://localhost:8000/health |

1. Open http://localhost:8000.
2. Drag the six files from the `sample_docs/` folder into the upload area.
3. Ask questions. These five show how the assistant behaves:

| Question | Expected behaviour |
| :--- | :--- |
| What was Q3 2026 consolidated revenue and operating margin? | Exact figures, with a citation to the file and page |
| Why did revenue decline by 14.5% in Q3 2026? | Corrects the false premise: revenue grew |
| Who approves a purchase of exactly KWD 5,000? | Reads the band edge correctly: the Finance Director |
| If a client needs 120 analyst hours a month, is the Fractional CFO tier enough and what would the total monthly cost be? | No: 80 hours are included, so 40 overtime hours at KWD 35. 1,850 + 40 × 35 = KWD 3,250, marked as calculated |
| Who is the CEO? | Says the documents do not contain this and does not guess |

Every answer lists the documents and pages it came from. Click a citation to see the source passage.

### Run the automated tests (no API key needed)

With the Python setup from Option A:
```bash
pip install -r requirements-dev.txt
pytest -q
```
Expected result: `86 passed`.

---

## 4. What It Does

- Accepts PDF, DOCX (including tables) and TXT files, up to 10 files of 20 MB each per upload.
- Answers questions only from the uploaded documents, with a citation for every fact.
- Refuses clearly when the documents do not contain the answer, instead of guessing.
- Catches misleading questions: false premises, targets presented as results, figures from the wrong period.
- Does simple arithmetic on document figures and labels the result as calculated.
- Handles follow-up questions in the same conversation ("and for Q2?").
- Streams answers word by word in the chat interface.
- Skips files that were already uploaded instead of indexing them twice.
- Switches to a backup LLM provider automatically if the first one fails.

## 5. What It Does Not Do

- It does not read scanned PDFs (images with no text layer); these are rejected with a clear error. OCR is out of scope.
- It does not support old `.doc` files, spreadsheets or images. Save as `.docx` or PDF first.
- It does not answer Arabic or other non-English questions. It refuses safely instead, because the retrieval models are English-only.
- It does not use outside knowledge, even when it seems obvious. For example, it will not convert KWD to USD without a rate in the documents.
- It does not keep conversation history after a restart. Uploaded documents do persist.
- It does not require login by default. API-key protection can be switched on in `.env` (`AUTH_ENABLED=true`).

---

## 6. Troubleshooting

| Symptom | Cause and fix |
| :--- | :--- |
| Answers fail with "No LLM provider is configured" (HTTP 503) | No key in `.env`. Add one key (section 2), then restart the app. With Docker, run `docker compose up` again. |
| An answer fails with a 401, 403 or quota message | The key is wrong or its free quota is used up. Check the key, or add a key from a second provider as a backup. |
| `docker compose` is not recognised, or cannot connect | Docker Desktop is not running. Start it and wait until it reports "running". |
| "Port 8000 is already in use" | Another program is using the port. Stop it, or with Python run `uvicorn app.main:app --port 8001` and open http://localhost:8001. |
| PowerShell blocks `Activate.ps1` | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once in that window, then activate again. |
| The first start is slow | It is downloading the local models once. Later starts are fast. |
| Start again with no documents | Use the reset button in the interface, or call `DELETE /api/v1/documents/reset`. |

---

## Further Reading

- [README.md](README.md): architecture, anti-hallucination design and configuration reference
- [deliverables/](deliverables/README.md): API specifications, Postman collection, verification report and the 61-question trap evaluation
