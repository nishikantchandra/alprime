# API Walkthrough (Requests and Real Responses)

The responses below were captured from the running Docker stack with all six `sample_docs/` files uploaded. IDs, scores and timings will differ on your machine.

- Live Swagger UI: `http://localhost:8000/docs`
- Live ReDoc: `http://localhost:8000/redoc`
- Static OpenAPI 3.1 spec: [`./openapi.json`](openapi.json)
- Postman v2.1 collection: [`./postman_collection.json`](postman_collection.json). Set the `base_url` variable; the upload request points at `sample_docs/`.

If `AUTH_ENABLED=true`, add `-H "X-API-Key: <your-key>"` to every `/api/v1/*` call. A missing or wrong key returns `401`.

---

## 1. Health and Configuration

```bash
curl -s http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "app_name": "AlPrime Document RAG Assistant",
  "version": "1.1.0",
  "qdrant_mode": "remote",
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "reranker_enabled": true,
  "llm_provider": "runware (deepseek:v4.1@flash-1017) -> gemini (gemini-3.5-flash-lite) -> groq (qwen/qwen3.8-27b) | reasoning: runware",
  "llm_configured": true,
  "auth_enabled": false,
  "indexed_documents": 6,
  "indexed_chunks": 20
}
```

`qdrant_mode` is `remote` under Docker Compose and `embedded` for a plain `uvicorn` run. `llm_provider` shows the failover order. Only providers with a key are listed.

---

## 2. Upload Documents (PDF, DOCX, TXT)

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "files=@sample_docs/AlPrime_Q3_2026_Financial_Performance.pdf" \
  -F "files=@sample_docs/AlPrime_Q2_2026_Management_Accounts.pdf" \
  -F "files=@sample_docs/AlPrime_Advisory_Fee_Schedule_2026.docx" \
  -F "files=@sample_docs/AlPrime_Staff_Expense_and_Approval_Policy.docx" \
  -F "files=@sample_docs/Kuwait_Market_Compliance_Policy.txt" \
  -F "files=@sample_docs/AlPrime_Client_Engagement_FAQ.txt"
```

- `201`: at least one file was newly indexed.
- Re-uploading identical content returns `200`. This holds even under another filename. The response says nothing was added:

```json
{
  "message": "This file is already indexed; nothing new was added.",
  "documents": [
    {
      "doc_id": "c8003ec2-affa-4335-87fd-0345125aaacb",
      "filename": "Kuwait_Market_Compliance_Policy.txt",
      "file_type": "txt",
      "page_count": 1,
      "chunk_count": 2,
      "char_count": null,
      "status": "already_indexed"
    }
  ],
  "total_chunks_indexed": 0,
  "duplicates_skipped": 1
}
```

#### Error codes

| Code | Cause |
| :--- | :--- |
| `400` | Unsupported extension |
| `413` | A file is over `MAX_UPLOAD_MB`, or the batch has more than `MAX_UPLOAD_FILES` files |
| `422` | The file is empty, corrupt, or a scanned PDF with no text layer |

A batch is all-or-nothing: if one file fails, nothing from the batch is kept.

---

## 3. List and Delete Documents

```bash
curl -s http://localhost:8000/api/v1/documents
```

```json
{
  "total_documents": 6,
  "total_chunks": 20,
  "documents": [
    {
      "doc_id": "d5743a0e-837b-4c29-a2b4-3d52ce1fdc1f",
      "filename": "AlPrime_Staff_Expense_and_Approval_Policy.docx",
      "file_type": "docx",
      "page_count": 1,
      "chunk_count": 3,
      "uploaded_at": "2026-11-11T08:47:52.586298Z"
    }
  ]
}
```

(Truncated to one document.)

- Delete one document: `curl -X DELETE http://localhost:8000/api/v1/documents/<doc_id>`. An unknown ID returns `404`.
- Delete everything: `curl -X DELETE http://localhost:8000/api/v1/documents/reset`.

---

## 4. Ask a Grounded Question

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Who approves a purchase of exactly KWD 5,000?", "session_id": "demo-1"}'
```

```json
{
  "answer": "A purchase of exactly KWD 5,000 requires approval from the Finance Director, with an approval turnaround of 2 business days [1].",
  "is_found": true,
  "response_type": "grounded",
  "refusal_reason": null,
  "status_detail": null,
  "documents_used": ["AlPrime_Staff_Expense_and_Approval_Policy.docx"],
  "citations": [
    {
      "document_name": "AlPrime_Staff_Expense_and_Approval_Policy.docx",
      "doc_id": "d5743a0e-837b-4c29-a2b4-3d52ce1fdc1f",
      "page": 1,
      "similarity_score": 0.6521,
      "rerank_score": 2.6993,
      "snippet": "AlPrime Capital - Staff Expense & Spend Approval Policy (v3.1) ... | Spend Amount (per transaction) | Required Approver | Approval Turnaround | ..."
    }
  ],
  "session_id": "demo-1"
}
```

#### How the response fits together
- `[1]` in `answer` refers to `citations[0]`.
- `citations` contains only the passages the answer actually cites, and the markers are renumbered to match.
- `snippet` is the exact retrieved text. The example is shortened here.

#### Optional request fields
- `doc_ids`: restricts the search to the listed documents.
- `top_k`: overrides how many evidence passages go to the model (default 6).
- `session_id`: turns on multi-turn follow-ups (see section 7).

---

## 5. Refusal When the Documents Do Not Contain the Answer

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Who is the CEO of AlPrime Capital?"}'
```

```json
{
  "answer": "I couldn't find relevant information in the uploaded documents to answer this question. The closest passages I found were in your 5 uploaded documents, but they don't actually state this, so I won't guess or fill the gap with outside knowledge. If the detail is in another file, upload it and ask again, or rephrase the question using the document's own terms.",
  "is_found": false,
  "response_type": "not_found",
  "refusal_reason": "insufficient_document_evidence",
  "status_detail": "Related passages were retrieved, but none of them states the answer, so the model declined rather than guess.",
  "documents_used": [],
  "citations": [],
  "session_id": null
}
```

#### Values of `refusal_reason`

| Value | Meaning | LLM called? |
| :--- | :--- | :--- |
| `empty_knowledge_base` | No documents are indexed | No |
| `low_retrieval_relevance` | Nothing passed the similarity and cross-encoder gates (e.g. *"What is the capital of France?"*) | No |
| `insufficient_document_evidence` | Passages were retrieved, but the model found that none of them answers the question | Yes |

The first sentence of every refusal is fixed, so clients can rely on it. The rest is built from templates and real filenames, never written by the model.

---

## 6. Streaming (Server-Sent Events)

```bash
curl -N -X POST "http://localhost:8000/api/v1/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the bad-debt provision cap?"}'
```

Captured event sequence (snippet shortened):

```text
event: status
data: {"stage": "searching", "message": "Searching & reranking uploaded documents..."}

event: status
data: {"stage": "generating", "message": "Drafting cited answer from 1 document(s)..."}

event: citations
data: {"documents_used": ["AlPrime_Q3_2026_Financial_Performance.pdf"], "citations": [{"document_name": "AlPrime_Q3_2026_Financial_Performance.pdf", "page": 3, "similarity_score": 0.6361, "rerank_score": 2.0381, "snippet": "... bad-debt provision remains capped at 0.8% of billed retainers. ..."}]}

event: status
data: {"stage": "writing", "message": "Writing response..."}

event: token
data: {"text": "The bad-debt provision is capped at 0.8% of billed retainers [1]."}

event: done
data: {"is_found": true, "response_type": "grounded", "refusal_reason": null, "status_detail": null, "final_answer": "The bad-debt provision is capped at 0.8% of billed retainers [1].", "session_id": null, "duration_ms": 7629}
```

#### How the stream behaves
- Tokens are held back until the answer is confirmed: a citation marker or about 160 characters has arrived. A refusal therefore never starts typing and then flips.
- On a refusal, no `citations` or `token` events are sent. The refusal arrives only in `done`.
- On a failure, an `error` event is sent if every LLM provider fails before answering.
- Failover: if a provider sends no answer text within 20 s, or ends its stream empty, the next provider answers instead.

---

## 7. Multi-Turn Follow-Ups and Transcript Export

```bash
curl -X POST http://localhost:8000/api/v1/query -H "Content-Type: application/json" \
  -d '{"question": "What was the consolidated revenue in Q3 2026?", "session_id": "demo-2"}'
curl -X POST http://localhost:8000/api/v1/query -H "Content-Type: application/json" \
  -d '{"question": "and the year before?", "session_id": "demo-2"}'
```

#### How follow-ups are handled
- The second question is rewritten from the session history into a standalone question, e.g. *"What was the consolidated revenue in Q3 2025?"*.
- Both the rewrite and the original wording are searched, and the results are merged.
- The answer is KWD 1,607,000.

#### Session endpoints
- Export the conversation: `GET /api/v1/sessions/demo-2/export?format=markdown`, or `format=json`. Each turn includes its question, answer, outcome and citations.
- Clear the session: `DELETE /api/v1/sessions/demo-2`. An unknown session returns `404`.
