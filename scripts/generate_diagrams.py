"""Generate clean high-DPI PNG architecture & flowchart diagrams plus ASCII text mirrors.

Run via:
    python scripts/generate_diagrams.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "deliverables" / "diagrams"

# AlPrime Capital visual palette
BG = "#f8fafc"
NAVY = "#0f172a"
NAVY_SOFT = "#1e293b"
GOLD = "#b8893a"
GOLD_BG = "#fef3c7"
GOLD_BORDER = "#f59e0b"
BLUE_BG = "#eff6ff"
BLUE_BORDER = "#3b82f6"
BLUE_FG = "#1e3a8a"
GREEN_BG = "#dcfce7"
GREEN_BORDER = "#16a34a"
GREEN_FG = "#14532d"
AMBER_BG = "#fffbeb"
AMBER_BORDER = "#d97706"
AMBER_FG = "#78350f"
RED_BG = "#fee2e2"
RED_BORDER = "#dc2626"
RED_FG = "#7f1d1d"
CARD_BG = "#ffffff"
CARD_BORDER = "#cbd5e1"
MUTED = "#475569"
TEXT = "#0f172a"
ARROW = "#475569"


@dataclass(frozen=True, slots=True)
class ThemeFonts:
    title: ImageFont.FreeTypeFont
    subtitle: ImageFont.FreeTypeFont
    section: ImageFont.FreeTypeFont
    card_title: ImageFont.FreeTypeFont
    body: ImageFont.FreeTypeFont
    mono: ImageFont.FreeTypeFont
    badge: ImageFont.FreeTypeFont


def _load_font(filename: str, size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts") / filename,
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()  # type: ignore[return-value]


def load_fonts() -> ThemeFonts:
    body_font = _load_font("segoeui.ttf", 13)
    return ThemeFonts(
        title=_load_font("segoeuib.ttf", 28),
        subtitle=_load_font("segoeui.ttf", 16),
        section=_load_font("segoeuib.ttf", 17),
        card_title=_load_font("segoeuib.ttf", 14),
        body=body_font,
        mono=_load_font("consola.ttf", 12),
        badge=_load_font("segoeuib.ttf", 12),
    )


def draw_header(
    draw: ImageDraw.ImageDraw,
    fonts: ThemeFonts,
    width: int,
    title: str,
    subtitle: str,
) -> None:
    draw.rectangle([0, 0, width, 88], fill=NAVY)
    draw.rectangle([0, 84, width, 88], fill=GOLD)
    draw.rounded_rectangle([28, 22, 72, 66], radius=10, fill=GOLD)
    draw.text((42, 28), "A", font=fonts.title, fill=NAVY)
    draw.text((90, 18), title, font=fonts.title, fill="#ffffff")
    draw.text((90, 54), subtitle, font=fonts.subtitle, fill="#cbd5e1")


def draw_group_box(
    draw: ImageDraw.ImageDraw,
    fonts: ThemeFonts,
    box: tuple[int, int, int, int],
    label: str,
    *,
    fill: str = "#f1f5f9",
    outline: str = "#cbd5e1",
    label_fill: str = NAVY,
) -> None:
    draw.rounded_rectangle(box, radius=14, fill=fill, outline=outline, width=2)
    draw.text((box[0] + 16, box[1] + 12), label, font=fonts.section, fill=label_fill)


def draw_card(
    draw: ImageDraw.ImageDraw,
    fonts: ThemeFonts,
    box: tuple[int, int, int, int],
    heading: str,
    lines: list[str],
    *,
    fill: str = CARD_BG,
    outline: str = CARD_BORDER,
    heading_color: str = NAVY,
    body_color: str = MUTED,
    mono: bool = False,
) -> None:
    draw.rounded_rectangle(box, radius=10, fill=fill, outline=outline, width=2)
    x0, y0 = box[0] + 14, box[1] + 11
    draw.text((x0, y0), heading, font=fonts.card_title, fill=heading_color)
    line_font = fonts.mono if mono else fonts.body
    cur_y = y0 + 22
    for line in lines:
        draw.text((x0, cur_y), line, font=line_font, fill=body_color)
        cur_y += 18


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = ARROW,
    label: str | None = None,
    fonts: ThemeFonts | None = None,
    label_offset: tuple[int, int] = (0, -18),
) -> None:
    draw.line([start, end], fill=color, width=3)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    tip = end
    base = (end[0] - ux * 11, end[1] - uy * 11)
    p1 = (base[0] + px * 6, base[1] + py * 6)
    p2 = (base[0] - px * 6, base[1] - py * 6)
    draw.polygon([tip, p1, p2], fill=color)
    if label and fonts:
        mx = (start[0] + end[0]) // 2 + label_offset[0]
        my = (start[1] + end[1]) // 2 + label_offset[1]
        bbox = draw.textbbox((mx, my), label, font=fonts.badge)
        pad = 4
        draw.rounded_rectangle(
            [bbox[0] - pad, bbox[1] - 2, bbox[2] + pad, bbox[3] + 2],
            radius=4,
            fill=BG,
            outline=CARD_BORDER,
            width=1,
        )
        draw.text((mx, my), label, font=fonts.badge, fill=color)


def draw_polyline_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    color: str = ARROW,
    label: str | None = None,
    fonts: ThemeFonts | None = None,
    label_pos: tuple[int, int] | None = None,
) -> None:
    for i in range(len(points) - 2):
        draw.line([points[i], points[i + 1]], fill=color, width=3)
    draw_arrow(draw, points[-2], points[-1], color=color)
    if label and fonts and label_pos:
        bbox = draw.textbbox(label_pos, label, font=fonts.badge)
        draw.rounded_rectangle(
            [bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2],
            radius=4,
            fill=BG,
            outline=CARD_BORDER,
            width=1,
        )
        draw.text(label_pos, label, font=fonts.badge, fill=color)


def build_system_and_api_diagram(fonts: ThemeFonts) -> Path:
    width, height = 1600, 920
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    draw_header(
        draw,
        fonts,
        width,
        "1. System Architecture, Project Structure & REST API Design",
        "AlPrime Document Assistant  •  FastAPI + Qdrant + Local FastEmbed ONNX + Multi-Provider LLM Router",
    )

    # Column 1: Clients
    draw_group_box(draw, fonts, (24, 114, 296, 892), "Clients & UI Layer", fill="#eef2ff", outline="#c7d2fe")
    draw_card(
        draw,
        fonts,
        (40, 158, 280, 276),
        "Web Chat UI  (GET /)",
        [
            "static/index.html, app.js, css",
            "Drag-drop / + Attach uploads",
            "Live SSE streaming & [1] pills",
            "Markdown / JSON export",
        ],
        fill=CARD_BG,
        outline=BLUE_BORDER,
    )
    draw_card(
        draw,
        fonts,
        (40, 292, 280, 392),
        "Swagger & ReDoc",
        [
            "GET /docs  (OpenAPI 3.1)",
            "GET /redoc",
            "deliverables/api_docs/",
        ],
        fill=CARD_BG,
        outline=CARD_BORDER,
    )
    draw_card(
        draw,
        fonts,
        (40, 408, 280, 514),
        "REST / Postman / Eval",
        [
            "postman_collection.json",
            "scripts/trap_eval.py",
            "cURL / SDK integrations",
        ],
        fill=CARD_BG,
        outline=CARD_BORDER,
    )
    draw_card(
        draw,
        fonts,
        (40, 530, 280, 874),
        "Project Package Layout",
        [
            "app/main.py     App factory",
            "app/config.py   Env settings",
            "app/api/        REST routes",
            "app/core/       Auth/Errors",
            "app/schemas/    I/O models",
            "app/services/   RAG engine",
            "sample_docs/    6 test docs",
            "tests/          86 tests",
            "Dockerfile + compose.yml",
        ],
        fill="#f8fafc",
        outline=CARD_BORDER,
        mono=True,
    )

    # Column 2: FastAPI API & Security Layer
    draw_group_box(
        draw,
        fonts,
        (336, 114, 782, 892),
        "FastAPI API & Guard Layer (app/api/, app/core/)",
        fill="#f1f5f9",
        outline="#94a3b8",
    )
    draw_card(
        draw,
        fonts,
        (354, 158, 764, 250),
        "Security & Middleware (security.py, main.py)",
        [
            "Optional X-API-Key constant-time check (AUTH_ENABLED)",
            "CORS policy  •  Lifespan DI for VectorStore / LLM / Sessions",
        ],
        fill=GOLD_BG,
        outline=GOLD_BORDER,
        heading_color=AMBER_FG,
        body_color=AMBER_FG,
    )

    draw_card(
        draw,
        fonts,
        (354, 266, 764, 406),
        "Document Endpoints (documents.py)",
        [
            "POST   /api/v1/documents/upload   201 new | 200 dupe",
            "GET    /api/v1/documents          list docs & chunks",
            "DELETE /api/v1/documents/{doc_id} delete doc vectors",
            "DELETE /api/v1/documents/reset    clear Qdrant index",
        ],
        fill=CARD_BG,
        outline=BLUE_BORDER,
        mono=True,
    )

    draw_card(
        draw,
        fonts,
        (354, 422, 764, 584),
        "Query, Streaming & Session Endpoints (query.py)",
        [
            "POST   /api/v1/query              grounded JSON Q&A",
            "POST   /api/v1/query/stream       SSE cit/tok/done",
            "GET    /api/v1/sessions/{id}/export  ?format=md|json",
            "DELETE /api/v1/sessions/{id}      clear session",
            "GET    /health                    readiness & status",
        ],
        fill=CARD_BG,
        outline=GREEN_BORDER,
        mono=True,
    )

    draw_card(
        draw,
        fonts,
        (354, 600, 764, 874),
        "Typed Error & Schema Contract (exceptions.py, schemas/)",
        [
            "400  unsupported_format   PDF / DOCX / TXT only",
            "401  unauthorized         Invalid X-API-Key",
            "404  not_found            Missing doc/session ID",
            "413  payload_too_large    >20 MB/file or >10 files",
            "422  empty_document       Blank/corrupt/scanned",
            "502  llm_provider_error   Upstream LLM failure",
            "503  service_unavailable  DB down / missing key",
        ],
        fill=RED_BG,
        outline=RED_BORDER,
        heading_color=RED_FG,
        body_color=RED_FG,
        mono=True,
    )

    # Column 3: AI & RAG Services
    draw_group_box(
        draw,
        fonts,
        (820, 114, 1222, 892),
        "Service Orchestration (app/services/)",
        fill="#f0fdf4",
        outline="#86efac",
    )
    draw_card(
        draw,
        fonts,
        (838, 158, 1204, 268),
        "Ingestion Services",
        [
            "document_parser.py: PDF pages, DOCX tables",
            "text_splitter.py: 800 chars, 150 overlap",
            "SHA-256 dedup check + batch rollback",
        ],
        fill=CARD_BG,
        outline=BLUE_BORDER,
    )
    draw_card(
        draw,
        fonts,
        (838, 284, 1204, 428),
        "RAG & Intent Orchestrator (rag_service.py)",
        [
            "query_intent.py: greeting/overview/reasoning",
            "session_store.py: scope memory + transcript",
            "Dual-query search (rewrite + original)",
            "SSE commit gate holds until real answer",
        ],
        fill=CARD_BG,
        outline=GREEN_BORDER,
    )
    draw_card(
        draw,
        fonts,
        (838, 444, 1204, 580),
        "Grounding, Citations & Templates",
        [
            "prompts.py: XML fence + sentinel guardrail",
            "citations.py: compact [1] + snippet filter",
            "replies.py: templated refusals & greetings",
        ],
        fill=CARD_BG,
        outline=AMBER_BORDER,
    )
    draw_card(
        draw,
        fonts,
        (838, 596, 1204, 722),
        "Observability (app/core/tracing.py)",
        [
            "1 JSON query_trace line per question",
            "Stage timings, cosine/rerank, provider calls",
            "Token counts logged (0 doc text logged)",
        ],
        fill=CARD_BG,
        outline=CARD_BORDER,
    )
    draw_card(
        draw,
        fonts,
        (838, 738, 1204, 874),
        "Verification & Tooling (tests/, scripts/)",
        [
            "86 offline pytest specs (no API key needed)",
            "scripts/trap_eval.py: 61 trap probes",
            "scripts/package_submission.py: secret check",
        ],
        fill="#f8fafc",
        outline=CARD_BORDER,
    )

    # Column 4: Storage & Inference
    draw_group_box(draw, fonts, (1260, 114, 1576, 892), "Storage & Models", fill="#fefce8", outline="#fde047")
    draw_card(
        draw,
        fonts,
        (1278, 158, 1558, 340),
        "FastEmbed ONNX (Local CPU)",
        [
            "Embeddings (384-dim):",
            "BAAI/bge-small-en-v1.5",
            "",
            "Cross-Encoder Reranker:",
            "ms-marco-MiniLM-L-6-v2",
            "Pre-cached in Dockerfile",
        ],
        fill=CARD_BG,
        outline=GOLD_BORDER,
    )
    draw_card(
        draw,
        fonts,
        (1278, 360, 1558, 536),
        "Qdrant Vector Database",
        [
            "Embedded Disk (./data/qdrant)",
            "or Docker Sidecar (:6333)",
            "Payload: doc_id, page_number,",
            "content_sha256, chunk_index,",
            "uploaded_at, source_filename",
        ],
        fill=CARD_BG,
        outline=BLUE_BORDER,
    )
    draw_card(
        draw,
        fonts,
        (1278, 556, 1558, 874),
        "RoutingLLMClient",
        [
            "1. Runware (DeepSeek V4.1 Flash)",
            "   low/high reasoning effort",
            "2. Google Gemini 3.5 Flash Lite",
            "   automatic fallback on quota",
            "3. Groq (Qwen 3.8 27B)",
            "   think / instruct failover",
            "4. OpenAI-compatible (optional)",
            "",
            "20s stall & empty-stream failover",
            "Temp = 0.0  •  Strips <think>",
        ],
        fill=CARD_BG,
        outline=GREEN_BORDER,
    )

    # Connecting arrows
    draw_arrow(draw, (296, 215), (336, 215), label="HTTP", fonts=fonts)
    draw_arrow(draw, (296, 342), (336, 342))
    draw_arrow(draw, (296, 462), (336, 462))
    draw_arrow(draw, (764, 336), (838, 212))
    draw_arrow(draw, (764, 502), (838, 356))
    draw_arrow(draw, (1204, 222), (1278, 222))
    draw_arrow(draw, (1204, 338), (1278, 430))
    draw_arrow(draw, (1204, 398), (1278, 675))

    out = OUT_DIR / "01_system_and_api_architecture.png"
    img.save(out, format="PNG")
    return out


def build_rag_and_guardrails_diagram(fonts: ThemeFonts) -> Path:
    width, height = 1600, 1020
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    draw_header(
        draw,
        fonts,
        width,
        "2. AI Pipelines: Document Ingestion & 5-Layer Anti-Hallucination RAG",
        "Deterministic ingestion with deduplication + multi-layer retrieval & synthesis guardrails",
    )

    # Top Horizontal Lane: AI Pipeline 1 (Document Ingestion)
    draw_group_box(
        draw,
        fonts,
        (28, 108, 1572, 304),
        "AI Pipeline 1: Multi-Format Ingestion, SHA-256 Deduplication & Atomic Indexing",
        fill="#eff6ff",
        outline="#93c5fd",
        label_fill=BLUE_FG,
    )
    draw_card(
        draw,
        fonts,
        (46, 152, 272, 282),
        "1. Upload & Validate",
        [
            "PDF, DOCX, TXT only",
            "<= 10 files / batch",
            "<= 20 MB streaming cap",
            "400 / 413 on violation",
        ],
    )
    draw_card(
        draw,
        fonts,
        (304, 152, 536, 282),
        "2. SHA-256 Deduplication",
        [
            "Hash raw file bytes",
            "Check Qdrant + batch",
            "Repeat -> already_indexed",
            "(HTTP 200 skip, no dupes)",
        ],
        fill=GOLD_BG,
        outline=GOLD_BORDER,
    )
    draw_card(
        draw,
        fonts,
        (568, 152, 824, 282),
        "3. Content Extraction",
        [
            "PDF: pypdf per-page text",
            "DOCX: paras + MD tables",
            "TXT: UTF-8/CP1252 fallback",
            "Blank/scanned PDF -> 422",
        ],
    )
    draw_card(
        draw,
        fonts,
        (856, 152, 1118, 282),
        "4. Provenance Chunking",
        [
            "Recursive character split",
            "800 chars, 150 overlap",
            "Tracks doc_id, page_number,",
            "filename, sha256, chunk_idx",
        ],
    )
    draw_card(
        draw,
        fonts,
        (1150, 152, 1554, 282),
        "5. Local Embed + Atomic Upsert",
        [
            "FastEmbed ONNX bge-small-en-v1.5",
            "Upsert vectors + payload indexes",
            "Batch failure triggers automatic rollback",
            "Success -> HTTP 201 + indexed counts",
        ],
        fill=GREEN_BG,
        outline=GREEN_BORDER,
    )

    draw_arrow(draw, (272, 217), (304, 217))
    draw_arrow(draw, (536, 217), (568, 217))
    draw_arrow(draw, (824, 217), (856, 217))
    draw_arrow(draw, (1118, 217), (1150, 217))

    # Main Section: AI Pipeline 2 (RAG + 5 Guardrail Layers)
    draw_group_box(
        draw,
        fonts,
        (28, 326, 1572, 992),
        "AI Pipeline 2: Query Execution & 5-Layer Zero-Hallucination Guardrails",
        fill="#f1f5f9",
        outline="#94a3b8",
    )

    # Row 1 in RAG pipeline
    draw_card(
        draw,
        fonts,
        (48, 376, 360, 492),
        "Query + Optional Filters",
        [
            "question, session_id, doc_ids, top_k",
            "Conversational intent check:",
            "Pure hi / thanks / help -> templated",
        ],
        fill=BLUE_BG,
        outline=BLUE_BORDER,
    )

    draw_card(
        draw,
        fonts,
        (410, 376, 770, 492),
        "Layer 1: Empty / Stale Scope Guard",
        [
            "Resolve doc_scope & recency target",
            "Empty KB -> no_documents_indexed",
            "Deleted selection -> selected_documents_missing",
        ],
        fill=AMBER_BG,
        outline=AMBER_BORDER,
    )

    draw_card(
        draw,
        fonts,
        (820, 376, 1180, 492),
        "Overview vs Follow-Up Search",
        [
            "Overview: lead chunks from every doc",
            "Follow-up: rewrite (thinking=off) +",
            "dual search (rewrite + original)",
        ],
        fill=CARD_BG,
        outline=BLUE_BORDER,
    )

    draw_card(
        draw,
        fonts,
        (1230, 376, 1552, 492),
        "Fast Templated Refusal Exit",
        [
            "Fixed lead sentence + real filenames",
            "Never generated by LLM",
            "0 hallucination risk, 0 LLM cost",
        ],
        fill=RED_BG,
        outline=RED_BORDER,
        heading_color=RED_FG,
        body_color=RED_FG,
    )

    draw_arrow(draw, (360, 434), (410, 434), label="Factual", fonts=fonts, label_offset=(-20, -22))
    draw_arrow(draw, (770, 434), (820, 434), label="Docs OK", fonts=fonts, label_offset=(-24, -22))
    draw_polyline_arrow(
        draw,
        [(590, 492), (590, 514), (1391, 514), (1391, 492)],
        color=RED_BORDER,
        label="Empty / Stale (0 LLM Calls)",
        fonts=fonts,
        label_pos=(1115, 502),
    )

    # Row 2: Retrieval & Reranking Gates (Layers 2 & 3)
    draw_card(
        draw,
        fonts,
        (48, 560, 500, 696),
        "Layer 2: Dense Vector Search + Cosine Gate",
        [
            "Qdrant similarity top-10 (bge-small-en-v1.5)",
            "Threshold: MIN_VECTOR_SCORE >= 0.45",
            "Removes off-topic noise before cross-encoder",
            "All rejected -> low_retrieval_relevance",
        ],
        fill=AMBER_BG,
        outline=AMBER_BORDER,
    )

    draw_card(
        draw,
        fonts,
        (550, 560, 1080, 696),
        "Layer 3: Cross-Encoder + MaxP Window + Lexical Rescue",
        [
            "Xenova/ms-marco-MiniLM-L-6-v2 scores top-10 (gate >= -6.0)",
            "Fail-gate chunks rescored on best 2-sentence window (raise-only)",
            "Bounded lexical rescue: keyword hit + cos >= 0.48 + rerank >= -8.8",
            "Keeps top-6 passages; 0 pass -> low_retrieval_relevance",
        ],
        fill=AMBER_BG,
        outline=AMBER_BORDER,
    )

    draw_card(
        draw,
        fonts,
        (1130, 560, 1552, 696),
        "Gates Reject Noise Deterministically",
        [
            "Off-topic probes score -10.6 to -11.4",
            "Genuine buried table answers rescued cleanly",
            "Blocked noise never reaches LLM -> 0 tokens spent",
        ],
        fill=RED_BG,
        outline=RED_BORDER,
        heading_color=RED_FG,
        body_color=RED_FG,
    )

    draw_polyline_arrow(
        draw,
        [(1000, 492), (1000, 535), (274, 535), (274, 560)],
        color=ARROW,
    )
    draw_arrow(draw, (500, 628), (550, 628), label="Candidates", fonts=fonts, label_offset=(-32, -22))
    draw_arrow(draw, (1080, 628), (1130, 628), label="0 Pass", fonts=fonts, label_offset=(-18, -22))

    # Row 3: Grounded Synthesis, Sentinel Commit & Finalization (Layers 4 & 5)
    draw_card(
        draw,
        fonts,
        (48, 756, 515, 962),
        "Layer 4: XML-Fenced Grounded Synthesis",
        [
            'Escape <, >, &, " in chunks & filenames',
            "Wrap inside <document_context> passive tags",
            "Temperature = 0.0  •  Strict citation rules [1]",
            "Prompt rule 11 forbids corpus-wide absence claims",
            "Routed via Runware DeepSeek -> Gemini -> Groq",
        ],
        fill=GOLD_BG,
        outline=GOLD_BORDER,
        heading_color=AMBER_FG,
        body_color=AMBER_FG,
    )

    draw_card(
        draw,
        fonts,
        (565, 756, 1080, 962),
        "Layer 5: Sentinel Holdback & SSE Commit Gate",
        [
            "Hold tokens until [1] marker or ~160 chars arrive",
            "If NO_RELEVANT_INFO_FOUND -> cancel stream before",
            "typing starts -> emit insufficient_document_evidence",
            "Strip <think> chains & unwrap `[1]` code backticks",
            "Renumber [1], [2] and filter unused source chunks",
        ],
        fill=AMBER_BG,
        outline=AMBER_BORDER,
    )

    draw_card(
        draw,
        fonts,
        (1130, 756, 1552, 962),
        "Verified Response + Audit Trail",
        [
            "Answer + deduplicated documents_used + snippets",
            "Save to InMemorySessionStore (condense + export)",
            "Emit 1 JSON query_trace line (timings & tokens)",
            "Available as JSON, SSE Stream, or Export (.md/.json)",
        ],
        fill=GREEN_BG,
        outline=GREEN_BORDER,
        heading_color=GREEN_FG,
        body_color=GREEN_FG,
    )

    draw_polyline_arrow(
        draw,
        [(815, 696), (815, 725), (280, 725), (280, 756)],
        color=GREEN_BORDER,
        label="Top-6 Evidence",
        fonts=fonts,
        label_pos=(480, 712),
    )
    draw_arrow(draw, (515, 859), (565, 859), color=ARROW)
    draw_arrow(
        draw,
        (1080, 859),
        (1130, 859),
        color=GREEN_BORDER,
        label="Grounded",
        fonts=fonts,
        label_offset=(-28, -22),
    )

    out = OUT_DIR / "02_rag_and_guardrails_pipeline.png"
    img.save(out, format="PNG")
    return out


def build_llm_and_api_lifecycle_diagram(fonts: ThemeFonts) -> Path:
    width, height = 1600, 920
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    draw_header(
        draw,
        fonts,
        width,
        "3. Multi-Provider LLM Routing, SSE Streaming Protocol & API Contract",
        "Complexity-aware reasoning depth, safe stream failover, and complete endpoint & error reference",
    )

    # Top Left: Reasoning Router
    draw_group_box(
        draw,
        fonts,
        (28, 108, 975, 494),
        "Multi-Provider LLM Router (app/services/llm_client.py)",
        fill="#f8fafc",
        outline="#94a3b8",
    )
    draw_card(
        draw,
        fonts,
        (48, 154, 330, 304),
        "Task Classification",
        [
            "1. Follow-up Rewrite (<= 320 tok)",
            "   thinking OFF (fast ~0.7s)",
            "2. Simple Lookup",
            "   light reasoning (effort = low)",
            "3. Analytical / Calc / Multi-Part",
            "   deep reasoning (effort = high)",
        ],
        fill=BLUE_BG,
        outline=BLUE_BORDER,
    )

    draw_card(
        draw,
        fonts,
        (382, 154, 654, 304),
        "Ordered Provider Chain",
        [
            "Primary:  Runware DeepSeek V4.1",
            "Fallback 1: Gemini 3.5 Flash Lite",
            "Fallback 2: Groq Qwen 3.8 27B",
            "Optional: OpenAI-compatible URL",
            "Unconfigured keys auto-skipped",
        ],
        fill=CARD_BG,
        outline=GOLD_BORDER,
    )

    draw_card(
        draw,
        fonts,
        (704, 154, 955, 304),
        "Stream Safety Guards",
        [
            "1. 20s first-token timeout",
            "2. Empty-stream detection",
            "3. Retry 429/503 spikes",
            "Failover ONLY before first",
            "visible token (no splicing)",
        ],
        fill=AMBER_BG,
        outline=AMBER_BORDER,
    )

    draw_card(
        draw,
        fonts,
        (48, 332, 466, 474),
        "Sanitization & Model Resilience",
        [
            "Strips <think>...</think> even mid-stream",
            "DeepSeek reasoning_content stays internal",
            "Auto-retries without reasoning params if rejected",
            "4096 token budget gives thinking models room",
        ],
        fill=CARD_BG,
        outline=CARD_BORDER,
    )

    draw_card(
        draw,
        fonts,
        (510, 332, 955, 474),
        "Per-Query Trace Telemetry (app/core/tracing.py)",
        [
            "Records: mode, docs_in_scope, top_similarity, top_rerank",
            "Timings: scope, condense, retrieve_rerank, first_token, llm",
            "Provider attempts & errors (silent failovers stay visible)",
            "Token usage breakdown: prompt / completion / reasoning",
        ],
        fill=GREEN_BG,
        outline=GREEN_BORDER,
    )

    draw_arrow(draw, (330, 229), (382, 229))
    draw_arrow(draw, (654, 229), (704, 229))
    draw_arrow(draw, (518, 304), (257, 332))
    draw_arrow(draw, (828, 304), (732, 332))

    # Top Right: SSE Event Flow
    draw_group_box(
        draw,
        fonts,
        (1010, 108, 1572, 494),
        "SSE Streaming Event Lifecycle (/api/v1/query/stream)",
        fill="#eff6ff",
        outline="#93c5fd",
    )
    draw_card(
        draw,
        fonts,
        (1032, 154, 1550, 226),
        "Step 1  event: status  (stage: searching / generating)",
        [
            "Immediate progress feedback while Qdrant + cross-encoder run",
        ],
        fill=CARD_BG,
        outline=BLUE_BORDER,
    )
    draw_card(
        draw,
        fonts,
        (1032, 238, 1550, 310),
        "Step 2  Commit Gate -> event: citations + status (writing)",
        [
            "Fires ONLY after real answer text or [1] arrives (not on refusals)",
        ],
        fill=GOLD_BG,
        outline=GOLD_BORDER,
    )
    draw_card(
        draw,
        fonts,
        (1032, 322, 1550, 394),
        "Step 3  event: token  (streamed text deltas)",
        [
            "Client renders smooth typewriter pace; tail checks partial sentinel",
        ],
        fill=CARD_BG,
        outline=CARD_BORDER,
    )
    draw_card(
        draw,
        fonts,
        (1032, 406, 1550, 478),
        "Step 4  event: done  (or event: error)  ->  Final citations + metrics",
        [
            "Contains is_found, response_type, refusal_reason, final_answer",
        ],
        fill=GREEN_BG,
        outline=GREEN_BORDER,
    )

    # Bottom Full Width: REST API & Response Discriminator Matrix
    draw_group_box(
        draw,
        fonts,
        (28, 514, 1572, 896),
        "Complete REST API Surface & Outcome Contract",
        fill="#ffffff",
        outline="#cbd5e1",
    )
    draw_card(
        draw,
        fonts,
        (48, 560, 775, 876),
        "Endpoints (OpenAPI 3.1 + Postman Collection)",
        [
            "GET    /                               Interactive Web UI",
            "GET    /health                         Readiness & DB/LLM info",
            "POST   /api/v1/documents/upload        Multi-file index (201/200)",
            "GET    /api/v1/documents               List uploaded doc metadata",
            "DELETE /api/v1/documents/{doc_id}      Delete single doc (or 404)",
            "DELETE /api/v1/documents/reset         Clear index & sessions",
            "POST   /api/v1/query                   Grounded JSON Q&A",
            "POST   /api/v1/query/stream            SSE token + citation stream",
            "GET    /api/v1/sessions/{id}/export    Transcript (?format=md|json)",
            "DELETE /api/v1/sessions/{id}           Clear session & transcript",
        ],
        fill="#f8fafc",
        outline=CARD_BORDER,
        mono=True,
    )

    draw_card(
        draw,
        fonts,
        (800, 560, 1185, 876),
        "QueryResponse Discriminators",
        [
            "response_type = 'grounded':",
            "  is_found=true, answer with [1][2],",
            "  documents_used, ranked citations + snippets",
            "",
            "response_type = 'not_found':",
            "  is_found=false, fixed safe refusal,",
            "  refusal_reason in:",
            "   • no_documents_indexed",
            "   • selected_documents_missing",
            "   • low_retrieval_relevance",
            "   • insufficient_document_evidence",
            "",
            "response_type = 'conversational':",
            "  Polite reply + loaded doc preview",
        ],
        fill=AMBER_BG,
        outline=AMBER_BORDER,
    )

    draw_card(
        draw,
        fonts,
        (1205, 560, 1552, 876),
        "Typed Exception -> HTTP Status",
        [
            "400  UnsupportedFormatError",
            "     (.doc, .exe, etc.)",
            "401  Invalid / Missing X-API-Key",
            "404  Document / Session Not Found",
            "413  FileTooLarge / TooManyFiles",
            "422  EmptyDocument / Corrupt PDF",
            "     or Blank Question validation",
            "502  LLMProviderError",
            "     (upstream providers failed)",
            "503  VectorStoreError / No Key",
        ],
        fill=RED_BG,
        outline=RED_BORDER,
        heading_color=RED_FG,
        body_color=RED_FG,
    )

    out = OUT_DIR / "03_llm_routing_and_api_lifecycle.png"
    img.save(out, format="PNG")
    return out


def write_ascii_diagrams() -> Path:
    ascii_doc = """# AlPrime Document RAG Assistant — Text / ASCII Architecture & Flowcharts

These plain-text diagrams mirror the PNG flowcharts in this folder so they remain
readable in any terminal, text editor, or offline viewer.

================================================================================
1. SYSTEM ARCHITECTURE & PROJECT STRUCTURE
================================================================================

  +-------------------------+        +------------------------------------------+
  | Clients                 |        | FastAPI Layer (app/main.py, app/api/)    |
  |  - Web UI (/)           |------->|  - X-API-Key Guard (core/security.py)    |
  |  - Swagger (/docs)      |        |  - CORS & Lifespan DI (dependencies.py)  |
  |  - Postman / cURL       |        |  - Typed Error Mapper (400..503)         |
  +-------------------------+        +--------------------+---------------------+
                                                          |
                +-----------------------------------------+----------------------------------------+
                | Ingestion (/api/v1/documents/*)                                                  | Query (/api/v1/query*, /sessions/*)
                v                                                                                  v
  +-------------------------------------+                            +--------------------------------------------------+
  | Document Ingestion Services         |                            | RAG & Guardrail Services                         |
  |  - document_parser.py (PDF/DOCX/TXT)|                            |  - query_intent.py (greeting/overview/reasoning) |
  |  - text_splitter.py (800 / 150)     |                            |  - rag_service.py (5-layer pipeline + SSE gate)  |
  |  - SHA-256 duplicate protection     |                            |  - citations.py (compact [1] + sentinel strip)   |
  |  - All-or-nothing batch rollback    |                            |  - replies.py (templated refusals & greetings)   |
  +-----------------+-------------------+                            |  - session_store.py (scope memory + export)      |
                    |                                                |  - tracing.py (structured JSON query_trace)      |
                    |                                                +------------------------+-------------------------+
                    |                                                                         |
                    v                                                                         v
  +--------------------------------------------------+               +--------------------------------------------------+
  | Vector Storage & Local ONNX Inference            |               | Multi-Provider LLM Router (llm_client.py)        |
  |  - Qdrant (Embedded Disk or Docker :6333)        |<------------->|  1. Runware DeepSeek V4.1 Flash (low/high think) |
  |  - FastEmbed bge-small-en-v1.5 (384-d embeddings)|               |  2. Google Gemini 3.5 Flash Lite                 |
  |  - FastEmbed ms-marco-MiniLM-L-6-v2 (reranker)   |               |  3. Groq Qwen 3.8 27B (+ 20s stall failover)     |
  +--------------------------------------------------+               +--------------------------------------------------+

================================================================================
2. DOCUMENT INGESTION PIPELINE
================================================================================

  [Upload PDF/DOCX/TXT]
         |
         v
  [Validate <=10 files, <=20MB, allowed extension] --fail--> HTTP 400 / 413
         |
         v
  [Compute SHA-256 Digest] --already in Qdrant/batch--> HTTP 200 status="already_indexed"
         |
         v (new file)
  [Extract Text (PDF per-page / DOCX Markdown tables / TXT)] --empty/scanned/corrupt--> HTTP 422
         |
         v
  [Recursive Split (800 chars, 150 overlap) + Metadata (doc_id, page, sha256)]
         |
         v
  [ONNX Embed (bge-small-en-v1.5) + Qdrant Upsert] --error--> [Atomic Batch Rollback]
         |
         v
  HTTP 201 Created (doc_id, page_count, chunk_count)

================================================================================
3. 5-LAYER ANTI-HALLUCINATION RAG & STREAMING PIPELINE
================================================================================

  [Question + optional session_id, doc_ids, top_k]
         |
         +--> [Pure Greeting / Thanks / Help?] --Yes--> [Templated Reply + Doc Count (0 LLM calls)]
         |
         v (Factual Query)
  [Layer 1: Empty / Stale doc_ids Guard] --None left--> [Templated Refusal (no_documents_indexed / selected_documents_missing)]
         |
         +--> [Whole-Doc Overview Prompt?] --Yes--> [Fetch Lead Chunks per Doc + Semantic Search]
         |
         v (Specific Topic)
  [Session History?] --Yes--> [Rewrite Follow-Up (Thinking Off)] --> [Dual Search: Rewrite + Original]
         |
         v
  [Layer 2: Dense Search Top-10 & Cosine Gate >= 0.45] --0 pass--> [Templated Refusal (low_retrieval_relevance)]
         |
         v
  [Layer 3: Cross-Encoder (>= -6.0) + MaxP 2-Sentence Window + Bounded Lexical Rescue] --0 pass--> [Refusal (low_retrieval_relevance)]
         |
         v (Top-6 Passages)
  [Layer 4: XML Escape + <document_context> Fence + Temp 0.0 + RoutingLLMClient]
         |
         v
  [Layer 5: SSE Commit Gate (Hold until [1] or 160 chars) + Sentinel Check (NO_RELEVANT_INFO_FOUND)]
         |
         +--> [Sentinel / Refusal] --> [Templated Refusal Naming Searched Docs (insufficient_document_evidence)]
         |
         v (Grounded)
  [Unwrap `[1]` Code Marks, Renumber [1][2], Narrow Citations] --> [Session Transcript + JSON QueryTrace]

================================================================================
4. REST API ENDPOINTS & ERROR CONTRACT
================================================================================

  GET    /                                 HTML Chat & Upload UI
  GET    /health                           Readiness, Qdrant mode, LLM chain, indexed counts
  POST   /api/v1/documents/upload          Index PDF/DOCX/TXT (201 new, 200 already_indexed)
  GET    /api/v1/documents                 Inventory of indexed files
  DELETE /api/v1/documents/{doc_id}        Remove a document (404 if missing)
  DELETE /api/v1/documents/reset           Wipe Qdrant collection & sessions
  POST   /api/v1/query                     Synchronous JSON RAG response
  POST   /api/v1/query/stream              SSE events: status -> citations -> token* -> done | error
  GET    /api/v1/sessions/{id}/export      Download transcript (?format=markdown|json, 404 if empty)
  DELETE /api/v1/sessions/{id}             Clear session history & transcript

  Status Codes: 200 OK | 201 Created | 400 Bad Ext | 401 Auth | 404 Missing | 413 Too Large | 422 Empty/Invalid | 502 LLM Fail | 503 DB/No Key
"""
    out = OUT_DIR / "ARCHITECTURE_FLOWCHARTS.txt"
    out.write_text(ascii_doc, encoding="utf-8")
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fonts = load_fonts()
    paths = [
        build_system_and_api_diagram(fonts),
        build_rag_and_guardrails_diagram(fonts),
        build_llm_and_api_lifecycle_diagram(fonts),
        write_ascii_diagrams(),
    ]
    for path in paths:
        print("Generated:", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
