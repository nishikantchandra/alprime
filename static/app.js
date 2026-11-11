(() => {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const uploadStatus = document.getElementById("uploadStatus");
  const composerAttachBtn = document.getElementById("composerAttachBtn");
  const composerToast = document.getElementById("composerToast");
  const docList = document.getElementById("docList");
  const docCount = document.getElementById("docCount");
  const resetBtn = document.getElementById("resetBtn");
  const exportBtn = document.getElementById("exportBtn");
  const exportFormatSelect = document.getElementById("exportFormatSelect");
  const newChatBtn = document.getElementById("newChatBtn");
  const queryForm = document.getElementById("queryForm");
  const questionInput = document.getElementById("questionInput");
  const sendBtn = document.getElementById("sendBtn");
  const stopBtn = document.getElementById("stopBtn");
  const chatMessages = document.getElementById("chatMessages");
  const streamToggle = document.getElementById("streamToggle");
  const apiKeyGroup = document.getElementById("apiKeyGroup");
  const apiKeyInput = document.getElementById("apiKeyInput");
  const healthBadge = document.getElementById("healthBadge");
  const sessionLabel = document.getElementById("sessionLabel");

  let sessionId = createSessionId();
  let activeController = null;
  let toastTimer = null;
  const transcriptTurns = [];
  updateSessionLabel();
  updateExportState();

  function createSessionId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID();
    }
    return "sess-" + Date.now() + "-" + Math.random().toString(16).slice(2, 8);
  }

  function updateSessionLabel() {
    sessionLabel.textContent = `Session: ${sessionId.slice(0, 8)} • Multi-turn follow-ups enabled`;
  }

  function updateExportState() {
    const empty = transcriptTurns.length === 0;
    if (exportBtn) exportBtn.disabled = empty;
    if (exportFormatSelect) exportFormatSelect.disabled = empty;
  }

  function clearTranscript() {
    transcriptTurns.length = 0;
    updateExportState();
  }

  function recordTranscriptTurn(entry) {
    transcriptTurns.push({
      turn: transcriptTurns.length + 1,
      timestamp: entry.timestamp || new Date().toISOString(),
      question: entry.question,
      answer: entry.answer || "",
      response_type: entry.response_type || "grounded",
      is_found: Boolean(entry.is_found),
      refusal_reason: entry.refusal_reason || null,
      status_detail: entry.status_detail || null,
      duration_ms: typeof entry.duration_ms === "number" ? entry.duration_ms : null,
      documents_used: entry.documents_used || [],
      citations: entry.citations || [],
    });
    updateExportState();
  }

  function buildMarkdownTranscript() {
    const docNames = Array.from(docList.querySelectorAll(".doc-name")).map((el) => el.textContent);
    const lines = [
      "# AlPrime Document Assistant — Conversation Transcript",
      `- **Session ID:** \`${sessionId}\``,
      `- **Exported At:** ${new Date().toISOString()}`,
      `- **Indexed Documents (${docNames.length}):** ${docNames.length ? docNames.join(", ") : "None"}`,
      "",
      "---",
      "",
    ];

    for (const item of transcriptTurns) {
      lines.push(`### Turn ${item.turn} — User (${item.timestamp})`);
      lines.push(item.question);
      lines.push("");

      const statusText =
        item.response_type === "grounded"
          ? "Verified Document Evidence"
          : item.response_type === "conversational"
          ? "Assistant Ready"
          : item.refusal_reason || item.response_type || "No Supporting Evidence Found";
      const timing = typeof item.duration_ms === "number" ? ` • ${(item.duration_ms / 1000).toFixed(2)}s` : "";

      lines.push(`### Turn ${item.turn} — Assistant (\`${statusText}${timing}\`)`);
      lines.push(item.answer);
      if (item.status_detail) {
        lines.push(`\n> *Note: ${item.status_detail}*`);
      }
      if (item.citations && item.citations.length) {
        lines.push("");
        lines.push("**Cited Sources:**");
        item.citations.forEach((cit, idx) => {
          const cleanSnippet = String(cit.snippet || "").replace(/\s+/g, " ").trim();
          lines.push(`- **[${idx + 1}] ${cit.document_name} (Page ${cit.page}):** ${cleanSnippet}`);
        });
      }
      lines.push("", "---", "");
    }
    return lines.join("\n");
  }

  function exportTranscript() {
    if (!transcriptTurns.length) return;
    const fmt = exportFormatSelect ? exportFormatSelect.value : "markdown";
    const shortId = sessionId.slice(0, 8);
    let content;
    let mime;
    let ext;

    if (fmt === "json") {
      const docNames = Array.from(docList.querySelectorAll(".doc-name")).map((el) => el.textContent);
      content = JSON.stringify(
        {
          session_id: sessionId,
          exported_at: new Date().toISOString(),
          indexed_documents: docNames,
          total_turns: transcriptTurns.length,
          turns: transcriptTurns,
        },
        null,
        2
      );
      mime = "application/json;charset=utf-8";
      ext = "json";
    } else {
      content = buildMarkdownTranscript();
      mime = "text/markdown;charset=utf-8";
      ext = "md";
    }

    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `alprime-transcript-${shortId}.${ext}`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function buildHeaders(extra = {}) {
    const headers = { ...extra };
    const key = apiKeyInput.value.trim();
    if (key) {
      headers["X-API-Key"] = key;
    }
    return headers;
  }

  function showUploadStatus(message, type) {
    uploadStatus.textContent = message;
    uploadStatus.className = `status-box ${type}`;
    uploadStatus.classList.remove("hidden");

    composerToast.textContent = message;
    composerToast.className = `composer-toast ${type}`;
    composerToast.classList.remove("hidden");
    if (toastTimer) clearTimeout(toastTimer);
    if (type !== "info") {
      toastTimer = setTimeout(() => {
        composerToast.classList.add("hidden");
      }, 4500);
    }
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function autoResizeTextarea() {
    questionInput.style.height = "auto";
    questionInput.style.height = `${Math.min(questionInput.scrollHeight, 140)}px`;
  }

  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (!res.ok) throw new Error("Health check failed");
      const data = await res.json();
      if (apiKeyGroup) {
        apiKeyGroup.classList.toggle("hidden", !data.auth_enabled);
      }
      if (data.llm_configured) {
        healthBadge.textContent = "System Ready";
        healthBadge.className = "status-pill status-ok";
      } else {
        healthBadge.textContent = "Configuration Required";
        healthBadge.className = "status-pill status-warn";
      }
    } catch {
      healthBadge.textContent = "Service Offline";
      healthBadge.className = "status-pill status-warn";
    }
  }

  // Signature of the last rendered list so background re-syncs don't rebuild the DOM needlessly.
  let lastDocSignature = null;

  // Re-sync the sidebar with what the server actually has. The knowledge base is shared and can
  // change outside this tab (Reset in another tab, Swagger /docs, scripts, container restarts),
  // so this runs on load, on tab focus, periodically, and right before every question.
  async function refreshDocuments() {
    try {
      const res = await fetch("/api/v1/documents", { headers: buildHeaders(), cache: "no-store" });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Failed to load documents");
      }
      const data = await res.json();
      const docs = data.documents || [];
      docCount.textContent = String(data.total_documents);
      const signature = docs.map((d) => `${d.doc_id}:${d.chunk_count}`).join("|");
      if (signature !== lastDocSignature) {
        lastDocSignature = signature;
        renderDocList(docs);
      }
      return docs;
    } catch (err) {
      lastDocSignature = null;
      docList.textContent = "";
      const li = document.createElement("li");
      li.className = "empty-docs";
      li.textContent = err.message;
      docList.appendChild(li);
      return null;
    }
  }

  function renderDocList(docs) {
    // Preserve the user's checkbox selection across re-renders; IDs that no longer exist drop out.
    const previouslySelected = new Set(
      Array.from(document.querySelectorAll(".doc-scope-checkbox:checked")).map((el) => el.value)
    );
    docList.textContent = "";
    if (!docs.length) {
      const empty = document.createElement("li");
      empty.className = "empty-docs";
      empty.textContent = "No documents indexed yet. Upload files above or click + in chat.";
      docList.appendChild(empty);
      return;
    }

    for (const doc of docs) {
      const li = document.createElement("li");
      li.className = "doc-item";

      const label = document.createElement("label");
      label.className = "doc-select-label";

      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.className = "doc-scope-checkbox";
      checkbox.value = doc.doc_id;
      checkbox.checked = previouslySelected.has(doc.doc_id);

      const info = document.createElement("div");
      const nameEl = document.createElement("div");
      nameEl.className = "doc-name";
      nameEl.textContent = doc.filename;

      const metaEl = document.createElement("div");
      metaEl.className = "doc-meta";
      metaEl.textContent = `${doc.file_type.toUpperCase()} • ${doc.page_count} pg • ${doc.chunk_count} chunks`;

      info.appendChild(nameEl);
      info.appendChild(metaEl);
      label.appendChild(checkbox);
      label.appendChild(info);

      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "btn btn-danger-text";
      delBtn.textContent = "Delete";
      delBtn.addEventListener("click", () => deleteDocument(doc.doc_id, doc.filename));

      li.appendChild(label);
      li.appendChild(delBtn);
      docList.appendChild(li);
    }
  }

  async function uploadFiles(files) {
    if (!files || !files.length) return;
    const formData = new FormData();
    for (const file of files) {
      formData.append("files", file);
    }

    showUploadStatus(`Uploading & indexing ${files.length} document(s)...`, "info");
    try {
      const res = await fetch("/api/v1/documents/upload", {
        method: "POST",
        headers: buildHeaders(),
        body: formData,
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || "Upload failed");
      }
      const wasEmpty = Number(docCount.textContent || "0") === 0;
      const fresh = data.documents.filter((d) => d.status !== "already_indexed");
      const dupes = data.documents.filter((d) => d.status === "already_indexed");
      const skipped = Number(data.duplicates_skipped || 0);
      const parts = [];
      if (fresh.length) {
        parts.push(
          `Indexed ${fresh.length} file(s): ${fresh.map((d) => d.filename).join(", ")} ` +
            `(${data.total_chunks_indexed} chunks).`
        );
      }
      if (skipped) {
        const dupeNames = dupes.map((d) => d.filename).join(", ");
        parts.push(
          `Skipped ${skipped} duplicate file(s)${dupeNames ? ` already indexed as ${dupeNames}` : ""}.`
        );
      }
      showUploadStatus(parts.join(" ") || data.message, fresh.length ? "ok" : "info");
      if (wasEmpty) {
        sessionId = createSessionId();
        updateSessionLabel();
      }
      await refreshDocuments();
    } catch (err) {
      showUploadStatus(err.message, "err");
    } finally {
      fileInput.value = "";
    }
  }

  async function deleteDocument(docId, filename) {
    try {
      const res = await fetch(`/api/v1/documents/${encodeURIComponent(docId)}`, {
        method: "DELETE",
        headers: buildHeaders(),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Delete failed");
      }
      showUploadStatus(`Removed '${filename}' from index.`, "info");
      await refreshDocuments();
    } catch (err) {
      showUploadStatus(err.message, "err");
    }
  }

  async function resetKnowledgeBase() {
    if (!confirm("Remove all uploaded documents and vectors from Qdrant?")) return;
    try {
      const res = await fetch("/api/v1/documents/reset", {
        method: "DELETE",
        headers: buildHeaders(),
      });
      if (!res.ok) throw new Error("Reset failed");
      sessionId = createSessionId();
      updateSessionLabel();
      clearTranscript();
      chatMessages.querySelectorAll(".msg-row").forEach((r) => r.remove());
      showUploadStatus("Knowledge base cleared and fresh chat session started.", "info");
      await refreshDocuments();
    } catch (err) {
      showUploadStatus(err.message, "err");
    }
  }

  function appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "msg-row msg-row-user";
    const div = document.createElement("div");
    div.className = "msg msg-user";
    div.textContent = text;
    row.appendChild(div);
    chatMessages.appendChild(row);
    scrollToBottom();
  }

  function setWorkingBadge(badgeEl, text) {
    badgeEl.className = "msg-badge badge-working";
    badgeEl.textContent = "";
    const dot = document.createElement("span");
    dot.className = "pulse-dot";
    const label = document.createElement("span");
    label.textContent = text;
    badgeEl.appendChild(dot);
    badgeEl.appendChild(label);
  }

  function applyOutcomeBadge(
    badgeEl,
    isFound,
    durationMs = null,
    responseType = null,
    refusalReason = null
  ) {
    const timeSuffix = typeof durationMs === "number" ? ` • ${(durationMs / 1000).toFixed(2)}s` : "";
    if (responseType === "conversational") {
      badgeEl.className = "msg-badge badge-chat";
      badgeEl.textContent = `Assistant Ready${timeSuffix}`;
      return;
    }
    if (isFound) {
      badgeEl.className = "msg-badge badge-grounded";
      badgeEl.textContent = `Verified Document Evidence${timeSuffix}`;
      return;
    }
    const reasonLabels = {
      no_documents_indexed: "Knowledge Base Empty",
      selected_documents_missing: "Selected Documents No Longer Exist",
      low_retrieval_relevance: "No Relevant Match in Documents",
      insufficient_document_evidence: "Not Stated in Uploaded Documents",
    };
    const label = reasonLabels[refusalReason] || "No Supporting Evidence Found";
    badgeEl.className = "msg-badge badge-notfound";
    badgeEl.textContent = `${label}${timeSuffix}`;
  }

  function createAssistantMessageShell() {
    const row = document.createElement("div");
    row.className = "msg-row";

    const avatar = document.createElement("div");
    avatar.className = "msg-avatar avatar-ai";
    avatar.textContent = "AI";

    const wrapper = document.createElement("div");
    wrapper.className = "msg msg-assistant";

    const header = document.createElement("div");
    header.className = "msg-header";

    const badge = document.createElement("div");
    setWorkingBadge(badge, "Searching uploaded documents...");

    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "copy-btn hidden";
    copyBtn.textContent = "Copy";

    header.appendChild(badge);
    header.appendChild(copyBtn);

    const body = document.createElement("div");
    body.className = "msg-body";

    const streamTextNode = document.createTextNode("");
    const cursor = document.createElement("span");
    cursor.className = "typing-cursor";
    body.appendChild(streamTextNode);
    body.appendChild(cursor);

    const detailNote = document.createElement("div");
    detailNote.className = "msg-detail hidden";

    const citationsContainer = document.createElement("div");
    citationsContainer.className = "citations-box hidden";

    wrapper.appendChild(header);
    wrapper.appendChild(body);
    wrapper.appendChild(detailNote);
    wrapper.appendChild(citationsContainer);
    row.appendChild(avatar);
    row.appendChild(wrapper);
    chatMessages.appendChild(row);
    scrollToBottom();

    return { badge, copyBtn, body, streamTextNode, cursor, detailNote, citationsContainer };
  }

  function renderStatusDetail(detailEl, statusDetail) {
    if (!statusDetail) {
      detailEl.textContent = "";
      detailEl.classList.add("hidden");
      return;
    }
    detailEl.textContent = statusDetail;
    detailEl.classList.remove("hidden");
  }

  function focusCitation(container, citIdx) {
    const details = container.querySelector("details");
    if (!details) return;
    details.open = true;
    const cards = details.querySelectorAll(".citation-card");
    cards.forEach((c) => c.classList.remove("citation-card-active"));
    const target = cards[citIdx];
    if (target) {
      target.classList.add("citation-card-active");
      target.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  function appendInlineRichTokens(
    parentEl,
    lineText,
    citations = [],
    onSelectCitation = null
  ) {
    const tokenRegex = /(\*\*[^*]+\*\*|\[Source:[^\]]+\]|\[\d+\](?!\())/g;
    let lastIdx = 0;
    for (const match of lineText.matchAll(tokenRegex)) {
      const idx = match.index ?? 0;
      if (idx > lastIdx) {
        parentEl.appendChild(document.createTextNode(lineText.slice(lastIdx, idx)));
      }
      const matched = match[0];
      if (matched.startsWith("**") && matched.endsWith("**")) {
        const strong = document.createElement("strong");
        strong.textContent = matched.slice(2, -2);
        parentEl.appendChild(strong);
      } else if (matched.startsWith("[Source:")) {
        const pill = document.createElement("span");
        pill.className = "inline-citation";
        pill.textContent = matched;
        parentEl.appendChild(pill);
      } else {
        const numMatch = matched.match(/^\[(\d+)\]$/);
        const citIdx = numMatch ? Number(numMatch[1]) - 1 : -1;
        if (citations && citIdx >= 0 && citIdx < citations.length) {
          const cit = citations[citIdx];
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "inline-citation inline-citation-num";
          btn.textContent = String(citIdx + 1);
          btn.title = `${cit.document_name} • Page ${cit.page}`;
          if (onSelectCitation) {
            btn.addEventListener("click", () => onSelectCitation(citIdx));
          }
          parentEl.appendChild(btn);
        } else {
          parentEl.appendChild(document.createTextNode(matched));
        }
      }
      lastIdx = idx + matched.length;
    }
    if (lastIdx < lineText.length) {
      parentEl.appendChild(document.createTextNode(lineText.slice(lastIdx)));
    }
  }

  // Safely render markdown headings, bullet lists, **bold** text, and interactive citation pills via DOM nodes.
  function renderRichAnswer(container, text, citations = [], citationsContainer = null) {
    container.textContent = "";
    const onSelectCitation = citationsContainer
      ? (citIdx) => focusCitation(citationsContainer, citIdx)
      : null;
    const lines = text.split("\n");
    let currentList = null;

    const flushList = () => {
      if (currentList) {
        container.appendChild(currentList);
        currentList = null;
      }
    };

    for (let i = 0; i < lines.length; i += 1) {
      const line = lines[i];
      const trimmed = line.trim();

      const headingMatch = trimmed.match(/^#{1,3}\s+(.+)$/);
      if (headingMatch) {
        flushList();
        const headingEl = document.createElement("div");
        headingEl.className = "msg-heading";
        appendInlineRichTokens(headingEl, headingMatch[1], citations, onSelectCitation);
        container.appendChild(headingEl);
        continue;
      }

      const bulletMatch = trimmed.match(/^[-*]\s+(.+)$/);
      if (bulletMatch) {
        if (!currentList) {
          currentList = document.createElement("ul");
          currentList.className = "msg-list";
        }
        const li = document.createElement("li");
        appendInlineRichTokens(li, bulletMatch[1], citations, onSelectCitation);
        currentList.appendChild(li);
        continue;
      }

      flushList();
      if (!trimmed) {
        container.appendChild(document.createTextNode("\n"));
        continue;
      }

      const lineDiv = document.createElement("div");
      appendInlineRichTokens(lineDiv, line, citations, onSelectCitation);
      container.appendChild(lineDiv);
    }
    flushList();
  }

  function enableCopyButton(copyBtn, rawText) {
    copyBtn.classList.remove("hidden");
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(rawText);
        copyBtn.textContent = "Copied";
        setTimeout(() => {
          copyBtn.textContent = "Copy";
        }, 1500);
      } catch {
        copyBtn.textContent = "Copy failed";
      }
    });
  }

  function renderCitations(container, citations, documentsUsed) {
    container.textContent = "";
    if (!citations || !citations.length) {
      container.classList.add("hidden");
      return;
    }
    container.classList.remove("hidden");
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    const countWord = citations.length === 1 ? "excerpt" : "excerpts";
    summary.textContent = `Sources (${documentsUsed.join(", ")}) — ${citations.length} ${countWord}`;
    details.appendChild(summary);

    citations.forEach((cit, idx) => {
      const card = document.createElement("div");
      card.className = "citation-card";
      card.dataset.citIdx = String(idx);

      const meta = document.createElement("div");
      meta.className = "citation-meta";
      meta.textContent = `[${idx + 1}] ${cit.document_name} • Page ${cit.page}`;

      const snippet = document.createElement("div");
      snippet.className = "citation-snippet";
      snippet.textContent = cit.snippet;

      card.appendChild(meta);
      card.appendChild(snippet);
      details.appendChild(card);
    });
    container.appendChild(details);
  }

  function getSelectedDocIds() {
    const checked = document.querySelectorAll(".doc-scope-checkbox:checked");
    return checked.length ? Array.from(checked).map((el) => el.value) : null;
  }

  // Smooth wall-clock typewriter animator shared by both SSE streaming and standard mode.
  function createTypewriter(shell, signal) {
    let pending = "";
    let charBudget = 0;
    let lastTime = performance.now();
    let streamClosed = false;
    let stopped = false;

    const finishedPromise = new Promise((resolve) => {
      function step(now) {
        if (signal.aborted || stopped) {
          resolve();
          return;
        }
        const dtSeconds = Math.min((now - lastTime) / 1000, 0.05);
        lastTime = now;

        if (pending.length > 0) {
          // Adaptive ChatGPT reading pace: ~85 chars/sec normally, smoothly scaling on long answers.
          const charsPerSec = Math.min(240, Math.max(85, pending.length * 0.45));
          charBudget += charsPerSec * dtSeconds;
          const whole = Math.floor(charBudget);
          if (whole > 0) {
            let take = Math.min(whole, pending.length);
            // Snap forward up to 6 chars to nearest space so partial words do not jitter.
            if (take < pending.length) {
              const nextSpace = pending.indexOf(" ", take);
              if (nextSpace !== -1 && nextSpace - take <= 6) {
                take = nextSpace + 1;
              }
            }
            charBudget = Math.max(0, charBudget - take);
            shell.streamTextNode.textContent += pending.slice(0, take);
            pending = pending.slice(take);
            scrollToBottom();
          }
          requestAnimationFrame(step);
          return;
        }

        if (streamClosed) {
          resolve();
          return;
        }
        requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    });

    return {
      push(chunk) {
        pending += chunk;
      },
      close() {
        streamClosed = true;
      },
      abort() {
        stopped = true;
      },
      waitUntilDrained() {
        return finishedPromise;
      },
    };
  }

  async function askJson(payload, shell, signal) {
    const t0 = performance.now();
    const res = await fetch("/api/v1/query", {
      method: "POST",
      headers: buildHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
      signal,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.detail || "Query request failed");
    }
    shell.cursor.remove();
    const durationMs = Math.round(performance.now() - t0);
    applyOutcomeBadge(
      shell.badge,
      data.is_found,
      durationMs,
      data.response_type,
      data.refusal_reason
    );
    renderCitations(shell.citationsContainer, data.citations, data.documents_used || []);
    renderRichAnswer(shell.body, data.answer, data.citations || [], shell.citationsContainer);
    renderStatusDetail(
      shell.detailNote,
      data.response_type === "not_found" ? data.status_detail : null
    );
    enableCopyButton(shell.copyBtn, data.answer);
    return {
      answer: data.answer,
      is_found: Boolean(data.is_found),
      response_type: data.response_type || "grounded",
      refusal_reason: data.refusal_reason || null,
      status_detail: data.response_type === "not_found" ? data.status_detail : null,
      duration_ms: durationMs,
      documents_used: data.documents_used || [],
      citations: data.citations || [],
    };
  }

  async function askStream(payload, shell, signal) {
    const res = await fetch("/api/v1/query/stream", {
      method: "POST",
      headers: buildHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
      signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Streaming query failed");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let citationsData = [];
    let docsUsed = [];
    let donePayload = null;
    let firstTokenSeen = false;

    const typewriter = createTypewriter(shell, signal);
    // The server fails over between LLM providers on its own; this only catches a dead
    // connection (no bytes at all for STREAM_IDLE_LIMIT_MS) so the UI never spins forever.
    const STREAM_IDLE_LIMIT_MS = 90000;
    const readWithIdleLimit = () =>
      new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          reader.cancel().catch(() => {});
          reject(new Error("The server stopped responding. Please try asking again."));
        }, STREAM_IDLE_LIMIT_MS);
        reader.read().then(
          (result) => { clearTimeout(timer); resolve(result); },
          (err) => { clearTimeout(timer); reject(err); }
        );
      });

    try {
      while (true) {
        const { value, done } = await readWithIdleLimit();
        if (done) break;
        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const rawEvent of events) {
          const lines = rawEvent.split("\n");
          let eventName = "message";
          let dataLine = "";
          for (const line of lines) {
            if (line.startsWith("event:")) eventName = line.slice(6).trim();
            else if (line.startsWith("data:")) dataLine += line.slice(5).trim();
          }
          if (!dataLine) continue;
          const parsed = JSON.parse(dataLine);

          if (eventName === "status") {
            setWorkingBadge(shell.badge, parsed.message || "Processing...");
          } else if (eventName === "citations") {
            citationsData = parsed.citations || [];
            docsUsed = parsed.documents_used || [];
          } else if (eventName === "token") {
            if (!firstTokenSeen) {
              firstTokenSeen = true;
              setWorkingBadge(shell.badge, "Writing response...");
            }
            typewriter.push(parsed.text || "");
          } else if (eventName === "done") {
            donePayload = parsed;
          } else if (eventName === "error") {
            typewriter.abort();
            throw new Error(parsed.detail || "Streaming error");
          }
        }
      }
    } catch (err) {
      typewriter.abort();
      throw err;
    }

    typewriter.close();
    await typewriter.waitUntilDrained();
    shell.cursor.remove();

    if (donePayload && !signal.aborted) {
      applyOutcomeBadge(
        shell.badge,
        Boolean(donePayload.is_found),
        donePayload.duration_ms,
        donePayload.response_type,
        donePayload.refusal_reason
      );
      const finalAnswer = donePayload.final_answer || shell.streamTextNode.textContent;
      renderCitations(shell.citationsContainer, citationsData, docsUsed);
      renderRichAnswer(shell.body, finalAnswer, citationsData, shell.citationsContainer);
      renderStatusDetail(
        shell.detailNote,
        donePayload.response_type === "not_found" ? donePayload.status_detail : null
      );
      enableCopyButton(shell.copyBtn, finalAnswer);
      return {
        answer: finalAnswer,
        is_found: Boolean(donePayload.is_found),
        response_type: donePayload.response_type || "grounded",
        refusal_reason: donePayload.refusal_reason || null,
        status_detail: donePayload.response_type === "not_found" ? donePayload.status_detail : null,
        duration_ms: donePayload.duration_ms ?? null,
        documents_used: docsUsed,
        citations: citationsData,
      };
    }
    if (!signal.aborted) {
      // Stream ended without a `done` event: surface it instead of a silent spinner.
      throw new Error("The connection closed before the answer finished. Please try asking again.");
    }
    return null;
  }

  async function handleSubmit(event) {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (!question || activeController) return;
    const askedAt = new Date().toISOString();

    appendUserMessage(question);
    questionInput.value = "";
    autoResizeTextarea();
    sendBtn.disabled = true;
    stopBtn.classList.remove("hidden");

    activeController = new AbortController();
    const shell = createAssistantMessageShell();
    // Make sure the selection reflects the server's current documents before asking.
    await refreshDocuments();
    const payload = {
      question,
      doc_ids: getSelectedDocIds(),
      session_id: sessionId,
    };

    try {
      const outcome = streamToggle.checked
        ? await askStream(payload, shell, activeController.signal)
        : await askJson(payload, shell, activeController.signal);
      if (outcome) {
        recordTranscriptTurn({ question, timestamp: askedAt, ...outcome });
      }
    } catch (err) {
      shell.cursor.remove();
      const partial = shell.streamTextNode.textContent.trim();
      if (err.name === "AbortError") {
        shell.badge.className = "msg-badge badge-notfound";
        shell.badge.textContent = "Generation stopped";
        recordTranscriptTurn({
          question,
          timestamp: askedAt,
          answer: partial || "[Generation stopped by user]",
          response_type: "stopped",
          is_found: false,
        });
      } else {
        shell.badge.className = "msg-badge badge-error";
        if (partial) {
          // Keep what was already shown; just flag that generation was cut off.
          shell.badge.textContent = "Response Interrupted";
          renderStatusDetail(shell.detailNote, `The answer was cut off: ${err.message}`);
          recordTranscriptTurn({
            question,
            timestamp: askedAt,
            answer: partial,
            response_type: "error",
            is_found: false,
            status_detail: `The answer was cut off: ${err.message}`,
          });
        } else {
          shell.badge.textContent = "Service Error";
          shell.streamTextNode.textContent = err.message;
          recordTranscriptTurn({
            question,
            timestamp: askedAt,
            answer: err.message,
            response_type: "error",
            is_found: false,
          });
        }
      }
    } finally {
      activeController = null;
      sendBtn.disabled = false;
      stopBtn.classList.add("hidden");
      scrollToBottom();
      // Pick up any server-side change (e.g. KB emptied) the answer just revealed.
      refreshDocuments();
    }
  }

  // Event bindings (Sidebar dropzone + Composer paperclip + Composer drag & drop)
  dropZone.addEventListener("click", () => fileInput.click());
  composerAttachBtn.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  [dropZone, queryForm].forEach((zone) => {
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.classList.add("dragover");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("dragover");
      uploadFiles(e.dataTransfer.files);
    });
  });

  fileInput.addEventListener("change", () => uploadFiles(fileInput.files));
  resetBtn.addEventListener("click", resetKnowledgeBase);
  queryForm.addEventListener("submit", handleSubmit);

  stopBtn.addEventListener("click", () => {
    if (activeController) {
      activeController.abort();
    }
  });

  questionInput.addEventListener("input", autoResizeTextarea);
  questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      queryForm.requestSubmit();
    }
  });

  if (exportBtn) {
    exportBtn.addEventListener("click", exportTranscript);
  }

  newChatBtn.addEventListener("click", async () => {
    if (activeController) activeController.abort();
    if (chatMessages.querySelector(".msg-row")) {
      await fetch(`/api/v1/sessions/${encodeURIComponent(sessionId)}`, {
        method: "DELETE",
        headers: buildHeaders(),
      }).catch(() => {});
    }
    sessionId = createSessionId();
    updateSessionLabel();
    clearTranscript();
    chatMessages.querySelectorAll(".msg-row").forEach((r) => r.remove());
  });

  document.querySelectorAll(".chip-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      questionInput.value = btn.dataset.prompt || "";
      autoResizeTextarea();
      questionInput.focus();
    });
  });

  apiKeyInput.addEventListener("change", refreshDocuments);

  // Keep the sidebar honest when the shared knowledge base changes outside this tab.
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") refreshDocuments();
  });
  window.addEventListener("focus", () => refreshDocuments());
  setInterval(() => {
    if (document.visibilityState === "visible" && !activeController) refreshDocuments();
  }, 15000);

  checkHealth();
  refreshDocuments();
})();
