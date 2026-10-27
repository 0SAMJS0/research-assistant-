"use strict";

const $ = (sel) => document.querySelector(sel);
const STORE_KEY = "mra.conversations.v1";
const KEY_KEY = "mra.apiKey";

const state = { conversations: [], activeId: null, config: null, busy: false };

/* ---------- storage (best effort; the app works without it) ---------- */
function load(key, fallback) {
  try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; } catch { return fallback; }
}
function save(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
}
function apiKey() { return load(KEY_KEY, ""); }

/* ---------- HTTP ---------- */
async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const key = apiKey();
  if (key) headers.set("X-API-Key", key);
  const res = await fetch(path, { ...options, headers });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* non-JSON error */ }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res;
}

/* ---------- safe markdown (escape first, then a small subset) ---------- */
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function safeUrl(url) {
  const u = url.replace(/&amp;/g, "&");
  if (/^https?:\/\//i.test(u) || u.startsWith("/api/")) return esc(u);
  return null;
}
function inline(text) {
  let s = text;
  s = s.replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`);
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, label, url) => {
    const href = safeUrl(url);
    return href ? `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>` : label;
  });
  s = s.replace(/(^|[\s(])(https?:\/\/[^\s<)]+)/g, (m, pre, url) => `${pre}<a href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`);
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|\W)\*([^*\n]+)\*(?=\W|$)/g, "$1<em>$2</em>");
  s = s.replace(/(^|\W)_([^_\n]+)_(?=\W|$)/g, "$1<em>$2</em>");
  return s;
}
function renderMarkdown(src) {
  const lines = esc(src).split("\n");
  const out = [];
  let i = 0;
  const isTableSep = (l) => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/.test(l);
  while (i < lines.length) {
    const line = lines[i];
    if (line.startsWith("```")) {
      const code = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) code.push(lines[i++]);
      i++;
      out.push(`<pre><code>${code.join("\n")}</code></pre>`);
      continue;
    }
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) { const n = Math.min(h[1].length + 2, 6); out.push(`<h${n}>${inline(h[2])}</h${n}>`); i++; continue; }
    if (line.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      const cells = (l) => l.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => inline(c.trim()));
      const head = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|")) rows.push(cells(lines[i++]));
      out.push(`<table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${rows
        .map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
      continue;
    }
    if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
      const ordered = /^\s*\d+\./.test(line);
      const items = [];
      while (i < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[i])) {
        items.push(`<li>${inline(lines[i].replace(/^\s*([-*]|\d+\.)\s+/, ""))}</li>`);
        i++;
      }
      out.push(ordered ? `<ol>${items.join("")}</ol>` : `<ul>${items.join("")}</ul>`);
      continue;
    }
    if (line.startsWith("&gt;")) {
      const quote = [];
      while (i < lines.length && lines[i].startsWith("&gt;")) quote.push(lines[i++].replace(/^&gt;\s?/, ""));
      out.push(`<blockquote>${inline(quote.join(" "))}</blockquote>`);
      continue;
    }
    if (!line.trim()) { i++; continue; }
    const para = [];
    while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|```|&gt;|\s*([-*]|\d+\.)\s)/.test(lines[i])) para.push(lines[i++]);
    if (!para.length) para.push(lines[i++]);
    out.push(`<p>${inline(para.join("<br>"))}</p>`);
  }
  return out.join("");
}

/* ---------- element helpers ---------- */
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) node.setAttribute(k, v);
  }
  for (const c of children) if (c) node.append(c);
  return node;
}
function fmt(v) { return v === null || v === undefined ? "n/a" : Number(v).toPrecision(4).replace(/\.?0+$/, ""); }

/* ---------- report downloads (authenticated) ---------- */
function toPath(url) { try { const u = new URL(url, location.origin); return u.pathname + u.search; } catch { return url; } }
async function openReport(url, download) {
  // Open the tab synchronously inside the click so popup blockers allow it; fill it after the authenticated fetch.
  const win = download ? null : window.open("", "_blank");
  if (win) win.opener = null;
  try {
    const res = await api(toPath(url));
    const blob = await res.blob();
    const href = URL.createObjectURL(blob);
    if (win) {
      win.location.href = href;
    } else {
      const a = el("a", { href, download: download ? "research-report.md" : "research-report.html" });
      document.body.append(a); a.click(); a.remove();
    }
    setTimeout(() => URL.revokeObjectURL(href), 60_000);
  } catch (err) {
    if (win) win.close();
    alert(`Could not open report: ${err.message}`);
  }
}
document.addEventListener("click", (e) => {
  const a = e.target.closest("a[href]");
  if (!a) return;
  const href = a.getAttribute("href");
  if (/\/api\/reports\/[a-f0-9]{32}/.test(href)) {
    e.preventDefault();
    openReport(href, href.includes("format=md"));
  }
});

/* ---------- rendering ---------- */
function renderExperimentTable(analysis) {
  const wrap = el("div");
  wrap.append(el("div", { class: "muted", text: `Files: ${analysis.files.join(", ")} · ${analysis.rows} rows · grouped by ${analysis.group_by || "all rows"}` }));
  for (const comp of analysis.comparisons) {
    wrap.append(el("div", { text: `${comp.metric} — ${comp.direction} is better · baseline ${comp.baseline ?? "n/a"} · best ${comp.winner ?? "n/a"}` }));
    const table = el("table", { class: "exp-table" });
    table.append(el("thead", {}, el("tr", {}, ...["Group", "Runs", "Final mean", "Final std", "Best observed", "Best epoch", "Δ vs baseline %"].map((h) => el("th", { text: h })))));
    const body = el("tbody");
    for (const g of comp.groups) {
      body.append(el("tr", {}, ...[g.group, g.runs, fmt(g.final_mean), fmt(g.final_std), fmt(g.best_observed), fmt(g.best_epoch), fmt(g.improvement_vs_baseline_pct)].map((v) => el("td", { text: String(v) }))));
    }
    table.append(body);
    wrap.append(table);
  }
  for (const w of analysis.warnings || []) wrap.append(el("div", { class: "muted", text: `Note: ${w}` }));
  return wrap;
}

function renderAssistant(data) {
  const node = el("div", { class: "msg assistant" });
  const body = el("div", { class: "body" });
  body.innerHTML = renderMarkdown(data.answer);
  node.append(body);
  const meta = el("div", { class: "meta" });

  if (data.tool_trace?.length) {
    const trace = el("div", { class: "trace" });
    for (const t of data.tool_trace) {
      trace.append(el("span", { class: `chip ${t.status}`, title: t.error || JSON.stringify(t.arguments), text: `${t.tool} · ${t.status} · ${Math.round(t.duration_ms)} ms` }));
    }
    meta.append(el("div", {}, el("h3", { text: "MCP tools executed" }), trace));
  } else {
    meta.append(el("div", { class: "muted", text: "No tools were called for this answer." }));
  }

  if (data.experiment_tables?.length) {
    const box = el("div", {}, el("h3", { text: "Experiment comparison" }));
    data.experiment_tables.forEach((t) => box.append(renderExperimentTable(t)));
    meta.append(box);
  }

  if (data.reports?.length) {
    const box = el("div", { class: "reports" });
    for (const r of data.reports) {
      box.append(el("button", { class: "btn small", text: `View report: ${r.title}`, onclick: () => openReport(r.html_url, false) }));
      box.append(el("button", { class: "btn small", text: "Download .md", onclick: () => openReport(r.markdown_url, true) }));
    }
    meta.append(el("div", {}, el("h3", { text: "Reports" }), box));
  }

  if (data.citations?.length) {
    const list = el("ol", { class: "citations" });
    for (const c of data.citations.slice(0, 30)) {
      const li = el("li", {}, el("span", { class: "kind", text: c.type.replace("_", " ") }));
      li.append(c.url ? el("a", { href: c.url, target: "_blank", rel: "noopener noreferrer", text: c.label }) : el("span", { text: c.label }));
      if (c.detail) li.append(el("div", { class: "detail", text: c.detail }));
      list.append(li);
    }
    meta.append(el("div", {}, el("h3", { text: "Sources" }), list));
  }
  if (data.citation_check?.unverified?.length) {
    meta.append(el("div", { class: "notice err", text: `Unverified citations: ${data.citation_check.unverified.join(", ")}` }));
  }
  node.append(meta);
  return node;
}

function renderConversation() {
  const box = $("#messages");
  box.replaceChildren();
  const conv = current();
  if (!conv || !conv.messages.length) {
    box.append(emptyState);
  } else {
    for (const m of conv.messages) {
      if (m.role === "user") box.append(el("div", { class: "msg user", text: m.text }));
      else if (m.role === "error") box.append(el("div", { class: "msg error", text: m.text }));
      else box.append(renderAssistant(m.data));
    }
  }
  box.scrollTop = box.scrollHeight;
  renderConversationList();
}

function renderConversationList() {
  const ul = $("#conversations");
  ul.replaceChildren();
  if (!state.conversations.length) { ul.append(el("li", { class: "muted", text: "No conversations yet" })); return; }
  for (const c of state.conversations) {
    ul.append(el("li", {}, el("button", { class: c.id === state.activeId ? "active" : "", text: c.title, title: c.title, onclick: () => { state.activeId = c.id; renderConversation(); closeSidebar(); } })));
  }
}

function current() { return state.conversations.find((c) => c.id === state.activeId); }
function persist() { save(STORE_KEY, state.conversations.slice(0, 30)); }

/* ---------- chat ---------- */
async function send(text) {
  if (state.busy || !text.trim()) return;
  let conv = current();
  if (!conv) {
    conv = { id: crypto.randomUUID(), threadId: null, title: text.slice(0, 60), messages: [] };
    state.conversations.unshift(conv);
    state.activeId = conv.id;
  }
  conv.messages.push({ role: "user", text });
  renderConversation();
  const loading = el("div", { class: "msg loading" }, el("span", { class: "spinner" }), "Researching — the agent is calling MCP tools…");
  $("#messages").append(loading);
  $("#messages").scrollTop = $("#messages").scrollHeight;
  setBusy(true);
  try {
    const res = await api("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, thread_id: conv.threadId || undefined }),
    });
    const data = await res.json();
    conv.threadId = data.thread_id;
    conv.messages.push({ role: "assistant", data });
  } catch (err) {
    conv.messages.push({ role: "error", text: err.status === 401 ? "Unauthorized: set a valid API key (top right)." : `Error: ${err.message}` });
  } finally {
    setBusy(false);
    persist();
    renderConversation();
  }
}
function setBusy(b) { state.busy = b; $("#send").disabled = b; $("#send").textContent = b ? "Working…" : "Send"; }

/* ---------- uploads ---------- */
async function upload(input, endpoint, statusEl, describe) {
  const file = input.files[0];
  if (!file) return;
  const max = (state.config?.max_upload_mb || 20) * 1024 * 1024;
  statusEl.hidden = false;
  if (file.size > max) { statusEl.className = "notice err"; statusEl.textContent = `File is larger than ${state.config?.max_upload_mb || 20} MB.`; input.value = ""; return; }
  statusEl.className = "notice busy";
  statusEl.replaceChildren(el("span", { class: "spinner" }), `Uploading ${file.name}…`);
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await api(endpoint, { method: "POST", body: form });
    const data = await res.json();
    statusEl.className = "notice ok";
    statusEl.textContent = describe(data);
    refreshFiles();
  } catch (err) {
    statusEl.className = "notice err";
    statusEl.textContent = err.status === 401 ? "Unauthorized: set your API key." : err.message;
  } finally {
    input.value = "";
  }
}

async function refreshFiles() {
  try {
    const docs = await (await api("/api/documents")).json();
    $("#documents").replaceChildren(...docs.map((d) => el("li", { title: d.document_id, text: `${d.filename} · ${d.pages ?? "?"} pages · ${d.chunks} chunks` })));
  } catch { $("#documents").replaceChildren(); }
  try {
    const files = await (await api("/api/experiments")).json();
    $("#experiments").replaceChildren(...files.map((f) => el("li", { text: f })));
  } catch { $("#experiments").replaceChildren(); }
}

async function refreshTools() {
  try {
    const data = await (await api("/api/tools")).json();
    $("#tools").replaceChildren(...data.tools.map((t) => el("li", { title: t.description }, el("strong", { text: t.name }), el("span", { text: t.title || "" }))));
  } catch (err) {
    $("#tools").replaceChildren(el("li", { class: "muted", text: err.status === 401 ? "Set API key to list tools" : `Unavailable: ${err.message}` }));
  }
}

async function refreshStatus() {
  const status = $("#status");
  try {
    state.config = await (await fetch("/api/config")).json();
    const ready = await fetch("/readyz");
    const body = await ready.json();
    if (!ready.ok) { status.className = "status err"; status.textContent = "Not ready"; status.title = JSON.stringify(body.checks); return; }
    if (!state.config.llm_configured) { status.className = "status warn"; status.textContent = "Tools ready · LLM not configured"; }
    else { status.className = "status ok"; status.textContent = `Ready · ${state.config.model}`; }
    status.title = JSON.stringify(body.checks);
  } catch {
    status.className = "status err"; status.textContent = "Offline";
  }
}

function closeSidebar() { $("#sidebar").classList.remove("open"); }

/* ---------- init ---------- */
const emptyState = $("#empty");
document.addEventListener("DOMContentLoaded", () => {
  state.conversations = load(STORE_KEY, []);
  state.activeId = state.conversations[0]?.id ?? null;
  renderConversation();

  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); const t = $("#input").value; $("#input").value = ""; send(t); });
  $("#input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); } });
  $("#examples").addEventListener("click", (e) => { const b = e.target.closest(".example"); if (b) { $("#input").value = b.textContent; $("#input").focus(); } });
  $("#new-chat").addEventListener("click", () => { state.activeId = null; renderConversation(); closeSidebar(); $("#input").focus(); });
  $("#toggle-sidebar").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
  $("#pdf-input").addEventListener("change", (e) => upload(e.target, "/api/upload/pdf", $("#pdf-status"),
    (d) => `${d.duplicate ? "Already indexed" : "Indexed"}: ${d.filename} (${d.pages} pages, ${d.chunks} chunks)`));
  $("#exp-input").addEventListener("change", (e) => upload(e.target, "/api/upload/experiment", $("#exp-status"),
    (d) => `${d.duplicate ? "Already stored" : "Stored"}: ${d.file_name} · ${d.rows} rows · metrics: ${d.metric_columns.join(", ")}`));

  const dialog = $("#key-dialog");
  $("#open-key").addEventListener("click", () => { $("#key-input").value = apiKey(); dialog.showModal(); });
  $("#key-cancel").addEventListener("click", () => dialog.close("cancel"));
  dialog.addEventListener("close", () => {
    if (dialog.returnValue === "save") { save(KEY_KEY, $("#key-input").value.trim()); refreshTools(); refreshFiles(); }
  });

  refreshStatus();
  refreshTools();
  refreshFiles();
});
