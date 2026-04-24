/* ── State ────────────────────────────────────────────────────────────────── */
let busy = false;

const $ = id => document.getElementById(id);

/* ── DOM refs ─────────────────────────────────────────────────────────────── */
const messages   = $('messages');
const queryInput = $('queryInput');
const sendBtn    = $('sendBtn');
const statusDot  = $('statusDot');
const statusLbl  = $('statusLabel');
const modelBadge = $('modelBadge');
const statText   = $('statText');
const statImages = $('statImages');
const docList    = $('docList');
const dropZone   = $('dropZone');
const fileInput  = $('fileInput');
const uploadStat = $('uploadStatus');
const traceBar   = $('traceBar');
const traceSteps = $('traceSteps');

/* ── Status polling ───────────────────────────────────────────────────────── */
async function refreshStatus() {
  try {
    const res  = await fetch('/api/status');
    const data = await res.json();

    statusDot.className = 'status-dot ' + (data.llm_connected ? 'ok' : 'err');
    statusLbl.textContent = data.llm_connected ? 'Ready' : 'LLM offline';
    modelBadge.textContent = data.model || 'unknown';
    statText.textContent   = data.stats?.text_chunks ?? '—';
    statImages.textContent = data.stats?.images ?? '—';
  } catch {
    statusDot.className   = 'status-dot err';
    statusLbl.textContent = 'Server offline';
  }
}

async function refreshDocs() {
  try {
    const res  = await fetch('/api/documents');
    const data = await res.json();
    const all  = [
      ...(data.documents || []).map(f => ({ name: f, type: 'doc' })),
      ...(data.images    || []).map(f => ({ name: f, type: 'img' })),
    ];
    docList.innerHTML = all.length
      ? all.map(f =>
          `<li class="file-item">
            <span class="fi-type">${f.type === 'doc' ? 'DOC' : 'IMG'}</span>
            ${escHtml(f.name)}
          </li>`).join('')
      : '<li class="file-item placeholder">No files yet</li>';
  } catch { /* silent */ }
}

/* ── Message helpers ──────────────────────────────────────────────────────── */
function appendMsg(role, html) {
  const wrap = document.createElement('div');
  wrap.className = `message ${role === 'user' ? 'user-msg' : 'assistant-msg'}`;
  wrap.innerHTML = `<div class="msg-bubble">${html}</div>`;
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;
  return wrap.querySelector('.msg-bubble');
}

function appendTyping() {
  return appendMsg('assistant',
    '<div class="typing-dots"><span></span><span></span><span></span></div>');
}

/* ── Simple markdown-ish formatter ───────────────────────────────────────── */
function fmt(text) {
  return escHtml(text)
    .replace(/\*\*(.+?)\*\*/g,  '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,      '<em>$1</em>')
    .replace(/`(.+?)`/g,        '<code>$1</code>')
    .replace(/\n/g,             '<br>');
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/* ── Source chips ─────────────────────────────────────────────────────────── */
function renderSources(sources, bubble) {
  if (!sources?.length) return;

  const chips = sources.map(s => {
    const typeClass = s.type === 'image' ? 'sc-img' : 'sc-text';
    const typeLabel = s.type === 'image' ? 'IMG' : 'TXT';
    return `<span class="source-chip" title="${escHtml(s.file)}">
      <span class="sc-type ${typeClass}">${typeLabel}</span>
      ${escHtml(s.title)}
      <span class="sc-score">${s.score}</span>
    </span>`;
  }).join('');

  const row = document.createElement('div');
  row.className = 'sources-row';
  row.innerHTML = chips;
  bubble.appendChild(row);

  // Thumbnails for image sources
  const imgSources = sources.filter(s => s.type === 'image' && s.thumbnail);
  if (imgSources.length) {
    const thumbRow = document.createElement('div');
    thumbRow.className = 'source-thumbs';
    imgSources.forEach(s => {
      const img = document.createElement('img');
      img.src = s.thumbnail;
      img.alt = s.title;
      img.className = 'source-thumb';
      thumbRow.appendChild(img);
    });
    bubble.appendChild(thumbRow);
  }

  messages.scrollTop = messages.scrollHeight;
}

/* ── Pipeline trace bar ───────────────────────────────────────────────────── */
function showTrace(steps) {
  if (!steps?.length) return;
  traceSteps.innerHTML = steps.map(s =>
    `<span class="trace-step">${escHtml(s)}</span>`).join('');
  traceBar.classList.remove('hidden');
}

/* ── Pipeline details panel ───────────────────────────────────────────────── */
function renderPipelineDetails(details, bubble) {
  const { hyde_document, expanded_queries, rrf_results } = details;

  const maxScore = rrf_results?.length
    ? Math.max(...rrf_results.map(r => r.rrf_score))
    : 1;

  const rrfRows = (rrf_results || []).map(r => {
    const pct  = maxScore > 0 ? (r.rrf_score / maxScore) * 100 : 0;
    const cls  = r.modality === 'image' ? 'rrf-img' : 'rrf-text';
    const lbl  = r.modality === 'image' ? 'IMG' : 'TXT';
    return `<div class="rrf-row">
      <span class="rrf-modality ${cls}">${lbl}</span>
      <span class="rrf-title" title="${escHtml(r.file)}">${escHtml(r.title || r.file)}</span>
      <div class="rrf-bar-wrap"><div class="rrf-bar" style="width:${pct.toFixed(1)}%"></div></div>
      <span class="rrf-score">${r.rrf_score}</span>
    </div>`;
  }).join('');

  const queryItems = (expanded_queries || []).map((q, i) =>
    `<li class="${i === 0 ? 'pd-q-original' : ''}">${escHtml(q)}</li>`
  ).join('');

  const el = document.createElement('details');
  el.className = 'pipeline-details';
  el.innerHTML = `
    <summary>Pipeline Details</summary>
    <div class="pd-section">
      <div class="pd-label">HyDE Document</div>
      <div class="pd-hyde-text">${escHtml(hyde_document || '—')}</div>
    </div>
    <div class="pd-section">
      <div class="pd-label">Expanded Queries (${(expanded_queries || []).length})</div>
      <ol class="pd-queries">${queryItems}</ol>
    </div>
    <div class="pd-section">
      <div class="pd-label">RRF Scores</div>
      <div class="pd-rrf-table">${rrfRows || '<span class="pd-empty">No results</span>'}</div>
    </div>`;
  bubble.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
}

/* ── Send message ─────────────────────────────────────────────────────────── */
async function send() {
  const query = queryInput.value.trim();
  if (!query || busy) return;

  busy = true;
  sendBtn.disabled = true;
  queryInput.value = '';
  autoResize();
  traceBar.classList.add('hidden');

  appendMsg('user', escHtml(query));
  const asBubble = appendTyping();

  try {
    const res = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });

    const reader    = res.body.getReader();
    const decoder   = new TextDecoder();
    let   tokenBuf  = '';
    let   firstTok  = true;
    let   sources   = null;
    let   pendingDetails = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      tokenBuf += decoder.decode(value, { stream: true });

      // Process complete SSE events
      const events = tokenBuf.split('\n\n');
      tokenBuf = events.pop(); // keep incomplete tail

      for (const raw of events) {
        const line = raw.trim();
        if (!line.startsWith('data:')) continue;
        const payload = JSON.parse(line.slice(5).trim());

        if (payload.type === 'pipeline_details') {
          pendingDetails = payload;
        }

        if (payload.type === 'trace') {
          showTrace(payload.steps);
        }

        if (payload.type === 'token') {
          if (firstTok) {
            asBubble.innerHTML = ''; // clear typing dots
            firstTok = false;
          }
          // Append as text node to avoid HTML injection
          const span = document.createElement('span');
          span.innerHTML = fmt(payload.content);
          asBubble.appendChild(span);
          messages.scrollTop = messages.scrollHeight;
        }

        if (payload.type === 'sources') {
          sources = payload.sources;
        }

        if (payload.type === 'done') {
          if (sources) renderSources(sources, asBubble);
          if (pendingDetails) renderPipelineDetails(pendingDetails, asBubble);
        }
      }
    }
  } catch (err) {
    asBubble.textContent = `Error: ${err.message}`;
  } finally {
    busy = false;
    sendBtn.disabled = false;
    queryInput.focus();
  }
}

/* ── Input events ─────────────────────────────────────────────────────────── */
function autoResize() {
  queryInput.style.height = 'auto';
  queryInput.style.height = Math.min(queryInput.scrollHeight, 140) + 'px';
}

queryInput.addEventListener('input', autoResize);
queryInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
sendBtn.addEventListener('click', send);

/* ── File upload ──────────────────────────────────────────────────────────── */
async function uploadFile(file) {
  uploadStat.className = 'upload-status';
  uploadStat.textContent = `Uploading ${file.name}…`;
  uploadStat.classList.remove('hidden');

  const form = new FormData();
  form.append('file', file);

  try {
    const res  = await fetch('/api/upload', { method: 'POST', body: form });
    const data = await res.json();

    if (!res.ok) {
      uploadStat.classList.add('err');
      uploadStat.textContent = data.error || 'Upload failed';
    } else {
      uploadStat.classList.add('ok');
      uploadStat.textContent = data.message || 'File indexed successfully';
      await Promise.all([refreshStatus(), refreshDocs()]);
    }
  } catch (err) {
    uploadStat.classList.add('err');
    uploadStat.textContent = `Network error: ${err.message}`;
  }

  setTimeout(() => uploadStat.classList.add('hidden'), 5000);
}

// Click to browse
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) uploadFile(fileInput.files[0]);
  fileInput.value = '';
});

// Drag & drop
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => {
  e.preventDefault(); dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});

/* ── Boot ─────────────────────────────────────────────────────────────────── */
(async () => {
  await Promise.all([refreshStatus(), refreshDocs()]);
  // Periodic status refresh every 30 s
  setInterval(refreshStatus, 30_000);
})();
