/* ── Signal Lab — app.js ────────────────────────────────────────────────── */

/* ── Theme colors (mirrors config.py / style.css) ─────────────────────── */
const THEMES = {
  green: {
    signal: '#00e87a', bgCard: '#0d1e15', bgRoot: '#030a06',
    border: '#163220', text: '#a8e0bc', textMuted: '#366048',
  },
  amber: {
    signal: '#f09c28', bgCard: '#1a1205', bgRoot: '#080601',
    border: '#3a2a0a', text: '#e0cfa8', textMuted: '#6b4e10',
  },
  red: {
    signal: '#e04040', bgCard: '#1a0808', bgRoot: '#080202',
    border: '#3a1212', text: '#e0a8a8', textMuted: '#6b2020',
  },
};

/* ── State ───────────────────────────────────────────────────────────────── */
let currentTheme = 'green';
let canvasMode   = 'idle';   // idle | authorized | uncertain | denied
let waveOffset   = 0;
let sessionLog   = [];
let sessionCount = 0;
const BITS = [1,0,1,1,0,1,0,0,1,1,0,1];

/* ── Live clock ──────────────────────────────────────────────────────────── */
function updateClock() {
  const now = new Date();
  document.getElementById('live-clock').textContent =
    now.toTimeString().slice(0,8);
}
updateClock();
setInterval(updateClock, 1000);

/* ── Animated title ──────────────────────────────────────────────────────── */
(function initTitle() {
  const FULL = 'PROJECT_RFFPLA';
  const el   = document.getElementById('title-text');
  let idx = 0;
  function type() {
    el.textContent = FULL.slice(0, idx);
    idx++;
    if (idx > FULL.length) {
      setTimeout(() => { idx = 0; type(); }, 2000);
    } else {
      setTimeout(type, 80);
    }
  }
  type();
})();

/* ── Tab switching ───────────────────────────────────────────────────────── */
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});

/* ── Threshold slider ────────────────────────────────────────────────────── */
const thrSlider   = document.getElementById('threshold');
const thrDisplay  = document.getElementById('thr-display');
thrSlider.addEventListener('input', () => {
  thrDisplay.textContent = thrSlider.value + '%';
});

const cmpThrSlider  = document.getElementById('cmp-threshold');
const cmpThrDisplay = document.getElementById('cmp-thr-display');
cmpThrSlider.addEventListener('input', () => {
  cmpThrDisplay.textContent = cmpThrSlider.value + '%';
});

/* ── Drag-and-drop helpers ───────────────────────────────────────────────── */
function setupDropZone(zoneId, inputId, onFile) {
  const zone  = document.getElementById(zoneId);
  const input = document.getElementById(inputId);

  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  });
  input.addEventListener('change', () => {
    if (input.files[0]) onFile(input.files[0]);
  });
}

/* ── Authentication upload ───────────────────────────────────────────────── */
setupDropZone('drop-zone', 'file-input', runPredict);

async function runPredict(file) {
  document.getElementById('upload-fname').textContent = file.name;
  document.getElementById('auth-msg').innerHTML = '';
  document.getElementById('analyzing-card').style.display = 'block';
  document.getElementById('charts-section').style.display = 'none';
  document.getElementById('tech-details').style.display = 'none';
  document.getElementById('auth-result-panel').innerHTML =
    '<div class="idle-card">AWAITING SIGNAL INPUT<span style="color:var(--border);font-size:32px;display:block;margin-top:12px;">○</span></div>';

  await delay(900);

  const fd = new FormData();
  fd.append('file', file);

  let data;
  try {
    const res = await fetch('/predict', { method: 'POST', body: fd });
    if (!res.ok) {
      const text = await res.text();
      console.error('Server error:', res.status, text);
      throw new Error(`HTTP ${res.status}: ${text.slice(0, 300)}`);
    }
    data = await res.json();
  } catch (err) {
    document.getElementById('analyzing-card').style.display = 'none';
    console.error('Predict failed:', err);
    showMsg('auth-msg', 'error', err.message || 'Network error — is the server running?');
    return;
  }

  document.getElementById('analyzing-card').style.display = 'none';

  if (data.error) {
    showMsg('auth-msg', 'error', data.error);
    return;
  }

  showResult(data, file.name);
}

function showResult(data, fileName) {
  const conf = data.confidence;
  const thr  = parseFloat(thrSlider.value);

  /* Theme */
  const theme = confToTheme(conf);
  applyTheme(theme);

  /* Canvas mode */
  if (conf >= 85) canvasMode = 'authorized';
  else if (conf >= 50) canvasMode = 'uncertain';
  else canvasMode = 'denied';

  /* Burst count */
  document.getElementById('burst-count').textContent = data.n_bursts;

  /* Result card */
  const { color, icon, label, badge } = confToStyle(conf);
  const barW = Math.min(100, conf).toFixed(1);
  sessionCount++;
  const sessionId = 'AUTH_S' + sessionCount;

  document.getElementById('auth-result-panel').innerHTML = `
    <div class="result-card" style="background:${color}1a;border:1px solid ${color}55;">
      <span class="result-badge" style="color:${color};border-color:${color}66;background:${color}1a;">${badge}</span>
      <span class="result-confval" style="color:${color};">
        ${conf.toFixed(1)}%<span class="cursor" style="color:${color};">_</span>
      </span>
      <span class="result-label" style="color:${color};">${icon} ${label}</span>
      <div class="conf-bar-wrap">
        <div class="conf-bar" style="width:${barW}%;background:${color};"></div>
      </div>
    </div>
    <div class="meta-grid">
      <span class="meta-lbl">DEVICE</span><span class="meta-val">CC1101 + ESP32</span>
      <span class="meta-lbl">SESSION</span><span class="meta-val">${sessionId}</span>
      <span class="meta-lbl">MODEL</span><span class="meta-val">v2+features 3ch</span>
      <span class="meta-lbl">ACCURACY</span><span class="meta-val">100.00%</span>
    </div>`;

  /* Charts */
  document.getElementById('charts-section').style.display = 'block';
  drawAmpChart(data.amp, theme, 'chart-amp');
  drawIQChart(data.i_ch, data.q_ch, theme, 'chart-iq');
  drawConfChart(data.probs, thr, theme, 'chart-conf');

  /* Technical details */
  const td = document.getElementById('tech-details');
  td.style.display = 'block';
  document.getElementById('tech-tbl').innerHTML = `
    <tr><td>File</td><td>${fileName}</td></tr>
    <tr><td>Duration</td><td>~${data.duration_s} s</td></tr>
    <tr><td>Bursts extracted</td><td>${data.n_windows}</td></tr>
    <tr><td>Sample count</td><td>${data.n_samples.toLocaleString()}</td></tr>
    <tr><td>Average score</td><td>${(data.confidence / 100).toFixed(4)}</td></tr>
    <tr><td>Std dev</td><td>${data.std.toFixed(2)}%</td></tr>
    <tr><td>Threshold (display)</td><td>${thr}%</td></tr>
    <tr><td>Threshold (model)</td><td>50% confidence</td></tr>`;

  /* Session log */
  const result = (conf >= thr && conf >= 50) ? 'PASS' : 'FAIL';
  sessionLog.unshift([data.timestamp, fileName, result, conf.toFixed(1) + '%', data.n_bursts]);
  renderSessionLog();
}

/* ── Compare mode ────────────────────────────────────────────────────────── */
let cmpFiles = { file_a: null, file_b: null };

setupDropZone('drop-zone-a', 'file-input-a', file => {
  cmpFiles.file_a = file;
  document.getElementById('fname-a').textContent = file.name;
  tryCompare();
});
setupDropZone('drop-zone-b', 'file-input-b', file => {
  cmpFiles.file_b = file;
  document.getElementById('fname-b').textContent = file.name;
  tryCompare();
});

async function tryCompare() {
  if (!cmpFiles.file_a || !cmpFiles.file_b) return;

  document.getElementById('cmp-msg').innerHTML = '';
  document.getElementById('cmp-results').style.display = 'none';
  document.getElementById('cmp-threshold-row').style.display = 'none';
  document.getElementById('cmp-analyzing').style.display = 'block';

  const fd = new FormData();
  fd.append('file_a', cmpFiles.file_a);
  fd.append('file_b', cmpFiles.file_b);

  let data;
  try {
    const res = await fetch('/predict_compare', { method: 'POST', body: fd });
    if (!res.ok) {
      const text = await res.text();
      console.error('Compare error:', res.status, text);
      throw new Error(`HTTP ${res.status}: ${text.slice(0, 300)}`);
    }
    data = await res.json();
  } catch (err) {
    document.getElementById('cmp-analyzing').style.display = 'none';
    console.error('Compare failed:', err);
    showMsg('cmp-msg', 'error', err.message || 'Network error — is the server running?');
    return;
  }

  document.getElementById('cmp-analyzing').style.display = 'none';

  if (data.error) {
    showMsg('cmp-msg', 'error', data.error);
    return;
  }

  const ra = data.file_a, rb = data.file_b;
  if (ra.error || rb.error) {
    if (ra.error) showMsg('cmp-msg', 'error', 'Signal A: ' + ra.error);
    if (rb.error) showMsg('cmp-msg', 'error', 'Signal B: ' + rb.error);
    return;
  }

  const thr = parseFloat(cmpThrSlider.value);
  const themeA = confToTheme(ra.confidence);
  const themeB = confToTheme(rb.confidence);

  document.getElementById('cmp-card-a').innerHTML = buildCmpCard('Signal A', ra, thr);
  document.getElementById('cmp-card-b').innerHTML = buildCmpCard('Signal B', rb, thr);
  document.getElementById('cmp-threshold-row').style.display = 'block';
  document.getElementById('cmp-results').style.display = 'block';

  drawOverlayChart(ra.amp, rb.amp, 'chart-overlay');
  drawIQChart(ra.i_ch, ra.q_ch, themeA, 'chart-iq-a', 'Signal A — I / Q');
  drawIQChart(rb.i_ch, rb.q_ch, themeB, 'chart-iq-b', 'Signal B — I / Q');
  buildCmpSummary(ra, rb, cmpFiles.file_a.name, cmpFiles.file_b.name, thr);
}

function buildCmpCard(label, d, thr) {
  const { color, icon, badge, label: lbl } = confToStyle(d.confidence);
  const barW = Math.min(100, d.confidence).toFixed(1);
  return `
    <div class="panel-hdr">◈ ${label.toUpperCase()}</div>
    <div class="result-card" style="background:${color}1a;border:1px solid ${color}55;">
      <span class="result-badge" style="color:${color};border-color:${color}66;background:${color}1a;">${badge}</span>
      <span class="result-confval" style="color:${color};">
        ${d.confidence.toFixed(1)}%<span class="cursor" style="color:${color};">_</span>
      </span>
      <span class="result-label" style="color:${color};">${icon} ${lbl}</span>
      <div class="conf-bar-wrap"><div class="conf-bar" style="width:${barW}%;background:${color};"></div></div>
    </div>
    <div class="meta-grid">
      <span class="meta-lbl">BURSTS</span><span class="meta-val">${d.n_bursts}</span>
      <span class="meta-lbl">STD DEV</span><span class="meta-val">${d.std.toFixed(1)}%</span>
    </div>`;
}

function buildCmpSummary(ra, rb, fnA, fnB, thr) {
  const decA = ra.confidence >= thr && ra.confidence >= 50 ? 'PASS' : (ra.confidence < thr ? 'LOW CONF' : 'FAIL');
  const decB = rb.confidence >= thr && rb.confidence >= 50 ? 'PASS' : (rb.confidence < thr ? 'LOW CONF' : 'FAIL');
  const rows = [
    ['Confidence', ra.confidence.toFixed(1) + '%', rb.confidence.toFixed(1) + '%'],
    ['Decision', decA, decB],
    ['Bursts extracted', ra.n_windows, rb.n_windows],
    ['Burst std dev', ra.std.toFixed(1) + '%', rb.std.toFixed(1) + '%'],
    ['Duration', '~' + ra.duration_s + ' s', '~' + rb.duration_s + ' s'],
  ];
  document.getElementById('cmp-summary-body').innerHTML =
    rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join('');
}

/* ── Canvas waveform ─────────────────────────────────────────────────────── */
(function initCanvas() {
  const cvs = document.getElementById('waveform-canvas');
  const ctx = cvs.getContext('2d');

  function resize() {
    cvs.width  = cvs.offsetWidth  || 400;
    cvs.height = cvs.offsetHeight || 160;
  }
  resize();
  window.addEventListener('resize', resize);

  function draw() {
    const t  = THEMES[currentTheme];
    const W  = cvs.width, H = cvs.height;
    ctx.fillStyle = t.bgCard;
    ctx.fillRect(0, 0, W, H);

    /* scanlines */
    for (let sy = 0; sy < H; sy += 3) {
      ctx.fillStyle = 'rgba(0,0,0,0.12)';
      ctx.fillRect(0, sy, W, 1);
    }

    ctx.beginPath();
    ctx.strokeStyle = t.signal;
    ctx.lineWidth   = 1.5;
    ctx.shadowColor = t.signal;
    ctx.shadowBlur  = 5;

    const highY = H * 0.25, lowY = H * 0.75, midY = H * 0.5;

    if (canvasMode === 'idle') {
      /* flat line with scan sweep */
      ctx.strokeStyle = t.border;
      ctx.shadowBlur  = 3;
      ctx.moveTo(0, midY); ctx.lineTo(W, midY);
      ctx.stroke();

      const cx = waveOffset % W;
      ctx.beginPath();
      ctx.strokeStyle = t.signal + '66';
      ctx.lineWidth   = 1;
      ctx.moveTo(cx, 0); ctx.lineTo(cx, H);
      ctx.stroke();

    } else if (canvasMode === 'denied') {
      for (let x = 0; x <= W; x++) {
        const t2 = (x + waveOffset) * 0.013;
        const y  = midY + H * 0.16 * Math.sin(t2 * 7.3)
                       + H * 0.11 * Math.sin(t2 * 11.17)
                       + H * 0.08 * Math.sin(t2 * 19.73);
        if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();

    } else {
      /* OOK digital waveform */
      const segW  = W / BITS.length;
      const off   = waveOffset % segW;
      const base  = Math.floor(waveOffset / segW);
      let prevY   = null;

      for (let i = -1; i <= BITS.length + 1; i++) {
        const idx  = ((base + i) % BITS.length + BITS.length) % BITS.length;
        let   bit  = BITS[idx];
        if (canvasMode === 'uncertain') {
          const seed = Math.sin(idx * 127.1);
          if (seed > 0.52) bit = 0.5;
        }
        const y  = bit === 1 ? highY : bit === 0.5 ? midY : lowY;
        const x1 = i * segW - off;
        const x2 = x1 + segW;
        if (x2 < 0) continue;
        if (x1 > W)  break;
        const dx1 = Math.max(0, x1);
        const dx2 = Math.min(W, x2);
        if (prevY === null) { ctx.moveTo(dx1, y); }
        else if (prevY !== y) { ctx.lineTo(dx1, y); }
        ctx.lineTo(dx2, y);
        prevY = y;
      }
      ctx.stroke();
    }

    waveOffset += 1.5;
    requestAnimationFrame(draw);
  }
  draw();
})();

/* ── Plotly chart helpers ─────────────────────────────────────────────────── */
const PLOTLY_CONFIG = { displayModeBar: false, responsive: true };

function plotLayout(theme, title) {
  const t = THEMES[theme];
  return {
    paper_bgcolor: t.bgCard,
    plot_bgcolor:  t.bgRoot,
    font:     { family: 'Space Mono, monospace', size: 9, color: t.text },
    margin:   { l: 44, r: 20, t: 18, b: 40 },
    height:   260,
    title:    { text: title || '', font: { size: 10, color: t.textMuted }, x: 0 },
    showlegend: false,
    xaxis: { showgrid: false, zeroline: false, color: t.textMuted, tickfont: { size: 8 } },
    yaxis: { gridcolor: t.border, gridwidth: 1, zerolinecolor: t.border,
             zeroline: false, color: t.textMuted, tickfont: { size: 8 } },
  };
}

function drawAmpChart(ampData, theme, divId) {
  const t   = THEMES[theme];
  const hex = t.signal.replace('#','');
  const r   = parseInt(hex.slice(0,2),16), g = parseInt(hex.slice(2,4),16), b = parseInt(hex.slice(4,6),16);
  const xs  = ampData.map((_,i) => i);
  Plotly.newPlot(divId, [{
    x: xs, y: ampData, type: 'scatter', mode: 'lines',
    line: { color: t.signal, width: 1.5 },
    fill: 'tozeroy', fillcolor: `rgba(${r},${g},${b},0.08)`,
    hovertemplate: 'Sample %{x}<br>Amp: %{y:.4f}<extra></extra>',
  }], {
    ...plotLayout(theme),
    xaxis: { ...plotLayout(theme).xaxis, title: { text: 'Sample', font: { size: 8 } } },
    yaxis: { ...plotLayout(theme).yaxis, title: { text: 'Amplitude', font: { size: 8 } } },
  }, PLOTLY_CONFIG);
}

function drawIQChart(iData, qData, theme, divId, title) {
  const t = THEMES[theme];
  const xs = iData.map((_,i) => i);
  Plotly.newPlot(divId, [
    { x: xs, y: iData, type: 'scatter', mode: 'lines', name: 'I (in-phase)',
      line: { color: t.signal, width: 1.5 }, showlegend: true },
    { x: xs, y: qData, type: 'scatter', mode: 'lines', name: 'Q (quadrature)',
      line: { color: '#f09c28', width: 1.5 }, showlegend: true },
  ], {
    ...plotLayout(theme, title),
    showlegend: true,
    legend: { font: { size: 8, color: t.textMuted }, bgcolor: 'transparent' },
    xaxis: { ...plotLayout(theme).xaxis, title: { text: 'Sample', font: { size: 8 } } },
    yaxis: { ...plotLayout(theme).yaxis, title: { text: 'Amplitude', font: { size: 8 } } },
  }, PLOTLY_CONFIG);
}

function drawConfChart(probs, thr, theme, divId) {
  const t  = THEMES[theme];
  const xs = probs.map((_,i) => i);
  Plotly.newPlot(divId, [{
    x: xs, y: probs, type: 'bar',
    marker: { color: t.signal },
    hovertemplate: 'Burst %{x}<br>Conf: %{y:.1f}%<extra></extra>',
  }], {
    ...plotLayout(theme),
    shapes: [{
      type: 'line', x0: 0, x1: probs.length - 1, y0: thr, y1: thr,
      line: { color: '#e04040', width: 1.5, dash: 'dash' },
    }],
    annotations: [{
      x: probs.length - 1, y: thr, text: `Threshold ${thr.toFixed(0)}%`,
      font: { size: 8, color: '#e04040' }, showarrow: false, xanchor: 'right',
    }],
    xaxis: { ...plotLayout(theme).xaxis, title: { text: 'Burst index', font: { size: 8 } } },
    yaxis: { ...plotLayout(theme).yaxis, range: [0,100], title: { text: 'Confidence (%)', font: { size: 8 } } },
  }, PLOTLY_CONFIG);
}

function drawOverlayChart(ampA, ampB, divId) {
  const layout = plotLayout('green', 'Amplitude Overlay');
  const xA = ampA.map((_,i) => i), xB = ampB.map((_,i) => i);
  Plotly.newPlot(divId, [
    { x: xA, y: ampA, type: 'scatter', mode: 'lines', name: 'Signal A (auth)',
      line: { color: '#00e87a', width: 1.5 }, showlegend: true },
    { x: xB, y: ampB, type: 'scatter', mode: 'lines', name: 'Signal B (rogue)',
      line: { color: '#e04040', width: 1.5 }, showlegend: true },
  ], {
    ...layout,
    paper_bgcolor: THEMES.green.bgRoot,
    plot_bgcolor:  THEMES.green.bgRoot,
    showlegend: true,
    legend: { font: { size: 8, color: THEMES.green.textMuted }, bgcolor: 'transparent' },
  }, PLOTLY_CONFIG);
}

/* ── Theme application ───────────────────────────────────────────────────── */
function applyTheme(theme) {
  currentTheme = theme;
  document.body.classList.remove('theme-amber', 'theme-red');
  if (theme !== 'green') document.body.classList.add('theme-' + theme);

  /* Update sidebar status */
  const status = document.getElementById('sys-status');
  if (theme === 'red') {
    status.className = 'threat-status';
    status.innerHTML = '<span class="threat-dot"></span>THREAT DETECTED';
  } else if (theme === 'amber') {
    status.className = 'uncertain-status';
    status.innerHTML = '<span class="uncertain-dot"></span>SYSTEM UNCERTAIN';
  } else {
    status.className = 'armed';
    status.innerHTML = '<span class="armed-dot"></span>SYSTEM ARMED';
  }

  /* Header threat label */
  const threat = document.getElementById('threat-label');
  threat.style.display = theme === 'red' ? 'inline' : 'none';
}

function confToTheme(conf) {
  if (conf >= 85) return 'green';
  if (conf >= 50) return 'amber';
  return 'red';
}

function confToStyle(conf) {
  if (conf >= 85) return { color: '#00e87a', icon: '●', label: 'AUTHORIZED',    badge: 'SIGNAL VERIFIED' };
  if (conf >= 50) return { color: '#f09c28', icon: '◑', label: 'UNCERTAIN',     badge: 'LOW CONFIDENCE' };
  return             { color: '#e04040', icon: '○', label: 'ACCESS DENIED', badge: 'THREAT DETECTED' };
}

/* ── Session log ─────────────────────────────────────────────────────────── */
function renderSessionLog() {
  const el = document.getElementById('session-log');
  if (!sessionLog.length) {
    el.innerHTML = '<div class="idle-card" style="padding:12px 8px;">NO ENTRIES THIS SESSION</div>';
    return;
  }
  const rows = sessionLog.map(([ts, fn, res, conf, bursts]) => {
    const cls = res === 'PASS' ? 'row-pass' : 'row-fail';
    return `<tr class="${cls}"><td>${ts}</td><td>${fn}</td><td>${res}</td><td>${conf}</td><td>${bursts}</td></tr>`;
  }).join('');
  el.innerHTML = `
    <div class="session-wrap">
      <table class="session-tbl">
        <thead><tr><th>TIME</th><th>FILE</th><th>RESULT</th><th>CONF</th><th>BURSTS</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

/* ── Utilities ───────────────────────────────────────────────────────────── */
function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

function showMsg(containerId, type, text) {
  const cls  = type === 'error' ? 'msg-error' : 'msg-info';
  const prev = document.getElementById(containerId).innerHTML;
  document.getElementById(containerId).innerHTML =
    prev + `<div class="${cls}">${text}</div>`;
}
