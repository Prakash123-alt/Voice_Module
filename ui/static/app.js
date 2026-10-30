// VoiceModule web UI - vanilla JS, no build step.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const TOKEN = new URLSearchParams(location.search).get("token");

const ui = {
  snap: null,             // last /api/status snapshot
  state: "IDLE",
  turns: [],              // per-turn latency breakdowns
  metrics: {},
  logs: [],
  streamingMsg: null,     // assistant bubble currently receiving deltas
  pendingTurn: null,
  config: null, original: null, devices: null,
  enroll: { phrases: [], count: 0, recording: false, calib: null },
  vadThreshold: 0.5,
};

// ------------------------------------------------------------------ helpers
async function api(method, path, body) {
  const headers = { "Content-Type": "application/json" };
  if (TOKEN) headers["X-UI-Token"] = TOKEN;
  const res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `${res.status} ${res.statusText}`);
  return data;
}
const recentToasts = new Map();
function toast(msg, kind = "info", ms = 4000) {
  const now = Date.now();
  if (now - (recentToasts.get(msg) || 0) < 3000) return;  // same message twice in a row -> show once
  recentToasts.set(msg, now);
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $("#toasts").append(el);
  setTimeout(() => el.remove(), ms);
}
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtMs = (v) => v == null ? "–" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`;
const time = (ts) => new Date((ts || Date.now() / 1000) * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const dbToPct = (db) => Math.max(0, Math.min(100, (db + 70) / 70 * 100));

// ------------------------------------------------------------------ navigation / theme
const TITLES = { live: "Live", metrics: "Metrics", enroll: "Voice profile", settings: "Settings", logs: "Logs" };
function show(view) {
  $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
  $("#viewTitle").textContent = TITLES[view];
  location.hash = view;
  if (view === "settings") loadSettings();
  if (view === "metrics") drawCharts();
  if (view === "enroll") enterEnroll();
  if (view === "logs") renderLogs();
}
$$(".nav-item").forEach((b) => b.addEventListener("click", () => show(b.dataset.view)));
const savedTheme = localStorage.getItem("theme");
if (savedTheme) document.documentElement.dataset.theme = savedTheme;
$("#themeBtn").onclick = () => {
  const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = t;
  localStorage.setItem("theme", t);
  drawCharts();
};

// ------------------------------------------------------------------ websocket
let ws, wsRetry = 0;
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws${TOKEN ? `?token=${TOKEN}` : ""}`);
  ws.onopen = () => {
    wsRetry = 0;
    setConn("on", "connected");
    resetChat(); ui.logs = []; ui.turns = []; renderLogs();
  };
  ws.onmessage = (m) => handle(JSON.parse(m.data));
  ws.onclose = () => {
    setConn("off", "reconnecting…");
    setTimeout(connect, Math.min(8000, 500 * 2 ** wsRetry++));
  };
}
function setConn(cls, text) {
  $("#connDot").className = `dot ${cls}`;
  $("#connText").textContent = text;
}

function handle(ev) {
  switch (ev.type) {
    case "status": renderStatus(ev); break;
    case "state": setState(ev.to); break;
    case "level": setLevel(ev.mic_db, ev.vad); break;
    case "speaker": addChip(ev.accepted ? "ok" : "bad", `${ev.accepted ? "you" : "other"} ${ev.score.toFixed(2)}`, `threshold ${ev.threshold}`); $("#thrLabel").textContent = `threshold ${ev.threshold}`; break;
    case "bargein": addChip("barge", `barge-in ${ev.score?.toFixed?.(2) ?? ""}`, `stopped in ${ev.stop_ms} ms`); break;
    case "transcript": onTranscript(ev); break;
    case "reply_delta": onDelta(ev.text); break;
    case "interrupted_reply": finishStreaming(ev.text, true); break;
    case "turn": onTurn(ev); break;
    case "metrics": ui.metrics = ev.summary; renderMetricCards(); break;
    case "log": addLog(ev); break;
    case "error": toast(ev.message, "error", 7000); break;
    case "mic_level": $("#micTestBar").style.width = `${dbToPct(ev.db)}%`; break;
    case "enroll_level": $("#enrollBar").style.width = `${dbToPct(ev.db)}%`; break;
  }
}

// ------------------------------------------------------------------ status / live
const HINTS = {
  IDLE: ["Idle", "Press Start to begin listening"],
  LISTENING: ["Listening", "Only your enrolled voice is accepted"],
  PROCESSING: ["Thinking…", "Transcribing and asking the model"],
  SPEAKING: ["Speaking", "Talk over me to interrupt"],
  INTERRUPTED: ["Interrupted", "Stopped - listening again"],
};
function setState(s) {
  ui.state = s;
  const orb = $("#orb");
  orb.className = `orb state-${s}${ui.snap?.muted ? " muted" : ""}`;
  const [label, hint] = HINTS[s] || [s, ""];
  $("#stateLabel").textContent = ui.snap?.muted && s !== "IDLE" ? `${label} (muted)` : label;
  $("#stateHint").textContent = hint;
  if (s !== "LISTENING" && s !== "SPEAKING") setLevel(-90, 0);
}

function renderStatus(s) {
  ui.snap = s;
  const pill = $("#statusPill");
  pill.textContent = s.status;
  pill.className = `pill ${s.status}`;
  const running = s.status === "running";
  const busy = s.status === "starting" || s.status === "stopping";
  const startBtn = $("#startBtn");
  startBtn.disabled = busy;
  startBtn.textContent = s.status === "starting" ? "Loading models…" : s.status === "stopping" ? "Stopping…" : running ? "■ Stop" : "▶ Start";
  startBtn.classList.toggle("danger", running);
  $("#muteBtn").disabled = !running;
  $("#muteBtn").classList.toggle("on", !!s.muted);
  $("#muteBtn").textContent = s.muted ? "🔇 Unmute" : "🎙 Mute";
  setState(running ? s.state : "IDLE");
  $("#factLlm").textContent = s.models?.llm || "–";
  $("#factStt").textContent = s.models?.stt || "–";
  $("#factTts").textContent = s.models?.tts || "–";
  $("#factAec").textContent = s.aec ? s.aec.replace("AEC", "").replace("WebRTC", "WebRTC AEC3") || s.aec : "–";
  $("#factBarge").textContent = running ? (s.barge_in ? "enabled" : "disabled (no AEC)") : "–";
  $("#keyState").textContent = s.api_key_set ? "set ✓" : "NOT SET";
  $("#keyState").className = s.api_key_set ? "ok-t" : "bad-t";

  const banner = $("#banner");
  let html = "", cls = "";
  if (!s.api_key_set) { html = "<span><b>No OpenRouter key.</b> Add <code>OPENROUTER_API_KEY</code> to <code>.env</code> and restart the UI.</span>"; cls = "error"; }
  else if (!s.profile) { html = `<span><b>Enroll your voice first.</b> The assistant only answers its enrolled user, so it needs a voice profile before it can start (takes ~1 minute).</span><button class="primary small" id="goEnroll">Enroll now →</button>`; }
  else if (s.status === "error") { html = `<span><b>Couldn't start:</b> ${esc(s.error || "")}</span><button class="ghost small" id="goLogs">Logs</button>`; cls = "error"; }
  banner.innerHTML = html;
  banner.className = `banner ${cls} ${html ? "" : "hidden"}`;
  $("#goEnroll")?.addEventListener("click", () => show("enroll"));
  $("#goLogs")?.addEventListener("click", () => show("logs"));
  startBtn.title = s.profile ? "Start / stop (Ctrl+Space)" : "Enroll your voice first";
  if (!s.profile && !running && !busy) startBtn.textContent = "Enroll to start";
}

function setLevel(db, vad) {
  $("#micBar").style.width = `${dbToPct(db)}%`;
  $("#micDb").textContent = db <= -89 ? "– dB" : `${db.toFixed(0)} dB`;
  $("#vadBar").style.width = `${Math.round(vad * 100)}%`;
  $("#vadVal").textContent = vad ? vad.toFixed(2) : "–";
}

function addChip(kind, text, title) {
  const box = $("#speakerChips");
  if (box.querySelector(".muted")) box.innerHTML = "";
  const c = document.createElement("span");
  c.className = `chip ${kind}`;
  c.textContent = text;
  c.title = title || "";
  box.prepend(c);
  while (box.children.length > 20) box.lastChild.remove();
}

// chat
function resetChat() {
  $("#chat").innerHTML = "";
  $("#chat").append(emptyChatEl);
  ui.streamingMsg = null;
}
const emptyChatEl = $("#emptyChat");
function addMsg(role, text, meta = "") {
  emptyChatEl.remove();
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = `<div class="body">${esc(text)}</div><div class="msg-meta">${meta}</div>`;
  $("#chat").append(el);
  scrollChat();
  return el;
}
function scrollChat() { const c = $("#chat"); c.scrollTop = c.scrollHeight; }
function onTranscript(ev) {
  if (ev.role === "user") {
    addMsg("user", ev.text, `${ev.source === "text" ? '<span class="tag typed">typed</span>' : ""}<span>${time(ev.ts)}</span>`);
  } else {
    finishStreaming(ev.text, false, ev.ts);
  }
}
function onDelta(text) {
  if (!ui.streamingMsg) {
    ui.streamingMsg = addMsg("assistant", "");
    ui.streamingMsg.classList.add("streaming");
    ui.streamingMsg._text = "";
  }
  ui.streamingMsg._text += text;
  ui.streamingMsg.querySelector(".body").textContent = ui.streamingMsg._text;
  scrollChat();
}
function finishStreaming(text, interrupted, ts) {
  let el = ui.streamingMsg;
  if (!el && interrupted && text) {
    // playback interrupted after the transcript bubble was already finalized: mark that
    // bubble "cut off" instead of rendering the reply a second time
    const last = [...document.querySelectorAll("#chat .msg.assistant")].pop();
    if (last && last.querySelector(".body").textContent === text) el = last;
  }
  if (!el) { if (!text) return; el = addMsg("assistant", text); }
  el.classList.remove("streaming");
  if (text && el === ui.streamingMsg) el.querySelector(".body").textContent = text;
  const meta = el.querySelector(".msg-meta");
  const t = el._turn;
  meta.innerHTML = `<span>${time(ts)}</span>${t ? `<span title="STT ${fmtMs(t.stt_ms)} · LLM ${fmtMs(t.llm_first_ms)} · TTS ${fmtMs(t.tts_ms)}">⏱ ${fmtMs(t.e2e_ms)}</span>` : ""}${interrupted ? '<span class="tag">cut off</span>' : ""}`;
  ui.streamingMsg = null;
}
function onTurn(t) {
  ui.turns.push(t);
  if (ui.turns.length > 60) ui.turns.shift();
  if (ui.streamingMsg) ui.streamingMsg._turn = t;
  $("#factLatency").textContent = `${fmtMs(t.e2e_ms)} to first audio`;
  if ($("#view-metrics").classList.contains("active")) drawCharts();
}

// controls
$("#startBtn").onclick = async () => {
  const running = ui.snap?.status === "running";
  if (!running && !ui.snap?.profile) { show("enroll"); return; }
  $("#startBtn").disabled = true;
  try {
    await api("POST", running ? "/api/stop" : "/api/start");
  } catch (e) { toast(e.message, "error", 8000); }
  finally { $("#startBtn").disabled = false; }
};
$("#muteBtn").onclick = () => api("POST", "/api/mute", { muted: !ui.snap?.muted }).catch((e) => toast(e.message, "error"));
// interruption is fully autonomous: verified user speech while PROCESSING/SPEAKING barges in
$("#clearChat").onclick = async () => { await api("POST", "/api/history/clear").catch(() => {}); resetChat(); $("#speakerChips").innerHTML = '<span class="muted small">Speaker decisions appear here.</span>'; };
$("#textForm").onsubmit = async (e) => {
  e.preventDefault();
  const input = $("#textInput");
  const text = input.value.trim();
  if (!text) return;
  try { await api("POST", "/api/text", { text }); input.value = ""; }
  catch (err) { toast(err.message, "warn"); }
};
document.addEventListener("keydown", (e) => {
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
  if (e.code === "Space" && e.ctrlKey) { e.preventDefault(); $("#startBtn").click(); }
  else if (!typing && e.key.toLowerCase() === "m" && !$("#muteBtn").disabled) $("#muteBtn").click();
});

// ------------------------------------------------------------------ metrics
const METRIC_DEFS = [
  ["e2e_first_audio", "Speech end → first audio", 1500],
  ["stt", "Speech-to-text", 600],
  ["llm_first_token", "LLM first token", 800],
  ["tts_sentence", "TTS per sentence", 250],
  ["speaker_verify", "Speaker check", 150],
  ["barge_verify", "Barge-in check", 100],
  ["barge_in_stop", "Barge-in stop", 100],
  ["dsp_frame", "DSP per 20 ms frame", 10],
];
function renderMetricCards() {
  const box = $("#metricCards");
  box.innerHTML = METRIC_DEFS.map(([k, label, target]) => {
    const m = ui.metrics[k];
    const cls = !m ? "" : m.p50_ms <= target ? "good" : m.p50_ms <= target * 1.5 ? "warn" : "bad";
    return `<div class="metric ${cls}"><div class="label">${label}</div>
      <div class="value">${m ? fmtMs(m.p50_ms) : "–"} ${m ? "<small>p50</small>" : ""}</div>
      <div class="sub">${m ? `p95 ${fmtMs(m.p95_ms)} · last ${fmtMs(m.last_ms)} · n=${m.n}` : "no data yet"}</div></div>`;
  }).join("");
}
function canvasCtx(id) {
  const c = $(id);
  const w = c.clientWidth || 600, h = Number(c.getAttribute("height"));
  const dpr = window.devicePixelRatio || 1;
  c.width = w * dpr; c.height = h * dpr; c.style.height = `${h}px`;
  const ctx = c.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.font = "11px Inter, Segoe UI, sans-serif";
  return { ctx, w, h };
}
function axes(ctx, w, h, maxV, pad) {
  ctx.strokeStyle = cssVar("--line"); ctx.fillStyle = cssVar("--muted"); ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const v = maxV * i / 4, y = h - pad.b - (h - pad.t - pad.b) * i / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    ctx.fillText(fmtMs(v), 4, y + 4);
  }
}
function emptyChart(ctx, w, h, msg) {
  ctx.fillStyle = cssVar("--muted"); ctx.textAlign = "center";
  ctx.fillText(msg, w / 2, h / 2); ctx.textAlign = "left";
}
function drawCharts() {
  if (!$("#view-metrics").classList.contains("active")) return;
  renderMetricCards();
  const pad = { l: 58, r: 12, t: 12, b: 22 };
  const turns = ui.turns.filter((t) => t.e2e_ms != null);
  { // e2e
    const { ctx, w, h } = canvasCtx("#e2eChart");
    if (!turns.length) emptyChart(ctx, w, h, "Talk to the assistant to see per-turn latency");
    else {
      const maxV = Math.max(2000, ...turns.map((t) => t.e2e_ms)) * 1.1;
      axes(ctx, w, h, maxV, pad);
      const y = (v) => h - pad.b - (h - pad.t - pad.b) * v / maxV;
      const x = (i) => pad.l + (w - pad.l - pad.r) * (turns.length === 1 ? 0.5 : i / (turns.length - 1));
      ctx.setLineDash([5, 4]); ctx.strokeStyle = cssVar("--ok");
      ctx.beginPath(); ctx.moveTo(pad.l, y(1500)); ctx.lineTo(w - pad.r, y(1500)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = cssVar("--ok"); ctx.fillText("target 1.5 s", w - pad.r - 70, y(1500) - 5);
      ctx.strokeStyle = cssVar("--accent"); ctx.lineWidth = 2; ctx.beginPath();
      turns.forEach((t, i) => (i ? ctx.lineTo(x(i), y(t.e2e_ms)) : ctx.moveTo(x(i), y(t.e2e_ms)))); ctx.stroke();
      turns.forEach((t, i) => {
        ctx.fillStyle = t.e2e_ms <= 1500 ? cssVar("--ok") : t.e2e_ms <= 2250 ? cssVar("--warn") : cssVar("--bad");
        ctx.beginPath(); ctx.arc(x(i), y(t.e2e_ms), 4, 0, 7); ctx.fill();
      });
    }
  }
  { // stacked
    const { ctx, w, h } = canvasCtx("#stackChart");
    if (!turns.length) emptyChart(ctx, w, h, "No turns yet");
    else {
      const total = (t) => (t.stt_ms || 0) + (t.llm_first_ms || 0) + (t.tts_ms || 0);
      const maxV = Math.max(1000, ...turns.map(total)) * 1.1;
      axes(ctx, w, h, maxV, pad);
      const bw = Math.min(36, (w - pad.l - pad.r) / turns.length * 0.7);
      const step = (w - pad.l - pad.r) / turns.length;
      const sc = (v) => (h - pad.t - pad.b) * v / maxV;
      turns.forEach((t, i) => {
        let yb = h - pad.b;
        const x0 = pad.l + step * i + (step - bw) / 2;
        for (const [k, col] of [["stt_ms", "#818cf8"], ["llm_first_ms", "#f59e0b"], ["tts_ms", "#34d399"]]) {
          const hh = sc(t[k] || 0);
          ctx.fillStyle = col; ctx.fillRect(x0, yb - hh, bw, hh); yb -= hh;
        }
        if (t.source === "text") { ctx.fillStyle = cssVar("--muted"); ctx.fillText("T", x0 + bw / 2 - 3, h - 6); }
      });
    }
  }
}
window.addEventListener("resize", () => drawCharts());

// ------------------------------------------------------------------ enrollment
function setStep(n) {
  $$(".step").forEach((s) => {
    const k = Number(s.dataset.step);
    s.classList.toggle("active", k === n);
    s.classList.toggle("done", k < n);
  });
  ["#enroll1", "#enroll2", "#enroll3"].forEach((id, i) => $(id).classList.toggle("hidden", i + 1 !== n));
}
async function enterEnroll() {
  try {
    const { config } = await api("GET", "/api/config");
    $("#enrollDevice").textContent = config.audio.input_device ?? "system default";
  } catch {}
}
$$(".step").forEach((s) => s.addEventListener("click", async () => {
  const n = Number(s.dataset.step);
  if (n === 3 && ui.snap?.profile) {
    try { showCalibration(await api("GET", "/api/calibrate")); setStep(3); } catch (e) { toast(e.message, "warn"); }
  } else if (n === 1) setStep(1);
}));
function mustBeStopped() {
  if (ui.snap?.status === "running" || ui.snap?.status === "starting") {
    toast("Stop the assistant first - it's using the microphone.", "warn");
    return false;
  }
  return true;
}
$("#micTestBtn").onclick = async () => {
  if (!mustBeStopped()) return;
  const btn = $("#micTestBtn");
  btn.disabled = true; btn.textContent = "Listening… speak now";
  try {
    const r = await api("POST", "/api/mictest", { seconds: 3 });
    const msg = r.ok ? `✓ Sounds good (${r.level_db} dBFS)` : r.problem === "too_quiet" ? `Too quiet (${r.level_db} dBFS) - wrong mic, or speak up` : `Clipping (peak ${r.peak}) - move back a little`;
    $("#micTestResult").innerHTML = `<span class="${r.ok ? "ok-t" : "bad-t"}">${msg}</span>`;
  } catch (e) { toast(e.message, "error"); }
  finally { btn.disabled = false; btn.textContent = "Test mic (3 s)"; $("#micTestBar").style.width = "0"; }
};
$("#toStep2").onclick = async () => {
  if (!mustBeStopped()) return;
  const btn = $("#toStep2");
  btn.disabled = true; btn.textContent = "Loading speaker model…";
  try {
    const r = await api("POST", "/api/enroll/start");
    ui.enroll = { ...ui.enroll, phrases: r.phrases, count: 0, max: r.max, min: r.min };
    renderTakes(); setStep(2);
  } catch (e) { toast(e.message, "error"); }
  finally { btn.disabled = false; btn.textContent = "Next →"; }
};
function renderTakes() {
  const { count, phrases, min, max } = ui.enroll;
  $("#takeDots").innerHTML = Array.from({ length: max }, (_, i) => `<i class="${i < count ? "done" : ""} ${i >= 8 ? "opt" : ""}"></i>`).join("");
  $("#phraseN").textContent = count >= max ? "All done" : `Phrase ${count + 1} of ${Math.max(8, count + 1)}`;
  $("#phraseText").textContent = count >= max ? "You've recorded the maximum. Press Finish." : `“${phrases[count % phrases.length]}”`;
  const fin = $("#finishEnroll");
  fin.disabled = count < min;
  fin.textContent = count < min ? `Finish (need ${min - count} more)` : `Finish with ${count} takes`;
  $("#recordBtn").disabled = count >= max;
}
$("#recordBtn").onclick = async () => {
  if (ui.enroll.recording) return;
  ui.enroll.recording = true;
  const btn = $("#recordBtn"), ring = $("#recordRing"), secs = 4;
  btn.classList.add("recording"); btn.disabled = true;
  $("#recordLabel").textContent = "Speak!";
  $("#takeMsg").textContent = "Recording… read the phrase now.";
  ring.style.transition = "none"; ring.style.strokeDashoffset = "339.3";
  requestAnimationFrame(() => { ring.style.transition = `stroke-dashoffset ${secs}s linear`; ring.style.strokeDashoffset = "0"; });
  try {
    const r = await api("POST", "/api/enroll/record", { seconds: secs });
    ui.enroll.count = r.count;
    $("#takeMsg").innerHTML = r.ok ? `<span class="ok-t">✓ Good take (${r.level_db} dBFS)</span>`
      : r.problem === "too_quiet" ? `<span class="bad-t">Too quiet (${r.level_db} dBFS) - try again a bit louder</span>`
      : `<span class="bad-t">Clipping - move back from the mic and try again</span>`;
    renderTakes();
  } catch (e) { toast(e.message, "error"); $("#takeMsg").textContent = ""; }
  finally {
    ui.enroll.recording = false;
    btn.classList.remove("recording"); btn.disabled = ui.enroll.count >= ui.enroll.max;
    $("#recordLabel").textContent = "Record";
    ring.style.transition = "none"; ring.style.strokeDashoffset = "339.3";
    $("#enrollBar").style.width = "0";
  }
};
$("#undoTake").onclick = async () => { const r = await api("POST", "/api/enroll/undo"); ui.enroll.count = r.count; renderTakes(); $("#takeMsg").textContent = "Last take removed."; };
$("#finishEnroll").onclick = async () => {
  try {
    const r = await api("POST", "/api/enroll/finish");
    const m = r.meta;
    toast(`Voice profile saved (${m.phrases} takes)`, "ok");
    showCalibration(r.calibration, m);
    setStep(3);
  } catch (e) { toast(e.message, "error"); }
};
$("#reEnroll").onclick = () => setStep(1);
function showCalibration(c, meta) {
  ui.enroll.calib = c;
  const slider = $("#thrSlider");
  slider.value = c.suggested;
  $("#thrValue").textContent = Number(c.suggested).toFixed(2);
  $("#thrSuggest").textContent = `suggested ${c.suggested} · current ${c.current}`;
  $("#enrollStats").innerHTML = `Your takes vs each other: mean <b>${c.genuine_mean.toFixed(3)}</b>, lowest <b>${c.genuine_min.toFixed(3)}</b>` +
    (meta ? ` · self-similarity ${meta.self_similarity_mean.toFixed(3)}` : "") +
    (c.genuine_min < 0.7 ? ' · <span class="bad-t">takes are inconsistent - consider re-recording in a quieter spot</span>' : "");
  drawHist();
}
$("#thrSlider").oninput = (e) => { $("#thrValue").textContent = Number(e.target.value).toFixed(2); drawHist(); };
$("#applyThr").onclick = async () => {
  const v = Number($("#thrSlider").value);
  try {
    const r = await api("PUT", "/api/config", { speaker: { threshold: v } });
    toast(`Threshold set to ${v.toFixed(2)}${r.restarted ? " (assistant restarted)" : ""}`, "ok");
    ui.enroll.calib.current = v; $("#thrSuggest").textContent = `suggested ${ui.enroll.calib.suggested} · current ${v}`;
  } catch (e) { toast(e.message, "error"); }
};
function drawHist() {
  const c = ui.enroll.calib;
  if (!c) return;
  const { ctx, w, h } = canvasCtx("#histChart");
  const pad = { l: 16, r: 16, t: 12, b: 24 }, lo = 0.3, hi = 1.0;
  const x = (v) => pad.l + (w - pad.l - pad.r) * (v - lo) / (hi - lo);
  ctx.strokeStyle = cssVar("--line"); ctx.fillStyle = cssVar("--muted");
  for (let v = 0.3; v <= 1.001; v += 0.1) { ctx.beginPath(); ctx.moveTo(x(v), pad.t); ctx.lineTo(x(v), h - pad.b); ctx.stroke(); ctx.fillText(v.toFixed(1), x(v) - 8, h - 6); }
  const thr = Number($("#thrSlider").value);
  ctx.fillStyle = "rgba(248,113,113,.08)"; ctx.fillRect(x(lo), pad.t, x(thr) - x(lo), h - pad.t - pad.b);
  c.genuine.forEach((g) => {
    ctx.fillStyle = g >= thr ? cssVar("--ok") : cssVar("--bad");
    ctx.fillRect(x(g) - 3, pad.t + 20, 6, h - pad.t - pad.b - 20);
  });
  ctx.strokeStyle = cssVar("--text"); ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x(thr), pad.t); ctx.lineTo(x(thr), h - pad.b); ctx.stroke();
  ctx.fillStyle = cssVar("--text"); ctx.fillText(`threshold ${thr.toFixed(2)}`, Math.min(x(thr) + 6, w - 100), pad.t + 10);
  const rejected = c.genuine.filter((g) => g < thr).length;
  ctx.fillStyle = rejected ? cssVar("--bad") : cssVar("--ok");
  ctx.fillText(rejected ? `${rejected} of your takes would be rejected` : "all your takes accepted", pad.l + 4, pad.t + 10);
}

// ------------------------------------------------------------------ settings
const SECTIONS = [
  ["Audio", [
    { k: "audio.input_device", label: "Microphone", type: "device", dir: "inputs" },
    { k: "audio.output_device", label: "Speaker", type: "device", dir: "outputs" },
    { k: "audio.input_gain", label: "Mic gain", type: "range", min: 0, max: 4, step: 0.1 },
    { k: "audio.output_gain", label: "Output volume", type: "range", min: 0, max: 2, step: 0.05 },
  ]],
  ["Speaker verification", [
    { k: "speaker.threshold", label: "Threshold", type: "range", min: 0.3, max: 0.95, step: 0.01, help: "Higher = stricter. Use Voice profile → Threshold to pick it from your data." },
    { k: "speaker.min_segment_ms", label: "Min utterance (ms)", type: "number", min: 100, max: 3000, help: "Shorter utterances are ignored." },
    { k: "speaker.backend", label: "Embedding model", type: "select", options: ["resemblyzer", "ecapa"], help: "Re-enroll after changing." },
  ]],
  ["Barge-in", [
    { k: "barge_in.enabled", label: "Allow interrupting", type: "bool" },
    { k: "barge_in.min_speech_ms", label: "Speech needed (ms)", type: "number", min: 100, max: 2000 },
    { k: "barge_in.verify_window_ms", label: "Verify window (ms)", type: "number", min: 250, max: 3000 },
    { k: "barge_in.threshold", label: "Echo-floor threshold", type: "number", min: 0.3, max: 0.95, step: 0.01,
      help: "Empty = speaker threshold − 0.25. Over-echo speech must also match you better than the TTS echo." },
  ]],
  ["Voice activity", [
    { k: "vad.threshold", label: "Speech probability", type: "range", min: 0.1, max: 0.9, step: 0.05 },
    { k: "vad.min_speech_ms", label: "Onset (ms)", type: "number", min: 20, max: 1000 },
    { k: "vad.hangover_ms", label: "End-of-speech silence (ms)", type: "number", min: 100, max: 3000, help: "Lower = faster replies, more cut-offs." },
    { k: "vad.pad_ms", label: "Pre-roll (ms)", type: "number", min: 0, max: 1000, help: "Raise if first words get clipped." },
    { k: "vad.max_segment_s", label: "Max utterance (s)", type: "number", min: 2, max: 60 },
  ]],
  ["Echo & noise", [
    { k: "aec.backend", label: "Echo cancellation", type: "select", options: ["webrtc", "speex", "gate", "none"], help: "none = headphones. gate disables barge-in." },
    { k: "aec.delay_ms", label: "Delay (ms or auto)", type: "text" },
    { k: "denoise.backend", label: "Noise suppression", type: "select", options: ["rnnoise", "deepfilternet", "none"] },
  ]],
  ["Speech-to-text", [
    { k: "stt.mode", label: "Mode", type: "select", options: ["local", "openrouter"] },
    { k: "stt.local_model", label: "Whisper model", type: "select", options: ["tiny.en", "base.en", "small.en", "medium.en", "tiny", "base", "small"], help: "tiny.en ≈ 2x faster than base.en on CPU." },
    { k: "stt.openrouter_model", label: "OpenRouter model", type: "text" },
    { k: "stt.language", label: "Language", type: "text" },
  ]],
  ["Language model", [
    { k: "llm.model", label: "OpenRouter model", type: "text" },
    { k: "llm.reasoning", label: "Reasoning", type: "reasoning", help: "Off is ~0.8 s faster to first word." },
    { k: "llm.temperature", label: "Temperature", type: "range", min: 0, max: 1.5, step: 0.05 },
    { k: "llm.max_tokens", label: "Max tokens", type: "number", min: 16, max: 4000 },
    { k: "llm.history_turns", label: "Memory (turns)", type: "number", min: 0, max: 50 },
    { k: "llm.system_prompt", label: "System prompt", type: "textarea" },
  ]],
  ["Text-to-speech", [
    { k: "tts.mode", label: "Mode", type: "select", options: ["local", "openrouter"] },
    { k: "tts.piper_model", label: "Piper voice", type: "text" },
    { k: "tts.openrouter_model", label: "OpenRouter model", type: "text" },
    { k: "tts.openrouter_voice", label: "OpenRouter voice", type: "text" },
    { k: "tts.min_chunk_chars", label: "Min chunk (chars)", type: "number", min: 1, max: 200 },
  ]],
];
const getPath = (obj, k) => k.split(".").reduce((o, p) => o?.[p], obj);
function fieldValue(f, cfg) {
  if (f.type === "reasoning") return cfg.llm.extra_body?.reasoning?.enabled !== false;
  return getPath(cfg, f.k);
}
async function loadSettings() {
  try {
    const [{ config, api_key_set }, devices] = await Promise.all([api("GET", "/api/config"), api("GET", "/api/devices").catch(() => null)]);
    ui.config = config; ui.original = JSON.parse(JSON.stringify(config)); ui.devices = devices;
    ui.vadThreshold = config.vad.threshold;
    $("#vadThr").style.left = `${config.vad.threshold * 100}%`;
    $("#keyState").textContent = api_key_set ? "set ✓" : "NOT SET";
    renderSettings();
  } catch (e) { toast(e.message, "error"); }
}
function inputFor(f, v) {
  const id = `f-${f.k.replace(".", "-")}`;
  switch (f.type) {
    case "bool": case "reasoning":
      return `<label class="switch"><input type="checkbox" id="${id}" ${v ? "checked" : ""}/><span></span></label>`;
    case "range":
      return `<div class="range-wrap"><input type="range" id="${id}" min="${f.min}" max="${f.max}" step="${f.step}" value="${v}"/><output>${v}</output></div>`;
    case "number":
      return `<input type="number" id="${id}" min="${f.min ?? ""}" max="${f.max ?? ""}" value="${v ?? ""}"/>`;
    case "select":
      return `<select id="${id}">${f.options.map((o) => `<option ${o === v ? "selected" : ""}>${o}</option>`).join("")}</select>`;
    case "textarea":
      return `<textarea id="${id}">${esc(v ?? "")}</textarea>`;
    case "device": {
      const list = ui.devices?.[f.dir] || [];
      const names = [...new Set(list.map((d) => d.name))];
      const opts = [`<option value="" ${v == null ? "selected" : ""}>System default</option>`];
      if (v != null && !names.includes(String(v))) opts.push(`<option value="${esc(v)}" selected>Matching “${esc(v)}”</option>`);
      names.forEach((n) => opts.push(`<option value="${esc(n)}" ${String(v) === n ? "selected" : ""}>${esc(n)}</option>`));
      return `<select id="${id}">${opts.join("")}</select>`;
    }
    default:
      return `<input type="text" id="${id}" value="${esc(v ?? "")}"/>`;
  }
}
function renderSettings() {
  const cfg = ui.config;
  $("#settingsForm").innerHTML = SECTIONS.map(([title, fields]) => `
    <div class="card"><div class="card-title">${title}</div>
      ${fields.map((f) => `<div class="field ${f.type === "textarea" ? "wide" : ""}" data-k="${f.k}">
        <label for="f-${f.k.replace(".", "-")}">${f.label}${f.help ? `<span class="help">${f.help}</span>` : ""}</label>
        ${inputFor(f, fieldValue(f, cfg))}</div>`).join("")}
    </div>`).join("");
  $$("#settingsForm input, #settingsForm select, #settingsForm textarea").forEach((el) =>
    el.addEventListener("input", () => {
      if (el.type === "range") el.nextElementSibling.textContent = el.value;
      updateDirty();
    }));
  updateDirty();
}
function readField(f) {
  const el = $(`#f-${f.k.replace(".", "-")}`);
  if (!el) return undefined;
  switch (f.type) {
    case "bool": case "reasoning": return el.checked;
    case "range": case "number": return el.value === "" ? null : Number(el.value);
    case "device": return el.value === "" ? null : el.value;
    case "text": if (f.k === "aec.delay_ms") return /^\d+$/.test(el.value.trim()) ? Number(el.value) : el.value.trim() || "auto";
    default: return el.value;
  }
}
function collectPatch() {
  const patch = {};
  for (const [, fields] of SECTIONS) for (const f of fields) {
    const v = readField(f);
    const orig = fieldValue(f, ui.original);
    const changed = JSON.stringify(v) !== JSON.stringify(orig ?? null) && !(v === null && orig == null);
    $(`.field[data-k="${f.k}"]`)?.classList.toggle("changed", changed);
    if (!changed) continue;
    if (f.type === "reasoning") {
      patch.llm = patch.llm || {};
      patch.llm.extra_body = { ...(ui.original.llm.extra_body || {}), reasoning: { ...(ui.original.llm.extra_body?.reasoning || {}), enabled: v } };
    } else {
      const [s, k] = f.k.split(".");
      (patch[s] = patch[s] || {})[k] = v;
    }
  }
  return patch;
}
function updateDirty() {
  const n = Object.values(collectPatch()).reduce((a, s) => a + Object.keys(s).length, 0);
  const btn = $("#saveSettings");
  btn.disabled = n === 0;
  btn.textContent = n ? `Save ${n} change${n > 1 ? "s" : ""}${ui.snap?.status === "running" ? " & restart" : ""}` : "Save";
}
$("#saveSettings").onclick = async () => {
  const btn = $("#saveSettings");
  btn.disabled = true; btn.textContent = "Saving…";
  try {
    const r = await api("PUT", "/api/config", collectPatch());
    ui.config = r.config; ui.original = JSON.parse(JSON.stringify(r.config));
    $("#vadThr").style.left = `${r.config.vad.threshold * 100}%`;
    renderSettings();
    toast(`Settings saved${r.restarted ? " - assistant restarted" : ""}`, "ok");
  } catch (e) { toast(e.message, "error", 7000); updateDirty(); }
};
$("#checkModels").onclick = async () => {
  const box = $("#checkResult");
  box.classList.remove("hidden");
  box.innerHTML = '<span class="muted">Checking OpenRouter…</span>';
  try {
    const r = await api("POST", "/api/check");
    box.innerHTML = `<div class="card-title">Model check (${r.total} models on OpenRouter)</div><div class="check-list">` +
      Object.entries(r.models).map(([role, m]) => `<div><b>${role.toUpperCase()}</b> <code>${esc(m.slug)}</code> ` +
        (m.ok ? '<span class="ok-t">✓ available</span>' : `<span class="bad-t">✗ not found</span>${m.candidates.length ? ` - try: ${m.candidates.map((c) => `<code>${esc(c)}</code>`).join(", ")}` : ""}`) + "</div>").join("") + "</div>";
  } catch (e) { box.innerHTML = `<span class="bad-t">${esc(e.message)}</span>`; }
};

// ------------------------------------------------------------------ logs
const LEVELS = { DEBUG: 10, INFO: 20, WARNING: 30, ERROR: 40, CRITICAL: 50 };
function logMatches(l) {
  const min = LEVELS[$("#logLevel").value];
  const q = $("#logSearch").value.toLowerCase();
  return LEVELS[l.level] >= min && (!q || l._text.toLowerCase().includes(q));
}
function logEl(l) {
  const d = document.createElement("div");
  d.className = "log-line";
  const kv = Object.entries(l.kv || {}).map(([k, v]) => `${k}=${v}`).join(" ");
  d.innerHTML = `<span class="t">${time(l.ts)}</span> <span class="lv ${l.level}">${l.level}</span> <span class="lg-name">${esc(l.logger)}</span>: ${esc(l.msg)} <span class="kv">${esc(kv)}</span>`;
  return d;
}
function addLog(l) {
  l._text = `${l.level} ${l.logger} ${l.msg} ${Object.entries(l.kv || {}).map(([k, v]) => `${k}=${v}`).join(" ")}`;
  ui.logs.push(l);
  if (ui.logs.length > 3000) ui.logs.shift();
  // runtime errors surface as toasts; UI-initiated actions already report their own errors
  if ((l.level === "ERROR" || l.level === "CRITICAL") && l.logger !== "ui") toast(`${l.msg} ${l.kv?.err || ""}`.trim(), "error", 6000);
  if (!$("#view-logs").classList.contains("active") || !logMatches(l)) return;
  const list = $("#logList");
  list.append(logEl(l));
  while (list.children.length > 1500) list.firstChild.remove();
  if ($("#logFollow").checked) list.scrollTop = list.scrollHeight;
}
function renderLogs() {
  const list = $("#logList");
  list.replaceChildren(...ui.logs.filter(logMatches).slice(-1500).map(logEl));
  if ($("#logFollow").checked) list.scrollTop = list.scrollHeight;
}
$("#logLevel").onchange = renderLogs;
$("#logSearch").oninput = renderLogs;
$("#logClear").onclick = () => { ui.logs = []; renderLogs(); };
$("#logDownload").onclick = () => {
  const text = ui.logs.map((l) => `${new Date(l.ts * 1000).toISOString()} ${l._text}`).join("\n");
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(new Blob([text], { type: "text/plain" })), download: "voicemodule-log.txt" });
  a.click(); URL.revokeObjectURL(a.href);
};

// ------------------------------------------------------------------ boot
renderMetricCards();
api("GET", "/api/config").then(({ config }) => { $("#vadThr").style.left = `${config.vad.threshold * 100}%`; $("#thrLabel").textContent = `threshold ${config.speaker.threshold}`; }).catch(() => {});
connect();
show(TITLES[location.hash.slice(1)] ? location.hash.slice(1) : "live");
