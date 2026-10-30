// VoiceModule Live client: browser AEC + duplex WS + local fast-path ducking.
const $ = (s) => document.querySelector(s);
let ws, audioCtx, capNode, playNode, micStream;
let playing = false, localDucked = false, speechMs = 0, curAssistantEl = null, ui_state = "IDLE";
const HINTS = { IDLE: "Press Start", LISTENING: "Listening — just talk", PROCESSING: "Thinking…",
                SPEAKING: "Speaking — talk over me to interrupt", INTERRUPTED: "Interrupted — listening", DUCKED: "Hearing you…" };

async function start() {
  micStream = await navigator.mediaDevices.getUserMedia({ audio: {
    echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
  audioCtx = new AudioContext();
  await audioCtx.resume();                     // autoplay policy: context may start suspended
  await audioCtx.audioWorklet.addModule("/static/capture-worklet.js");
  await audioCtx.audioWorklet.addModule("/static/playback-worklet.js");
  const src = audioCtx.createMediaStreamSource(micStream);
  capNode = new AudioWorkletNode(audioCtx, "capture");
  const micAn = audioCtx.createAnalyser(); micAn.fftSize = 2048;
  src.connect(micAn);                          // tap for the visualizer
  src.connect(capNode);
  const mute = audioCtx.createGain();          // worklet must be in the graph to run, at gain 0
  mute.gain.value = 0;
  capNode.connect(mute).connect(audioCtx.destination);
  playNode = new AudioWorkletNode(audioCtx, "playback");
  const outAn = audioCtx.createAnalyser(); outAn.fftSize = 2048;
  playNode.connect(outAn).connect(audioCtx.destination);
  Viz.init($("#scope"), micAn, outAn);
  playNode.port.onmessage = (e) => { if (e.data.played !== undefined && ws?.readyState === 1)
    ws.send(JSON.stringify({ type: "played", played_ms: e.data.played })); };
  capNode.port.onmessage = (e) => {
    if (ws?.readyState === 1) ws.send(e.data.pcm);
    $("#micFill").style.width = Math.min(100, e.data.rms * 600) + "%";
    $("#micDb").textContent = (20 * Math.log10(Math.max(e.data.rms, 1e-9))).toFixed(0) + " dB";
    localFastPath(e.data.rms);
  };
  connect();
}

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { $("#conn").textContent = "connected"; $("#startBtn").textContent = "■ Stop"; };
  ws.onclose = () => { $("#conn").textContent = "disconnected"; $("#startBtn").textContent = "▶ Start"; playing = false; };
  ws.onmessage = (e) => {
    if (typeof e.data === "string") return onEvent(JSON.parse(e.data));
    playNode.port.postMessage({ pcm: e.data }, [e.data]);   // int16 16kHz audio out
    playing = true;
  };
}

// LOCAL FAST PATH: ~120 ms of loud-ish mic while playing -> duck now, don't wait for server
function localFastPath(rms) {
  if (!playing) { speechMs = 0; if (localDucked) unduck(); return; }
  speechMs = rms > 0.02 ? speechMs + 20 : 0;
  if (speechMs >= 120 && !localDucked) { duck(); }
}

function duck() { localDucked = true; playNode.port.postMessage({ gain: 0.2 }); $("#orb").className = "ducked"; }
function unduck() { localDucked = false; playNode.port.postMessage({ gain: 1.0 }); }

function onEvent(ev) {
  switch (ev.type) {
    case "state": setState(ev.to); break;
    case "speech_started": duck(); break;                    // server tier-1 agrees
    case "duck": duck(); break;
    case "restore": unduck(); setState("SPEAKING"); break;
    case "interrupted":
      playNode.port.postMessage({ cmd: "flush" });           // hard stop: clear ring + 15 ms fade
      playing = false; speechMs = 0; unduck();
      addChip("barge", `barge-in ${ev.score ?? ""}`, `played ${ev.played_ms} ms`);
      setState("INTERRUPTED"); break;
    case "turn_complete": playing = false; setState("LISTENING"); break;
    case "transcript": onTranscript(ev); break;
    case "reply_delta": onDelta(ev.text); break;
    case "speaker":
      Viz.setScore(ev.score || 0);
      if (!ev.accepted) { Viz.setState("REJECTED"); setTimeout(() => Viz.setState(ui_state || "LISTENING"), 900); }
      addChip(ev.accepted ? "you" : "other", `${ev.accepted ? "you" : "other"} ${ev.score}`, ev.barge ? "interrupt check" : ""); break;
    case "error": addMsg("assistant", `⚠ ${ev.message}`); break;
  }
}

function setState(s) {
  ui_state = s;
  const orb = $("#orb");
  if (s !== "SPEAKING" && localDucked) unduck();
  orb.className = s.toLowerCase();
  Viz.setState(s === "DUCKED" ? "SPEAKING" : s);
  $("#state").textContent = s[0] + s.slice(1).toLowerCase();
  $("#hint").textContent = HINTS[s] || s;
}

function addMsg(role, text, meta = "") {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = `<div class="body"></div><div class="meta">${meta}</div>`;
  el.querySelector(".body").textContent = text;
  $("#chat").appendChild(el); $("#chat").scrollTop = 1e9;
  return el;
}
function onDelta(t) {
  if (!curAssistantEl) curAssistantEl = addMsg("assistant", "");
  curAssistantEl.querySelector(".body").textContent += t;
  $("#chat").scrollTop = 1e9;
}
function onTranscript(ev) {
  if (ev.role === "user") { curAssistantEl = null; addMsg("user", ev.text); }
  else { if (curAssistantEl) curAssistantEl.querySelector(".body").textContent = ev.text;
         else addMsg("assistant", ev.text); curAssistantEl = null; }
}
function addChip(kind, label, title) {
  const c = $("#chips");
  if (c.querySelector(".small")) c.innerHTML = "";
  const el = document.createElement("span");
  el.className = `chip ${kind}`; el.textContent = label; el.title = title;
  c.prepend(el);
  while (c.children.length > 24) c.lastChild.remove();
}

$("#startBtn").onclick = async () => {
  if (ws && ws.readyState === 1) { ws.close(); micStream?.getTracks().forEach(t => t.stop());
    audioCtx?.close(); ws = null; setState("IDLE"); return; }
  $("#startBtn").disabled = true;
  try { await start(); } catch (e) { $("#conn").textContent = "mic error: " + e.message; }
  $("#startBtn").disabled = false;
};
$("#vizBtn").onclick = () => {
  const next = Viz.styles[(Viz.styles.indexOf(Viz.getStyle()) + 1) % Viz.styles.length];
  Viz.setStyle(next);
  $("#vizBtn").textContent = "style: " + next;
};
setState("IDLE");
