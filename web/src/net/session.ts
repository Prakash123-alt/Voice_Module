import { MicCapture } from "../audio/capture";
import { PcmPlayer } from "../audio/player";
import type {
  AgentState, ClientMsg, ServerMsg,
} from "./types";

/**
 * Owns the whole client call: session token, WS, mic capture, playback,
 * auto-reconnect (same sessionId -> server memory survives the drop).
 */
export class VoiceSession {
  state: AgentState = "idle";
  sessionId?: string;
  private ws?: WebSocket;
  private mic = new MicCapture();
  private player = new PcmPlayer();
  private reconnectAttempts = 0;
  private wantClose = false;
  private bufferedAudio: ArrayBuffer[] = [];
  private open = false;

  onState: (s: AgentState) => void = () => {};
  onPartial: (t: string) => void = () => {};
  onTranscript: (t: string, lang?: string) => void = () => {};
  onLlmDelta: (t: string) => void = () => {};
  onSentence: (t: string) => void = () => {};
  onSpeakerRejected: () => void = () => {};
  onError: (code: string, msg: string) => void = () => {};
  onConnectionChange: (s: "connecting" | "open" | "reconnecting" | "closed") => void = () => {};
  onDisclosure: (t: string) => void = () => {};

  async start() {
    this.wantClose = false;
    this.onConnectionChange("connecting");
    try {
      await this.mic.start();
    } catch {
      this.onError("mic_denied",
        "Microphone access was denied. Check browser permissions and retry.");
      this.onConnectionChange("closed");
      return;
    }
    await this.player.ensure();
    this.mic.onPcm = pcm => {
      if (this.open) this.ws?.send(pcm);
      else this.bufferedAudio.push(pcm);
    };
    this.mic.onLocalSpeech = () => {
      if (this.state === "speaking" || this.state === "thinking") {
        // <200 ms local cut, server VAD confirms shortly after
        this.player.flush();
        this.send({ type: "barge_in" });
      }
    };
    await this.connect();
  }

  private async connect() {
    // Voice Firewall: the master token (entered once in the UI, kept in
    // localStorage) is required to mint a session — /ws then uses the
    // per-session token returned here, so the master token never travels
    // in a WebSocket URL.
    const master = localStorage.getItem("vfw_master_token") ?? "";
    const res = await fetch("/api/session", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(master ? { Authorization: `Bearer ${master}` } : {}),
      },
      body: JSON.stringify(this.sessionId ? { sessionId: this.sessionId } : {}),
    });
    if (!res.ok) throw new Error(`session ${res.status}`);
    const { sessionId, token, disclosure } = await res.json();
    this.sessionId = sessionId;
    this.onDisclosure(disclosure);

    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws?token=${token}`);
    ws.binaryType = "arraybuffer";
    this.ws = ws;

    ws.onopen = () => {
      this.open = true;
      this.reconnectAttempts = 0;
      this.onConnectionChange("open");
      this.send({ type: "start" });
      for (const p of this.bufferedAudio) ws.send(p);
      this.bufferedAudio = [];
    };
    ws.onmessage = e => {
      if (typeof e.data === "string") this.handle(JSON.parse(e.data));
      else this.player.push(e.data as ArrayBuffer);
    };
    ws.onclose = () => {
      this.open = false;
      if (this.wantClose) { this.onConnectionChange("closed"); return; }
      // auto-reconnect with backoff; server-side memory keyed by sessionId
      const delay = Math.min(4000, 500 * 2 ** this.reconnectAttempts++);
      this.onConnectionChange("reconnecting");
      setTimeout(() => { if (!this.wantClose) void this.connect(); }, delay);
    };
    ws.onerror = () => ws.close();
  }

  private handle(m: ServerMsg) {
    switch (m.type) {
      case "state": this.state = m.state; this.onState(m.state); break;
      case "partial": this.onPartial(m.text); break;
      case "transcript": this.onTranscript(m.text, m.language); break;
      case "llm_delta": this.onLlmDelta(m.text); break;
      case "sentence": this.onSentence(m.text); break;
      case "audio_start": this.player.setRate(m.sampleRate); break;
      case "interrupt": this.player.flush(); break;
      case "speaker_rejected": this.onSpeakerRejected(); break;
      case "error": this.onError(m.code, m.message); break;
      case "ready": break;
      case "audio_end": case "tool": case "pong": break;
    }
  }

  send(m: ClientMsg) {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(m));
  }

  sendText(text: string) { this.send({ type: "text", text }); }
  setMuted(m: boolean) { this.mic.setMuted(m); this.send({ type: "mute", muted: m }); }

  async stop() {
    this.wantClose = true;
    this.send({ type: "stop" });
    // give the stop frame a beat to flush, then tear down locally
    await new Promise(r => setTimeout(r, 200));
    this.ws?.close();
    this.mic.stop();
    await this.player.stop();
    this.state = "idle";
    this.onState("idle");
    this.onConnectionChange("closed");
  }
}
