import { useCallback, useEffect, useState } from "react";
import { VoiceSession } from "../net/session";
import type { AgentState } from "../net/types";
import { Captions, type CaptionLine } from "./Captions";
import { Orb } from "./Orb";

type Conn = "connecting" | "open" | "reconnecting" | "closed";

export function CallView() {
  const [session] = useState(() => new VoiceSession());
  const [state, setState] = useState<AgentState>("idle");
  const [conn, setConn] = useState<Conn>("closed");
  const [lines, setLines] = useState<CaptionLine[]>([]);
  const [partial, setPartial] = useState("");
  const [agentDraft, setAgentDraft] = useState("");
  const [muted, setMuted] = useState(false);
  const [notice, setNotice] = useState("");
  const [disclosure, setDisclosure] = useState("");
  const [text, setText] = useState("");
  const inCall = conn === "open" || conn === "connecting" || conn === "reconnecting";

  useEffect(() => {
    session.onState = s => { setState(s); if (s === "speaking") setAgentDraft(""); };
    session.onPartial = setPartial;
    session.onTranscript = t => {
      setPartial("");
      setLines(l => [...l, { role: "user", text: t }]);
    };
    session.onLlmDelta = d => setAgentDraft(a => a + d);
    session.onSentence = s => {
      setAgentDraft("");
      setLines(l => [...l, { role: "agent", text: s }]);
    };
    session.onSpeakerRejected = () =>
      setNotice("Voice not recognized — only the enrolled speaker can talk to the agent.");
    session.onError = (_c, m) => setNotice(m);
    session.onDisclosure = setDisclosure;
    session.onConnectionChange = setConn;
  }, [session]);

  const toggle = useCallback(async () => {
    setNotice("");
    if (inCall) {
      await session.stop();
      setLines([]); setPartial(""); setAgentDraft("");
    } else {
      await session.start();
    }
  }, [inCall, session]);

  const toggleMute = () => {
    const m = !muted;
    setMuted(m);
    session.setMuted(m);
  };

  const submitText = (e: React.FormEvent) => {
    e.preventDefault();
    const t = text.trim();
    if (!t) return;
    setText("");
    setLines(l => [...l, { role: "user", text: t }]);
    session.sendText(t);
  };

  // keyboard: space toggles call when the panel is focused, Esc ends it
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && inCall) void session.stop();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [inCall, session]);

  return (
    <div className="call-view">
      <Orb state={inCall ? state : "idle"} />

      {conn === "reconnecting" &&
        <div className="banner warn">Connection dropped — reconnecting…</div>}
      {notice && <div className="banner info">{notice}</div>}
      {disclosure && inCall && <div className="banner info">{disclosure}</div>}

      <Captions lines={[
        ...lines,
        ...(agentDraft ? [{ role: "agent" as const, text: agentDraft, partial: true }] : []),
      ]} livePartial={partial} />

      <div className="controls">
        {inCall ? (
          <>
            <button className={`btn ${muted ? "btn-on" : ""}`}
              onClick={toggleMute} aria-pressed={muted}>
              {muted ? "Unmute" : "Mute"}
            </button>
            <button className="btn btn-danger" onClick={toggle}>End call</button>
          </>
        ) : (
          <button className="btn btn-primary" onClick={toggle}>Start call</button>
        )}
      </div>

      {!inCall && (
        <form className="text-fallback" onSubmit={e => e.preventDefault()}>
          <input
            type="password"
            defaultValue={localStorage.getItem("vfw_master_token") ?? ""}
            onChange={e => localStorage.setItem("vfw_master_token", e.target.value)}
            placeholder="Firewall session token…"
            aria-label="Firewall session token"
          />
        </form>
      )}

      {inCall && (
        <form className="text-fallback" onSubmit={submitText}>
          <input
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder="Or type a message…"
            aria-label="Type a message"
          />
          <button className="btn" type="submit">Send</button>
        </form>
      )}
    </div>
  );
}
