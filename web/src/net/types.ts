/** Mirrors server/src/protocol.ts — keep in sync. */
export type AgentState = "idle" | "listening" | "thinking" | "speaking";

export type ClientMsg =
  | { type: "start"; sessionId?: string }
  | { type: "stop" }
  | { type: "text"; text: string }
  | { type: "barge_in" }
  | { type: "mute"; muted: boolean }
  | { type: "ping"; t: number };

export type ServerMsg =
  | { type: "ready"; sessionId: string; disclosure: string }
  | { type: "state"; state: AgentState }
  | { type: "partial"; text: string; language?: string }
  | { type: "transcript"; text: string; language?: string }
  | { type: "llm_delta"; text: string }
  | { type: "sentence"; text: string }
  | { type: "audio_start"; sampleRate: number }
  | { type: "audio_end" }
  | { type: "interrupt" }
  | { type: "speaker_rejected"; similarity: number }
  | { type: "tool"; name: string; status: "start" | "done" | "error" }
  | { type: "error"; code: string; message: string }
  | { type: "pong"; t: number };
