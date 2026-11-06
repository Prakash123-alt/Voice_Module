import type { AgentState } from "../net/types";

const LABEL: Record<AgentState, string> = {
  idle: "Tap to start",
  listening: "Listening",
  thinking: "Thinking",
  speaking: "Speaking",
};

export function Orb({ state }: { state: AgentState }) {
  return (
    <div className="orb-wrap">
      <div className={`orb orb-${state}`} aria-hidden="true">
        <div className="orb-core" />
      </div>
      <div className="orb-label" role="status">{LABEL[state]}</div>
    </div>
  );
}
