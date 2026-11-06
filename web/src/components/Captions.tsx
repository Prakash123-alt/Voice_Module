import { useEffect, useRef } from "react";

export interface CaptionLine {
  role: "user" | "agent";
  text: string;
  partial?: boolean;
}

export function Captions({ lines, livePartial }: {
  lines: CaptionLine[]; livePartial: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.scrollTo({ top: ref.current.scrollHeight });
  }, [lines, livePartial]);
  return (
    <div className="captions" ref={ref} aria-live="polite">
      {lines.map((l, i) => (
        <div key={i} className={`cap cap-${l.role}`}>
          <span className="cap-who">{l.role === "user" ? "You" : "Agent"}</span>
          <span>{l.text}</span>
        </div>
      ))}
      {livePartial && (
        <div className="cap cap-user cap-partial">
          <span className="cap-who">You</span>
          <span>{livePartial}…</span>
        </div>
      )}
    </div>
  );
}
