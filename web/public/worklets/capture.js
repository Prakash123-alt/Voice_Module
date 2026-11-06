// Mic capture worklet: forwards Float32 frames to the main thread and reports
// per-block RMS so the client can fire a <200 ms barge-in locally.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.muted = false;
    this.port.onmessage = e => {
      if (e.data.type === "mute") this.muted = e.data.muted;
    };
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];
    if (!this.muted) {
      let sum = 0;
      for (let i = 0; i < ch.length; i++) sum += ch[i] * ch[i];
      const rms = Math.sqrt(sum / ch.length);
      // copy: the input buffer is reused by the audio thread
      const data = ch.slice(0);
      this.port.postMessage({ type: "pcm", data, rms }, [data.buffer]);
    }
    return true;
  }
}
registerProcessor("capture", Capture);
