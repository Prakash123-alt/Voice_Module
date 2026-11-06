// Playback worklet: ring buffer of s16le PCM arriving at a fixed source rate,
// linearly resampled to the context rate. "clear" drains instantly (barge-in).
class Player extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ring = new Float32Array(44100 * 30); // up to 30 s buffered
    this.readIdx = 0;
    this.writeIdx = 0;
    this.srcRate = 22050;
    this.pos = 0; // fractional read position
    this.last = 0;
    this.port.onmessage = e => {
      const m = e.data;
      if (m.type === "push") {
        const s16 = new Int16Array(m.pcm);
        for (let i = 0; i < s16.length; i++) {
          const next = (this.writeIdx + 1) % this.ring.length;
          if (next === this.readIdx) break; // full: drop (backpressure)
          this.ring[this.writeIdx] = s16[i] / 32768;
          this.writeIdx = next;
        }
      } else if (m.type === "clear") {
        this.readIdx = this.writeIdx = 0;
        this.pos = 0; this.last = 0;
      } else if (m.type === "rate") {
        this.srcRate = m.rate;
      }
    };
  }
  available() {
    return (this.writeIdx - this.readIdx + this.ring.length) % this.ring.length;
  }
  process(_in, outputs) {
    const out = outputs[0][0];
    const step = this.srcRate / sampleRate;
    for (let i = 0; i < out.length; i++) {
      if (this.available() < 2) { out[i] = 0; continue; }
      const i0 = this.readIdx;
      const i1 = (i0 + 1) % this.ring.length;
      const frac = this.pos - Math.floor(this.pos);
      out[i] = this.ring[i0] * (1 - frac) + this.ring[i1] * frac;
      this.pos += step;
      while (this.pos >= 1) {
        this.pos -= 1;
        this.readIdx = (this.readIdx + 1) % this.ring.length;
        if (this.available() < 2) break;
      }
    }
    return true;
  }
}
registerProcessor("player", Player);
