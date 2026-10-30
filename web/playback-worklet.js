// Playback: ring buffer, <= ~1 s queued; "stop" = clear + 15 ms fade. Gain for ducking.
class Playback extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Float32Array(16000 * 2);   // 2 s ring @16k (resampled here? server sends 16k)
    this._w = 0; this._r = 0; this._gain = 1; this._fade = 0; this._played = 0;
    this.port.onmessage = (e) => {
      const d = e.data;
      if (d.pcm) {
        const s = new Int16Array(d.pcm);
        for (let i = 0; i < s.length; i++) {
          this._buf[this._w % this._buf.length] = s[i] / 32768;
          this._w++;
        }
        if (this._w - this._r > this._buf.length) this._r = this._w - this._buf.length; // overflow: drop oldest
      } else if (d.cmd === "flush") {
        this._r = this._w; this._fade = Math.floor(sampleRate * 0.015);
      } else if (d.gain !== undefined) {
        this._gain = d.gain;
      }
    };
    setInterval(() => this.port.postMessage({ played: this._played / 16 }), 50); // ms @16k
  }
  process(_, outputs) {
    const out = outputs[0][0];
    for (let i = 0; i < out.length; i++) {
      let v = 0;
      if (this._r < this._w) {
        v = this._buf[this._r % this._buf.length];
        this._r++; this._played++;
        if (this._fade > 0) { v *= this._fade / (sampleRate * 0.015); this._fade--; }
      }
      out[i] = v * this._gain;
    }
    return true;
  }
}
registerProcessor("playback", Playback);
