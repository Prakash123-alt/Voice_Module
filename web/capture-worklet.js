// Capture: browser pipeline (already AEC'd by getUserMedia) -> 16 kHz PCM16 -> main.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Float32Array(0);
    this._rms = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    // linear resample contextRate -> 16k
    const ratio = sampleRate / 16000;
    const n = Math.floor((this._buf.length + ch.length) / ratio);
    const src = new Float32Array(this._buf.length + ch.length);
    src.set(this._buf); src.set(ch, this._buf.length);
    const out = new Int16Array(n);
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const v = src[Math.floor(i * ratio)];
      out[i] = Math.max(-1, Math.min(1, v)) * 32767;
      sum += v * v;
    }
    const used = Math.floor(n * ratio);
    this._buf = src.slice(used);
    if (n) this.port.postMessage({ pcm: out.buffer, rms: Math.sqrt(sum / n) }, [out.buffer]);
    return true;
  }
}
registerProcessor("capture", Capture);
