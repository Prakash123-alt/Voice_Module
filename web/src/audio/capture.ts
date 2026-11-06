/**
 * Mic capture: getUserMedia (browser AEC + noise suppression -> the "AEC"
 * stage of the pipeline) -> AudioWorklet -> s16le 16 kHz -> callback.
 * Also exposes a fast local VAD trigger for sub-200 ms barge-in.
 */
export class MicCapture {
  private ctx?: AudioContext;
  private stream?: MediaStream;
  private node?: AudioWorkletNode;
  private resamplePos = 0;
  private speechBlocks = 0;

  /** Fires with s16le 16 kHz chunks ready to send on the socket. */
  onPcm: (pcm: ArrayBuffer) => void = () => {};
  /** Fires on early local speech detection (~3 blocks, <100 ms). */
  onLocalSpeech: () => void = () => {};

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,   // AEC: keeps agent audio out of the mic
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    // request 16 kHz directly; browsers that ignore it get resampled below
    this.ctx = new AudioContext({ sampleRate: 16000 });
    await this.ctx.audioWorklet.addModule("/worklets/capture.js");
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, "capture");
    this.node.port.onmessage = e => this.handle(e.data);
    src.connect(this.node);
    this.node.connect(this.ctx.destination); // required on some browsers
  }

  setMuted(muted: boolean) {
    this.node?.port.postMessage({ type: "mute", muted });
    this.speechBlocks = 0;
  }

  private handle(m: { type: string; data: Float32Array; rms: number }) {
    if (m.type !== "pcm") return;
    // local fast VAD: ~3 consecutive loud blocks (~80 ms at 128-frame blocks)
    if (m.rms > 0.02) {
      if (++this.speechBlocks === 3) this.onLocalSpeech();
    } else {
      this.speechBlocks = 0;
    }
    this.onPcm(this.toPcm16(m.data));
  }

  private carry = new Float32Array(0);

  /** Float32 at ctx rate -> s16le at 16 kHz. Carries the tail of each block
   * so no samples are dropped between worklet deliveries. */
  private toPcm16(input: Float32Array): ArrayBuffer {
    const inRate = this.ctx!.sampleRate;
    const step = inRate / 16000;
    const all = new Float32Array(this.carry.length + input.length);
    all.set(this.carry);
    all.set(input, this.carry.length);
    const outLen = Math.max(
      0, Math.floor((all.length - 1 - this.resamplePos) / step));
    const s16 = new Int16Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const idx = this.resamplePos + i * step;
      const i0 = Math.floor(idx), f = idx - i0;
      const a = all[i0], b = all[Math.min(i0 + 1, all.length - 1)];
      const s = Math.max(-1, Math.min(1, a + (b - a) * f));
      s16[i] = s * 32767;
    }
    const used = this.resamplePos + outLen * step;
    const cut = Math.floor(used);
    this.carry = all.slice(cut);
    this.resamplePos = used - cut;
    return s16.buffer;
  }

  stop() {
    this.node?.disconnect();
    this.stream?.getTracks().forEach(t => t.stop());
    void this.ctx?.close();
    this.node = undefined; this.stream = undefined; this.ctx = undefined;
  }
}
