/** Streaming PCM player backed by an AudioWorklet ring buffer. */
export class PcmPlayer {
  private ctx?: AudioContext;
  private node?: AudioWorkletNode;

  async ensure() {
    if (this.ctx) return;
    this.ctx = new AudioContext();
    await this.ctx.audioWorklet.addModule("/worklets/player.js");
    this.node = new AudioWorkletNode(this.ctx, "player");
    this.node.connect(this.ctx.destination);
  }

  setRate(rate: number) {
    this.node?.port.postMessage({ type: "rate", rate });
  }

  push(pcm: ArrayBuffer) {
    this.node?.port.postMessage({ type: "push", pcm }, [pcm]);
  }

  /** Drop everything buffered — barge-in must sound within ~200 ms. */
  flush() {
    this.node?.port.postMessage({ type: "clear" });
  }

  async stop() {
    this.flush();
    this.node?.disconnect();
    await this.ctx?.close();
    this.ctx = undefined; this.node = undefined;
  }
}
