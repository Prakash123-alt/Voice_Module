// Oscilloscope visualizer: CRT phosphor | amber LED matrix | blue spectrum.
// Reads mic + playback AnalyserNodes. window.Viz exposes {setState, setScore, setStyle}.
const Viz = (() => {
  let canvas, ctx, micAn, outAn, style = "crt";
  let st = { state: "IDLE", score: 0, dB: -90, f0: 0 };
  let amp = 0, peak = 0, trail;
  const STYLES = ["crt", "led", "spectrum"];

  function init(el, micAnalyser, outAnalyser) {
    canvas = el; ctx = canvas.getContext("2d");
    micAn = micAnalyser; outAn = outAnalyser;
    fit(); addEventListener("resize", fit);
    requestAnimationFrame(draw);
  }
  function fit() {
    const dpr = devicePixelRatio || 1, r = canvas.getBoundingClientRect();
    canvas.width = r.width * dpr; canvas.height = r.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    trail = document.createElement("canvas");
    trail.width = canvas.width; trail.height = canvas.height;
    trail.getContext("2d").scale(dpr, dpr);
  }
  const tb = new Float32Array(2048), fb = new Uint8Array(1024);

  function levels() {
    const an = (st.state === "SPEAKING" && outAn) ? outAn : micAn;
    if (!an) return { amp: 0, f0: 0, bass: 0, high: 0 };
    an.getFloatTimeDomainData(tb);
    let e = 0; for (const v of tb) e += v * v;
    const a = Math.sqrt(e / tb.length);
    let f0 = 0;
    if (a > 0.01) f0 = estimateF0(tb, an.context.sampleRate);
    an.getByteFrequencyData(fb);
    const band = (lo, hi) => fb.slice(lo, hi).reduce((s, v) => s + v, 0) / (hi - lo) / 255;
    return { amp: a, f0, bass: band(2, 20), high: band(80, 256) };
  }

  const COLORS = {
    IDLE: [90, 110, 140], LISTENING: [96, 165, 250], PROCESSING: [245, 158, 11],
    SPEAKING: [52, 211, 153], INTERRUPTED: [248, 113, 113], REJECTED: [180, 60, 60],
    DUCKED: [52, 211, 153],
  };
  const rgb = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

  function draw() {
    requestAnimationFrame(draw);
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w) return;
    const l = levels();
    amp += (l.amp * 8 - amp) * (l.amp > amp ? 0.5 : 0.06);        // fast attack, slow release
    peak = Math.max(l.amp * 8, peak * 0.995);
    st.f0 += (l.f0 - st.f0) * 0.2; st.dB += ((20 * Math.log10(Math.max(l.amp, 1e-6))) - st.dB) * 0.2;
    const col = COLORS[st.state] || COLORS.IDLE;
    const dens = st.f0 ? Math.max(0.5, Math.min(2.5, st.f0 / 220)) : 1;
    const t = performance.now() / 1000;

    if (style === "crt") {
      const tc = trail.getContext("2d");
      tc.globalCompositeOperation = "copy"; tc.drawImage(canvas, 0, 0);
      tc.globalCompositeOperation = "source-over";
      ctx.fillStyle = "rgba(4,10,6,0.28)"; ctx.fillRect(0, 0, w, h);          // persistence fade
      ctx.globalAlpha = 0.55; ctx.drawImage(trail, 0, 0, w, h); ctx.globalAlpha = 1;
      ctx.shadowBlur = 14; ctx.shadowColor = "#39ff88";
      ctx.strokeStyle = rgb(col, 0.95); ctx.lineWidth = 2;
      ctx.beginPath();
      for (let x = 0; x < w; x++) {
        const ph = x / w * Math.PI * 2 * 8 * dens + t * 3;
        const y = h / 2 + Math.sin(ph) * amp * h * 0.35 + Math.sin(t * 0.7 + x * 0.01) * 2;
        x ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.stroke(); ctx.shadowBlur = 0;
      ctx.fillStyle = "rgba(120,255,170,0.05)";                              // scanlines
      for (let y = 0; y < h; y += 4) ctx.fillRect(0, y, w, 1);
      vignette(w, h);
    } else if (style === "led") {
      ctx.fillStyle = "#0a0803"; ctx.fillRect(0, 0, w, h);
      const cols = 64, rows = 18, cw = w / cols, rh = h / rows;
      for (let c = 0; c < cols; c++) {
        const ph = c / cols * Math.PI * 2 * 8 * dens + t * 3;
        const lvl = Math.abs(Math.sin(ph)) * amp + l.bass * 0.4;
        const lit = Math.min(rows, Math.round(lvl * rows * 1.4));
        for (let r = 0; r < rows; r++) {
          const on = rows - r <= lit;
          ctx.fillStyle = on ? rgb(col, 0.4 + 0.6 * (rows - r) / rows) : "rgba(60,40,10,0.25)";
          ctx.fillRect(c * cw + 1, r * rh + 1, cw - 2, rh - 2);
        }
      }
    } else {                                                                   // spectrum
      ctx.fillStyle = "#04070f"; ctx.fillRect(0, 0, w, h);
      const bars = 96, bw = w / bars;
      for (let i = 0; i < bars; i++) {
        const v = (fb[Math.floor(i * fb.length / bars / 2)] / 255) * (0.3 + amp);
        const bh = Math.min(h * 0.9, v * h * 2);
        const grad = ctx.createLinearGradient(0, h - bh, 0, h);
        grad.addColorStop(0, rgb(col)); grad.addColorStop(1, rgb(col, 0.15));
        ctx.fillStyle = grad;
        ctx.fillRect(i * bw + 1, h - bh, bw - 2, bh);
      }
      ctx.strokeStyle = rgb(col, 0.4);
      ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
    }
    hud(w, col, l);
  }

  function vignette(w, h) {
    const g = ctx.createRadialGradient(w/2, h/2, h/3, w/2, h/2, w*0.7);
    g.addColorStop(0, "rgba(0,0,0,0)"); g.addColorStop(1, "rgba(0,0,0,0.45)");
    ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
  }

  function hud(w, col, l) {
    ctx.fillStyle = rgb(col, 0.9); ctx.font = "11px ui-monospace,monospace";
    ctx.fillText(`${st.dB.toFixed(0)} dB`, 8, 15);
    ctx.fillText(st.f0 > 40 ? `${st.f0.toFixed(0)} Hz` : "— Hz", 60, 15);
    if (st.score) ctx.fillText(`match ${st.score.toFixed(2)}`, 112, 15);
    ctx.fillText(st.state, w - ctx.measureText(st.state).width - 8, 15);
  }

  return {
    init, styles: STYLES,
    setStyle: (s) => { if (STYLES.includes(s)) style = s; },
    getStyle: () => style,
    setState: (s) => { st.state = s; },
    setScore: (v) => { st.score = v; },
  };
})();
