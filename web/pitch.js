// F0 estimation via autocorrelation (YIN-lite): fast enough per animation frame on 2048 samples.
function estimateF0(buf, sampleRate) {
  const N = buf.length;
  let rms = 0;
  for (let i = 0; i < N; i++) rms += buf[i] * buf[i];
  rms = Math.sqrt(rms / N);
  if (rms < 0.01) return 0;                       // too quiet -> unvoiced
  const minLag = Math.floor(sampleRate / 500);    // 500 Hz ceiling
  const maxLag = Math.min(Math.floor(sampleRate / 60), N - 2); // 60 Hz floor
  let best = 0, bestLag = 0;
  let prev = 0, rising = false;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let sum = 0;
    for (let i = 0; i < N - lag; i++) sum += buf[i] * buf[i + lag];
    const c = sum / (N - lag);
    if (c > prev && c > best) { best = c; bestLag = lag; rising = true; }
    else if (rising && c < prev * 0.6 && bestLag) break;      // first strong peak
    prev = c;
  }
  if (!bestLag || best < 0.01) return 0;
  // parabolic interpolation for sub-sample accuracy
  const c = (l) => { let s = 0; for (let i = 0; i < N - l; i++) s += buf[i] * buf[i + l]; return s / (N - l); };
  const y0 = c(bestLag - 1), y1 = best, y2 = c(bestLag + 1);
  const shift = (y2 - y0) / Math.max(2 * (2 * y1 - y2 - y0), 1e-9);
  return sampleRate / (bestLag + Math.max(-0.5, Math.min(0.5, shift)));
}
