// 说 · Shuo — 屏幕底部胶囊浮层（录音中 / 识别中）
//
// 只显示固定波形动画和转圈，不读真实音频；不抢焦点。

const { listen } = window.__TAURI__.event;

listen("busy-state", (event) => {
  const { mode, label } = event.payload || {};
  const recording = mode === "recording";
  const processing = mode === "processing";
  document.body.classList.toggle("recording", recording);
  document.body.classList.toggle("processing", processing);
  document.body.classList.toggle("idle", !recording && !processing);
  if (label) document.getElementById("label").textContent = label;
});
