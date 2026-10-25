// 说 · Shuo — 前端逻辑（纯静态，无构建步骤）
//
// 与 Rust 端约定：
//   命令：bootstrap / save_config / set_hotkey / start_record / stop_record
//        / copy_text / list_mics / list_presets / get_preset_prompt
//        / get_custom_prompts / save_custom_prompts / get_history / clear_history
//        / restart_recorder / reload_engine / open_user_dir / open_log / hide_window / quit_app
//   事件：engine-status / rec / processing / asr / final / asr-error / notice / hotkey / recorder
//
// 文案：`locales/*.json`（Python 版 i18n 生成，27 种语言）+ 本文件 EXTRA 里补的新键
// （其它语言回退英文）。
//
// SPDX-License-Identifier: GPL-3.0-or-later

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

/* ─────────────────────────── i18n ─────────────────────────── */

const EXTRA = {
  en: {
    "result.asr": "Original",
    "result.final": "Polished",
    "record.hold": "Hold to talk, release to stop",
    "history.title": "History",
    "history.empty": "No history yet",
    "history.disabled": "History is off (Settings → Save History)",
    "btn.copy": "Copy",
    "btn.record_key": "Record",
    "btn.refresh": "Refresh",
    "btn.apply": "Apply",
    "btn.save_as": "Save as",
    "btn.reload": "Reload",
    "btn.open_config": "Open config folder",
    "btn.open_log": "Open log",
    "settings.hotkey_key": "Global hotkey (Ctrl+V paste)",
    "settings.hotkey_terminal_key": "Terminal hotkey (Ctrl+Shift+V paste)",
    "settings.hotkey_tip": "Hold to talk: press to start, release to stop. The two hotkeys are independent and can be switched off separately.",
    "settings.llm": "LLM post-processing",
    "settings.llm_enable": "Enable LLM",
    "settings.llm_preset": "Prompt preset",
    "settings.llm_saved": "My prompts",
    "settings.llm_custom": "Custom prompt",
    "settings.prompt_name": "Prompt name",
    "settings.llm_hint": "The LLM turns spoken words into written text (punctuation, filler removal). When off, the raw ASR text is used.",
    "settings.single": "Single-model output",
    "settings.single_enable": "Enable (fine-tuned unified model)",
    "settings.single_preset": "Optimization prompt",
    "settings.single_hint": "The fine-tuned unified model applies the prompt while transcribing (recognition + polish in one pass); the LLM stage is skipped. Requires a unified model package (finetune/ pipeline); on the base package the prompt only biases recognition.",
    "settings.single_state_unified": "Unified package detected — prompts are executed as instructions.",
    "settings.single_state_base": "Base package (no unified.json) — prompt only biases recognition. Build a unified package with the finetune/ pipeline.",
    "status.single": "single-model",
    "badge.single": "single-model",
    "settings.engine": "Models & engine",
    "settings.interface": "Interface",
    "rec.capturing": "Press a new hotkey (mouse/keyboard)…",
    "toast.copied": "Copied",
    "toast.saved": "Saved",
    "toast.hotkey": "Hotkey updated",
    "toast.history_cleared": "History cleared",
    "toast.mic_switched": "Microphone switched",
    "toast.need_name": "Enter a prompt name first",
    "toast.hotkey_invalid": "Unsupported hotkey",
    "status.ready": "ready",
    "status.loading": "loading",
    "status.off": "off",
    "status.error": "error",
    "meta.processing": "Transcribing…",
    "meta.done": "Done",
    "confirm.clear": "Clear all history? This cannot be undone.",
    "banner.hotkey": "Global hotkey unavailable",
    "about.version": "Version",
    "about.engine": "ASR: Qwen3-ASR-0.6B (ONNX CPU) · LLM: Qwen3.5-2B (ONNX)",
    "about.license": "GPL-3.0-or-later · Tauri 3 · Rust + ort",
  },
  zh: {
    "result.asr": "识别原文",
    "result.final": "整理结果",
    "record.hold": "按住说话，松开结束",
    "history.title": "历史记录",
    "history.empty": "还没有记录",
    "history.disabled": "历史未开启（设置 → 保存历史）",
    "btn.copy": "复制",
    "btn.record_key": "录制",
    "btn.refresh": "刷新",
    "btn.apply": "应用",
    "btn.save_as": "另存为",
    "btn.reload": "重新加载",
    "btn.open_config": "打开配置目录",
    "btn.open_log": "打开日志",
    "settings.hotkey_key": "全局热键（Ctrl+V 上屏）",
    "settings.hotkey_terminal_key": "终端热键（Ctrl+Shift+V 上屏）",
    "settings.hotkey_tip": "按住说话：按下开始，松开结束。两个热键互相独立，可分别开关。",
    "settings.llm": "LLM 后处理",
    "settings.llm_enable": "启用 LLM",
    "settings.llm_preset": "提示词预设",
    "settings.llm_saved": "我的提示词",
    "settings.llm_custom": "自定义提示词",
    "settings.prompt_name": "提示词名称",
    "settings.llm_hint": "LLM 会把口语转成书面文本（补标点、去口癖）。关闭时直接输出 ASR 原文。",
    "settings.single": "单模型直出",
    "settings.single_enable": "启用（微调统一模型）",
    "settings.single_preset": "优化提示词",
    "settings.single_hint": "微调统一模型按提示词一步直出优化文本（识别+整理），不再经过 LLM。需要 unified 模型包（finetune/ 产线）；基座包上提示词仅作识别偏置。",
    "settings.single_state_unified": "已检测到统一包——提示词会被当作任务指令执行。",
    "settings.single_state_base": "当前为基座包（无 unified.json）——提示词仅作识别偏置。可用 finetune/ 产线制作统一包。",
    "status.single": "直出",
    "badge.single": "单模型直出",
    "settings.engine": "模型与引擎",
    "settings.interface": "界面",
    "rec.capturing": "请按下新的热键（鼠标/键盘）…",
    "toast.copied": "已复制",
    "toast.saved": "已保存",
    "toast.hotkey": "热键已更新",
    "toast.history_cleared": "历史已清空",
    "toast.mic_switched": "麦克风已切换",
    "toast.need_name": "请先填写提示词名称",
    "toast.hotkey_invalid": "不支持这个热键",
    "status.ready": "就绪",
    "status.loading": "加载中",
    "status.off": "未启用",
    "status.error": "出错",
    "meta.processing": "识别中…",
    "meta.done": "完成",
    "confirm.clear": "确定清空所有历史记录？此操作不可撤销。",
    "banner.hotkey": "全局热键不可用",
    "about.version": "版本",
    "about.engine": "ASR：Qwen3-ASR-0.6B（ONNX CPU） · LLM：Qwen3.5-2B（ONNX）",
    "about.license": "GPL-3.0-or-later · Tauri 3 · Rust + ort",
  },
};

const I18N = { en: {}, cur: {}, lang: "en" };

async function loadLocale(lang) {
  const get = async (code) => {
    const r = await fetch(`locales/${code}.json`, { cache: "no-store" });
    if (!r.ok) throw new Error(`locale ${code}: ${r.status}`);
    return r.json();
  };
  I18N.en = { ...(await get("en")), ...EXTRA.en };
  I18N.cur = lang === "en" ? {} : { ...(await get(lang)), ...(EXTRA[lang] || {}) };
  I18N.lang = lang;
}

function t(key, vars) {
  let s = I18N.cur[key] ?? I18N.en[key] ?? key;
  if (vars) {
    for (const [k, v] of Object.entries(vars)) s = s.split(`{${k}}`).join(String(v));
  }
  return s;
}

function isZh() {
  return String(I18N.lang).toLowerCase().startsWith("zh") || I18N.lang === "yue";
}

/* ─────────────────────────── 状态 ─────────────────────────── */

const state = {
  cfg: {},
  status: {},
  history: [],
  presets: [],
  presetsAsr: [],
  custom: {},
  micList: [],
  locales: [],
  recording: false,
  recStart: 0,
  raf: 0,
  capturing: false,
  current: { asr: "", final: "" },
  lastRecordS: 0,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/* ─────────────────────────── 工具 ─────────────────────────── */

function toast(msg, kind = "", ms = 2600) {
  const box = $("#toasts");
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  box.appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    el.style.transition = "opacity .2s";
    setTimeout(() => el.remove(), 220);
  }, ms);
}

function fmtSec(s) {
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}

function displaySpec(spec) {
  if (!spec) return "";
  const parts = spec.split("+").filter(Boolean);
  return parts
    .map((p) => {
      const low = p.toLowerCase();
      const named = {
        ctrl: "Ctrl", control: "Ctrl", shift: "Shift", alt: "Alt", option: "Alt",
        win: "Win", super: "Super", meta: "Meta", cmd: "Cmd", command: "Cmd",
        xbutton1: "XButton1", xbutton2: "XButton2", mouse4: "XButton1", mouse5: "XButton2",
        esc: "Esc", tab: "Tab", enter: "Enter", return: "Enter", space: "Space",
        backspace: "Backspace", pageup: "PageUp", pagedown: "PageDown", up: "Up",
        down: "Down", left: "Left", right: "Right", insert: "Insert", delete: "Delete",
        home: "Home", end: "End", pause: "Pause", capslock: "CapsLock",
        numlock: "NumLock", scrolllock: "ScrollLock", printscreen: "PrintScreen",
      };
      if (named[low]) return named[low];
      if (/^f\d{1,2}$/.test(low)) return low.toUpperCase();
      if (low.length === 1) return low.toUpperCase();
      return p;
    })
    .join("+");
}

function langName(code) {
  if (!code) return "";
  const key = `asr_lang.${code}`;
  const v = I18N.cur[key] ?? I18N.en[key];
  return v || code;
}

const ASR_LANGS = [
  "auto", "zh", "en", "yue", "ja", "de", "ko", "ru", "fr", "pt", "ar", "it", "es",
  "hi", "id", "th", "tr", "uk", "vi", "cs", "da", "fil", "fi", "is", "ms", "no",
  "pl", "sv",
];

/* ─────────────────────────── 静态文案 ─────────────────────────── */

function applyStaticText() {
  $$("[data-i18n]").forEach((el) => {
    el.textContent = t(el.dataset.i18n);
  });
  $$("[data-i18n-title]").forEach((el) => {
    el.title = t(el.dataset.i18nTitle);
  });
  $$("[data-i18n-placeholder]").forEach((el) => {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  });
  document.title = `${t("app.title")} · Shuo`;
  $("#record-hint").textContent = t("record.hold");
  $("#loading-text").textContent = t("loading.model");
  $("#about-line").textContent = `${t("about.version")} ${state.version || ""} · ${t("about.license")}`;
  $("#model-hint").textContent = state.modelHint || "";
  renderHotkeyBadge();
  renderStatus(state.status);
}

/* ─────────────────────────── 渲染 ─────────────────────────── */

function setChip(el, kind, label) {
  el.classList.remove("ok", "pending", "err", "off");
  if (kind) el.classList.add(kind);
  const span = el.querySelector("span");
  if (span) span.textContent = label;
}

function renderStatus(s) {
  state.status = s || {};
  const asr = state.status;
  const asrLabel = asr.asr_ready
    ? `${t("status.ready")} · ASR`
    : asr.error
      ? `${t("status.error")} · ASR`
      : `${t("status.loading")} · ASR`;
  setChip($("#chip-asr"), asr.asr_ready ? "ok" : asr.error ? "err" : "pending", asrLabel);

  const llmOn = !!(state.cfg && state.cfg.llm_enabled);
  const singleOn = !!(state.cfg && state.cfg.single_model_enabled);
  const llmLabel = !llmOn || singleOn
    ? `LLM · ${t("status.off")}`
    : asr.llm_ready
      ? `LLM · ${t("status.ready")}`
      : `${t("status.loading")} · LLM`;
  setChip($("#chip-llm"), !llmOn || singleOn ? "off" : asr.llm_ready ? "ok" : "pending", llmLabel);

  const chipSingle = $("#chip-single");
  if (chipSingle) {
    chipSingle.hidden = !singleOn;
    if (singleOn) {
      const sLabel = `${t("status.single")} · ${asr.asr_unified ? t("status.ready") : t("settings.single_state_base").split("——")[0]}`;
      setChip(chipSingle, asr.asr_unified ? "ok" : "pending", asr.asr_unified ? `${t("status.single")} · ${t("status.ready")}` : sLabel);
    }
  }
  renderSingleModelState();

  const dn = !!state.cfg && !!state.cfg.denoise;
  const dnLabel = dn
    ? asr.denoise_ready
      ? `${t("settings.denoise")} · ${t("status.ready")}`
      : `${t("settings.denoise")} · ${t("status.loading")}`
    : `${t("settings.denoise")} · ${t("status.off")}`;
  setChip($("#chip-denoise"), dn ? (asr.denoise_ready ? "ok" : "pending") : "off", dnLabel);

  $("#dir-asr").textContent = asr.asr_dir || "—";
  $("#dir-llm").textContent = asr.llm_dir || "—";

  const banner = $("#banner");
  if (asr.error) {
    $("#banner-text").textContent = asr.error;
    banner.hidden = false;
  } else {
    banner.hidden = true;
  }

  const loading = (asr.loading && !asr.asr_ready) || asr.llm_loading;
  $("#loading").hidden = !loading;
  if (loading) {
    $("#loading-text").textContent = asr.llm_loading && asr.asr_ready
      ? "LLM"
      : t("loading.model");
  }

  const canRecord = !!asr.asr_ready;
  $("#record").disabled = !canRecord || state.recording;

  const bits = [];
  if (asr.asr_ready) bits.push("ASR");
  if (asr.llm_ready) bits.push("LLM");
  if (asr.denoise_ready) bits.push(t("settings.denoise"));
  $("#engine-line").textContent = bits.length
    ? bits.join(" · ")
    : asr.error
      ? t("status.error")
      : t("loading.model");
}

function renderHotkeyBadge() {
  const c = state.cfg;
  const mainOn = c.hotkey_enabled !== false;
  const termOn = c.hotkey_terminal_enabled !== false;
  const mainDisp = displaySpec(c.hotkey || "");
  const termDisp = displaySpec(c.hotkey_terminal || "");
  $("#hotkey-value").textContent = mainOn ? mainDisp || "—" : "—";
  $("#hotkey-value-terminal").textContent = termOn ? termDisp || "—" : "—";
  const joined = [mainOn ? mainDisp : "", termOn ? termDisp : ""].filter(Boolean).join(" / ");
  const badge = $("#hotkey-badge");
  badge.hidden = !joined;
  badge.textContent = joined;
  $("#hotkey-hint").textContent = joined ? t("hint.hotkey", { key: joined }) : "";
}

function renderResult() {
  const asr = state.current.asr;
  const final = state.current.final;
  $("#asr-text").textContent = asr || "—";
  $("#asr-text").classList.toggle("muted", !asr);
  $("#final-text").textContent = final || "—";
  $("#final-text").classList.toggle("muted", !final);
  $("#btn-copy").disabled = !final;
  const lang = state.current.language;
  $("#asr-lang").textContent = lang ? langName(lang) : "";
}

function renderMeta(text) {
  $("#meta").textContent = text || "";
}

function renderHistory() {
  const ul = $("#history");
  ul.innerHTML = "";
  const items = state.history;
  $("#history-count").textContent = items.length ? String(items.length) : "";
  const emptyEl = $("#history-empty");
  if (!state.cfg.save_history) {
    emptyEl.textContent = t("history.disabled");
    emptyEl.hidden = items.length > 0;
  } else {
    emptyEl.textContent = t("history.empty");
    emptyEl.hidden = items.length > 0;
  }
  for (const item of items) {
    const li = document.createElement("li");
    const text = item.llm_text || item.asr_text || "";
    const sub = item.llm_text ? item.asr_text : "";

    const top = document.createElement("div");
    top.className = "h-top";
    const p = document.createElement("span");
    p.className = "h-text";
    p.textContent = text;
    top.append(p);
    li.append(top);

    const meta = document.createElement("div");
    meta.className = "h-meta";
    const time = document.createElement("span");
    time.className = "h-time";
    time.textContent = item.time || "";
    const s = document.createElement("span");
    s.className = "h-sub";
    s.textContent = sub;
    const copy = document.createElement("button");
    copy.className = "icon-btn sm h-copy";
    copy.title = t("btn.copy");
    copy.innerHTML = '<svg><use href="#i-copy"/></svg>';
    copy.addEventListener("click", (e) => {
      e.stopPropagation();
      copyText(text);
    });
    meta.append(time, s, copy);
    li.append(meta);

    li.addEventListener("click", () => copyText(text));
    ul.append(li);
  }
}

/* ─────────────────────────── 录音 ─────────────────────────── */

function startTimer() {
  state.recStart = performance.now();
  cancelAnimationFrame(state.raf);
  const tick = () => {
    if (!state.recording) return;
    const s = (performance.now() - state.recStart) / 1000;
    $("#timer").textContent = fmtSec(s);
    state.raf = requestAnimationFrame(tick);
  };
  state.raf = requestAnimationFrame(tick);
}

function onRecEvent(payload) {
  const active = !!payload.active;
  state.recording = active;
  document.body.classList.toggle("recording", active);
  $("#record").disabled = !state.status.asr_ready || active;
  const btn = $("#record");
  btn.querySelector(".mic use").setAttribute("href", active ? "#i-stop" : "#i-mic");
  if (active) {
    $("#record-hint").textContent = t("record.hold");
    startTimer();
  } else {
    cancelAnimationFrame(state.raf);
    if (payload.duration_s > 0) {
      state.lastRecordS = payload.duration_s;
      $("#timer").textContent = fmtSec(payload.duration_s);
    } else {
      $("#timer").textContent = "0.0s";
    }
    $("#record-hint").textContent = t("record.hold");
    renderMeta("");
  }
}

async function pressRecord() {
  if (state.recording) return;
  try {
    await invoke("start_record");
  } catch (e) {
    toast(String(e), "err");
  }
}

async function releaseRecord() {
  if (!state.recording) return;
  try {
    await invoke("stop_record");
  } catch (e) {
    toast(String(e), "err");
  }
}

function bindRecordButton() {
  const btn = $("#record");
  btn.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || btn.disabled) return;
    e.preventDefault();
    btn.setPointerCapture(e.pointerId);
    pressRecord();
  });
  const up = (e) => {
    if (e && e.button !== undefined && e.button !== 0) return;
    releaseRecord();
  };
  btn.addEventListener("pointerup", up);
  btn.addEventListener("pointercancel", up);
  // 键盘：空格 / 回车按住也能说
  btn.addEventListener("keydown", (e) => {
    if ((e.key === " " || e.key === "Enter") && !e.repeat) {
      e.preventDefault();
      pressRecord();
    }
  });
  btn.addEventListener("keyup", (e) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      releaseRecord();
    }
  });
}

/* ─────────────────────────── 复制 / 插入 ─────────────────────────── */

async function copyText(text) {
  if (!text) return;
  try {
    await invoke("copy_text", { text });
    toast(t("toast.copied"), "ok", 1400);
  } catch (e) {
    toast(String(e), "err");
  }
}

/* ─────────────────────────── 设置 ─────────────────────────── */

async function patchConfig(patch) {
  state.cfg = await invoke("save_config", { patch });
  return state.cfg;
}

function presetName(p) {
  return isZh() ? p.name_zh : p.name_en;
}

function fillLocales() {
  const sel = $("#sel-ui-lang");
  sel.innerHTML = "";
  for (const { code, name } of state.locales) {
    const o = document.createElement("option");
    o.value = code;
    o.textContent = name;
    sel.append(o);
  }
  sel.value = state.cfg.language || "en";
}

function fillAsrLangs() {
  const sel = $("#sel-asr-lang");
  sel.innerHTML = "";
  for (const code of ASR_LANGS) {
    const o = document.createElement("option");
    o.value = code;
    o.textContent = langName(code);
    sel.append(o);
  }
  sel.value = state.cfg.asr_lang || "auto";
}

function fillMics() {
  const sel = $("#sel-mic");
  sel.innerHTML = "";
  const def = document.createElement("option");
  def.value = "";
  def.textContent = t("settings.mic_default");
  sel.append(def);
  const saved = state.cfg.mic_device || "";
  let found = !saved;
  for (const [name] of state.micList) {
    const o = document.createElement("option");
    o.value = name;
    o.textContent = name;
    sel.append(o);
    if (name === saved) found = true;
  }
  if (saved && !found) {
    const o = document.createElement("option");
    o.value = saved;
    o.textContent = `${saved} (${t("status.off")})`;
    sel.append(o);
  }
  sel.value = saved;
}

function fillPresets() {
  const sel = $("#sel-preset");
  sel.innerHTML = "";
  for (const p of state.presets) {
    const o = document.createElement("option");
    o.value = p.key;
    o.textContent = presetName(p);
    sel.append(o);
  }
  const custom = document.createElement("option");
  custom.value = "custom";
  custom.textContent = t("settings.llm_custom");
  sel.append(custom);
  sel.value = state.cfg.llm_preset || "standard";
}

function fillSinglePresets() {
  const sel = $("#sel-single-preset");
  if (!sel) return;
  sel.innerHTML = "";
  for (const p of state.presetsAsr) {
    const o = document.createElement("option");
    o.value = p.key;
    o.textContent = presetName(p);
    sel.append(o);
  }
  const custom = document.createElement("option");
  custom.value = "custom";
  custom.textContent = t("settings.llm_custom");
  sel.append(custom);
  sel.value = state.cfg.single_model_preset || "polish_zh";
}

async function updateSinglePromptArea() {
  const ta = $("#single-prompt");
  if (!ta) return;
  const preset = state.cfg.single_model_preset || "polish_zh";
  if (preset === "custom") {
    ta.value = state.cfg.single_model_prompt || "";
    ta.readOnly = false;
    ta.hidden = false;
    ta.classList.remove("disabled");
  } else {
    try {
      ta.value = await invoke("get_asr_preset_prompt", { key: preset });
    } catch {
      ta.value = "";
    }
    ta.readOnly = true;
    ta.hidden = false;
    ta.classList.add("disabled");
  }
}

function fillCustomPrompts() {
  const sel = $("#sel-custom");
  sel.innerHTML = "";
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "—";
  sel.append(none);
  for (const name of Object.keys(state.custom)) {
    const o = document.createElement("option");
    o.value = name;
    o.textContent = name;
    sel.append(o);
  }
}

function syncSettings() {
  const c = state.cfg;
  $("#chk-denoise").checked = !!c.denoise;
  $("#chk-auto-type").checked = !!c.auto_type;
  $("#chk-history").checked = !!c.save_history;
  $("#chk-remove-punc").checked = !!c.remove_punc;
  $("#chk-llm").checked = !!c.llm_enabled;
  const chkSingle = $("#chk-single");
  if (chkSingle) chkSingle.checked = !!c.single_model_enabled;
  $("#row-remove-punc").style.display = isZh() ? "" : "none";
  $("#chk-hotkey").checked = c.hotkey_enabled !== false;
  $("#chk-hotkey-terminal").checked = c.hotkey_terminal_enabled !== false;
  $("#hotkey-value-terminal").textContent = c.hotkey_terminal || "—";
  fillMics();
  fillAsrLangs();
  fillPresets();
  fillSinglePresets();
  fillCustomPrompts();
  fillLocales();
  renderHotkeyBadge();
  updatePromptArea();
  updateSinglePromptArea();
  renderSingleModelState();
}

function renderSingleModelState() {
  const el = $("#single-model-state");
  if (!el) return;
  const unified = !!(state.status && state.status.asr_unified);
  el.textContent = unified
    ? "✔ " + t("settings.single_state_unified")
    : "⚠ " + t("settings.single_state_base");
}

async function updatePromptArea() {
  const preset = state.cfg.llm_preset || "standard";
  const ta = $("#llm-prompt");
  if (preset === "custom") {
    ta.value = state.cfg.llm_system_prompt || "";
    ta.readOnly = false;
    ta.classList.remove("disabled");
  } else {
    try {
      ta.value = await invoke("get_preset_prompt", { key: preset });
    } catch {
      ta.value = "";
    }
    ta.readOnly = true;
  }
}

function bindSettings() {
  const open = () => {
    $("#drawer").hidden = false;
    $("#overlay").hidden = false;
    $("#drawer").setAttribute("aria-hidden", "false");
    syncSettings();
  };
  const close = () => {
    $("#drawer").hidden = true;
    $("#overlay").hidden = true;
    $("#drawer").setAttribute("aria-hidden", "true");
    stopCapture();
  };
  $("#btn-settings").addEventListener("click", open);
  $("#btn-close-settings").addEventListener("click", close);
  $("#overlay").addEventListener("click", close);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("#drawer").hidden) close();
  });

  $("#chk-denoise").addEventListener("change", async (e) => {
    await patchConfig({ denoise: e.target.checked });
    renderStatus(state.status);
    try {
      await invoke("restart_recorder");
    } catch (err) {
      toast(String(err), "err");
    }
  });
  $("#chk-auto-type").addEventListener("change", (e) => patchConfig({ auto_type: e.target.checked }));
  $("#chk-history").addEventListener("change", async (e) => {
    await patchConfig({ save_history: e.target.checked });
    renderHistory();
  });
  $("#chk-remove-punc").addEventListener("change", (e) => patchConfig({ remove_punc: e.target.checked }));
  $("#chk-llm").addEventListener("change", async (e) => {
    await patchConfig({ llm_enabled: e.target.checked });
    renderStatus(state.status);
  });
  const chkSingle = $("#chk-single");
  if (chkSingle) {
    chkSingle.addEventListener("change", async (e) => {
      await patchConfig({ single_model_enabled: e.target.checked });
      renderStatus(state.status);
    });
  }
  const selSingle = $("#sel-single-preset");
  if (selSingle) {
    selSingle.addEventListener("change", async (e) => {
      await patchConfig({ single_model_preset: e.target.value });
      await updateSinglePromptArea();
    });
  }
  const taSingle = $("#single-prompt");
  if (taSingle) {
    taSingle.addEventListener("change", async () => {
      if ((state.cfg.single_model_preset || "") !== "custom") return;
      await patchConfig({ single_model_prompt: taSingle.value });
      toast(t("toast.saved"), "ok", 1200);
    });
  }

  $("#sel-mic").addEventListener("change", async (e) => {
    const dev = e.target.value || null;
    try {
      await patchConfig({ mic_device: dev });
      await invoke("restart_recorder");
      toast(t("toast.mic_switched"), "ok", 1400);
    } catch (err) {
      toast(String(err), "err");
    }
  });
  $("#btn-refresh-mic").addEventListener("click", async () => {
    state.micList = await invoke("list_mics");
    fillMics();
  });
  $("#sel-asr-lang").addEventListener("change", (e) => patchConfig({ asr_lang: e.target.value }));
  $("#chk-hotkey").addEventListener("change", async (e) => {
    await patchConfig({ hotkey_enabled: e.target.checked });
    renderHotkeyBadge();
  });
  $("#chk-hotkey-terminal").addEventListener("change", async (e) => {
    await patchConfig({ hotkey_terminal_enabled: e.target.checked });
    renderHotkeyBadge();
  });

  $("#sel-preset").addEventListener("change", async (e) => {
    const key = e.target.value;
    await patchConfig({ llm_preset: key });
    await updatePromptArea();
  });

  $("#llm-prompt").addEventListener("change", async () => {
    if ((state.cfg.llm_preset || "") !== "custom") return;
    const text = $("#llm-prompt").value;
    await patchConfig({ llm_system_prompt: text });
    toast(t("toast.saved"), "ok", 1200);
  });

  $("#sel-custom").addEventListener("change", async (e) => {
    const name = e.target.value;
    if (!name) return;
    const body = state.custom[name] || "";
    $("#llm-prompt").value = body;
    await patchConfig({ llm_preset: "custom", llm_system_prompt: body });
    $("#sel-preset").value = "custom";
    toast(t("toast.saved"), "ok", 1200);
  });

  $("#btn-apply-custom").addEventListener("click", async () => {
    const name = $("#sel-custom").value;
    if (!name) return;
    const body = state.custom[name] || "";
    $("#llm-prompt").value = body;
    await patchConfig({ llm_preset: "custom", llm_system_prompt: body });
    $("#sel-preset").value = "custom";
    toast(t("toast.saved"), "ok", 1200);
  });

  $("#btn-save-custom").addEventListener("click", async () => {
    const name = $("#custom-name").value.trim();
    if (!name) {
      toast(t("toast.need_name"), "err");
      return;
    }
    state.custom[name] = $("#llm-prompt").value;
    try {
      await invoke("save_custom_prompts", { prompts: state.custom });
      fillCustomPrompts();
      $("#sel-custom").value = name;
      toast(t("toast.saved"), "ok", 1400);
    } catch (e) {
      toast(String(e), "err");
    }
  });

  $("#sel-ui-lang").addEventListener("change", async (e) => {
    const lang = e.target.value;
    await patchConfig({ language: lang });
    await loadLocale(lang);
    applyStaticText();
    syncSettings();
    renderResult();
    renderHistory();
    renderStatus(state.status);
  });

  $("#btn-reload").addEventListener("click", async () => {
    await invoke("reload_engine");
    toast(t("status.loading"), "", 1600);
  });
  $("#btn-open-dir").addEventListener("click", () => invoke("open_user_dir"));
  $("#btn-open-log").addEventListener("click", () => invoke("open_log"));
  $("#banner-log").addEventListener("click", () => invoke("open_log"));
  $("#banner-close").addEventListener("click", () => {
    $("#banner").hidden = true;
  });

  $("#btn-clear-history").addEventListener("click", async () => {
    if (!window.confirm(t("confirm.clear"))) return;
    await invoke("clear_history");
    state.history = [];
    renderHistory();
    toast(t("toast.history_cleared"), "ok", 1400);
  });

  $("#btn-capture").addEventListener("click", () => {
    if (state.capturing) stopCapture();
    else startCapture("main");
  });
  $("#btn-capture-terminal").addEventListener("click", () => {
    if (state.capturing) stopCapture();
    else startCapture("terminal");
  });
  bindResultActions();
}

function bindResultActions() {
  $$("[data-copy]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const which = btn.dataset.copy;
      copyText(which === "asr" ? state.current.asr : state.current.final);
    });
  });
  $("#btn-copy").addEventListener("click", () => copyText(state.current.final));
  $("#history").addEventListener("click", () => {});
}

/* ─────────────────────────── 热键录制 ─────────────────────────── */

const CODE_TO_TOKEN = {
  Backquote: "`", Minus: "-", Equal: "=", BracketLeft: "[", BracketRight: "]",
  Backslash: "\\", Semicolon: ";", Quote: "'", Comma: ",", Period: ".", Slash: "/",
  Escape: "esc", Tab: "tab", Enter: "enter", NumpadEnter: "enter",
  Backspace: "backspace", Space: "space", Insert: "insert", Delete: "delete",
  Home: "home", End: "end", PageUp: "pageup", PageDown: "pagedown",
  ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
  Pause: "pause", CapsLock: "capslock", NumLock: "numlock", ScrollLock: "scrolllock",
  PrintScreen: "printscreen",
};

function tokenFromEvent(e) {
  if (/^Key[A-Z]$/.test(e.code)) return e.code.slice(3).toLowerCase();
  if (/^Digit\d$/.test(e.code)) return e.code.slice(5);
  if (/^Numpad\d$/.test(e.code)) return `num${e.code.slice(6)}`;
  if (/^F\d{1,2}$/.test(e.code)) {
    const n = Number(e.code.slice(1));
    return n >= 1 && n <= 24 ? `f${n}` : null;
  }
  return CODE_TO_TOKEN[e.code] || null;
}

function modsOf(e) {
  const mods = [];
  if (e.ctrlKey) mods.push("ctrl");
  if (e.altKey) mods.push("alt");
  if (e.shiftKey) mods.push("shift");
  if (e.metaKey) mods.push("win");
  return mods;
}

let captureCtx = null;

function startCapture(which = "main") {
  if (state.capturing) return;
  state.capturing = true;
  const btn = which === "terminal" ? $("#btn-capture-terminal") : $("#btn-capture");
  const prev = btn.textContent;
  btn.textContent = t("btn.cancel");
  $("#hotkey-tip").textContent = t("rec.capturing");

  const finish = async (spec) => {
    stopCapture();
    try {
      await invoke("set_hotkey", { spec, which });
      if (which === "terminal") state.cfg.hotkey_terminal = spec;
      else state.cfg.hotkey = spec;
      renderHotkeyBadge();
      toast(t("toast.hotkey"), "ok", 1600);
    } catch (err) {
      toast(t("toast.hotkey_invalid") + ": " + String(err), "err");
    }
  };

  const onKey = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.key === "Escape") {
      stopCapture();
      return;
    }
    const token = tokenFromEvent(e);
    if (!token) return; // 只按下修饰键时继续等
    const mods = modsOf(e);
    if (!/^f\d{1,2}$/.test(token) && mods.length === 0 && token.length === 1) return;
    finish([...mods, token].join("+"));
  };

  const onMouse = (e) => {
    if (e.button !== 3 && e.button !== 4) return;
    e.preventDefault();
    e.stopPropagation();
    const token = e.button === 3 ? "xbutton1" : "xbutton2";
    finish([...modsOf(e), token].join("+"));
  };

  window.addEventListener("keydown", onKey, true);
  window.addEventListener("mousedown", onMouse, true);

  captureCtx = {
    cleanup: () => {
      btn.textContent = prev;
      $("#hotkey-tip").textContent = t("settings.hotkey_tip");
      state.capturing = false;
      window.removeEventListener("keydown", onKey, true);
      window.removeEventListener("mousedown", onMouse, true);
    },
  };
}

function stopCapture() {
  if (captureCtx) {
    captureCtx.cleanup();
    captureCtx = null;
  }
}

/* ─────────────────────────── 事件 ─────────────────────────── */

async function bindEvents() {
  await listen("engine-status", (e) => renderStatus(e.payload));
  await listen("rec", (e) => onRecEvent(e.payload));
  await listen("recorder", (e) => {
    if (!e.payload.ok) toast(e.payload.error || "recorder error", "err");
  });
  await listen("processing", () => {
    state.current = { asr: "", final: "", language: "" };
    renderResult();
    $("#meta-llm").textContent = "";
    renderMeta(t("meta.processing"));
    $("#btn-copy").disabled = true;
  });
  await listen("asr", (e) => {
    const p = e.payload;
    state.current.asr = p.text || "";
    state.current.language = p.language || "";
    state.current.final = p.text || "";
    $("#meta-llm").textContent = "";
    renderResult();
    const parts = [];
    if (p.language) parts.push(langName(p.language));
    if (p.audio_s) parts.push(`${Number(p.audio_s).toFixed(1)}s`);
    if (p.asr_s) parts.push(`ASR ${Number(p.asr_s).toFixed(2)}s`);
    renderMeta(parts.join(" · "));
  });
  await listen("final", (e) => {
    const p = e.payload;
    state.current.asr = p.asr_text || state.current.asr;
    state.current.final = p.text || p.llm_text || p.asr_text || "";
    renderResult();
    const parts = [];
    if (p.llm_text) parts.push(t("result.final"));
    if (p.llm_s) parts.push(`LLM ${Number(p.llm_s).toFixed(2)}s`);
    if (p.single_model) parts.push(t("badge.single"));
    renderMeta(parts.join(" · "));
    $("#meta-llm").textContent = p.llm_text
      ? `LLM ${Number(p.llm_s || 0).toFixed(1)}s`
      : p.single_model
        ? t("badge.single")
        : "";

    if (state.cfg.save_history) {
      state.history.unshift({
        asr_text: p.asr_text,
        llm_text: p.llm_text || null,
        time: p.time || "",
      });
      if (state.history.length > 500) state.history.length = 500;
      renderHistory();
    }
  });
  await listen("asr-error", (e) => {
    renderMeta("");
    toast(e.payload.error || "ASR error", "err", 4200);
  });
  await listen("notice", (e) => toast(String(e.payload), "", 3200));
  await listen("hotkey", (e) => {
    const p = e.payload || {};
    const badge = $("#hotkey-badge");
    if (p.ok) {
      badge.classList.remove("err");
      badge.title = p.backend || "";
      if (p.display) state.hotkeyDisplay = p.display;
      else state.hotkeyDisplay = "";
      renderHotkeyBadge();
    } else {
      badge.classList.add("err");
      if (p.error) toast(`${t("banner.hotkey")}: ${p.error}`, "err", 5000);
    }
  });
}

/* ─────────────────────────── 启动 ─────────────────────────── */

async function loadLocalesIndex() {
  try {
    const r = await fetch("locales/index.json", { cache: "no-store" });
    if (r.ok) {
      state.locales = await r.json();
    } else {
      throw new Error(String(r.status));
    }
  } catch {
    state.locales = [{ code: "en", name: "English" }, { code: "zh", name: "中文" }];
  }
  state.locales.sort((a, b) => a.name.localeCompare(b.name));
}

async function main() {
  bindRecordButton();
  await loadLocalesIndex();

  const boot = await invoke("bootstrap");
  state.cfg = boot.config || {};
  state.status = boot.status || {};
  state.history = boot.history || [];
  state.version = boot.version || "";
  state.modelHint = boot.model_hint || "";
  state.platform = boot.platform || "";
  state.hotkeySupported = !!boot.hotkey_supported;

  const lang = state.cfg.language || "en";
  try {
    await loadLocale(lang);
  } catch (e) {
    console.error(e);
    await loadLocale("en").catch(() => {});
  }

  try {
    state.presets = await invoke("list_presets");
  } catch {
    state.presets = [];
  }
  try {
    state.presetsAsr = await invoke("list_asr_presets");
  } catch {
    state.presetsAsr = [];
  }
  try {
    state.custom = await invoke("get_custom_prompts");
  } catch {
    state.custom = {};
  }
  try {
    state.micList = await invoke("list_mics");
  } catch {
    state.micList = [];
  }

  applyStaticText();
  renderResult();
  renderHistory();
  bindSettings();
  renderHotkeyBadge();

  if (!state.hotkeySupported) {
    $("#hotkey-badge").hidden = true;
  }

  await bindEvents();
  renderStatus(state.status);

  // 抽屉打开时才拉一次历史（bootstrap 已经带了一份）
  if (!state.history.length) {
    invoke("get_history").then((h) => {
      state.history = h || [];
      renderHistory();
    });
  }
}

main().catch((e) => {
  console.error(e);
  const el = document.getElementById("loading-text");
  if (el) el.textContent = String(e);
});
