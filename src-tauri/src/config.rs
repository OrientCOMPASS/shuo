// 说 · Shuo — 配置与历史（与 Python 版共用 `~/.shuo/`，字段兼容）
//
// SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub fn user_dir() -> PathBuf {
  let home = std::env::var_os("USERPROFILE")
    .or_else(|| std::env::var_os("HOME"))
    .map(PathBuf::from)
    .unwrap_or_else(|| PathBuf::from("."));
  let dir = home.join(".shuo");
  let _ = std::fs::create_dir_all(&dir);
  dir
}

pub fn config_path() -> PathBuf {
  user_dir().join("config.json")
}

pub fn history_path() -> PathBuf {
  user_dir().join("history.json")
}

pub fn log_path() -> PathBuf {
  user_dir().join("shuo.log")
}

pub fn default_prompt() -> String {
  "请优化以下文本：1. 修正标点符号；2. 去除口语中的口癖和犹豫词（如嗯、啊、呃、那个、就是说、然后、对吧、你知道吗、反正、就是、大概、好像等）；3. 保持原意和原始表达方式。只输出修正后的文本。".to_string()
}

fn d_hotkey() -> String {
  "f2".into()
}
fn d_language() -> String {
  "en".into()
}
fn d_asr_lang() -> String {
  "auto".into()
}
fn d_preset() -> String {
  crate::llm::PRESET_STANDARD.into()
}
fn d_hotkey_terminal() -> String {
  "f3".into()
}
fn d_true() -> bool {
  true
}
fn d_asr_backend() -> String {
  "auto".into()
}
fn d_single_preset() -> String {
  crate::asr::UPRESET_POLISH_ZH.into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
  pub hotkey: String,
  /// 界面语言：`zh` / `en`（其余 27 种语言在设置里可选，见 ui/locales）
  pub language: String,
  /// ASR 识别语言（auto / zh / en / ...，传给模型）
  pub asr_lang: String,
  pub denoise: bool,
  pub auto_type: bool,
  pub remove_punc: bool,
  pub save_history: bool,
  /// 麦克风设备名（None = 系统默认）
  pub mic_device: Option<String>,
  pub llm_enabled: bool,
  /// LLM 提示词预设 key；`custom` 时使用 `llm_system_prompt`
  pub llm_preset: String,
  pub llm_system_prompt: String,
  /// 单模型直出：微调后的 ASR 模型按 system 段提示词直接输出优化文本，跳过 LLM 后处理
  #[serde(default)]
  pub single_model_enabled: bool,
  /// 单模型直出提示词预设 key（见 `asr::unified_preset_list`）；`custom` 时用 `single_model_prompt`
  #[serde(default = "d_single_preset")]
  pub single_model_preset: String,
  /// 自定义单模型提示词（进 chat 模板 system 段；效果取决于微调数据是否覆盖）
  #[serde(default = "crate::asr::default_unified_prompt")]
  pub single_model_prompt: String,
  /// 启动后直接收到托盘（默认否）
  pub start_hidden: bool,
  /// 模型目录覆盖（空 = 自动查找）
  pub asr_model_dir: Option<String>,
  pub llm_model_dir: Option<String>,
  /// ASR 推理后端："auto"（有 GGUF 模型优先 GGUF）/ "onnx" / "gguf"
  #[serde(default = "d_asr_backend")]
  pub asr_backend: String,
  /// 主热键开关（关掉后这个键不再触发录音）
  #[serde(default = "d_true")]
  pub hotkey_enabled: bool,
  /// 终端热键开关
  #[serde(default = "d_true")]
  pub hotkey_terminal_enabled: bool,
  /// 终端热键：用这个键录音，识别后用 Ctrl+Shift+V 上屏（主热键用 Ctrl+V）
  #[serde(default = "d_hotkey_terminal")]
  pub hotkey_terminal: String,
  /// 保留 Python 版写入的其它字段，避免来回切换时丢配置
  #[serde(flatten)]
  pub extra: BTreeMap<String, serde_json::Value>,
}

impl Default for Config {
  fn default() -> Self {
    Self {
      hotkey: d_hotkey(),
      language: d_language(),
      asr_lang: d_asr_lang(),
      denoise: false,
      auto_type: true,
      remove_punc: false,
      save_history: true,
      mic_device: None,
      llm_enabled: false,
      llm_preset: d_preset(),
      llm_system_prompt: default_prompt(),
      single_model_enabled: false,
      single_model_preset: d_single_preset(),
      single_model_prompt: crate::asr::default_unified_prompt(),
      hotkey_enabled: true,
      hotkey_terminal_enabled: true,
      hotkey_terminal: d_hotkey_terminal(),
      start_hidden: false,
      asr_model_dir: None,
      llm_model_dir: None,
      asr_backend: d_asr_backend(),
      extra: BTreeMap::new(),
    }
  }
}

pub fn load_config() -> Config {
  match std::fs::read_to_string(config_path()) {
    Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
      crate::log_line(&format!("配置解析失败（使用默认值）：{e}"));
      Config::default()
    }),
    Err(_) => Config::default(),
  }
}

impl Config {
  /// 单模型直出时进 chat 模板 system 段的提示词。
  /// 预设 key 非法时回退 `polish_zh`；`custom` 用用户自定义文本（可为空 = 逐字转写）。
  pub fn single_model_context(&self) -> String {
    if self.single_model_preset == crate::asr::UPRESET_CUSTOM {
      self.single_model_prompt.trim().to_string()
    } else {
      crate::asr::unified_preset_prompt(&self.single_model_preset)
        .or_else(|| crate::asr::unified_preset_prompt(crate::asr::UPRESET_POLISH_ZH))
        .unwrap_or("")
        .to_string()
    }
  }
}

pub fn save_config(cfg: &Config) -> Result<(), String> {
  let s = serde_json::to_string_pretty(cfg).map_err(|e| e.to_string())?;
  std::fs::write(config_path(), s).map_err(|e| format!("保存配置失败：{e}"))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryItem {
  #[serde(default)]
  pub asr_text: String,
  #[serde(default)]
  pub llm_text: Option<String>,
  #[serde(default)]
  pub time: String,
}

pub fn load_history() -> Vec<HistoryItem> {
  std::fs::read_to_string(history_path())
    .ok()
    .and_then(|s| serde_json::from_str(&s).ok())
    .unwrap_or_default()
}

pub fn save_history(items: &[HistoryItem]) -> Result<(), String> {
  let s = serde_json::to_string_pretty(items).map_err(|e| e.to_string())?;
  std::fs::write(history_path(), s).map_err(|e| format!("保存历史失败：{e}"))
}

/// 追加一条历史（最多 500 条，最新的在末尾；前端倒序展示）。
pub fn add_history(asr_text: &str, llm_text: Option<&str>) {
  let now = chrono_time();
  let mut items = load_history();
  items.push(HistoryItem {
    asr_text: asr_text.to_string(),
    llm_text: llm_text.map(|s| s.to_string()),
    time: now,
  });
  if items.len() > 500 {
    let cut = items.len() - 500;
    items.drain(..cut);
  }
  if let Err(e) = save_history(&items) {
    crate::log_line(&e);
  }
}

/// `YYYY-MM-DD HH:MM:SS`（本地时区），不引入 chrono：
/// 通过 `date` 命令在 Linux 取时区偏移；Windows 下用系统时间近似 UTC+8 之外的场景见 README。
pub fn chrono_time() -> String {
  use std::time::{SystemTime, UNIX_EPOCH};
  let secs = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .map(|d| d.as_secs() as i64)
    .unwrap_or(0);
  // 本地时区偏移（分钟）：从 `date +%z` 解析，失败则按 +08:00
  let offset_min = local_offset_minutes();
  let t = secs + offset_min * 60;
  let days = t.div_euclid(86_400);
  let sod = t.rem_euclid(86_400);
  let (y, m, d) = civil_from_days(days);
  format!(
    "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
    y,
    m,
    d,
    sod / 3600,
    (sod % 3600) / 60,
    sod % 60
  )
}

fn local_offset_minutes() -> i64 {
  #[cfg(unix)]
  {
    if let Ok(out) = std::process::Command::new("date").arg("+%z").output()
      && out.status.success()
    {
      let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
      if s.len() == 5 {
        let sign = if s.starts_with('-') { -1 } else { 1 };
        let h: i64 = s[1..3].parse().unwrap_or(8);
        let m: i64 = s[3..5].parse().unwrap_or(0);
        return sign * (h * 60 + m);
      }
    }
  }
  // Windows 及其它平台：不引入额外时区依赖，固定 +08:00（本应用主要用户群）。
  // 时间戳只用于展示，不影响功能。
  8 * 60
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
  let z = z + 719_468;
  let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
  let doe = z - era * 146_097;
  let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
  let y = yoe + era * 400;
  let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  let mp = (5 * doy + 2) / 153;
  let d = doy - (153 * mp + 2) / 5 + 1;
  let m = if mp < 10 { mp + 3 } else { mp - 9 };
  (if m <= 2 { y + 1 } else { y }, m, d)
}
