// 说 · Shuo — 全局热键（按住说话：按下 / 松开都要）
//
// 平台实现：
//   Windows — 低级键鼠钩子（WH_KEYBOARD_LL / WH_MOUSE_LL），支持鼠标侧键，
//             和 Python 版 pynput 的行为一致（按下开始、松开结束）。
//   Linux   — xdg-desktop-portal 的 GlobalShortcuts（Wayland 唯一正规全局热键），
//             portal 不可用时用 evdev 直接读 /dev/input/event*（X11 / 无桌面也能用）。
//   其它    — 明确报告不可用。
//
// 键位串与 Python 版保持一致：`f2`、`ctrl+shift+a`、`xbutton1`（鼠标侧键）、
// 也接受 `win+space` / `super+space`。
//
// SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::state::AppState;

static RECORDING: AtomicBool = AtomicBool::new(false);

pub fn set_recording(on: bool) {
  RECORDING.store(on, Ordering::Relaxed);
}

pub fn is_recording() -> bool {
  RECORDING.load(Ordering::Relaxed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trigger {
  /// 键盘主键（已规范化的 token，如 `f2` / `a` / `space`）
  Key,
  /// 鼠标侧键：1 = xbutton1（后退），2 = xbutton2（前进）
  Mouse(u8),
}

#[derive(Debug, Clone)]
pub struct Parsed {
  /// 主键 token（小写；鼠标时为空串）
  pub token: String,
  pub mouse: Option<u8>,
  pub ctrl: bool,
  pub shift: bool,
  pub alt: bool,
  pub meta: bool,
  /// 原始键位串（保存到配置的就是它，保持与 Python 版兼容）
  pub raw: String,
}

impl Parsed {
  pub fn trigger(&self) -> Trigger {
    match self.mouse {
      Some(btn) => Trigger::Mouse(btn),
      None => Trigger::Key,
    }
  }
}

const NAMED: &[(&str, &str)] = &[
  ("esc", "esc"),
  ("escape", "esc"),
  ("tab", "tab"),
  ("enter", "enter"),
  ("return", "enter"),
  ("backspace", "backspace"),
  ("space", "space"),
  ("insert", "insert"),
  ("ins", "insert"),
  ("delete", "delete"),
  ("del", "delete"),
  ("home", "home"),
  ("end", "end"),
  ("pageup", "pageup"),
  ("pgup", "pageup"),
  ("pagedown", "pagedown"),
  ("pgdn", "pagedown"),
  ("up", "up"),
  ("down", "down"),
  ("left", "left"),
  ("right", "right"),
  ("pause", "pause"),
  ("capslock", "capslock"),
  ("numlock", "numlock"),
  ("scrolllock", "scrolllock"),
  ("printscreen", "printscreen"),
  ("`", "`"),
  ("grave", "`"),
  ("-", "-"),
  ("=", "="),
  ("[", "["),
  ("]", "]"),
  ("\\", "\\"),
  (";", ";"),
  ("'", "'"),
  (",", ","),
  (".", "."),
  ("/", "/"),
];

/// 解析热键串（与 Python 版一致：`f2`、`ctrl+shift+a`、`xbutton1`）。
pub fn parse_spec(spec: &str) -> Option<Parsed> {
  let raw = spec.trim().to_string();
  let low = raw.to_lowercase();
  if low.is_empty() {
    return None;
  }
  let parts: Vec<&str> = low.split('+').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
  let key = parts.last()?.to_string();
  let mut ctrl = false;
  let mut shift = false;
  let mut alt = false;
  let mut meta = false;
  for m in &parts[..parts.len() - 1] {
    match *m {
      "ctrl" | "control" => ctrl = true,
      "shift" => shift = true,
      "alt" | "option" => alt = true,
      "win" | "super" | "meta" | "cmd" | "command" => meta = true,
      _ => {}
    }
  }

  let mut mouse = None;
  let token = match key.as_str() {
    "xbutton1" | "mouse4" | "back" => {
      mouse = Some(1);
      String::new()
    }
    "xbutton2" | "mouse5" | "forward" => {
      mouse = Some(2);
      String::new()
    }
    other => normalize_token(other)?,
  };

  // 无修饰键时只允许功能键 / 鼠标键（避免普通字母把输入吃成录音）
  let is_fn = token.starts_with('f')
    && token[1..].parse::<u32>().map(|i| (1..=24).contains(&i)).unwrap_or(false);
  if mouse.is_none() && !is_fn && !(ctrl || shift || alt || meta) {
    return None;
  }

  // 鼠标键 + 修饰键的组合在 portal 里不支持，Windows / evdev 可以；保持宽松。
  Some(Parsed { token, mouse, ctrl, shift, alt, meta, raw })
}

fn normalize_token(t: &str) -> Option<String> {
  let s = t.trim();
  if let Some(n) = s.strip_prefix('f')
    && let Ok(i) = n.parse::<u32>()
  {
    return (1..=24).contains(&i).then(|| format!("f{i}"));
  }
  if s.len() == 1 {
    let c = s.chars().next()?;
    if c.is_ascii_alphanumeric() {
      return Some(c.to_ascii_lowercase().to_string());
    }
  }
  NAMED.iter().find(|(alias, _)| *alias == s).map(|(_, canon)| (*canon).to_string())
}

/// 规范化显示（`Ctrl+Shift+A` / `F2` / `XButton1`）。
pub fn normalize_spec(spec: &str) -> String {
  let Some(p) = parse_spec(spec) else {
    return spec.trim().to_string();
  };
  let mut parts = Vec::new();
  if p.ctrl {
    parts.push("Ctrl".to_string());
  }
  if p.alt {
    parts.push("Alt".to_string());
  }
  if p.shift {
    parts.push("Shift".to_string());
  }
  if p.meta {
    parts.push(if cfg!(windows) { "Win" } else { "Super" }.to_string());
  }
  match p.mouse {
    Some(1) => parts.push("XButton1".into()),
    Some(2) => parts.push("XButton2".into()),
    _ => {
      let t = &p.token;
      let disp = if t.len() == 1 {
        t.to_ascii_uppercase()
      } else if t.starts_with('f') && t[1..].chars().all(|c| c.is_ascii_digit()) {
        t.to_ascii_uppercase()
      } else {
        match t.as_str() {
          "esc" => "Esc".into(),
          "tab" => "Tab".into(),
          "enter" => "Enter".into(),
          "backspace" => "Backspace".into(),
          "space" => "Space".into(),
          "insert" => "Insert".into(),
          "delete" => "Delete".into(),
          "home" => "Home".into(),
          "end" => "End".into(),
          "pageup" => "PageUp".into(),
          "pagedown" => "PageDown".into(),
          "up" => "Up".into(),
          "down" => "Down".into(),
          "left" => "Left".into(),
          "right" => "Right".into(),
          other => other.to_string(),
        }
      };
      parts.push(disp)
    }
  }
  parts.join("+")
}

pub struct Hotkey {
  pub spec: String,
  stop: Arc<AtomicBool>,
  join: Option<std::thread::JoinHandle<()>>,
  #[cfg(windows)]
  tid: Arc<std::sync::atomic::AtomicU32>,
}

impl Hotkey {
  pub fn stop(mut self) {
    self.stop.store(true, Ordering::Relaxed);
    #[cfg(windows)]
    {
      // 热键线程阻塞在 GetMessageW，需要投递 WM_QUIT 才能退出
      let tid = self.tid.load(Ordering::Relaxed);
      if tid != 0 {
        hotkey_windows::wake(tid);
      }
    }
    if let Some(j) = self.join.take() {
      let _ = j.join();
    }
  }
}

#[cfg(windows)]
#[path = "hotkey_windows.rs"]
mod hotkey_windows;
#[cfg(target_os = "linux")]
#[path = "hotkey_linux.rs"]
mod hotkey_linux;

/// 一条热键绑定：键位串 + 该键触发的录音用哪种上屏方式。
pub fn start(
  state: Arc<AppState>,
  list: Vec<(String, crate::typing::PasteMode)>,
) -> Hotkey {
  let stop = Arc::new(AtomicBool::new(false));
  if list.is_empty() {
    // 两个热键都被用户关掉了：正常状态，不报错
    return Hotkey {
      spec: String::new(),
      stop,
      join: None,
      #[cfg(windows)]
      tid: Arc::new(std::sync::atomic::AtomicU32::new(0)),
    };
  }
  let mut parsed: Vec<(Parsed, crate::typing::PasteMode)> = Vec::new();
  let mut seen: Vec<String> = Vec::new();
  for (spec, mode) in list {
    let spec = spec.trim().to_string();
    if spec.is_empty() {
      continue;
    }
    let Some(p) = parse_spec(&spec) else {
      crate::log_line(&format!("热键格式无效，已忽略：{spec}"));
      continue;
    };
    let key = normalize_spec(&spec).to_lowercase();
    if seen.contains(&key) {
      crate::log_line(&format!("热键重复，已忽略：{spec}"));
      continue;
    }
    seen.push(key);
    parsed.push((p, mode));
  }
  if parsed.is_empty() {
    crate::state::emit_hotkey(&state, false, "none", "", Some("没有可用的全局热键"));
    return Hotkey {
      spec: String::new(),
      stop,
      join: None,
      #[cfg(windows)]
      tid: Arc::new(std::sync::atomic::AtomicU32::new(0)),
    };
  }
  let display = parsed
    .iter()
    .map(|(p, _)| normalize_spec(&p.raw))
    .collect::<Vec<_>>()
    .join(" / ");
  let spec = parsed
    .iter()
    .map(|(p, _)| p.raw.clone())
    .collect::<Vec<_>>()
    .join("+");

  #[cfg(windows)]
  {
    let tid = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let join =
      hotkey_windows::spawn(parsed, state.clone(), stop.clone(), tid.clone(), display.clone());
    if join.is_some() {
      crate::log_line(&format!("热键已注册：{display}"));
      crate::state::emit_hotkey(&state, true, "win-hook", &display, None);
    } else {
      crate::log_line("热键注册失败：安装键鼠钩子失败");
      crate::state::emit_hotkey(&state, false, "win-hook", &display, Some("安装键鼠钩子失败"));
    }
    return Hotkey { spec, stop, join, tid };
  }

  #[cfg(target_os = "linux")]
  {
    let join = hotkey_linux::spawn(parsed, state.clone(), stop.clone(), display);
    return Hotkey { spec, stop, join };
  }

  #[cfg(all(not(windows), not(target_os = "linux")))]
  {
    let _ = (parsed, display);
    crate::state::emit_hotkey(&state, false, "none", "", Some("当前平台不支持全局热键"));
    Hotkey { spec: String::new(), stop, join: None }
  }
}
