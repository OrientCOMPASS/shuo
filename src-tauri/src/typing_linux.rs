// 说 · Shuo — Linux 自动输入（上屏）
//
// 为什么不是逐字符 keysym 注入：GNOME 的 RemoteDesktop 门户把 keysym 交给合成器后，
// **当前输入法（libpinyin）会把它当键盘输入处理** —— 实测注入 `abc123` 上屏成
// 「阿拔出23」。中文更糟（keysym 被拼音候选吃掉）。
//
// 可靠通路（按顺序）：
//   1. 剪贴板 + 粘贴快捷键（Shift+Insert，GTK/Qt/终端/Chromium 都认）：
//      文本不经过输入法，中文/英文/符号都能原样上屏；先把原剪贴板存起来，粘完再还原。
//   2. RemoteDesktop 门户（只用来发快捷键，不发正文）——首次授权一次。
//   3. wtype / ydotool / xdotool 发同样的快捷键（X11 / wlroots / uinput 环境）。
//   4. 都失败：只把文本放进剪贴板，并明确告诉用户手动粘贴。
//
// 注意：真正的「输入法通路」（IBus 引擎 commit_text）可以彻底避开模拟按键，但需要
// 用户切到「Shuo Voice」输入法（或应用临时切换引擎再切回），代价见
// docs/scaffold-comparison.md 的讨论；默认不启用。
//
// SPDX-License-Identifier: GPL-3.0-or-later

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Duration;

use ashpd::desktop::{PersistMode, Session};
use ashpd::desktop::remote_desktop::{
  DeviceType, KeyState, RemoteDesktop, SelectDevicesOptions, StartOptions,
};
use enumflags2::BitFlags;

use crate::typing::PasteMode;

#[path = "uinput.rs"]
mod uinput;

struct PortalText {
  rd: RemoteDesktop,
  session: Session<RemoteDesktop>,
}

static PORTAL: OnceLock<Mutex<Option<PortalText>>> = OnceLock::new();
/// 授权失败后的冷却时间，避免每次上屏都弹框；30s 后允许再次申请。
static NEXT_PORTAL_TRY: Mutex<Option<std::time::Instant>> = Mutex::new(None);

// evdev keycode（门户的 NotifyKeyboardKeycode 用这个，能正确带修饰键；
// keysym 只适合字符，带 Ctrl/Shift 的组合会被输入法/合成器吃掉）
const KEY_LEFTCTRL: i32 = 29;
const KEY_LEFTSHIFT: i32 = 42;
const KEY_V: i32 = 47;
const KEY_INSERT: i32 = 110;

fn has(cmd: &str) -> bool {
  Command::new("sh")
    .args(["-c", &format!("command -v {cmd} >/dev/null 2>&1")])
    .status()
    .map(|s| s.success())
    .unwrap_or(false)
}

fn run_with_stdin(cmd: &str, args: &[&str], text: &str) -> Result<(), String> {
  let mut child = Command::new(cmd)
    .args(args)
    .stdin(Stdio::piped())
    .spawn()
    .map_err(|e| format!("启动 {cmd} 失败：{e}"))?;
  if let Some(stdin) = child.stdin.as_mut() {
    stdin.write_all(text.as_bytes()).map_err(|e| e.to_string())?;
  }
  let status = child.wait().map_err(|e| e.to_string())?;
  if status.success() {
    Ok(())
  } else {
    Err(format!("{cmd} 退出码 {:?}", status.code()))
  }
}

pub fn type_text(text: &str, mode: PasteMode) -> Result<(), String> {
  if text.is_empty() {
    return Ok(());
  }

  // 1. 剪贴板 + 粘贴快捷键
  match clipboard_paste(text, mode) {
    Ok(()) => return Ok(()),
    Err(e) => crate::log_line(&format!("剪贴板上屏失败：{e}")),
  }

  // 2. 外部工具直接打字（ASCII 环境 / 没有门户时）
  if has("wtype") && run_with_stdin("wtype", &["-"], text).is_ok() {
    return Ok(());
  }
  if has("ydotool") && run_with_stdin("ydotool", &["type", "--file", "-"], text).is_ok() {
    return Ok(());
  }
  if has("xdotool")
    && run_with_stdin("xdotool", &["type", "--clearmodifiers", "--file", "-"], text).is_ok()
  {
    return Ok(());
  }

  // 3. 只放剪贴板
  copy_text(text)?;
  Err("没有可用的上屏通道，已把文本复制到剪贴板，请按 Ctrl+V / Shift+Insert 粘贴".into())
}

fn clipboard_paste(text: &str, mode: PasteMode) -> Result<(), String> {
  let previous = read_clipboard();
  copy_text(text)?;

  let sent = send_paste_shortcut(mode);
  match sent {
    Ok(()) => {
      // 目标应用可能异步读剪贴板，延迟后再还原
      if let Some(prev) = previous.clone() {
        std::thread::spawn(move || {
          std::thread::sleep(Duration::from_millis(800));
          let _ = write_clipboard(&prev);
        });
      }
      Ok(())
    }
    Err(e) => {
      if let Some(prev) = previous {
        let _ = write_clipboard(&prev);
      }
      Err(e)
    }
  }
}

fn uinput_paste(mode: PasteMode) -> Result<(), String> {
  use std::sync::Mutex;
  use std::sync::OnceLock;
  static VK: OnceLock<Mutex<Option<uinput::VirtualKeyboard>>> = OnceLock::new();
  let cell = VK.get_or_init(|| Mutex::new(None));
  let mut guard = cell.lock().map_err(|_| "uinput 状态锁损坏".to_string())?;
  if guard.is_none() {
    *guard = Some(uinput::VirtualKeyboard::open()?);
    crate::log_line("uinput 虚拟键盘已就绪");
  }
  let codes: Vec<u16> = paste_keys(mode).iter().map(|c| *c as u16).collect();
  guard.as_ref().unwrap().tap_combo(&codes)
}

/// 发 Ctrl+V（GUI 应用通用；终端请用 Ctrl+Shift+V，见文档）。
fn send_paste_shortcut(mode: PasteMode) -> Result<(), String> {
  // 首选 uinput：不需要门户授权，也不会每次启动弹「共享」
  if uinput_paste(mode).is_ok() {
    return Ok(());
  }

  let portal = ensure_portal();
  if portal.is_ok() {
    match portal_paste(mode) {
      Ok(()) => return Ok(()),
      Err(e) => crate::log_line(&format!("门户粘贴快捷键失败：{e}")),
    }
  }
  // X11 / wlroots / uinput
  let keys = paste_keys(mode);
  if has("xdotool")
    && Command::new("xdotool")
      .args(["key", "--clearmodifiers", paste_combo_name(&keys)])
      .status()
      .map(|s| s.success())
      .unwrap_or(false)
  {
    return Ok(());
  }
  if has("ydotool") {
    let seq: Vec<String> = keys
      .iter()
      .map(|c| format!("{c}:1"))
      .chain(keys.iter().rev().map(|c| format!("{c}:0")))
      .collect();
    let mut args = vec!["key"];
    args.extend(seq.iter().map(String::as_str));
    if Command::new("ydotool")
      .args(&args)
      .status()
      .map(|s| s.success())
      .unwrap_or(false)
    {
      return Ok(());
    }
  }
  portal.map_err(|e| format!("没有可用的粘贴快捷键通道：{e}"))
}

fn ensure_portal() -> Result<(), String> {
  let cell = PORTAL.get_or_init(|| Mutex::new(None));
  let mut guard = cell.lock().map_err(|_| "门户状态锁已损坏".to_string())?;
  if guard.is_some() {
    return Ok(());
  }
  {
    let next = NEXT_PORTAL_TRY.lock().unwrap();
    if let Some(t) = *next
      && std::time::Instant::now() < t
    {
      return Err("门户授权冷却中（稍后再试）".into());
    }
  }
  let token = std::fs::read_to_string(token_path())
    .ok()
    .map(|s| s.trim().to_string())
    .filter(|s| !s.is_empty());
  // 先用记住的 token；如果 token 被撤回/失效，清掉并立刻重新申请（GNOME 会再弹一次共享框）
  let first = futures_lite::future::block_on(connect_portal(token.as_deref()));
  let result = match first {
    Ok(p) => Ok(p),
    Err(e) if token.is_some() => {
      crate::log_line(&format!("门户授权已失效（{e}），重新申请"));
      let _ = std::fs::remove_file(token_path());
      futures_lite::future::block_on(connect_portal(None))
    }
    Err(e) => Err(e),
  };
  match result {
    Ok(p) => {
      *guard = Some(p);
      *NEXT_PORTAL_TRY.lock().unwrap() = None;
      Ok(())
    }
    Err(e) => {
      *NEXT_PORTAL_TRY.lock().unwrap() =
        Some(std::time::Instant::now() + std::time::Duration::from_secs(30));
      Err(e)
    }
  }
}

/// 上屏快捷键：主热键 → Ctrl+V（图形界面）；终端热键 → Ctrl+Shift+V（终端/TUI）。
fn paste_keys(mode: PasteMode) -> Vec<i32> {
  match mode {
    PasteMode::CtrlV => vec![KEY_LEFTCTRL, KEY_V],
    PasteMode::CtrlShiftV => vec![KEY_LEFTCTRL, KEY_LEFTSHIFT, KEY_V],
  }
}

fn paste_combo_name(keys: &[i32]) -> &'static str {
  match keys {
    [KEY_LEFTCTRL, KEY_V] => "ctrl+v",
    [KEY_LEFTSHIFT, KEY_INSERT] => "shift+Insert",
    _ => "ctrl+shift+v",
  }
}

fn portal_paste(mode: PasteMode) -> Result<(), String> {
  let keys = paste_keys(mode);
  let cell = PORTAL.get_or_init(|| Mutex::new(None));
  let mut guard = cell.lock().map_err(|_| "门户状态锁已损坏".to_string())?;
  let result = {
    let Some(p) = guard.as_ref() else {
      return Err("门户未连接".into());
    };
    futures_lite::future::block_on(async {
      for &code in &keys {
        p.rd
          .notify_keyboard_keycode(&p.session, code, KeyState::Pressed, Default::default())
          .await
          .map_err(|e| format!("{e}"))?;
        async_io::Timer::after(Duration::from_millis(12)).await;
      }
      for &code in keys.iter().rev() {
        p.rd
          .notify_keyboard_keycode(&p.session, code, KeyState::Released, Default::default())
          .await
          .map_err(|e| format!("{e}"))?;
        async_io::Timer::after(Duration::from_millis(12)).await;
      }
      Ok(())
    })
  };
  if result.is_err() {
    // 会话被撤回/失效：丢掉连接和 token，下次上屏会重新弹授权
    *guard = None;
    let _ = std::fs::remove_file(token_path());
  }
  result
}

fn token_path() -> std::path::PathBuf {
  crate::config::user_dir().join("portal_restore_token")
}

async fn connect_portal(token: Option<&str>) -> Result<PortalText, String> {
  let rd = RemoteDesktop::new().await.map_err(|e| format!("{e}"))?;
  let session = rd
    .create_session(Default::default())
    .await
    .map_err(|e| format!("创建会话失败：{e}"))?;
  // 持久化授权：把上次的 restore token 带上，GNOME 就不再每次弹「共享」对话框
  rd.select_devices(
    &session,
    SelectDevicesOptions::default()
      .set_devices(BitFlags::from_flag(DeviceType::Keyboard))
      .set_persist_mode(PersistMode::ExplicitlyRevoked)
      .set_restore_token(token),
  )
  .await
  .map_err(|e| format!("选择键盘设备失败：{e}"))?;
  let start = rd
    .start(&session, None, StartOptions::default())
    .await
    .map_err(|e| format!("请求授权失败（{e}）"))?;
  let devices = start.response().map_err(|e| format!("授权结果无效：{e}"))?;
  if !devices.devices().contains(DeviceType::Keyboard) {
    return Err("用户未授权键盘控制".into());
  }
  if let Some(t) = devices.restore_token() {
    let _ = std::fs::write(token_path(), t);
    crate::log_line("RemoteDesktop 授权已记住（restore token）");
  }
  crate::log_line("RemoteDesktop 键盘注入已授权（仅用于粘贴快捷键）");
  Ok(PortalText { rd, session })
}

fn read_clipboard() -> Option<String> {
  if has("wl-paste") {
    let out = Command::new("wl-paste").args(["--no-newline", "--type", "text"]).output().ok()?;
    if out.status.success() {
      return Some(String::from_utf8_lossy(&out.stdout).to_string());
    }
  }
  if has("xclip") {
    let out = Command::new("xclip")
      .args(["-selection", "clipboard", "-o"])
      .output()
      .ok()?;
    if out.status.success() {
      return Some(String::from_utf8_lossy(&out.stdout).to_string());
    }
  }
  None
}

fn write_clipboard(text: &str) -> Result<(), String> {
  copy_text(text)
}

pub fn copy_text(text: &str) -> Result<(), String> {
  if has("wl-copy") {
    // 同时写 CLIPBOARD 和 PRIMARY：Shift+Insert 在部分应用里读 PRIMARY
    let ok1 = run_with_stdin("wl-copy", &["--type", "text/plain"], text).is_ok();
    let _ = run_with_stdin("wl-copy", &["--primary", "--type", "text/plain"], text);
    if ok1 {
      return Ok(());
    }
  }
  if has("xclip") {
    return run_with_stdin("xclip", &["-selection", "clipboard"], text);
  }
  Err("没有可用的剪贴板工具（安装 wl-clipboard 或 xclip）".into())
}
