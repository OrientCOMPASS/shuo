// 说 · Shuo — Linux 全局热键（支持多条绑定）
//
// 两条通路：
// 1. xdg-desktop-portal 的 GlobalShortcuts（Wayland 官方通道，按下/松开都有信号）。
//    多条绑定放在**同一个会话**里，一次授权即可。
// 2. evdev（/dev/input/event*）—— portal 不可用、或绑定里含鼠标侧键时使用。
//
// 每条绑定带 PasteMode：主热键 Ctrl+V 上屏，终端热键 Ctrl+Shift+V 上屏。
//
// SPDX-License-Identifier: GPL-3.0-or-later

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::state::AppState;
use crate::typing::PasteMode;

use super::Parsed;

// evdev 常量（与 linux/input-event-codes.h 一致）
const EV_KEY: u16 = 0x01;
const BTN_LEFT: u32 = 0x110;
const BTN_SIDE: u32 = 0x113;
const BTN_EXTRA: u32 = 0x114;
const KEY_A: u32 = 30;
const KEY_SPACE: u32 = 57;

const MOD_CTRL: u32 = 1;
const MOD_SHIFT: u32 = 2;
const MOD_ALT: u32 = 4;
const MOD_META: u32 = 8;

pub(super) fn spawn(
  bindings: Vec<(Parsed, PasteMode)>,
  state: Arc<AppState>,
  stop: Arc<AtomicBool>,
  display: String,
) -> Option<JoinHandle<()>> {
  std::thread::Builder::new()
    .name("shuo-hotkey".into())
    .spawn(move || {
      let has_mouse = bindings.iter().any(|(p, _)| p.mouse.is_some());
      if !has_mouse {
        match futures_lite::future::block_on(portal_loop(&bindings, &state, &stop, &display)) {
          Ok(()) => {
            crate::log_line("热键线程退出（门户）");
            return;
          }
          Err(reason) => {
            crate::log_line(&format!("GlobalShortcuts 不可用：{reason}；改用 evdev"));
          }
        }
      }
      if !evdev_loop(&bindings, &state, &stop, &display) {
        crate::state::emit_hotkey(
          &state,
          false,
          "evdev",
          &display,
          Some("没有可读取的 /dev/input/event*（需要 input 组或 udev 规则）"),
        );
      }
      crate::log_line("热键线程退出（evdev）");
    })
    .ok()
}

// ─────────────────────────── portal ───────────────────────────

async fn portal_loop(
  bindings: &[(Parsed, PasteMode)],
  state: &Arc<AppState>,
  stop: &AtomicBool,
  display: &str,
) -> Result<(), String> {
  use ashpd::desktop::global_shortcuts::{GlobalShortcuts, NewShortcut};
  use futures_lite::StreamExt;

  let gp = GlobalShortcuts::new().await.map_err(|e| format!("{e}"))?;
  let session = gp
    .create_session(Default::default())
    .await
    .map_err(|e| format!("创建会话失败：{e}"))?;

  let mut shortcuts = Vec::new();
  let mut by_id: HashMap<String, PasteMode> = HashMap::new();
  for (i, (spec, mode)) in bindings.iter().enumerate() {
    let id = format!("ptt-{i}");
    by_id.insert(id.clone(), *mode);
    let preferred = portal_trigger(spec);
    shortcuts.push(
      NewShortcut::new(id, "说 · 按住说话")
        .preferred_trigger(preferred.as_deref()),
    );
  }

  let request = match with_stop(
    gp.bind_shortcuts(&session, &shortcuts, None, Default::default()),
    stop,
  )
  .await
  {
    Some(Ok(req)) => req,
    Some(Err(e)) => return Err(format!("绑定失败：{e}")),
    None => return Err("用户取消或绑定超时".into()),
  };

  let bound = request.response().map_err(|e| format!("读取绑定结果失败：{e}"))?;
  for s in bound.shortcuts() {
    crate::log_line(&format!("热键已绑定（门户）：{} → {}", s.id(), s.trigger_description()));
  }
  crate::state::emit_hotkey(state, true, "portal", display, None);

  let mut activated = gp
    .receive_activated()
    .await
    .map_err(|e| format!("订阅 Activated 失败：{e}"))?;
  let mut deactivated = gp
    .receive_deactivated()
    .await
    .map_err(|e| format!("订阅 Deactivated 失败：{e}"))?;

  enum Ev {
    Tick,
    Act(Option<ashpd::desktop::global_shortcuts::Activated>),
    Deact(Option<ashpd::desktop::global_shortcuts::Deactivated>),
  }

  let mut recording = false;
  loop {
    if stop.load(Ordering::Relaxed) {
      break;
    }
    let tick = async {
      async_io::Timer::after(Duration::from_millis(150)).await;
      Ev::Tick
    };
    let act = async { Ev::Act(activated.next().await) };
    let deact = async { Ev::Deact(deactivated.next().await) };
    match futures_lite::future::or(futures_lite::future::or(tick, act), deact).await {
      Ev::Tick => {}
      Ev::Act(Some(ev)) => {
        if let Some(mode) = by_id.get(ev.shortcut_id())
          && !recording
        {
          recording = true;
          crate::log_line(&format!("热键按下（门户）：{}", ev.shortcut_id()));
          state.press(*mode);
        }
      }
      Ev::Deact(Some(ev)) => {
        if by_id.contains_key(ev.shortcut_id()) && recording {
          recording = false;
          crate::log_line("热键松开（门户）：结束录音");
          state.release();
        }
      }
      Ev::Act(None) | Ev::Deact(None) => return Err("门户信号流已结束".into()),
    }
  }
  if recording {
    state.release();
  }
  let _ = session.close().await;
  Ok(())
}

/// 等一个 future，同时每 100 ms 检查一次 stop。
async fn with_stop<F: std::future::Future>(fut: F, stop: &AtomicBool) -> Option<F::Output> {
  futures_lite::future::or(
    async { Some(fut.await) },
    async {
      loop {
        if stop.load(Ordering::Relaxed) {
          return None;
        }
        async_io::Timer::after(Duration::from_millis(100)).await;
      }
    },
  )
  .await
}

/// 门户触发器语法（XDG shortcuts spec）：`CTRL+SHIFT+a` / `F2`。
fn portal_trigger(spec: &Parsed) -> Option<String> {
  if spec.mouse.is_some() {
    return None;
  }
  let mut parts = Vec::new();
  if spec.ctrl {
    parts.push("CTRL");
  }
  if spec.alt {
    parts.push("ALT");
  }
  if spec.shift {
    parts.push("SHIFT");
  }
  if spec.meta {
    parts.push("LOGO");
  }
  let key = match spec.token.as_str() {
    "esc" => "Escape".to_string(),
    "tab" => "Tab".to_string(),
    "enter" => "Return".to_string(),
    "backspace" => "BackSpace".to_string(),
    "space" => "space".to_string(),
    "insert" => "Insert".to_string(),
    "delete" => "Delete".to_string(),
    "home" => "Home".to_string(),
    "end" => "End".to_string(),
    "pageup" => "Page_Up".to_string(),
    "pagedown" => "Page_Down".to_string(),
    "up" => "Up".to_string(),
    "down" => "Down".to_string(),
    "left" => "Left".to_string(),
    "right" => "Right".to_string(),
    "pause" => "Pause".to_string(),
    "capslock" => "Caps_Lock".to_string(),
    "numlock" => "Num_Lock".to_string(),
    "scrolllock" => "Scroll_Lock".to_string(),
    "printscreen" => "Print".to_string(),
    t if t.starts_with('f') && t[1..].chars().all(|c| c.is_ascii_digit()) => {
      t.to_ascii_uppercase()
    }
    t => t.to_string(),
  };
  parts.push(&key);
  Some(parts.join("+"))
}

// ─────────────────────────── evdev ───────────────────────────

struct Dev {
  file: File,
  name: String,
  mods: u32,
}

struct Want {
  code: u32,
  mouse: bool,
  mode: PasteMode,
  spec: Parsed,
}

fn evdev_loop(
  bindings: &[(Parsed, PasteMode)],
  state: &Arc<AppState>,
  stop: &AtomicBool,
  display: &str,
) -> bool {
  let mut wants = Vec::new();
  for (spec, mode) in bindings {
    let (code, mouse) = match spec.mouse {
      Some(1) => (BTN_SIDE, true),
      Some(2) => (BTN_EXTRA, true),
      Some(_) => (BTN_SIDE, true),
      None => match evdev_code(&spec.token) {
        Some(c) => (c, false),
        None => {
          crate::log_line(&format!("evdev 没有对应的键：{}", spec.token));
          continue;
        }
      },
    };
    wants.push(Want { code, mouse, mode: *mode, spec: spec.clone() });
  }
  if wants.is_empty() {
    return false;
  }
  let need_mouse = wants.iter().any(|w| w.mouse);

  let mut devs: HashMap<PathBuf, Dev> = HashMap::new();
  let mut mods = 0u32;
  let mut recording = false;
  let mut active_code: u32 = 0;
  let mut warned_denied = false;
  let mut announced = false;
  let mut last_scan = Instant::now() - Duration::from_secs(10);

  loop {
    if stop.load(Ordering::Relaxed) {
      break;
    }

    if last_scan.elapsed() >= Duration::from_secs(2) {
      last_scan = Instant::now();
      let denied = scan_devices(&mut devs, need_mouse);
      if !devs.is_empty() && !announced {
        announced = true;
        crate::state::emit_hotkey(state, true, "evdev", display, None);
        crate::log_line(&format!("热键已注册（evdev）：{display}"));
      }
      if devs.is_empty() && denied && !warned_denied {
        warned_denied = true;
        crate::log_line(
          "无法读取 /dev/input/event*（权限不足）。修复：sudo usermod -aG input $USER 后重新登录，\
           或安装 udev 规则 70-shuo-hotkey.rules",
        );
        crate::state::emit_hotkey(
          state,
          false,
          "evdev",
          display,
          Some("全局热键需要 input 组权限（详见 ~/.shuo/shuo.log）"),
        );
      }
    }

    if devs.is_empty() {
      std::thread::sleep(Duration::from_millis(50));
      continue;
    }

    let mut got = false;
    let mut dead: Vec<PathBuf> = Vec::new();
    for (path, dev) in devs.iter_mut() {
      let mut evs = [unsafe { std::mem::zeroed::<libc::input_event>() }; 64];
      let buf = unsafe {
        std::slice::from_raw_parts_mut(
          evs.as_mut_ptr() as *mut u8,
          evs.len() * std::mem::size_of::<libc::input_event>(),
        )
      };
      match dev.file.read(buf) {
        Ok(0) => dead.push(path.clone()),
        Ok(n) => {
          got = true;
          let count = n / std::mem::size_of::<libc::input_event>();
          for ev in &evs[..count] {
            if ev.type_ != EV_KEY {
              continue;
            }
            let code = ev.code as u32;
            let down = ev.value == 1;
            let up = ev.value == 0;

            if let Some(bit) = mod_bit(code) {
              if down {
                mods |= bit;
                dev.mods |= bit;
              } else if up {
                dev.mods &= !bit;
              }
              continue;
            }

            if recording && up && code == active_code {
              recording = false;
              crate::log_line("热键松开（evdev）：结束录音");
              state.release();
              continue;
            }

            if !recording && down {
              for w in &wants {
                if w.code != code {
                  continue;
                }
                if w.mouse || mods_match(&w.spec, mods) {
                  recording = true;
                  active_code = code;
                  crate::log_line(&format!("热键按下（evdev）：{}", dev.name));
                  state.press(w.mode);
                  break;
                }
              }
            }
          }
        }
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
        Err(_) => dead.push(path.clone()),
      }
    }
    for p in dead {
      devs.remove(&p);
    }

    mods = devs.values().fold(0, |acc, d| acc | d.mods);

    if !got {
      std::thread::sleep(Duration::from_millis(5));
    }
  }

  if recording {
    state.release();
  }
  !devs.is_empty()
}

fn mods_match(spec: &Parsed, mods: u32) -> bool {
  (!spec.ctrl || mods & MOD_CTRL != 0)
    && (!spec.shift || mods & MOD_SHIFT != 0)
    && (!spec.alt || mods & MOD_ALT != 0)
    && (!spec.meta || mods & MOD_META != 0)
}

fn mod_bit(code: u32) -> Option<u32> {
  Some(match code {
    29 | 97 => MOD_CTRL,
    42 | 54 => MOD_SHIFT,
    56 | 100 => MOD_ALT,
    125 | 126 => MOD_META,
    _ => return None,
  })
}

fn evdev_code(token: &str) -> Option<u32> {
  Some(match token {
    "a" => 30,
    "b" => 48,
    "c" => 46,
    "d" => 32,
    "e" => 18,
    "f" => 33,
    "g" => 34,
    "h" => 35,
    "i" => 23,
    "j" => 36,
    "k" => 37,
    "l" => 38,
    "m" => 50,
    "n" => 49,
    "o" => 24,
    "p" => 25,
    "q" => 16,
    "r" => 19,
    "s" => 31,
    "t" => 20,
    "u" => 22,
    "v" => 47,
    "w" => 17,
    "x" => 45,
    "y" => 21,
    "z" => 44,
    "1" => 2,
    "2" => 3,
    "3" => 4,
    "4" => 5,
    "5" => 6,
    "6" => 7,
    "7" => 8,
    "8" => 9,
    "9" => 10,
    "0" => 11,
    "f1" => 59,
    "f2" => 60,
    "f3" => 61,
    "f4" => 62,
    "f5" => 63,
    "f6" => 64,
    "f7" => 65,
    "f8" => 66,
    "f9" => 67,
    "f10" => 68,
    "f11" => 87,
    "f12" => 88,
    "f13" => 183,
    "f14" => 184,
    "f15" => 185,
    "f16" => 186,
    "f17" => 187,
    "f18" => 188,
    "f19" => 189,
    "f20" => 190,
    "f21" => 191,
    "f22" => 192,
    "f23" => 193,
    "f24" => 194,
    "esc" => 1,
    "tab" => 15,
    "enter" => 28,
    "backspace" => 14,
    "space" => 57,
    "insert" => 110,
    "delete" => 111,
    "home" => 102,
    "end" => 107,
    "pageup" => 104,
    "pagedown" => 109,
    "up" => 103,
    "down" => 108,
    "left" => 105,
    "right" => 106,
    "pause" => 119,
    "capslock" => 58,
    "numlock" => 69,
    "scrolllock" => 70,
    "printscreen" => 99,
    "`" => 41,
    "-" => 12,
    "=" => 13,
    "[" => 26,
    "]" => 27,
    "\\" => 43,
    ";" => 39,
    "'" => 40,
    "," => 51,
    "." => 52,
    "/" => 53,
    "xbutton1" | "mouse4" | "back" => BTN_SIDE,
    "xbutton2" | "mouse5" | "forward" => BTN_EXTRA,
    _ => return None,
  })
}

fn scan_devices(devs: &mut HashMap<PathBuf, Dev>, need_mouse: bool) -> bool {
  let mut denied = false;
  let Ok(dir) = std::fs::read_dir("/dev/input") else {
    return false;
  };
  for entry in dir.flatten() {
    let path = entry.path();
    if !path
      .file_name()
      .and_then(|n| n.to_str())
      .map(|n| n.starts_with("event"))
      .unwrap_or(false)
    {
      continue;
    }
    if devs.contains_key(&path) {
      continue;
    }
    let file = match std::fs::OpenOptions::new()
      .read(true)
      .custom_flags(libc::O_NONBLOCK)
      .open(&path)
    {
      Ok(f) => f,
      Err(e) => {
        if e.kind() == std::io::ErrorKind::PermissionDenied {
          denied = true;
        }
        continue;
      }
    };

    if !has_cap(&file, EV_KEY as u32, if need_mouse { BTN_LEFT } else { KEY_A })
      && !(has_cap(&file, EV_KEY as u32, BTN_SIDE) || has_cap(&file, EV_KEY as u32, BTN_EXTRA))
      && !has_cap(&file, EV_KEY as u32, KEY_SPACE)
    {
      continue;
    }

    let name = path
      .file_name()
      .and_then(|n| n.to_str())
      .map(|n| {
        std::fs::read_to_string(format!("/sys/class/input/{n}/device/name"))
          .ok()
          .map(|s| s.trim().to_string())
          .filter(|s| !s.is_empty())
          .unwrap_or_else(|| n.to_string())
      })
      .unwrap_or_else(|| path.display().to_string());

    crate::log_line(&format!("监听输入设备：{name} ({})", path.display()));
    devs.insert(path, Dev { file, name, mods: 0 });
  }
  denied
}

fn has_cap(file: &File, ev: u32, code: u32) -> bool {
  let mut bits = [0u8; 96];
  let req = (2u64 << 30)
    | ((bits.len() as u64) << 16)
    | ((b'E' as u64) << 8)
    | (0x20 + ev as u64);
  let r = unsafe { libc::ioctl(file.as_raw_fd(), req as libc::c_ulong, bits.as_mut_ptr()) };
  if r < 0 {
    return false;
  }
  let idx = (code / 8) as usize;
  idx < bits.len() && bits[idx] & (1 << (code % 8)) != 0
}
