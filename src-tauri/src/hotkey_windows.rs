// 说 · Shuo — Windows 全局热键（低级键鼠钩子，支持多条绑定）
//
// 为什么不用 RegisterHotKey：它只给「按下」不给「松开」，而本应用是按住说话。
//
// 每条绑定带一个 PasteMode：主热键用 Ctrl+V 上屏，终端热键用 Ctrl+Shift+V 上屏。
//
// SPDX-License-Identifier: GPL-3.0-or-later

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::thread::JoinHandle;

use windows::Win32::Foundation::{LPARAM, LRESULT, WPARAM};
use windows::Win32::UI::Input::KeyboardAndMouse::{VK_LWIN, VK_RWIN};
use windows::Win32::UI::WindowsAndMessaging::{
  CallNextHookEx, GetMessageW, KBDLLHOOKSTRUCT, LLKHF_INJECTED, LLMHF_INJECTED, MSG, MSLLHOOKSTRUCT, PM_NOREMOVE,
  PeekMessageW, PostThreadMessageW, SetWindowsHookExW, UnhookWindowsHookEx, WH_KEYBOARD_LL,
  WH_MOUSE_LL, WM_KEYDOWN, WM_SYSKEYDOWN, WM_XBUTTONDOWN, WM_XBUTTONUP,
};

use super::Parsed;
use crate::state::AppState;
use crate::typing::PasteMode;

const VK_CTRL: u32 = 0x11;
const VK_SHIFT: u32 = 0x10;
const VK_ALT: u32 = 0x12;

const MOD_CTRL: u32 = 1;
const MOD_SHIFT: u32 = 2;
const MOD_ALT: u32 = 4;
const MOD_META: u32 = 8;

struct Ctx {
  bindings: Vec<(Parsed, PasteMode)>,
  state: Arc<AppState>,
  active: AtomicBool,
  active_vk: AtomicU32,
  active_mouse: AtomicU8,
  mods: AtomicU32,
}

thread_local! {
  static CTX: RefCell<Option<Arc<Ctx>>> = const { RefCell::new(None) };
}

fn vk_of(token: &str) -> Option<u32> {
  let s = token;
  if let Some(n) = s.strip_prefix('f')
    && let Ok(i) = n.parse::<u32>()
    && (1..=24).contains(&i)
  {
    return Some(0x70 + i - 1);
  }
  if s.len() == 1 {
    let c = s.chars().next()?;
    if c.is_ascii_alphabetic() {
      return Some(c.to_ascii_uppercase() as u32);
    }
    if c.is_ascii_digit() {
      return Some(c as u32);
    }
  }
  Some(match s {
    "space" => 0x20,
    "tab" => 0x09,
    "esc" => 0x1B,
    "enter" => 0x0D,
    "backspace" => 0x08,
    "insert" => 0x2D,
    "delete" => 0x2E,
    "home" => 0x24,
    "end" => 0x23,
    "pageup" => 0x21,
    "pagedown" => 0x22,
    "up" => 0x26,
    "down" => 0x28,
    "left" => 0x25,
    "right" => 0x27,
    "pause" => 0x13,
    "capslock" => 0x14,
    "numlock" => 0x90,
    "scrolllock" => 0x91,
    "printscreen" => 0x2C,
    "`" => 0xC0,
    "-" => 0xBD,
    "=" => 0xBB,
    "[" => 0xDB,
    "]" => 0xDD,
    "\\" => 0xDC,
    ";" => 0xBA,
    "'" => 0xDE,
    "," => 0xBC,
    "." => 0xBE,
    "/" => 0xBF,
    _ => return None,
  })
}

fn update_mods(ctx: &Ctx, vk: u32, down: bool) {
  let bit = match vk {
    VK_CTRL | 0xA2 | 0xA3 => MOD_CTRL,
    VK_SHIFT | 0xA0 | 0xA1 => MOD_SHIFT,
    VK_ALT | 0xA4 | 0xA5 => MOD_ALT,
    v if v == VK_LWIN.0 as u32 || v == VK_RWIN.0 as u32 => MOD_META,
    _ => return,
  };
  let mut m = ctx.mods.load(Ordering::Relaxed);
  if down {
    m |= bit;
  } else {
    m &= !bit;
  }
  ctx.mods.store(m, Ordering::Relaxed);
}

fn mods_ok(ctx: &Ctx, p: &Parsed) -> bool {
  let m = ctx.mods.load(Ordering::Relaxed);
  (!p.ctrl || m & MOD_CTRL != 0)
    && (!p.shift || m & MOD_SHIFT != 0)
    && (!p.alt || m & MOD_ALT != 0)
    && (!p.meta || m & MOD_META != 0)
}

fn begin(ctx: &Ctx, mode: PasteMode, vk: u32, mouse: u8) {
  if !ctx.active.swap(true, Ordering::Relaxed) {
    ctx.active_vk.store(vk, Ordering::Relaxed);
    ctx.active_mouse.store(mouse, Ordering::Relaxed);
    crate::log_line("热键按下：开始录音");
    ctx.state.press(mode);
  }
}

fn end_if(ctx: &Ctx, vk: u32, mouse: u8) {
  if ctx.active.load(Ordering::Relaxed)
    && ctx.active_vk.load(Ordering::Relaxed) == vk
    && ctx.active_mouse.load(Ordering::Relaxed) == mouse
  {
    ctx.active.store(false, Ordering::Relaxed);
    crate::log_line("热键松开：结束录音");
    ctx.state.release();
  }
}

unsafe extern "system" fn keyboard_proc(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
  if code >= 0
    && let Some(ctx) = CTX.with(|c| c.borrow().clone())
  {
    let kb = unsafe { &*(lparam.0 as *const KBDLLHOOKSTRUCT) };
    let injected = kb.flags.0 & LLKHF_INJECTED.0 != 0;
    if !injected {
      let vk = kb.vkCode;
      let msg = wparam.0 as u32;
      let down = msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN;
      update_mods(&ctx, vk, down);

      for (spec, mode) in &ctx.bindings {
        if spec.mouse.is_some() {
          continue;
        }
        let Some(want) = vk_of(&spec.token) else { continue };
        if vk != want {
          continue;
        }
        if down {
          if mods_ok(&ctx, spec) {
            begin(&ctx, *mode, vk, 0);
          }
        } else {
          end_if(&ctx, vk, 0);
        }
      }
    }
  }
  unsafe { CallNextHookEx(None, code, wparam, lparam) }
}

unsafe extern "system" fn mouse_proc(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
  if code >= 0
    && let Some(ctx) = CTX.with(|c| c.borrow().clone())
  {
    let ms = unsafe { &*(lparam.0 as *const MSLLHOOKSTRUCT) };
    let injected = ms.flags & LLMHF_INJECTED != 0;
    let msg = wparam.0 as u32;
    if !injected && (msg == WM_XBUTTONDOWN || msg == WM_XBUTTONUP) {
      let xbutton = ((ms.mouseData >> 16) & 0xFFFF) as u8;
      let down = msg == WM_XBUTTONDOWN;
      for (spec, mode) in &ctx.bindings {
        if spec.mouse != Some(xbutton) {
          continue;
        }
        if down {
          if mods_ok(&ctx, spec) {
            begin(&ctx, *mode, 0, xbutton);
          }
        } else {
          end_if(&ctx, 0, xbutton);
        }
      }
    }
  }
  unsafe { CallNextHookEx(None, code, wparam, lparam) }
}

/// 让阻塞在 GetMessageW 的热键线程退出。
pub(super) fn wake(tid: u32) {
  unsafe {
    let _ = PostThreadMessageW(tid, 0x0012 /* WM_QUIT */, WPARAM(0), LPARAM(0));
  }
}

pub(super) fn spawn(
  bindings: Vec<(Parsed, PasteMode)>,
  state: Arc<AppState>,
  stop: Arc<AtomicBool>,
  tid: Arc<AtomicU32>,
  _display: String,
) -> Option<JoinHandle<()>> {
  std::thread::Builder::new()
    .name("shuo-hotkey".into())
    .spawn(move || unsafe {
      use windows::Win32::System::Threading::GetCurrentThreadId;

      let mut msg = MSG::default();
      let _ = PeekMessageW(&mut msg, None, 0, 0, PM_NOREMOVE);
      tid.store(GetCurrentThreadId(), Ordering::Relaxed);

      let need_keys = bindings.iter().any(|(p, _)| p.mouse.is_none());
      let need_mouse = bindings.iter().any(|(p, _)| p.mouse.is_some());

      let ctx = Arc::new(Ctx {
        bindings,
        state,
        active: AtomicBool::new(false),
        active_vk: AtomicU32::new(0),
        active_mouse: AtomicU8::new(0),
        mods: AtomicU32::new(0),
      });
      CTX.with(|c| *c.borrow_mut() = Some(ctx.clone()));

      let key_hook = if need_keys {
        Some(SetWindowsHookExW(WH_KEYBOARD_LL, Some(keyboard_proc), None, 0))
      } else {
        None
      };
      let mouse_hook = if need_mouse {
        Some(SetWindowsHookExW(WH_MOUSE_LL, Some(mouse_proc), None, 0))
      } else {
        None
      };
      let any_ok = key_hook.as_ref().is_some_and(|h| h.is_ok())
        || mouse_hook.as_ref().is_some_and(|h| h.is_ok());
      if !any_ok {
        crate::log_line("安装键鼠钩子失败（热键不可用）");
        return;
      }

      // 阻塞式消息循环：钩子回调由系统投递到本线程，GetMessageW 负责泵消息
      loop {
        let r = GetMessageW(&mut msg, None, 0, 0);
        if r.0 <= 0 || stop.load(Ordering::Relaxed) {
          break;
        }
      }

      CTX.with(|c| *c.borrow_mut() = None);
      if let Some(Ok(h)) = key_hook {
        let _ = UnhookWindowsHookEx(h);
      }
      if let Some(Ok(h)) = mouse_hook {
        let _ = UnhookWindowsHookEx(h);
      }
      if ctx.active.load(Ordering::Relaxed) {
        ctx.state.release();
      }
      crate::log_line("热键线程退出");
    })
    .ok()
}
