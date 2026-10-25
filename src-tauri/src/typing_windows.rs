// 说 · Shuo — Windows 自动输入（SendInput / Unicode）
//
// SPDX-License-Identifier: GPL-3.0-or-later

use windows::Win32::UI::Input::KeyboardAndMouse::{
  INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYBD_EVENT_FLAGS, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE,
  SendInput,
};

const CHUNK: usize = 64;

pub fn type_text(text: &str, _mode: crate::typing::PasteMode) -> Result<(), String> {
  let units: Vec<u16> = text.encode_utf16().collect();
  for chunk in units.chunks(CHUNK) {
    let mut inputs: Vec<INPUT> = Vec::with_capacity(chunk.len() * 2);
    for &u in chunk {
      inputs.push(make_input(u, KEYEVENTF_UNICODE));
      inputs.push(make_input(u, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP));
    }
    let sent = unsafe { SendInput(&inputs, std::mem::size_of::<INPUT>() as i32) };
    if sent as usize != inputs.len() {
      return Err("自动输入被系统拒绝（SendInput 返回不足）".into());
    }
  }
  Ok(())
}

fn make_input(scan: u16, flags: KEYBD_EVENT_FLAGS) -> INPUT {
  INPUT {
    r#type: INPUT_KEYBOARD,
    Anonymous: INPUT_0 {
      ki: KEYBDINPUT {
        wVk: Default::default(),
        wScan: scan,
        dwFlags: flags,
        time: 0,
        dwExtraInfo: 0,
      },
    },
  }
}

pub fn copy_text(text: &str) -> Result<(), String> {
  // 剪贴板走前端 navigator.clipboard；这里给一个 PowerShell 兜底
  let mut child = std::process::Command::new("cmd")
    .args(["/C", "clip"])
    .stdin(std::process::Stdio::piped())
    .spawn()
    .map_err(|e| format!("启动 clip 失败：{e}"))?;
  use std::io::Write;
  if let Some(stdin) = child.stdin.as_mut() {
    stdin
      .write_all(text.as_bytes())
      .map_err(|e| format!("写入剪贴板失败：{e}"))?;
  }
  let _ = child.wait();
  Ok(())
}
