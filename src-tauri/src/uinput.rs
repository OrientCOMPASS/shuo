// 说 · Shuo — uinput 虚拟键盘（Linux 上屏用，绕开 RemoteDesktop 门户授权）
//
// 只发粘贴快捷键（Ctrl+V / Ctrl+Shift+V），不逐字符打。xdotool 只能作用于 XWayland，
// ydotool 需要额外守护进程；直接在进程内建一个 uinput 键盘最简单：
//   /dev/uinput 需要可写（deb 的 udev 规则会加 TAG+="uaccess"）。
//
// SPDX-License-Identifier: GPL-3.0-or-later

use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::time::Duration;

const EV_SYN: u16 = 0;
const EV_KEY: u16 = 1;

const UI_DEV_CREATE: libc::c_ulong = 0x5501;
const UI_DEV_DESTROY: libc::c_ulong = 0x5502;
const UI_DEV_SETUP: libc::c_ulong = 0x405c_5503;
const UI_SET_EVBIT: libc::c_ulong = 0x4004_5564;
const UI_SET_KEYBIT: libc::c_ulong = 0x4004_5565;

#[repr(C)]
struct InputId {
  bustype: u16,
  vendor: u16,
  product: u16,
  version: u16,
}

#[repr(C)]
struct UinputSetup {
  id: InputId,
  name: [u8; 80],
  ff_effects_max: u32,
}

pub struct VirtualKeyboard {
  file: File,
}

impl VirtualKeyboard {
  pub fn open() -> Result<Self, String> {
    let file = OpenOptions::new()
      .write(true)
      .custom_flags(libc::O_NONBLOCK)
      .open("/dev/uinput")
      .map_err(|e| format!("打开 /dev/uinput 失败：{e}"))?;
    let fd = file.as_raw_fd();
    unsafe {
      if libc::ioctl(fd, UI_SET_EVBIT, EV_KEY as libc::c_ulong) < 0 {
        return Err("uinput: UI_SET_EVBIT 失败".into());
      }
      // Ctrl / Shift / V / Insert
      for code in [29u32, 42, 47, 110] {
        if libc::ioctl(fd, UI_SET_KEYBIT, code as libc::c_ulong) < 0 {
          return Err("uinput: UI_SET_KEYBIT 失败".into());
        }
      }
      let mut setup = UinputSetup {
        id: InputId { bustype: 0x03, vendor: 0x1234, product: 0x5678, version: 1 },
        name: [0u8; 80],
        ff_effects_max: 0,
      };
      let name = b"Shuo Virtual Keyboard";
      setup.name[..name.len()].copy_from_slice(name);
      if libc::ioctl(fd, UI_DEV_SETUP, &setup as *const UinputSetup) < 0 {
        return Err("uinput: UI_DEV_SETUP 失败".into());
      }
      if libc::ioctl(fd, UI_DEV_CREATE) < 0 {
        return Err("uinput: UI_DEV_CREATE 失败".into());
      }
    }
    // 设备节点创建需要一点时间
    std::thread::sleep(Duration::from_millis(120));
    Ok(Self { file })
  }

  fn emit(&self, type_: u16, code: u16, value: i32) {
    let ev = libc::input_event {
      time: libc::timeval { tv_sec: 0, tv_usec: 0 },
      type_,
      code,
      value,
    };
    unsafe {
      libc::write(
        self.file.as_raw_fd(),
        &ev as *const libc::input_event as *const libc::c_void,
        std::mem::size_of::<libc::input_event>(),
      );
    }
  }

  /// 按下一组键，再反序松开（组合键）。
  pub fn tap_combo(&self, codes: &[u16]) -> Result<(), String> {
    for &c in codes {
      self.emit(EV_KEY, c, 1);
      self.emit(EV_SYN, 0, 0);
      std::thread::sleep(Duration::from_millis(10));
    }
    for &c in codes.iter().rev() {
      self.emit(EV_KEY, c, 0);
      self.emit(EV_SYN, 0, 0);
      std::thread::sleep(Duration::from_millis(10));
    }
    Ok(())
  }
}

impl Drop for VirtualKeyboard {
  fn drop(&mut self) {
    unsafe {
      libc::ioctl(self.file.as_raw_fd(), UI_DEV_DESTROY);
    }
  }
}
