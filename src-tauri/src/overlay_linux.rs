// 说 · Shuo — Linux 底部状态胶囊（XWayland 置顶小窗 helper）
//
// GNOME/Mutter 不支持 wlr-layer-shell，普通 Wayland 窗口又无法「置顶且不抢焦点」。
// XWayland 窗口由 mutter 按 EWMH 管理：keep_above + accept_focus=false + 输入区域清空
// 就能做到置顶、不抢焦点、鼠标穿透。helper 是一个很小的 Python GTK3 脚本
// （系统自带 python3-gi；不额外装 GNOME 扩展，也不用注销重登录）。
//
// 协议：应用通过 Unix socket 发一行 recording / processing / idle。
//
// SPDX-License-Identifier: GPL-3.0-or-later

use std::io::Write;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use tauri::Manager;

static TX: OnceLock<Mutex<Option<UnixStream>>> = OnceLock::new();
static CHILD: OnceLock<Mutex<Option<Child>>> = OnceLock::new();

fn socket_path() -> PathBuf {
  let dir = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "/tmp".into());
  PathBuf::from(dir).join("shuo-overlay.sock")
}

fn script_path(app: &tauri::AppHandle) -> Option<PathBuf> {
  if let Ok(p) = std::env::var("SHUO_OVERLAY") {
    let p = PathBuf::from(p);
    if p.is_file() {
      return Some(p);
    }
  }
  if let Ok(res) = app.path().resource_dir() {
    let p = res.join("shuo-overlay.py");
    if p.is_file() {
      return Some(p);
    }
  }
  let dev = Path::new(env!("CARGO_MANIFEST_DIR")).join("../linux/shuo-overlay.py");
  dev.is_file().then_some(dev)
}

/// XWayland 的 xauth 文件由 mutter 动态生成，名字带随机后缀。
fn xauthority() -> Option<PathBuf> {
  if let Ok(p) = std::env::var("XAUTHORITY") {
    let p = PathBuf::from(p);
    if p.is_file() {
      return Some(p);
    }
  }
  let home = std::env::var_os("HOME").map(PathBuf::from);
  if let Some(h) = home {
    let p = h.join(".Xauthority");
    if p.is_file() {
      return Some(p);
    }
  }
  let uid = unsafe { libc::getuid() };
  let dir = PathBuf::from(format!("/run/user/{uid}"));
  let entries = std::fs::read_dir(dir).ok()?;
  for entry in entries.flatten() {
    let name = entry.file_name();
    if name.to_string_lossy().starts_with(".mutter-Xwaylandauth") {
      return Some(entry.path());
    }
  }
  None
}

pub fn start(app: &tauri::AppHandle) {
  let Some(script) = script_path(app) else {
    crate::log_line("[overlay] 没找到 shuo-overlay.py，跳过状态胶囊");
    return;
  };
  let mut cmd = Command::new("python3");
  cmd
    .arg(&script)
    .env("GDK_BACKEND", "x11")
    .env(
      "DISPLAY",
      std::env::var("DISPLAY").unwrap_or_else(|_| ":0".into()),
    )
    .stdin(Stdio::null())
    .stdout(Stdio::null())
    .stderr(Stdio::piped());
  if let Some(auth) = xauthority() {
    cmd.env("XAUTHORITY", auth);
  }
  match cmd.spawn() {
    Ok(mut child) => {
      if let Some(err) = child.stderr.take() {
        std::thread::spawn(move || {
          use std::io::BufRead;
          for line in std::io::BufReader::new(err).lines().map_while(Result::ok) {
            let line = line.trim();
            if !line.is_empty() {
              crate::log_line(&format!("[overlay] {line}"));
            }
          }
        });
      }
      *CHILD.get_or_init(|| Mutex::new(None)).lock().unwrap() = Some(child);
      std::thread::spawn(connect_loop);
      crate::log_line("[overlay] 状态胶囊已启动");
    }
    Err(e) => crate::log_line(&format!("[overlay] 启动失败：{e}")),
  }
}

fn connect_loop() {
  for _ in 0..50 {
    match UnixStream::connect(socket_path()) {
      Ok(s) => {
        let mut guard = TX.get_or_init(|| Mutex::new(None)).lock().unwrap();
        if guard.is_none() {
          *guard = Some(s);
        }
        return;
      }
      Err(_) => std::thread::sleep(Duration::from_millis(100)),
    }
  }
  crate::log_line("[overlay] 胶囊未就绪（检查 python3-gi / gir1.2-gtk-3.0）");
}

pub fn send(mode: &str) {
  let cell = TX.get_or_init(|| Mutex::new(None));
  let mut guard = cell.lock().unwrap();
  if guard.is_none() {
    if let Ok(s) = UnixStream::connect(socket_path()) {
      *guard = Some(s);
    }
  }
  if let Some(stream) = guard.as_mut() {
    let ok = stream
      .write_all(format!("{mode}\n").as_bytes())
      .and_then(|_| stream.flush())
      .is_ok();
    if !ok {
      *guard = None;
    }
  }
}
