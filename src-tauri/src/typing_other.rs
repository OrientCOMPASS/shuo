// 说 · Shuo — 其它平台：明确不可用
//
// SPDX-License-Identifier: GPL-3.0-or-later

pub fn type_text(_text: &str) -> Result<(), String> {
  Err("当前平台不支持自动输入".into())
}

pub fn copy_text(_text: &str) -> Result<(), String> {
  Err("当前平台不支持剪贴板操作".into())
}
