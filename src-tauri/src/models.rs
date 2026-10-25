// 说 · Shuo — 模型目录解析
//
// 打包后模型不进安装包（ASR 2.5 GB + LLM 1.4 GB），按以下顺序查找：
//   1. 配置里的显式路径
//   2. 环境变量 SHUO_MODEL_DIR（目录下放两个模型文件夹）
//   3. ~/.shuo/models/
//   4. 可执行文件上溯 4 层的仓库根（开发态）
//   5. 当前工作目录上溯 3 层
//   6. Linux 安装包资源目录 /usr/lib/<product>/models
//
// SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};

use crate::config::Config;

pub const ASR_DIR_NAME: &str = "Qwen3-ASR-0.6B-ONNX-CPU";
pub const LLM_DIR_NAME: &str = "Qwen3.5-2B-ONNX-OPT";

#[derive(Debug, Clone)]
pub struct ModelPaths {
  pub asr: Option<PathBuf>,
  pub llm: Option<PathBuf>,
  /// 已尝试过的路径（诊断用）
  pub tried: Vec<String>,
}

pub fn asr_ok(dir: &Path) -> bool {
  dir.join("onnx_models/encoder_conv.onnx").is_file()
    && dir.join("onnx_models/encoder_transformer.onnx").is_file()
    && dir.join("onnx_models/embed_tokens.bin").is_file()
    && dir.join("tokenizer.json").is_file()
}

pub fn llm_ok(dir: &Path) -> bool {
  crate::llm::LlmPipeline::is_available(dir)
}

fn candidates(kind: &str, explicit: Option<&str>) -> Vec<PathBuf> {
  let mut out = Vec::new();
  if let Some(p) = explicit.map(str::trim).filter(|p| !p.is_empty()) {
    let p = PathBuf::from(p);
    out.push(if p.file_name().is_some_and(|n| n == kind) { p } else { p.join(kind) });
  }
  if let Ok(root) = std::env::var("SHUO_MODEL_DIR") {
    let p = PathBuf::from(root);
    out.push(p.join(kind));
    out.push(p.clone());
  }
  out.push(crate::config::user_dir().join("models").join(kind));
  if let Ok(exe) = std::env::current_exe() {
    let mut dir = exe.parent().map(Path::to_path_buf);
    for _ in 0..5 {
      if let Some(d) = dir {
        out.push(d.join(kind));
        out.push(d.join("models").join(kind));
        dir = d.parent().map(Path::to_path_buf);
      }
    }
  }
  if let Ok(cwd) = std::env::current_dir() {
    let mut dir = Some(cwd);
    for _ in 0..4 {
      if let Some(d) = dir {
        out.push(d.join(kind));
        out.push(d.join("models").join(kind));
        dir = d.parent().map(Path::to_path_buf);
      }
    }
  }
  // 开发态：src-tauri/../Qwen...
  out.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(kind));
  #[cfg(target_os = "linux")]
  {
    out.push(PathBuf::from("/usr/lib/Shuo/models").join(kind));
    out.push(PathBuf::from("/usr/share/shuo/models").join(kind));
  }
  out
}

pub fn resolve(cfg: &Config) -> ModelPaths {
  let mut tried = Vec::new();
  let mut asr = None;
  for c in candidates(ASR_DIR_NAME, cfg.asr_model_dir.as_deref()) {
    if asr_ok(&c) {
      asr = Some(c);
      break;
    }
    tried.push(c.display().to_string());
  }
  let mut llm = None;
  for c in candidates(LLM_DIR_NAME, cfg.llm_model_dir.as_deref()) {
    if llm_ok(&c) {
      llm = Some(c);
      break;
    }
    tried.push(c.display().to_string());
  }
  ModelPaths { asr, llm, tried }
}

pub fn describe(paths: &ModelPaths) -> String {
  let a = paths.asr.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "未找到".into());
  let l = paths.llm.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "未找到".into());
  format!("ASR：{a}\nLLM：{l}")
}
