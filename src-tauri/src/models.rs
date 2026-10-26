// 说 · Shuo — 模型目录解析
//
// 打包后模型不进安装包（ONNX ASR 约 2.5 GB / GGUF 1.7B 约 2.4 GB），按以下顺序查找：
//   1. 配置里的显式路径
//   2. 环境变量 SHUO_MODEL_DIR（目录下放两个模型文件夹）
//   3. ~/.shuo/models/
//   4. 可执行文件上溯 4 层的仓库根（开发态）
//   5. 当前工作目录上溯 3 层
//   6. Linux 安装包资源目录 /usr/lib/<product>/models
//
// ASR 槽位认两种目录名（先统一包后基座包）：
//   * `Qwen3-ASR-0.6B-ONNX-CPU-unified/` —— finetune/ 产线（LoRA → int8 ONNX 补丁）
//     输出的「单模型直出」包：目录带 `unified.json` 标记，system 段提示词会被
//     模型当任务指令执行（识别 + 文本优化一步直出）；
//   * `Qwen3-ASR-0.6B-ONNX-CPU/`         —— Daumee 基座包（提示词只作识别偏置）。
// 两种布局完全相同（onnx_models/ + tokenizer.json），运行时可无缝切换。
//
// SPDX-License-Identifier: GPL-3.0-or-later

use std::path::{Path, PathBuf};

use crate::config::Config;

pub const ASR_DIR_NAME: &str = "Qwen3-ASR-0.6B-ONNX-CPU";
pub const ASR_DIR_NAME_UNIFIED: &str = "Qwen3-ASR-0.6B-ONNX-CPU-unified";
/// GGUF 后端（llama.cpp）模型目录：官方 ggml-org/Qwen3-ASR-1.7B-GGUF 原样解压，
/// 或 finetune gguf-e2 产线的统一包（*-unified，带 unified.json 标记）。
/// 目录内 `mmproj` 前缀的 .gguf 是音频编码器，其余最大的 .gguf 是 LM
/// （qsa 的自动发现约定，官方/E2 两种命名都兼容）。
pub const GGUF_DIR_NAME: &str = "Qwen3-ASR-1.7B-GGUF";
pub const GGUF_DIR_NAME_UNIFIED: &str = "Qwen3-ASR-1.7B-GGUF-unified";
/// 统一包标记文件（patch_onnx_lora.py / gguf-e2 产线写入：基座、适配器、提示词集、时间戳）
pub const UNIFIED_MARKER: &str = "unified.json";

/// GGUF 模型对（LM + mmproj 音频编码器）
#[derive(Debug, Clone)]
pub struct GgufPaths {
  pub lm: PathBuf,
  pub mmproj: PathBuf,
}

/// ASR 推理后端
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
  Onnx,
  Gguf,
}

#[derive(Debug, Clone)]
pub struct ModelPaths {
  pub asr: Option<PathBuf>,
  /// 解析到的 ASR 目录是否为统一包（带 unified.json，提示词可当任务指令）
  pub asr_unified: bool,
  /// GGUF 模型对（llama.cpp 后端用）
  pub gguf: Option<GgufPaths>,
  /// GGUF 目录是否为统一包
  pub gguf_unified: bool,
  /// 已尝试过的路径（诊断用）
  pub tried: Vec<String>,
}

pub fn asr_ok(dir: &Path) -> bool {
  dir.join("onnx_models/encoder_conv.onnx").is_file()
    && dir.join("onnx_models/encoder_transformer.onnx").is_file()
    && dir.join("onnx_models/embed_tokens.bin").is_file()
    && dir.join("tokenizer.json").is_file()
}

pub fn is_unified(dir: &Path) -> bool {
  dir.join(UNIFIED_MARKER).is_file()
}

/// 扫描 GGUF 目录：`mmproj*` 前缀 → 音频编码器；其余取最大者 → LM。
pub fn gguf_scan(dir: &Path) -> Option<GgufPaths> {
  let mut lm: Option<(u64, PathBuf)> = None;
  let mut mm: Option<(u64, PathBuf)> = None;
  let entries = std::fs::read_dir(dir).ok()?;
  for e in entries.flatten() {
    let p = e.path();
    if !p.is_file() || p.extension().is_none_or(|x| x != "gguf") {
      continue;
    }
    let size = e.metadata().map(|m| m.len()).unwrap_or(0);
    let name = p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
    if name.starts_with("mmproj") {
      if mm.as_ref().is_none_or(|(s, _)| size > *s) {
        mm = Some((size, p));
      }
    } else if lm.as_ref().is_none_or(|(s, _)| size > *s) {
      lm = Some((size, p));
    }
  }
  Some(GgufPaths { lm: lm?.1, mmproj: mm?.1 })
}

fn candidates(kinds: &[&str], explicit: Option<&str>) -> Vec<PathBuf> {
  let mut out = Vec::new();
  if let Some(p) = explicit.map(str::trim).filter(|p| !p.is_empty()) {
    let p = PathBuf::from(p);
    if p.file_name().is_some_and(|n| kinds.iter().any(|k| *k == n)) {
      out.push(p.clone());
    } else {
      for kind in kinds {
        out.push(p.join(kind));
      }
      // 显式路径本身就是一个可用模型目录（asr_ok/gguf_scan 兜底判定）
      out.push(p);
    }
  }
  if let Ok(root) = std::env::var("SHUO_MODEL_DIR") {
    let p = PathBuf::from(root);
    for kind in kinds {
      out.push(p.join(kind));
    }
    out.push(p.clone());
  }
  for kind in kinds {
    out.push(crate::config::user_dir().join("models").join(kind));
  }
  if let Ok(exe) = std::env::current_exe() {
    let mut dir = exe.parent().map(Path::to_path_buf);
    for _ in 0..5 {
      if let Some(d) = dir {
        for kind in kinds {
          out.push(d.join(kind));
          out.push(d.join("models").join(kind));
        }
        dir = d.parent().map(Path::to_path_buf);
      }
    }
  }
  if let Ok(cwd) = std::env::current_dir() {
    let mut dir = Some(cwd);
    for _ in 0..4 {
      if let Some(d) = dir {
        for kind in kinds {
          out.push(d.join(kind));
          out.push(d.join("models").join(kind));
        }
        dir = d.parent().map(Path::to_path_buf);
      }
    }
  }
  // 开发态：src-tauri/../Qwen...
  for kind in kinds {
    out.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(kind));
  }
  #[cfg(target_os = "linux")]
  {
    for kind in kinds {
      out.push(PathBuf::from("/usr/lib/Shuo/models").join(kind));
      out.push(PathBuf::from("/usr/share/shuo/models").join(kind));
    }
  }
  out
}

pub fn resolve(cfg: &Config) -> ModelPaths {
  let mut tried = Vec::new();
  let mut asr = None;
  for c in candidates(&[ASR_DIR_NAME_UNIFIED, ASR_DIR_NAME], cfg.asr_model_dir.as_deref()) {
    if asr_ok(&c) {
      asr = Some(c);
      break;
    }
    tried.push(c.display().to_string());
  }
  let asr_unified = asr.as_ref().is_some_and(|p| is_unified(p));
  let mut gguf = None;
  for c in candidates(&[GGUF_DIR_NAME_UNIFIED, GGUF_DIR_NAME], cfg.asr_model_dir.as_deref()) {
    if c.is_dir()
      && let Some(g) = gguf_scan(&c)
    {
      gguf = Some(g);
      break; // candidates 已按「统一包优先、就近优先」排序，首个命中即用
    }
  }
  // gguf_unified：以最终解析出的 LM 所在目录判断
  let gguf_unified = gguf
    .as_ref()
    .and_then(|g| g.lm.parent())
    .is_some_and(is_unified);
  ModelPaths { asr, asr_unified, gguf, gguf_unified, tried }
}

/// 后端选择："auto" = 有 GGUF 模型优先 GGUF（1.7B 主路径），否则 ONNX；
/// 显式 "gguf"/"onnx" 时若对应模型缺失，回退到另一个可用后端（日志说明）。
pub fn pick_backend(cfg: &Config, paths: &ModelPaths) -> Option<Backend> {
  let want_gguf = match cfg.asr_backend.as_str() {
    "gguf" => true,
    "onnx" => false,
    _ => paths.gguf.is_some(),
  };
  if want_gguf && paths.gguf.is_some() {
    Some(Backend::Gguf)
  } else if paths.asr.is_some() {
    Some(Backend::Onnx)
  } else if paths.gguf.is_some() {
    Some(Backend::Gguf)
  } else {
    None
  }
}

/// 单模型直出能力：按所选后端取对应槽位的 unified 标记
pub fn unified_for(backend: Backend, paths: &ModelPaths) -> bool {
  match backend {
    Backend::Onnx => paths.asr_unified,
    Backend::Gguf => paths.gguf_unified,
  }
}

pub fn describe(paths: &ModelPaths) -> String {
  let a = paths
    .asr
    .as_ref()
    .map(|p| {
      let tag = if paths.asr_unified { "（统一包·单模型直出）" } else { "" };
      format!("{}{tag}", p.display())
    })
    .unwrap_or_else(|| "未找到".into());
  let g = paths
    .gguf
    .as_ref()
    .map(|g| {
      let tag = if paths.gguf_unified { "（统一包·单模型直出）" } else { "" };
      format!("{} + {}{tag}", g.lm.display(), g.mmproj.display())
    })
    .unwrap_or_else(|| "未找到".into());
  format!("ASR(ONNX)：{a}\nASR(GGUF)：{g}")
}
