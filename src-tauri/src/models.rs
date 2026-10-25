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
pub const LLM_DIR_NAME: &str = "Qwen3.5-2B-ONNX-OPT";
/// 统一包标记文件（patch_onnx_lora.py 写入：基座、适配器、提示词集、时间戳）
pub const UNIFIED_MARKER: &str = "unified.json";

#[derive(Debug, Clone)]
pub struct ModelPaths {
  pub asr: Option<PathBuf>,
  /// 解析到的 ASR 目录是否为统一包（带 unified.json，提示词可当任务指令）
  pub asr_unified: bool,
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

pub fn is_unified(dir: &Path) -> bool {
  dir.join(UNIFIED_MARKER).is_file()
}

pub fn llm_ok(dir: &Path) -> bool {
  crate::llm::LlmPipeline::is_available(dir)
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
      // 显式路径本身就是一个可用模型目录（asr_ok/llm_ok 兜底判定）
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
  let mut llm = None;
  for c in candidates(&[LLM_DIR_NAME], cfg.llm_model_dir.as_deref()) {
    if llm_ok(&c) {
      llm = Some(c);
      break;
    }
    tried.push(c.display().to_string());
  }
  ModelPaths { asr, asr_unified, llm, tried }
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
  let l = paths.llm.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "未找到".into());
  format!("ASR：{a}\nLLM：{l}")
}
