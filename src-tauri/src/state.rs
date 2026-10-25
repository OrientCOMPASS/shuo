// 说 · Shuo — 应用状态与识别线程
//
// 单一线程串行处理识别任务（ASR → LLM），与 Python 版的队列语义一致。
//
// SPDX-License-Identifier: GPL-3.0-or-later

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};

use serde::Serialize;
use tauri::{AppHandle, Emitter};

use crate::asr::AsrPipeline;
use crate::audio::RecorderHandle;
use crate::config::{Config, HistoryItem};
use crate::denoise::Denoiser;
use crate::llm::{LlmPipeline, Sampling, preset_prompt};
use crate::models::{self, ModelPaths};

pub struct Job {
  pub samples: Vec<f32>,
  pub rate: u32,
  pub duration_s: f64,
  pub paste_mode: crate::typing::PasteMode,
}

#[derive(Debug, Clone, Serialize)]
pub struct Status {
  pub loading: bool,
  pub asr_ready: bool,
  pub llm_ready: bool,
  pub llm_loading: bool,
  pub asr_dir: Option<String>,
  pub llm_dir: Option<String>,
  pub error: Option<String>,
  pub denoise_ready: bool,
  /// 当前 ASR 目录是否为统一包（微调产物，system 段提示词可当任务指令）
  pub asr_unified: bool,
}

#[derive(Clone)]
pub struct Sink {
  app: AppHandle,
  tx: Sender<Job>,
}

impl Sink {
  pub fn emit_record(&self, active: bool, duration_s: f64) {
    let _ = self.app.emit("rec", RecEvent { active, duration_s });
    if active {
      crate::hotkey::set_recording(true);
      crate::set_busy(&self.app, Some(crate::BusyMode::Recording));
    } else if duration_s >= 0.1 {
      // 松开后马上进入「识别中」，避免闪一下消失
      crate::set_busy(&self.app, Some(crate::BusyMode::Processing));
    } else {
      crate::set_busy(&self.app, None);
    }
  }

  pub fn submit(&self, job: Job) {
    if self.tx.send(job).is_err() {
      crate::log_line("识别队列已关闭");
    }
  }
}

#[derive(Clone, Serialize)]
struct RecEvent {
  active: bool,
  duration_s: f64,
}

/// 热键状态上报（前端用来显示「全局热键：F2（门户）」这类信息）。
pub fn emit_hotkey(
  state: &AppState,
  ok: bool,
  backend: &str,
  spec: &str,
  error: Option<&str>,
) {
  let _ = state.app.emit(
    "hotkey",
    serde_json::json!({
      "ok": ok,
      "backend": backend,
      "spec": spec,
      "display": crate::hotkey::normalize_spec(spec),
      "error": error,
    }),
  );
}

pub struct AppState {
  pub app: AppHandle,
  pub cfg: Mutex<Config>,
  pub paths: Mutex<ModelPaths>,
  pub asr: Mutex<Option<AsrPipeline>>,
  pub llm: Mutex<Option<LlmPipeline>>,
  pub denoiser: Mutex<Option<Denoiser>>,
  pub status: Mutex<Status>,
  pub recorder: Mutex<Option<RecorderHandle>>,
  pub hotkey: Mutex<Option<crate::hotkey::Hotkey>>,
  pub history: Mutex<Vec<HistoryItem>>,
  pub sink: Sink,
  next_id: AtomicU64,
}

impl AppState {
  pub fn new(app: AppHandle, cfg: Config) -> Arc<Self> {
    let paths = models::resolve(&cfg);
    let (tx, rx) = channel::<Job>();
    let status = Status {
      loading: true,
      asr_ready: false,
      llm_ready: false,
      llm_loading: false,
      asr_dir: paths.asr.as_ref().map(|p| p.display().to_string()),
      llm_dir: paths.llm.as_ref().map(|p| p.display().to_string()),
      error: None,
      denoise_ready: false,
      asr_unified: paths.asr_unified,
    };
    let state = Arc::new(Self {
      app: app.clone(),
      cfg: Mutex::new(cfg),
      paths: Mutex::new(paths),
      asr: Mutex::new(None),
      llm: Mutex::new(None),
      denoiser: Mutex::new(None),
      status: Mutex::new(status),
      recorder: Mutex::new(None),
      hotkey: Mutex::new(None),
      history: Mutex::new(crate::config::load_history()),
      sink: Sink { app, tx },
      next_id: AtomicU64::new(1),
    });
    spawn_loader(state.clone());
    spawn_worker(state.clone(), rx);
    state
  }

  pub fn config(&self) -> Config {
    self.cfg.lock().unwrap().clone()
  }

  pub fn emit_status(&self) {
    let s = self.status.lock().unwrap().clone();
    let _ = self.app.emit("engine-status", s);
  }

  pub fn set_status<F: FnOnce(&mut Status)>(&self, f: F) {
    {
      let mut s = self.status.lock().unwrap();
      f(&mut s);
    }
    self.emit_status();
  }

  pub fn next_job_id(&self) -> u64 {
    self.next_id.fetch_add(1, Ordering::Relaxed)
  }

  /// 配置变更后重新解析模型路径并（按需）加载 LLM。
  pub fn reload_models(self: &Arc<Self>) {
    let cfg = self.config();
    let paths = models::resolve(&cfg);
    {
      let mut p = self.paths.lock().unwrap();
      *p = paths.clone();
    }
    self.set_status(|s| {
      s.asr_dir = paths.asr.as_ref().map(|p| p.display().to_string());
      s.llm_dir = paths.llm.as_ref().map(|p| p.display().to_string());
      s.asr_unified = paths.asr_unified;
    });
    if cfg.llm_enabled && self.llm.lock().unwrap().is_none() {
      spawn_llm_loader(self.clone());
    }
  }

  /// 按配置（重新）启动录音。
  pub fn restart_recorder(self: &Arc<Self>) -> Result<(), String> {
    let old = self.recorder.lock().unwrap().take();
    if let Some(h) = old {
      let _ = h.stop_signal();
    }
    let cfg = self.config();
    match crate::audio::start_recorder(self.sink.clone(), cfg.mic_device.clone()) {
      Ok(h) => {
        crate::log_line(&format!("使用麦克风：{} ({} Hz)", h.device, h.rate));
        *self.recorder.lock().unwrap() = Some(h);
        let _ = self.app.emit(
          "recorder",
          serde_json::json!({ "ok": true, "error": serde_json::Value::Null }),
        );
        Ok(())
      }
      Err(e) => {
        crate::log_line(&format!("麦克风不可用：{e}"));
        let _ = self.app.emit(
          "recorder",
          serde_json::json!({ "ok": false, "error": e }),
        );
        Err(e)
      }
    }
  }

  pub fn press(&self, mode: crate::typing::PasteMode) {
    if let Some(h) = self.recorder.lock().unwrap().as_ref() {
      h.press(mode);
    }
  }

  pub fn release(&self) {
    if let Some(h) = self.recorder.lock().unwrap().as_ref() {
      h.release();
    }
  }

  pub fn start_hotkey(self: &Arc<Self>) {
    let cfg = self.config();
    let old = self.hotkey.lock().unwrap().take();
    if let Some(h) = old {
      h.stop();
    }
    // 主热键 = Ctrl+V 上屏；终端热键 = Ctrl+Shift+V 上屏
    let mut list = Vec::new();
    if cfg.hotkey_enabled {
      list.push((cfg.hotkey.clone(), crate::typing::PasteMode::CtrlV));
    }
    if cfg.hotkey_terminal_enabled {
      list.push((cfg.hotkey_terminal.clone(), crate::typing::PasteMode::CtrlShiftV));
    }
    let h = crate::hotkey::start(self.clone(), list);
    *self.hotkey.lock().unwrap() = Some(h);
  }
}

fn spawn_loader(state: Arc<AppState>) {
  std::thread::Builder::new()
    .name("shuo-loader".into())
    .spawn(move || {
      let cfg = state.config();
      let paths = state.paths.lock().unwrap().clone();
      let threads = crate::threads();

      let Some(asr_dir) = paths.asr.clone() else {
        state.set_status(|s| {
          s.loading = false;
          s.error = Some(format!(
            "找不到 ASR 模型（Qwen3-ASR-0.6B-ONNX-CPU）。已尝试：\n{}",
            paths.tried.join("\n")
          ));
        });
        return;
      };

      crate::log_line(&format!("加载 ASR 模型：{}", asr_dir.display()));
      state.set_status(|s| {
        s.loading = true;
        s.error = None;
      });
      match AsrPipeline::load(&asr_dir, threads) {
        Ok(p) => {
          *state.asr.lock().unwrap() = Some(p);
          let unified = state.paths.lock().unwrap().asr_unified;
          crate::log_line(if unified {
            "ASR 模型就绪（统一包：支持单模型直出）"
          } else {
            "ASR 模型就绪（基座包）"
          });
          if cfg.single_model_enabled && !unified {
            crate::log_line(
              "提示：已开启单模型直出，但当前模型包不是微调统一包（缺 unified.json），\
               提示词只会作为识别偏置，不会执行文本优化；统一包产线见 finetune/README.md",
            );
          }
          state.set_status(|s| {
            s.asr_ready = true;
            s.loading = false;
          });
        }
        Err(e) => {
          crate::log_line(&format!("ASR 加载失败：{e}"));
          state.set_status(|s| {
            s.loading = false;
            s.error = Some(format!("ASR 模型加载失败：{e}"));
          });
          return;
        }
      }

      // 降噪模型（可选）
      if let Some(path) = crate::denoise_path(&state.app) {
        match Denoiser::load(&path) {
          Ok(d) => {
            *state.denoiser.lock().unwrap() = Some(d);
            crate::log_line("降噪模型就绪");
            state.set_status(|s| s.denoise_ready = true);
          }
          Err(e) => crate::log_line(&format!("降噪模型加载失败：{e}")),
        }
      }

      if cfg.llm_enabled {
        if let Some(dir) = paths.llm.clone() {
          state.set_status(|s| s.llm_loading = true);
          match LlmPipeline::load(&dir, threads) {
            Ok(p) => {
              *state.llm.lock().unwrap() = Some(p);
              crate::log_line("LLM 模型就绪");
              state.set_status(|s| {
                s.llm_ready = true;
                s.llm_loading = false;
              });
            }
            Err(e) => {
              crate::log_line(&format!("LLM 加载失败：{e}"));
              state.set_status(|s| {
                s.llm_loading = false;
                s.error = Some(format!("LLM 模型加载失败：{e}"));
              });
            }
          }
        }
      }
    })
    .ok();
}

fn spawn_llm_loader(state: Arc<AppState>) {
  std::thread::Builder::new()
    .name("shuo-llm-loader".into())
    .spawn(move || {
      if state.llm.lock().unwrap().is_some() {
        return;
      }
      let paths = state.paths.lock().unwrap().clone();
      let Some(dir) = paths.llm else { return };
      state.set_status(|s| s.llm_loading = true);
      match LlmPipeline::load(&dir, crate::threads()) {
        Ok(p) => {
          *state.llm.lock().unwrap() = Some(p);
          crate::log_line("LLM 模型就绪");
          state.set_status(|s| {
            s.llm_ready = true;
            s.llm_loading = false;
          });
        }
        Err(e) => {
          crate::log_line(&format!("LLM 加载失败：{e}"));
          state.set_status(|s| {
            s.llm_loading = false;
            s.error = Some(format!("LLM 模型加载失败：{e}"));
          });
        }
      }
    })
    .ok();
}

fn spawn_worker(state: Arc<AppState>, rx: std::sync::mpsc::Receiver<Job>) {
  std::thread::Builder::new()
    .name("shuo-infer".into())
    .spawn(move || {
      while let Ok(job) = rx.recv() {
        let id = state.next_job_id();
        let _ = state.app.emit("processing", id);
        crate::set_busy(&state.app, Some(crate::BusyMode::Processing));
        let cfg = state.config();
        let paste_mode = job.paste_mode;
        let result = transcribe(&state, &cfg, job);
        match result {
          Ok((asr, llm_text, llm_s)) => {
            crate::set_busy(&state.app, None);
            let _ = state.app.emit(
              "asr",
              serde_json::json!({
                "id": id,
                "text": asr.text,
                "language": asr.language,
                "asr_s": asr.asr_s,
                "audio_s": asr.audio_s,
                "chunks": asr.chunks,
              }),
            );
            let final_text = llm_text.clone().unwrap_or_else(|| asr.text.clone());
            if cfg.save_history {
              crate::config::add_history(&asr.text, llm_text.as_deref());
              let mut h = state.history.lock().unwrap();
              h.push(crate::config::HistoryItem {
                asr_text: asr.text.clone(),
                llm_text: llm_text.clone(),
                time: crate::config::chrono_time(),
              });
              // 与历史文件一致：内存里也最多留 500 条，避免长跑无限增长
              if h.len() > 500 {
                let cut = h.len() - 500;
                h.drain(..cut);
              }
            }
            let final_text = finalize_text(&state, &cfg, &final_text);
            let _ = state.app.emit(
              "final",
              serde_json::json!({
                "id": id,
                "asr_text": asr.text,
                "llm_text": llm_text,
                "llm_s": llm_s,
                "single_model": cfg.single_model_enabled,
                "text": final_text,
                "time": crate::config::chrono_time(),
              }),
            );
            if cfg.auto_type && !final_text.trim().is_empty() {
              if let Err(e) = crate::typing::type_text(&state.app, &final_text, paste_mode) {
                crate::log_line(&format!("自动输入失败：{e}"));
                let _ = state.app.emit("notice", e);
              }
            }
          }
          Err(e) => {
            crate::set_busy(&state.app, None);
            crate::log_line(&format!("识别失败：{e}"));
            let _ = state.app.emit("asr-error", serde_json::json!({ "id": id, "error": e }));
          }
        }
      }
    })
    .ok();
}

fn transcribe(
  state: &Arc<AppState>,
  cfg: &Config,
  job: Job,
) -> Result<(crate::asr::AsrResult, Option<String>, f64), String> {
  let mut samples = job.samples;
  if job.rate != 16000 {
    samples = crate::dsp::resample(&samples, job.rate, 16000)?;
  }

  // 降噪路径：16k → 48k → 降噪 → 16k
  if cfg.denoise {
    let mut guard = state.denoiser.lock().unwrap();
    if let Some(d) = guard.as_mut() {
      let up = crate::dsp::resample(&samples, 16000, 48000)?;
      let den = d.denoise(&up)?;
      samples = crate::dsp::resample(&den, 48000, 16000)?;
    }
  }

  let lang_owned = cfg.asr_lang.clone();
  let language = if lang_owned == "auto" || lang_owned.is_empty() {
    None
  } else {
    Some(lang_owned.as_str())
  };

  // 单模型直出：用户提示词进 chat 模板 system 段（context 通道），识别即优化，
  // 跳过 LLM 后处理段。基座包（无 unified.json）上提示词只作识别偏置，功能不炸。
  let single = cfg.single_model_enabled;
  let context = if single { cfg.single_model_context() } else { String::new() };

  let asr_result = {
    let mut guard = state.asr.lock().unwrap();
    let pipeline = guard.as_mut().ok_or("ASR 模型尚未加载完成")?;
    if cfg.denoise {
      pipeline.transcribe_single(&samples, language, &context, 512)?
    } else {
      pipeline.transcribe(&samples, language, &context, 512, 30)?
    }
  };

  if single || !cfg.llm_enabled || asr_result.text.trim().is_empty() {
    return Ok((asr_result, None, 0.0));
  }

  // LLM 未加载则按需加载
  if state.llm.lock().unwrap().is_none() {
    let paths = state.paths.lock().unwrap().clone();
    if let Some(dir) = paths.llm {
      state.set_status(|s| s.llm_loading = true);
      match LlmPipeline::load(&dir, crate::threads()) {
        Ok(p) => {
          *state.llm.lock().unwrap() = Some(p);
          state.set_status(|s| {
            s.llm_ready = true;
            s.llm_loading = false;
          });
        }
        Err(e) => {
          state.set_status(|s| s.llm_loading = false);
          return Ok((asr_result, None, 0.0));
        }
      }
    }
  }

  let system = if cfg.llm_preset == "custom" {
    cfg.llm_system_prompt.clone()
  } else {
    preset_prompt(&cfg.llm_preset)
      .map(|s| s.to_string())
      .unwrap_or_else(|| preset_prompt(crate::llm::PRESET_STANDARD).unwrap().to_string())
  };

  let t0 = std::time::Instant::now();
  let mut guard = state.llm.lock().unwrap();
  let Some(pipeline) = guard.as_mut() else {
    return Ok((asr_result, None, 0.0));
  };
  match pipeline.generate(&system, &asr_result.text, Sampling::default()) {
    Ok(text) if !text.trim().is_empty() => {
      let llm_s = t0.elapsed().as_secs_f64();
      Ok((asr_result, Some(text.trim().to_string()), llm_s))
    }
    Ok(_) => Ok((asr_result, None, t0.elapsed().as_secs_f64())),
    Err(e) => {
      crate::log_line(&format!("LLM 推理失败：{e}"));
      Ok((asr_result, None, t0.elapsed().as_secs_f64()))
    }
  }
}

fn finalize_text(_state: &Arc<AppState>, cfg: &Config, text: &str) -> String {
  if cfg.language == "zh" && cfg.remove_punc {
    const PUNC: &[char] = &[
      '，', '。', '！', '？', '、', '；', '：', '“', '”', '‘', '’', '（', '）', '【', '】', '《',
      '》', '…', '—', ',', '.', '!', '?', ';', ':', '"', '\'', '(', ')', '[', ']', '{', '}', '<',
      '>', '~', '·', '～',
    ];
    let mut out = String::with_capacity(text.len());
    let mut last_space = false;
    for ch in text.chars() {
      let c = if PUNC.contains(&ch) || ch.is_whitespace() { ' ' } else { ch };
      if c == ' ' {
        if !last_space {
          out.push(' ');
        }
        last_space = true;
      } else {
        out.push(c);
        last_space = false;
      }
    }
    return out.trim().to_string();
  }
  text.trim().to_string()
}
