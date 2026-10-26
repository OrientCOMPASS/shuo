# 说 · Shuo

Local speech recognition GUI on a **fine-tuned Qwen3-ASR single model** (this branch: llama.cpp/GGUF, 1.7B): ASR + prompt-driven text optimization in one pass. The standalone LLM post-processing stage (Qwen3.5-2B ONNX) was **removed** on this branch — see `finetune/README.md`.

Two implementations share the same `~/.shuo/` (config.json / history.json / shuo.log / custom_prompts.json):

- **Tauri 3 (new, primary)** — `src-tauri/`, all-Rust inference (`ort` + `cpal`), static HTML/CSS/JS UI, no Python at runtime.
- **Python / PySide6 (legacy)** — `asr_gui.py`, `onnx_infer/`, kept for reference and A/B comparison.

## Commands

| Action | Command |
|--------|---------|
| Run Tauri | `cd src-tauri && cargo run --release` |
| Build Tauri (Linux) | `cd src-tauri && cargo tauri build --bundles deb appimage` |
| 发版（日期版本） | `./scripts/release.sh`（版本 = `YY.MM.DD`，tag `vYY.MM.DD`；同日重复加 `-2`）|
| 版本注入（CI） | `bash ./scripts/set-version.sh <版本>`（只由 CI 从 tag 调用；仓库里不提交版本改动）|
| Build Tauri (Windows) | `powershell -ExecutionPolicy Bypass -File src-tauri\build.ps1` |
| Rust tests | `cd src-tauri && cargo test --release`（真实模型用例由 `SHUO_ASR_DIR` / `SHUO_GGUF_LM`+`SHUO_GGUF_MM`+`SHUO_GGUF_WAV` 环境变量门控，缺失自动跳过） |
| Run Python (legacy) | `cd legacy/2026-10-25-python && PYTHONPATH=../.. python asr_gui.py` |
| Install Python deps (legacy) | `pip install -r legacy/2026-10-25-python/requirements.txt` |
| Download ASR model (GGUF, 本分支优先) | `hf download ggml-org/Qwen3-ASR-1.7B-GGUF --include "Qwen3-ASR-1.7B-Q8_0.gguf" "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf" --local-dir Qwen3-ASR-1.7B-GGUF`（或从 pre-release `model-unified-1.7B-pre` 下载微调统一包） |
| Download ASR model | `hf download Daumee/Qwen3-ASR-0.6B-ONNX-CPU` (clone) | 
| Test ASR (Python) | `cd legacy/2026-10-25-python && python test_asr.py` |
| Test LLM (Python) | `cd legacy/2026-10-25-python && python test_llm.py` |
| Build Python installer | `legacy/2026-10-25-python/build.ps1` (requires 7-Zip) |
| i18n extract | `python -m i18n.build_i18n_locales extract` (scans `.py` files for `i18n.tr(...)`) |
| i18n generate | `python -m i18n.build_i18n_locales generate` (regenerates all 27 locale files) |
| i18n sync UI | `python -m i18n.build_i18n_locales sync-ui` (copies 27 locales + index.json into `src-tauri/ui/locales/`) |
| 微调数据 | `python finetune/prepare_data.py --source fleurs --optimizer stub --tasks polish_zh,translate_en`（详见 finetune/README.md） |
| 微调训练 | `python finetune/sft_lora.py --model Qwen/Qwen3-ASR-0.6B --train data/train.jsonl --out out/smoke --device cpu --lora` |
| 提示词对表 | `python finetune/check_prompt_sync.py`（prompts.py ↔ asr.rs 逐字一致，CI 强制） |
| ONNX 补丁 | `python finetune/patch_onnx_lora.py --model-dir … --adapter … --out-dir …-unified`（Gate 1 探针先行：inspect_onnx_lora.py） |
| 补丁包复测 | `python finetune/verify_unified_onnx.py --model-dir …-unified --audio x.wav --check-set unified` |

**China mirror:** Set `$env:HF_ENDPOINT = "https://hf-mirror.com"` before downloads.

## Architecture

### Tauri 3 (`src-tauri/`)

| File | Role |
|------|------|
| `src/main.rs` | Entry, commands (`bootstrap`/`save_config`/`set_hotkey`/`start_record`/`insert_text`/`list_asr_presets`…), tray, portal app-id registration |
| `src/asr.rs` | ASR pipeline (ONNX/ort): mel → conv/transformer encoders → decoder init/step → tokens; `prompt_ids()` 纯函数构造 chat 模板（system 段带 context=单模型直出任务开关）；`unified_preset_prompt()` 预设表（与 `finetune/prompts.py` 逐字对表） |
| `src/gguf_asr.rs` | ASR pipeline (GGUF/llama.cpp，本分支主路径): llama-cpp-sys-2 静态链接（openmp+mtmd），LM+mmproj 加载、mtmd 音频 tokenize/prefill、贪心解码、`language X<asr_text>` 解析；与 AsrPipeline 同接口（transcribe/transcribe_single → AsrResult） |
| `src/mel.rs` | Mel filters/log-mel/VAD splitting/AGC — numerically equal to librosa (max diff 2e-5) |
| `src/denoise.rs` | Streaming ONNX denoiser (NFT_960, 48 kHz) |
| `src/audio.rs` | cpal always-on capture + 0.5 s pre-buffer + 500 ms release debounce |
| `src/hotkey.rs` | Spec parsing (keyboard + mouse side buttons), platform dispatch |
| `src/hotkey_windows.rs` | Low-level keyboard/mouse hooks (press **and** release, injected-event filter) |
| `src/hotkey_linux.rs` | GlobalShortcuts portal (Wayland, primary) + evdev fallback (X11 / mouse buttons) |
| `src/typing_windows.rs` | SendInput + `KEYEVENTF_UNICODE` |
| `src/typing_linux.rs` | uinput virtual keyboard for paste shortcuts (no portal prompt) → RemoteDesktop (persistent restore token, re-request on revoke) → wtype/ydotool/xdotool → clipboard |
| `src/config.rs` | `~/.shuo/config.json` + history, field-compatible with Python; 单模型直出字段 `single_model_{preset,prompt}`（enable 开关已随双模型工作流移除；`verbatim` 预设=旧的不优化行为）+ `asr_backend`（auto/onnx/gguf）+ `Config::single_model_context()`；已废弃字段（llm_*、single_model_enabled）经 `extra` flatten 原样保留 |
| `src/models.rs` | Model dir resolution; ONNX 槽位优先 `Qwen3-ASR-0.6B-ONNX-CPU-unified/`（`unified.json` 标记）；GGUF 槽位 `Qwen3-ASR-1.7B-GGUF[-unified]/`（mmproj 前缀=音频编码器，最大 .gguf=LM）；`pick_backend()`/`unified_for()` |
| `src/state.rs` | AppState, loader/worker threads, events `engine-status`/`rec`/`asr`/`final`/`hotkey`; `AsrEngine` 双后端（Onnx/Gguf，加载失败自动回退）；context（优化提示词）恒随识别请求进 system 段，识别结果即最终文本（无 LLM 段），`final` 事件带 `unified` 标志；`Status.asr_backend/asr_unified` |
| `ui/` | Static frontend (index.html / style.css / app.js / locales)；设置抽屉含「单模型直出」区（开关/预设/自定义提示词）与「直出」状态 chip；ASR chip/引擎行显示后端（ONNX/GGUF） |
| `linux/70-shuo-hotkey.rules` | udev rules for evdev + uinput (`/dev/uinput` uaccess), installed by deb |
| `linux/shuo-overlay.py` | XWayland status capsule helper (bottom, always-on-top, no focus); bundled as resource |

### Python / PySide6 (legacy, `legacy/2026-10-25-python/`)

| File | Role |
|------|------|
| `legacy/2026-10-25-python/onnx_infer/asr.py` | ASR pipeline (0.6B): mel spectrogram → ONNX encoder → decoder init/step → text |
| `legacy/2026-10-25-python/onnx_infer/denoise.py` | Streaming ONNX denoiser (NFT_960, 48kHz input) |
| `legacy/2026-10-25-python/onnx_infer/denoise.onnx` | Denoiser model file |
| `legacy/2026-10-25-python/onnx_infer/llm.py` | Qwen3.5-2B ONNX post-processing (punctuation, de-um, correction) |
| `legacy/2026-10-25-python/asr_gui.py` | Entry point — PySide6 GUI, audio recording, hotkey, config, result queue |
| `legacy/2026-10-25-python/hotkey.py` | pynput global hotkey listener, persisted to `~/.shuo/config.json` |
| `i18n/__init__.py` | Minimal i18n: `i18n.load("zh")`, `i18n.tr("key")`, English fallback |
| `i18n/build_i18n_locales.py` | Scans `.py` for translation keys + generates all 27 locale files |

## Key facts

- **Model dirs in .gitignore:** `Qwen3-ASR-1.7B-GGUF[-unified]/`, `Qwen3-ASR-0.6B-ONNX-CPU[-unified]/` — must be downloaded separately（或从 pre-release 获取统一包）
- **Model lookup (Tauri):** config path → `SHUO_MODEL_DIR` (dir containing both model folders) → `~/.shuo/models/` → exe ancestors → cwd ancestors → repo dir → `/usr/lib/Shuo/models`；ASR 两个槽位：ONNX 先试 `Qwen3-ASR-0.6B-ONNX-CPU-unified/` 再试基座包名；GGUF 先试 `Qwen3-ASR-1.7B-GGUF-unified/` 再试 `Qwen3-ASR-1.7B-GGUF/`（目录内 mmproj 前缀 .gguf=音频编码器，其余最大 .gguf=LM）
- **ASR 后端选择（本分支）：** `asr_backend` 配置 auto/onnx/gguf；auto=有 GGUF 模型优先 GGUF（1.7B 主路径）；显式后端加载失败自动回退另一个（日志说明）；`Status.asr_backend` 上报 UI
- **llama.cpp 构建依赖：** cmake + C/C++ 编译器 + libclang（bindgen）；Linux 运行期依赖 libgomp1（deb depends 已加）；Windows 运行期依赖 VC++ 2015-2022 运行库（vcomp140，OpenMP）
- **单模型直出契约**：任务提示词 `finetune/prompts.py` ↔ `src-tauri/src/asr.rs::unified_preset_prompt` **逐字一致**（CI `finetune/check_prompt_sync.py` 强制）；提示词进 chat 模板 system 段（context 通道），基座只当识别偏置，微调统一包才执行指令；GGUF 后端同一契约（`gguf_asr.rs` 的 prompt 模板与 `asr.rs::prompt_ids` 逐 token 同构）
- **CI 布局**：`ci.yml`（push 必跑：rust ubuntu/windows 构建单测 + finetune 脚本纯逻辑校验 + UI 语法）；`finetune.yml`（仅手动：cpu-smoke/real-mini/gpu-train/onnx-inspect/onnx-patch/ort-e2e/gguf-e2/gguf-e2e）；`release.yml`（tag 发版）
- **Config at `~/.shuo/config.json`**: hotkey/hotkey_terminal (+ _enabled), language, ASR lang, denoise, auto_type, mic_device, asr_backend, single_model_{preset,prompt}
- **Log at `~/.shuo/shuo.log`**: UTF-8, `logger = logging.getLogger("shuo")` (Python) / `log_line()` (Rust)
- **History at `~/.shuo/history.json`**: max 500 entries, newest first
- **PyAudio MME only (Python):** GUI filters to MME devices; Chinese Windows may return GBK-encoded device names (fixed in app)
- **ASR output regex (Python):** `^language\s+(\w+)\s*<asr_text>(.*)` in `legacy/2026-10-25-python/onnx_infer/asr.py:147`
- **Recording:** Pre-buffer 0.5s rolling buffer, AGC peak-normalizes to -1 dBFS; release debounce 500ms
- **Auto-type:** Windows SendInput + KEYEVENTF_UNICODE; Linux clipboard + paste shortcut via uinput (primary) or RemoteDesktop portal (fallback)
- **Global hotkey:** Windows = low-level hooks (RegisterHotKey can't detect release); Linux = `GlobalShortcuts` portal (needs host app-id registration + `com.shuo.desktop.desktop`), evdev as fallback
- **Build:** Tauri bundler (`deb`/`appimage`/`nsis`); Python uses PyInstaller via `build.ps1`; hooks in `hooks/` block torch/triton at collection time
- **Theme:** Qt system color scheme (Python) / `prefers-color-scheme` (web UI)
- **No test framework (Python):** standalone `.py` test scripts (not pytest)
- **Audio denoiser model** at `src-tauri/resources/denoise.onnx` (bundled); legacy copy at `legacy/2026-10-25-python/onnx_infer/denoise.onnx`

## Inference threading

Tauri（本分支）：单一推理 worker 线程串行处理识别任务（context 恒注入，无第二段模型）。
legacy Python 版的 LLM 串行化说明（`_inference_lock`、prefix embedding 缓存）仅适用于
`legacy/2026-10-25-python/`，Tauri 版已无 LLM 段。

## i18n key patterns

- `settings.mic`, `settings.mic_default`, `settings.asr_lang_label`, `settings.asr_lang_tip`, `settings.ui_lang`
- `asr_lang.auto`, `asr_lang.zh`, ..., `asr_lang.sv` (27 languages)
- Dynamic keys in `i18n/build_i18n_locales.py` (`cmd_extract()`) — agent must add new ones there if introducing runtime-dynamic keys
- Tauri-only UI strings live in `src-tauri/ui/app.js` → `EXTRA` (zh/en; other locales fall back to English)
- Scaffold comparison (Tauri vs Electron, measured) lives in `docs/scaffold-comparison.md`; the Electron prototype was removed on 2026-10-25

## Build output

- Tauri: `src-tauri/target/release/bundle/{deb,appimage}/*`; Windows `bundle/nsis/Shuo_<version>_x64-setup.exe`
- Python `build.ps1`:
  - `dist/Shuo_yyyy-MM-dd-HHmm.exe` (full installer with models, 7z SFX)
  - `dist/Shuo_nomodel_yyyy-MM-dd-HHmm.exe` (without model dirs, for upgrade use)
