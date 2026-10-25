# 说 · Shuo

Local speech recognition GUI: Qwen3-ASR (ONNX CPU) + Qwen3.5 text post-processing. No GPU/PyTorch/cloud.

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
| Rust tests | `cd src-tauri && cargo test --release` (LLM e2e: `-- --ignored llm_end_to_end`) |
| Run Python (legacy) | `cd legacy/2026-10-25-python && PYTHONPATH=../.. python asr_gui.py` |
| Install Python deps (legacy) | `pip install -r legacy/2026-10-25-python/requirements.txt` |
| Download LLM | `hf download onnx-community/Qwen3.5-2B-ONNX-OPT --include "config.json" "tokenizer.json" "tokenizer_config.json" "generation_config.json" "chat_template.jinja" "processor_config.json" "preprocessor_config.json" "onnx/embed_tokens_q4f16.onnx" "onnx/embed_tokens_q4f16.onnx_data" "onnx/decoder_model_merged_q4f16.onnx" "onnx/decoder_model_merged_q4f16.onnx_data" --local-dir Qwen3.5-2B-ONNX-OPT` |
| Download ASR model | `hf download Daumee/Qwen3-ASR-0.6B-ONNX-CPU` (clone) | 
| Test ASR (Python) | `cd legacy/2026-10-25-python && python test_asr.py` |
| Test LLM (Python) | `cd legacy/2026-10-25-python && python test_llm.py` |
| Build Python installer | `legacy/2026-10-25-python/build.ps1` (requires 7-Zip) |
| i18n extract | `python -m i18n.build_i18n_locales extract` (scans `.py` files for `i18n.tr(...)`) |
| i18n generate | `python -m i18n.build_i18n_locales generate` (regenerates all 27 locale files) |
| i18n sync UI | `python -m i18n.build_i18n_locales sync-ui` (copies 27 locales + index.json into `src-tauri/ui/locales/`) |

**China mirror:** Set `$env:HF_ENDPOINT = "https://hf-mirror.com"` before downloads.

## Architecture

### Tauri 3 (`src-tauri/`)

| File | Role |
|------|------|
| `src/main.rs` | Entry, commands (`bootstrap`/`save_config`/`set_hotkey`/`start_record`/`insert_text`…), tray, portal app-id registration |
| `src/asr.rs` | ASR pipeline: mel → conv/transformer encoders → decoder init/step → tokens |
| `src/mel.rs` | Mel filters/log-mel/VAD splitting/AGC — numerically equal to librosa (max diff 2e-5) |
| `src/llm.rs` | Qwen3.5-2B ONNX post-processing, sampling, Qwen chat template (`<think>`/`</think>`), presets |
| `src/denoise.rs` | Streaming ONNX denoiser (NFT_960, 48 kHz) |
| `src/audio.rs` | cpal always-on capture + 0.5 s pre-buffer + 500 ms release debounce |
| `src/hotkey.rs` | Spec parsing (keyboard + mouse side buttons), platform dispatch |
| `src/hotkey_windows.rs` | Low-level keyboard/mouse hooks (press **and** release, injected-event filter) |
| `src/hotkey_linux.rs` | GlobalShortcuts portal (Wayland, primary) + evdev fallback (X11 / mouse buttons) |
| `src/typing_windows.rs` | SendInput + `KEYEVENTF_UNICODE` |
| `src/typing_linux.rs` | uinput virtual keyboard for paste shortcuts (no portal prompt) → RemoteDesktop (persistent restore token, re-request on revoke) → wtype/ydotool/xdotool → clipboard |
| `src/config.rs` | `~/.shuo/config.json` + history, field-compatible with Python |
| `src/state.rs` | AppState, loader/worker threads, events `engine-status`/`rec`/`asr`/`final`/`hotkey` |
| `ui/` | Static frontend (index.html / style.css / app.js / locales) |
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

- **Model dirs in .gitignore:** `Qwen3-ASR-0.6B-ONNX-CPU/`, `Qwen3.5-*-ONNX-OPT/` — must be downloaded separately
- **Model lookup (Tauri):** config path → `SHUO_MODEL_DIR` (dir containing both model folders) → `~/.shuo/models/` → exe ancestors → cwd ancestors → repo dir → `/usr/lib/Shuo/models`
- **Config at `~/.shuo/config.json`**: hotkey/hotkey_terminal (+ _enabled), language, ASR lang, denoise, auto_type, mic_device, LLM settings
- **Log at `~/.shuo/shuo.log`**: UTF-8, `logger = logging.getLogger("shuo")` (Python) / `log_line()` (Rust)
- **History at `~/.shuo/history.json`**: max 500 entries, newest first
- **PyAudio MME only (Python):** GUI filters to MME devices; Chinese Windows may return GBK-encoded device names (fixed in app)
- **ASR output regex (Python):** `^language\s+(\w+)\s*<asr_text>(.*)` in `legacy/2026-10-25-python/onnx_infer/asr.py:147`
- **Recording:** Pre-buffer 0.5s rolling buffer, AGC peak-normalizes to -1 dBFS; release debounce 500ms
- **Auto-type:** Windows SendInput + KEYEVENTF_UNICODE; Linux clipboard + paste shortcut via uinput (primary) or RemoteDesktop portal (fallback)
- **LLM presets:** standard / moderate / aggressive / aggressive_no_punc / translate
- **LLM prompt:** must match `chat_template.jinja` with `enable_thinking=false` exactly (ends with `<think>\n\n</think>\n\n`); guarded by `llm::tests::chat_prompt_token_ids_match_python`
- **Global hotkey:** Windows = low-level hooks (RegisterHotKey can't detect release); Linux = `GlobalShortcuts` portal (needs host app-id registration + `com.shuo.desktop.desktop`), evdev as fallback
- **Build:** Tauri bundler (`deb`/`appimage`/`nsis`); Python uses PyInstaller via `build.ps1`; hooks in `hooks/` block torch/triton at collection time
- **Theme:** Qt system color scheme (Python) / `prefers-color-scheme` (web UI)
- **No test framework (Python):** standalone `.py` test scripts (not pytest)
- **Audio denoiser model** at `src-tauri/resources/denoise.onnx` (bundled); legacy copy at `legacy/2026-10-25-python/onnx_infer/denoise.onnx`

## LLM threading

`legacy/2026-10-25-python/onnx_infer/llm.py` 的 `_inference_lock` serializes all `call_llm()` invocations. The model singleton loads lazily at first call. Cache prefix embeddings per system prompt — changing prompt triggers re-cache. Tauri: single inference worker thread serializes ASR→LLM.

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
