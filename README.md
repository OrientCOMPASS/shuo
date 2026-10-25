# 说 · Shuo

本地语音识别 GUI，基于 Qwen3-ASR-0.6B ONNX CPU 推理 + Qwen3.5-2B ONNX 文本优化。无需 GPU，无需 PyTorch，无需云端。

On-device speech recognition GUI powered by Qwen3-ASR-0.6B ONNX CPU pipeline with Qwen3.5-2B ONNX text post-processing. No GPU, no PyTorch, no cloud.

## 两种实现 / Two implementations

| 实现 | 代码 | 说明 |
|---|---|---|
| **Tauri 3（新，主推）** | `src-tauri/` | 全 Rust 推理（`ort` + `cpal`），前端纯静态 HTML/CSS/JS；安装包不依赖 Python；Windows NSIS / Linux deb + AppImage |
| Python / PySide6（旧） | `asr_gui.py`、`onnx_infer/` | 原版实现，保留作参考与对照 |

两者共用 `~/.shuo/`（`config.json`、`history.json`、`shuo.log`、`custom_prompts.json`），配置字段兼容，可以来回切换。

Both share `~/.shuo/`, so configs are interchangeable.

### Tauri 版快速开始

```bash
# 1) 模型放到仓库根目录（或任意位置，用 config 的 asr_model_dir / llm_model_dir 指定）
#    Qwen3-ASR-0.6B-ONNX-CPU/   ≈2.5 GB
#    Qwen3.5-2B-ONNX-OPT/       ≈1.4 GB

# 2) 运行
cd src-tauri
cargo run --release

# 3) 打包（Linux）
cargo tauri build --bundles deb appimage     # dist: target/release/bundle/{deb,appimage}

# 4) 打包（Windows，需在 Windows 上）
powershell -ExecutionPolicy Bypass -File src-tauri\build.ps1
```

- **全局热键**：两个独立热键、各自带开关——
  - 主热键（默认 `F2`）→ 识别后用 **Ctrl+V** 上屏，适合图形界面；
  - 终端热键（默认 `F3`）→ 识别后用 **Ctrl+Shift+V** 上屏，适合终端 / TUI。
- **热键实现（Linux）**：优先走 `xdg-desktop-portal` 的 **GlobalShortcuts**（GNOME/KDE Wayland 官方通道，首次需在系统对话框里允许一次）；portal 不可用或要绑鼠标侧键时用 **evdev**（`/dev/input/event*`，需要 input 组权限或 `70-shuo-hotkey.rules`，deb 已自带）。
- **自动输入（Linux）**：剪贴板 + 粘贴快捷键，快捷键由触发录音的热键决定（Ctrl+V / Ctrl+Shift+V）。**首选 uinput 虚拟键盘**（不需要任何授权框，deb 的 udev 规则会放开 `/dev/uinput`）；uinput 不可用时退回 RemoteDesktop 门户（持久化 restore token，被撤回会自动重新申请）。
- **状态胶囊（Linux）**：屏幕底部（任务栏上方）显示「录音中 + 波形 / 识别中 + 转圈」，XWayland 小窗置顶、不抢焦点、鼠标穿透；不需要 GNOME 扩展，也不需要注销重登录。
- **自动输入（Windows）**：`SendInput` + `KEYEVENTF_UNICODE`，不经过剪贴板，终端 / Vim 也能用。
- **模型查找顺序**：配置项 → `SHUO_MODEL_DIR` 环境变量（目录下放两个模型文件夹）→ `~/.shuo/models/` → 可执行文件上溯 → 仓库目录。


## 快速开始 / Quick Start

| 平台 / Platform | 指南 / Guide | ASR 模型下载 |
|---|---|---|
| Windows | [中文](./docs/windows.md) / [English](./docs/windows-en.md) | `hf download Daumee/Qwen3-ASR-0.6B-ONNX-CPU` (clone) |
| Linux | [中文](./docs/linux.md) / [English](./docs/linux-en.md) | `hf download Daumee/Qwen3-ASR-0.6B-ONNX-CPU` (clone) |

下载后解压到项目根目录。

## 项目文件 / Project Structure

```
├── src-tauri/                       # Tauri 3 主版本（Rust 推理 + 静态前端）
│   ├── src/                         #   ASR / LLM / 降噪 / 音频 / 热键 / 上屏
│   ├── ui/                          #   index.html / style.css / app.js / locales
│   ├── resources/denoise.onnx       #   降噪模型（打包进 deb/NSIS）
│   ├── linux/                       #   udev 规则（evdev 热键后备）
│   ├── build.ps1                    #   Windows NSIS 打包
│   └── tauri.conf.json              #   deb / AppImage / NSIS 配置
├── legacy/2026-10-25-python/        # Python / PySide6 原版（归档，仅参考）
├── i18n/                            # 翻译生成/同步工具（共享）
├── locales/                         # 27 种语言源文件（共享，构建时同步到 ui/locales）
├── docs/                            # 文档（scaffold-comparison 等）
├── Qwen3-ASR-0.6B-ONNX-CPU/        # ASR 模型（gitignore，需下载）
├── Qwen3.5-2B-ONNX-OPT/            # LLM 模型（gitignore，需下载）
└── onnx_infer/                      # （已移到 legacy，仅历史）
```

## 配置 / Configuration

应用启动后自动创建 `~/.shuo/` 目录：

App creates `~/.shuo/` on first launch:

| 文件 File | 说明 Description |
|---|---|
| `config.json` | 语言、快捷键、自动输入 Language, hotkey, auto-type |
| `history.json` | 识别历史（最多 500 条）History (max 500) |
| `shuo.log` | 运行日志（UTF-8）Runtime log (UTF-8) |

- 默认快捷键：鼠标侧键（后退），可在左上角「快捷键」按钮自定义
- Default hotkey: mouse side button (back), customizable via top-left button
- 右上角下拉框切换中文 / English
- Language switch: top-right dropdown

## 依赖 / Dependencies

| 库 Library | 用途 Purpose |
|---|---|
| PySide6 | GUI 框架 Framework |
| qtawesome | Font Awesome 图标 Icons |
| pynput | 全局热键监听 Global hotkey listener |
| PyAudio | 音频录制 Audio recording |
| onnxruntime | ONNX 模型推理 Model inference |
| numpy | 数值计算 Numerical computing |
| tokenizers | 文本分词 Text tokenization |

## 许可证 / License

代码 Code：GPL-3.0

### Qt / PySide6

本应用使用 PySide6（Qt for Python），依据 **LGPL v3** 协议发布。可在专有软件中动态链接，无需开源。

This app uses PySide6 (Qt for Python) under **LGPL v3**. Dynamic linking in proprietary software is permitted without open-sourcing your code.

完整协议 Full license: https://www.gnu.org/licenses/lgpl-3.0.html

### 第三方库 / Third-party Libraries

| 库 Library | 协议 License |
|---|---|
| qtawesome | MIT |
| pynput | LGPL-3.0 |
| PyAudio | MIT |
| onnxruntime | MIT |
| numpy | BSD-3 |
| tokenizers | Apache-2.0 |

### 模型 / Models

模型权重版权归原作者所有，详见 [NOTICE](./NOTICE)。

Model weights: copyright belongs to original authors. See [NOTICE](./NOTICE).

文本优化模型 Text model：[Qwen3.5-2B-ONNX-OPT](https://huggingface.co/onnx-community/Qwen3.5-2B-ONNX-OPT)（Apache 2.0）
