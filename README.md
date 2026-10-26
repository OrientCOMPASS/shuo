# 说 · Shuo

本地语音识别 GUI，基于微调 Qwen3-ASR（1.7B，llama.cpp/GGUF）**单模型直出**：语音识别与提示词驱动的文本优化一步完成。无需云端，无需 PyTorch。

On-device speech recognition GUI powered by a fine-tuned Qwen3-ASR (1.7B, llama.cpp/GGUF) **single model**: recognition and prompt-driven text optimization in one pass. No cloud, no PyTorch.

## 单模型直出（微调统一模型）

参考 [Qwen3-subtitle-assistant](https://github.com/OrientCOMPASS/Qwen3-subtitle-assistant) 的
S2TT 微调路线，本仓库的 `finetune/` 产线把 Qwen3-ASR（AuT 音频编码器来自
Qwen3-Omni）用 LoRA 微调成**一个模型完成「语音识别 + 按提示词优化文本」**——
用户设置的提示词进 chat 模板 system 段（context 通道），模型直出优化后的文本。
**本分支起应用只有这一条路径：独立的 LLM 后处理段（Qwen3.5-2B ONNX）已整体移除**
（旧双模型工作流见 `main` 分支与 `legacy/`）：

* 任务预设：逐字转写 / 中文书面化 / 英文书面化 / 直出中文 / 直出英文 / **简略（压缩要点）/ 详细（保留全部细节）** / 自定义提示词；
* 基座模型不执行 prompt 指令（官方防指令注入设计），**必须配合微调统一包**
  （目录带 `unified.json` 标记）才能兑现任务开关；基座包上运行不报错，
  提示词退化为识别偏置（选「逐字转写」预设可完全消除偏置）；
* 统一包产线全在 CI（LoRA 训练 → 权重落地 → 运行时行为复测 → pre-release 发布），
  方法与判定标准见 [`finetune/README.md`](./finetune/README.md)。

### 推理后端（本分支：llama.cpp/GGUF + 1.7B）

本分支（`feat/gguf-1.7b`）把 ASR 推理框架从 ort/ONNX 迁移到 **llama.cpp/GGUF**
（`llama-cpp-sys-2` 源码静态链接进可执行文件，无 DLL/so 伴生），并换用
**Qwen3-ASR-1.7B**（论文所称开源 ASR SOTA，ggml-org 官方 GGUF）：

| 后端 | 模型目录 | 说明 |
|---|---|---|
| `gguf`（本分支优先） | `Qwen3-ASR-1.7B-GGUF/`（官方包原样解压）或 `Qwen3-ASR-1.7B-GGUF-unified/`（gguf-e2 产线统一包） | 目录内 `mmproj` 前缀 `.gguf`=音频编码器，其余最大 `.gguf`=LM |
| `onnx`（保留） | `Qwen3-ASR-0.6B-ONNX-CPU[-unified]/` | Daumee 包；ort 另承担降噪模型 |

配置 `asr_backend`：`auto`（默认，有 GGUF 模型优先 GGUF）/ `onnx` / `gguf`；
所选后端模型缺失时自动回退另一个可用后端（日志说明）。UI 状态 chip 与引擎行
会显示当前后端（`ASR (GGUF)` / `ASR·ONNX`）。ort-e2e 与 gguf-e2e 两个 CI 模式
分别在真实模型上验证两条运行时链路（finetune.yml 手动触发）。

### 产物下载与体验测试（CI artifact + pre-release）

无需本地构建，全部从 CI / Release 获取：

| 产物 | 获取位置 | 内容 |
|---|---|---|
| 安装包（Windows NSIS / Linux deb + AppImage） | `release.yml` 手动 dispatch（不带 tag 只构建不发布），run 页 artifacts：`windows-nsis` / `linux-deb` / `linux-appimage` | 本分支构建（llama.cpp 静态链接） |
| 统一模型包（微调 1.7B GGUF，约 1.36 GiB） | ① **pre-release** `model-unified-1.7B-pre`（Releases 页，`gguf-publish` 模式发布，含 SHA256SUMS 与发布说明）；② 或任一 `gguf-e2` run 的 artifact `shuo-unified-gguf-1.7B` | LM Q4_K_M + mmproj q8_0 + `unified.json` 标记 + INSTALL.md |

体验步骤：安装应用 → 模型包整体放到 `~/.shuo/models/Qwen3-ASR-1.7B-GGUF-unified/`
（目录名保持一致）→ 设置 → 单模型直出 → 开启并选任务预设（书面化 / 翻译 /
简略 / 详细）→ 按住热键说话。基座包（无 unified.json）上开启该模式不报错，
提示词退化为识别偏置（日志有提示）。

## 两种实现 / Two implementations

| 实现 | 代码 | 说明 |
|---|---|---|
| **Tauri 3（新，主推）** | `src-tauri/` | 全 Rust 推理（`ort` + `cpal`），前端纯静态 HTML/CSS/JS；安装包不依赖 Python；Windows NSIS / Linux deb + AppImage |
| Python / PySide6（旧） | `asr_gui.py`、`onnx_infer/` | 原版实现，保留作参考与对照 |

两者共用 `~/.shuo/`（`config.json`、`history.json`、`shuo.log`），配置字段兼容，可以来回切换（旧版写入的 `llm_*`、`single_model_enabled` 等废弃字段由 Tauri 版原样保留不丢失）。

Both share `~/.shuo/`, so configs are interchangeable.

### Tauri 版快速开始

```bash
# 1) 模型放到仓库根目录（或任意位置，用 config 的 asr_model_dir 指定）
#    Qwen3-ASR-1.7B-GGUF/            ≈2.4 GB（官方 ggml-org GGUF：Q8_0 LM + mmproj，本分支优先）
#    Qwen3-ASR-1.7B-GGUF-unified/    ≈1.4 GB（微调统一包，见「产物下载与体验测试」）
#    Qwen3-ASR-0.6B-ONNX-CPU/        ≈2.5 GB（ONNX 后备后端，可选）

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
- **单模型直出**（设置 → 单模型直出）：微调统一包按 system 段提示词一步直出优化文本
  （识别 + 整理），这是本分支唯一的识别路径（独立 LLM 后处理段已移除）；
  任务预设与自定义提示词见 `finetune/prompts.py`。
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
│   ├── src/                         #   ASR（GGUF/ONNX 双后端）/ 降噪 / 音频 / 热键 / 上屏
│   ├── ui/                          #   index.html / style.css / app.js / locales
│   ├── resources/denoise.onnx       #   降噪模型（打包进 deb/NSIS）
│   ├── linux/                       #   udev 规则（evdev 热键后备）
│   ├── build.ps1                    #   Windows NSIS 打包
│   └── tauri.conf.json              #   deb / AppImage / NSIS 配置
├── finetune/                        # 单模型直出微调产线（数据/训练/评测/ONNX 补丁，见其 README）
├── legacy/2026-10-25-python/        # Python / PySide6 原版（归档，仅参考）
├── i18n/                            # 翻译生成/同步工具（共享）
├── locales/                         # 27 种语言源文件（共享，构建时同步到 ui/locales）
├── docs/                            # 文档（scaffold-comparison 等）
├── Qwen3-ASR-1.7B-GGUF[-unified]/  # ASR 模型 GGUF（gitignore，需下载/pre-release）
├── Qwen3-ASR-0.6B-ONNX-CPU[-unified]/ # ASR 模型 ONNX 后备（gitignore，需下载）
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

识别与文本优化模型 ASR/optimization model：[Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)（Apache 2.0；GGUF 转换版见 ggml-org）
