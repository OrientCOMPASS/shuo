# Python / PySide6 版（2026-10-25 归档）

这是「说 · Shuo」的 Python 原版实现（PySide6 + onnxruntime + PyAudio），**已归档**，保留作参考与行为对照，不再作为主版本维护。

主版本在仓库根目录的 **`src-tauri/`**（Tauri 3 + 全 Rust 推理，安装包不依赖 Python）。

## 归档内容

| 文件/目录 | 说明 |
|---|---|
| `asr_gui.py` | PySide6 主窗口、录音、结果队列、设置 |
| `hotkey.py` | pynput 全局热键（按下 / 松开） |
| `onnx_infer/` | Python 推理：ASR / LLM / 降噪（含 `denoise.onnx`） |
| `test_asr.py` / `test_llm.py` | 独立测试脚本 |
| `requirements.txt` | Python 依赖 |
| `build.ps1` | PyInstaller + 7z SFX 打包脚本 |
| `hooks/` | PyInstaller hooks（屏蔽 torch/triton） |
| `7z/` | 7-Zip SFX 模块（打包用） |
| `docs/` | 原 Windows / Linux 使用指南 |

## 还想跑旧版的话

共享资源（`i18n/`、`locales/`、模型目录）仍在仓库根目录。Python 代码按 `__file__` 的上一级找模型，所以：

```bash
cd legacy/2026-10-25-python
ln -s ../../Qwen3-ASR-0.6B-ONNX-CPU .
ln -s ../../Qwen3.5-2B-ONNX-OPT .
pip install -r requirements.txt
PYTHONPATH=../.. python asr_gui.py
```

> 模型目录体积很大（ASR ≈2.5 GB，LLM ≈1.4 GB），且不在 git 里；没有模型时只能看界面，不能识别。

## 与 Tauri 版的关系

- 共用 `~/.shuo/`（`config.json` / `history.json` / `shuo.log` / `custom_prompts.json`），配置字段兼容。
- Tauri 版的 `src-tauri/src/asr.rs`、`llm.rs` 是对这里 `onnx_infer/asr.py`、`llm.py` 的 Rust 移植，数值已对齐（mel 2e-5、encoder 8.7e-7、LLM prompt 逐 token 一致）。
- 归档日期：**2026-10-25**
