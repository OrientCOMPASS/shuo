# Changelog

## 未发布（feat/gguf-1.7b：推理框架迁移 + 1.7B）

### 新增 — 风格任务提示词 + CI 产物下载 + pre-release 分发
- **风格轴任务预设**：`brief_zh`（简略风格：压缩为要点，尽量简短。）与 `detail_zh`（详细风格：保留全部细节，输出完整书面文本。）——UI 下拉自动出现（预设表驱动）；`prompts.py` ↔ `asr.rs` 对表扩展；评测新增**去空白长度比判据**（brief ≤0.80、detail ≥0.95，`eval_unified.py::TASK_LEN`，FLEURS 字符流转写下唯一公平口径）；stub 优化器补风格分支（CI 纯逻辑单测覆盖确定性与长度形态）
- **对照表扩容**：`table_zh.tsv` 240 → **480 行**（120 源句 × 4 任务：polish_zh / translate_en / brief_zh 人工压缩要点（均值长度比 0.652）/ detail_zh），real-mini 可单跑风格轴（--tasks brief_zh,detail_zh）或四任务合训
- **CI 产物下载（实际体验测试）**：`gguf-e2` 产出 artifact `shuo-unified-gguf-<尺寸>`（LM Q4_K_M + mmproj q8_0 + `unified.json` 标记 + INSTALL.md，落盘即用）；安装包走 `release.yml` 手动 dispatch（artifacts：windows-nsis / linux-deb / linux-appimage）
- **pre-release 分发**：新增 `gguf-publish` 模式——从 gguf-e2 跨 OS 缓存还原统一包，生成 unified.json / SHA256SUMS / 发布说明，发布到 GitHub pre-release（tag `model-unified-<尺寸>-pre`，重跑=资产覆盖更新）

### 实测记录（2026-10-25，全链路 CI 跑通）
- **real-mini 1.7B**（run 40237414373）：120 句对照表 + LoRA r=32，240 步 / 48.3 min / loss 0.236，四项判定全过——verbatim 相似度 0.906（泄漏 0/20）、polish_zh 0.957（口癖 0）、translate_en 拉丁 **95.5%**、静音双模式空；全面优于 0.6B（loss 0.347 / latin 92.5%），无「拍鸟、群」级断词漂移
- **gguf-e2**（run 40242476360）：1.7B 真实适配器 → merge(2.04B) → 官方转换器 → **Q4_K_M 1.03GiB + mmproj q8_0 0.33GiB = 1.36GiB**；llama-server 探针（训练未见过音频）：带 context 直出英文 "Open hours: 9:00 am to 5:00 pm."、不带 context 仍中文转写（任务开关成立）、静音双模式 `language None`+空（穿越量化全链）
- **gguf-e2e**（run 40235119900）：Rust `GgufAsr`（llama-cpp-sys-2 静态链接）加载官方 1.7B Q8_0，参考 wav 转写逐字正确（4 vCPU 2.5s），context 冒烟过
- **CI 双平台**：llama.cpp 源码静态编译 ubuntu + windows 两腿全绿（bindgen 需 libclang，CI 已自动化）

### 变更 — ASR 推理框架迁移到 llama.cpp/GGUF（换用 Qwen3-ASR-1.7B）
- **新后端 `src/gguf_asr.rs`**：llama-cpp-sys-2（=0.1.157，openmp+mtmd）**源码静态链接**进可执行文件（无 DLL/so 伴生），加载官方 `ggml-org/Qwen3-ASR-1.7B-GGUF`（LM+mmproj）；mtmd 音频 tokenize/prefill + 贪心解码 + `language X<asr_text>` 解析；与 ONNX 路径同接口（AsrResult），长音频复用静音切分；原生 C 日志桥接（默认只放行 ERROR，`SHUO_LLAMA_LOG=1` 全量）
- **双后端调度**：`config.json` 新增 `asr_backend`（auto/onnx/gguf，默认 auto=有 GGUF 模型优先 GGUF）；`models.rs` GGUF 目录探测（`Qwen3-ASR-1.7B-GGUF[-unified]/`，mmproj 前缀约定）；`state.rs` `AsrEngine` 枚举统一调度，加载失败自动回退另一后端；`Status.asr_backend` 上报，UI chip/引擎行显示 ONNX/GGUF
- **保留**：ort 继续承担降噪模型与 LLM 后处理段（双模型工作流不变）；ONNX ASR 路径完整保留（`asr_backend: "onnx"` 可显式回退）
- **打包**：deb depends 增加 `libgomp1`（llama.cpp OpenMP）；Windows 运行期依赖 VC++ 2015-2022 运行库（vcomp140）
- **CI**：ci.yml / release.yml 各腿补 libclang 准备（bindgen）；finetune.yml 新增 `gguf-e2`（LoRA 合并→llama.cpp 官方转换器→Q4_K_M→llama-server 探针，跨 OS 缓存产物）与 `gguf-e2e`（Rust 后端 + 官方 1.7B Q8_0 真实转写 cargo test）
- **finetune**：新增 `gguf_server_probe.py`（qsa E1/E2 同款探针，context/静音/对照断言，expect=any/latin/cjk）

## 未发布（main 开发中）

### 新增 — 单模型直出（微调 Qwen3-ASR，识别 + 提示词优化一步完成）
- **`finetune/` 微调产线**（方法论移植自 Qwen3-subtitle-assistant 的 S2TT 实验线，全链路已在其 CI 实测）：
  - `prepare_data.py`：FLEURS（默认 zh_cn）/自备 wav → 多任务 JSONL（prompt 进 chat 模板 system 段）；优化后端 none/stub/api/table；静音样本混入（保「静音→空输出」行为）
  - `sft_lora.py`：官方数据管线 + peft LoRA + CPU 兜底；`eval_unified.py`：任务生效/未遗忘/整条泄漏/静音双模式四项评测
  - `inspect_onnx_lora.py` + `patch_onnx_lora.py`：Gate 1/2——LoRA ΔW 按值匹配写回 Daumee int8 ONNX 包（适配 `onnx_models/` 子目录、外置 `.data`、decoder_init/step 共享权重双写），产出 `*-unified/` 目录 + `unified.json` 标记
  - `verify_unified_onnx.py`：Gate 3——onnxruntime 上与 Rust 管线同构的行为复测（支持 context）
  - `prompts.py` + `check_prompt_sync.py`：任务提示词唯一事实源，与 Rust 端逐字对表（CI 强制）
- **运行时（ort 保持不变）**：`asr.rs` 提示词构造重构为纯函数 `prompt_ids()`，system 段支持 context；`config.json` 新字段 `single_model_{enabled,preset,prompt}`（旧配置兼容）；`models.rs` 优先解析统一包；`state.rs` 单模型路径跳过 LLM 段
- **前端（Tauri UI）**：设置抽屉新增「单模型直出」区（开关/任务预设/自定义提示词/包状态提示），顶栏「直出」chip，结果卡「单模型直出」徽标
- **CI**：新增 `ci.yml`（push 必跑：rust ubuntu+windows 构建单测、finetune 脚本纯逻辑校验、提示词对表、UI 语法）与 `finetune.yml`（手动：cpu-smoke / real-mini / gpu-train / onnx-inspect / onnx-patch / ort-e2e）

## 26.10.25 — 2026-10-25

> 版本号自本版起改为日期 `YY.MM.DD`，tag 为 `v26.10.25`；同日多次发版自动加 `-2`、`-3`。

### 新增 — Tauri 3 全 Rust 重写（`src-tauri/`）
- **推理全部搬到 Rust**：`ort` 加载 Qwen3-ASR-0.6B（mel → conv → transformer → decoder init/step）与 Qwen3.5-2B（embed + decoder + 混合线性注意力状态），不再需要 Python / PyTorch / onnxruntime-python
- **数值对齐**：mel 滤波器与 librosa 一致（max diff 3.7e-9），mel 谱 2.1e-5，encoder 8.7e-7；LLM chat prompt 与 `chat_template.jinja`（`enable_thinking=false`）逐 token 对齐（`llm::tests::chat_prompt_token_ids_match_python`）
- **原生音频**：cpal 常开采集 + 0.5 s 预缓冲 + 500 ms 收尾，AGC 归一化到 -1 dBFS
- **历史记录**：默认开启，文件与内存都限制 500 条；关闭时界面明确提示
- **界面**：小录音按钮 + 顶部结果卡片 + 下方历史占主要空间；历史时间单独一行，避免与正文重叠
- **状态胶囊**：底部（任务栏上方）显示「录音中 + 不规则波形」/「识别中 + 转圈」；XWayland 小窗置顶、不抢焦点、鼠标穿透，不需要 GNOME 扩展或注销
- **Linux 上屏改用 uinput 虚拟键盘**：不再每次启动弹 RemoteDesktop「共享」授权；uinput 不可用时退回门户（restore token 记住授权，被撤回会自动重新申请）
- **前端**：纯静态 HTML/CSS/JS（`src-tauri/ui/`），复用 27 种语言包；设置抽屉、历史、结果卡片、托盘
- **Windows 热键**：低级键鼠钩子（`WH_KEYBOARD_LL` / `WH_MOUSE_LL`），支持按住说话与鼠标侧键，过滤自身注入事件
- **Linux 热键**：优先 xdg-desktop-portal `GlobalShortcuts`（GNOME/KDE Wayland，按下+松开信号），portal 不可用时 evdev 直读 `/dev/input/event*`；deb 自带 udev 规则
- **两个独立热键**（各带开关）：主热键（默认 F2）→ Ctrl+V 上屏（图形界面）；终端热键（默认 F3）→ Ctrl+Shift+V 上屏（终端 / TUI）
- **Linux 上屏**：剪贴板 + 粘贴快捷键（门户/工具发键）；逐字符 keysym 因会被当前输入法改写而不再使用
- **Windows 上屏**：`SendInput` + `KEYEVENTF_UNICODE`（与 Python 版 `send_unicode_text` 同一套）
- **打包**：Linux deb（16 MB，含降噪模型、udev 规则、门户桌面文件）/ AppImage（96 MB）；Windows NSIS 配置 + `build.ps1`
- **门户集成**：启动时注册 host app id（`com.shuo.desktop`），并确保 `com.shuo.desktop.desktop` 存在（AppImage 也能用）

### 变更
- 与 Python 版共用 `~/.shuo/`（config/history/log/custom_prompts），配置字段兼容
- 模型不随安装包分发：配置路径 → `SHUO_MODEL_DIR` → `~/.shuo/models` → 可执行文件上溯 → 仓库目录

### 记录
- 新增 [Tauri vs Electron 脚手架对比](docs/scaffold-comparison.md)：同一套推理/前端下比较输入、上屏、包体、内存；结论保留 Tauri，Electron 对照实现已删除

## 0.1.11 — 2026-06-27


### 新增
- **麦克风选择下拉框**：工具栏可选择录音设备，选择持久化到配置
- **预缓冲录音**：持续维护 0.3s 滚动缓冲，按下时回溯截取，解决 MME 启动慢丢前音问题
- **自动增益控制 (AGC)**：录音后 peak 归一化到 -1 dBFS
- **自动采样率适配**：按设备原生采样率录音，ASR pipeline 自动重采样到 16kHz
- **中文设备名修复**：修复 PortAudio 在中文 Windows 上返回 GBK 乱码设备名
- **设置菜单**：齿轮图标下拉菜单，集中管理所有设置项

### 优化
- **UI 精简**：去掉底部栏和 hint 文字，录音按钮移入 toolbar，窗口更紧凑
- **下拉框统一**：ASR 语言和麦克风下拉框统一 140px 宽度，带文字标签说明用途
- **设备列表只显示 MME**：移除 WASAPI（兼容性问题），去掉 API 前缀
- **i18n**：新增 `settings.mic` / `settings.mic_default` / `settings.asr_lang_label` / `settings.asr_lang_tip` / `settings.ui_lang` 到全部 28 个语言文件

## 0.1.10 — 2026-06-03

### 修复
- **首次启动语言显示错误**：配置为中文但界面显示英文，因 `i18n.load()` 在 UI 控件创建后才调用，已提前至 `__init__` 开头
- **托盘右键退出未随语言切换刷新**：`quit_action` 未存为成员，`_switch_lang` 未更新其文字

## 0.1.9 — 2026-06-03

### 新增
- **27 语言全覆盖**：UI 界面 + ASR 识别语言下拉框全部国际化，每个语言一个 locale 文件，默认英文界面
- **ASR 识别语言选择**：工具栏下拉框，支持 27 种语言 + 自动检测，传入模型作为语言参数
- **识别排队**：识别中再录音自动排队，当前识别完成后自动处理下一条
- **中文专属去标点**：中文界面下可开启"去标点替换为空格"，其他语言隐藏

### 修复
- **线程崩溃**：识别中再按侧键导致 QThread 销毁时崩溃

### 优化
- **历史加载顺序**：最新记录显示在最上面
- **README 目录结构**：文件夹在前、文件在后，按字母序排列
- **i18n 回落机制**：英文兜底，新增语言不会显示 key 名

## 0.1.8 — 2026-06-03

### 修复
- **i18n 翻译不完整**：清理 locales 中 12 个未使用的过期 key；切换语言时遗漏刷新历史记录、清空、关于、退出等控件，已补全

### 优化
- **onnx_inference.py 移至根目录**：不再放在 `Qwen3-ASR-0.6B-ONNX-CPU/` 下载目录内，避免被覆盖；模型目录完全 gitignore
- **根目录恢复 mel_filters.npy**：下载目录中的 npy 含 pickle 数据导致加载失败，改用根目录自带的纯净版本，无需 `allow_pickle`

## 0.1.7 — 2026-06-03

### 修复
- **内存泄漏**：录音 PCM buffer 即时清空、旧线程 `wait()` + `deleteLater()`、临时 WAV 用完即删、卡片上限 100 条自动淘汰
- **ONNX 内存**：关闭 `enable_cpu_mem_arena`，推理内存从数百 MB 降到几十 MB，不再随录音次数增长

## 0.1.6 — 2026-06-03

### 隐私
- 日志不再记录识别内容
- 历史记录默认关闭

## 0.1.5 — 2026-06-03

### 优化
- **砍掉 librosa 依赖**：音频加载改用 stdlib `wave`，STFT 改用 `numpy.fft`，mel 滤波器预计算为 `mel_filters.npy`
- **砍掉 scipy / soundfile / numba**：全部替换为 numpy + stdlib，只用 `onnxruntime + numpy + tokenizers` 三个外部依赖

### 新增
- 应用图标 `shuo.ico`

## 0.1.4 — 2026-06-03

### 修复
- 打包命令加回 `--hidden-import librosa`，修复运行时模块缺失

## 0.1.3 — 2026-06-03

### 优化
- 打包命令加 `--exclude-module torch/sklearn/numba/llvmlite/tensorflow`，产物从 7.5GB 降到 2.7GB
- 打包命令从多行改为单行，避免 PowerShell 续行符问题

## 0.1.2 — 2026-06-03

### 修复
- 打包命令写入 README，支持 Windows/macOS/Linux
- 补 `--hidden-import librosa`

## 0.1.1 — 2026-06-03

### 新增
- 语言切换改为按钮式：🌐 图标 + ZH/EN 文字，点击弹出菜单

### 修复
- 补充 `QGridLayout` 导入，修复关于对话框无法打开
- 语言下拉框翻译 key 未加载的问题

## 0.1.0 — 2026-06-03

### 新增
- 系统托盘图标，支持最小化到托盘
- 全局热键监听（键盘 + 鼠标侧键）
- 中/英文界面切换，完整 i18n 覆盖（47 个 key）
- 关于对话框，含依赖列表和许可证信息
- 识别结果保存到本地历史（最多 500 条）
- 自动输入模式：识别完成后自动粘贴到前台窗口

### 优化
- **主题系统重写**：显式颜色方案，切换瞬间响应，亮/暗模式完全跟随 Windows 系统设置
- **UI 全面美化**：圆形录音按钮（系统强调色）、卡片式结果项（左 accent 色条 + hover 复制按钮）、旋转弧线加载动画、原生 Win11 风格滚动条
- **启动加速**：去掉冗余初始化，模型后台加载，界面立即可见
- **窗口美化**：恢复最小化按钮、系统阴影、Win11 圆角，标题栏跟随主题色

### 修复
- 主题切换时背景色不跟随的问题
- 录音按钮 hover 颜色消失的问题
- 结果卡片 hover 时文字跳动的问题
- 浅色主题下复制图标不可见的问题
- 翻译文本未加载导致界面显示 key 名的问题
