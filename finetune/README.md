# Qwen3-ASR「单模型直出」微调产线（识别 + 提示词驱动文本优化）

目标：把 Qwen3-ASR（AuT 音频编码器来自 Qwen3-Omni，LM 为 Qwen3 解码器）微调成
**一个模型完成语音识别 + 按用户提示词优化文本**，替代「ASR → LLM(2B) 后处理」
双模型管线（main 分支保留双模型可切换；**本分支 feat/gguf-1.7b 已把独立 LLM
后处理段整体移除，应用收敛为单模型一条路径**，qsa E3 同款手术）。方法论与全部关键结论移植自
[Qwen3-subtitle-assistant](https://github.com/OrientCOMPASS/Qwen3-subtitle-assistant)
（下称 qsa）的 S2TT 实验线——该仓库已在 CI 全链路实测：LoRA 微调 → 任务开关成立
（带 prompt 直出目标语言、不带 prompt 仍转写、静音双模式为空）→ int8 ONNX 权重
补丁（250/250 张量）→ 产品运行时复测通过 → GGUF 产线（合并→转换→量化→直出）。

## 为什么必须微调、以及为什么可行

* **提示词通道存在但基座不执行指令**：Qwen3-ASR 的 chat 模板 system 段（context
  通道）会被基座当作识别先验/热词偏置；官方论文明确 SFT 阶段把模型训练成
  "不遵循 prompt 里的自然语言指令"（防指令注入）。qsa T4 双路径实测（量化 GGUF +
  未量化原始权重）复现了该结论：注入翻译指令前后输出几乎逐字相同。
  **所以「按提示词优化文本」不能靠提示词工程，只能改权重。**
* **改权重是安全的**：官方微调脚本的数据管线 `target = ex["text"]` 不校验 text
  是否为音频转写——把样本写成「音频 + prompt=任务指令 + text=期望输出」即可把
  模型训练成按指令直出。AuT 音频塔表征语言无关，主要动的是 LM 解码器在该表征
  条件下的输出分布；LoRA 小步微调 + 混入 verbatim/静音样本可保住原生能力。
* **任务开关**：prompt 进 system 段 ⇒ 一部分样本带任务 prompt（学优化/翻译）、
  一部分不带（学原生转写），一套权重靠 context 切换任务；`--asr-ratio` 控遗忘、
  `--silence-samples` 保「静音→空输出」（qsa 实测不混静音样本 4 epoch 就训坏）。

## 任务集（提示词 = 训练/推理契约）

`prompts.py` 是**唯一事实源**；Rust 推理端 `src-tauri/src/asr.rs` 的
`unified_preset_prompt()` 必须与之逐字一致（CI `check_prompt_sync.py` 强制对表）：

| key | system 段提示词 | 行为 |
|---|---|---|
| `verbatim` | （空串） | 原生逐字转写（抗遗忘锚点） |
| `polish_zh` | 转为书面文本：修正标点，去除口癖，保持原意。 | 中文书面化 |
| `polish_en` | Rewrite as clean written text: fix punctuation, remove fillers, keep the meaning. | 英文书面化 |
| `translate_zh` | translate to Chinese | 直出中文 |
| `translate_en` | translate to English | 直出英文 |
| `brief_zh` | 简略风格：压缩为要点，尽量简短。 | 风格轴·简略（压缩要点） |
| `detail_zh` | 详细风格：保留全部细节，输出完整书面文本。 | 风格轴·详细（保留全部细节） |

风格轴（brief/detail）的评测判据是**去空白字符数比**，且分母是**该任务的参考目标**
（不是源转写）：brief 输出/目标 ∈ [0.30, 1.20]（学得完美 ≈1.0；病态信号是泄漏成
源文全文 ≈1.5+，0.6B 欠训练实测 1.28）、detail ≥0.95（`eval_unified.py::TASK_LEN`；
FLEURS 字符流转写含空格，全字符口径没法比）。造表契约另有口径：brief 目标/源文
≤0.80（`build_table` 断言）。在朗读语料上 detail 目标与 polish 同文（本无冗余可删），
对比度由 brief 承担；口语数据上两者才真正分化。

自定义提示词（UI custom）：效果取决于微调数据是否覆盖相同/相近指令；放量训练
（gpu-train）时把常用自定义指令加进 `--tasks` 即可一并学会。

输出格式与基座一致：`language {源语言}<asr_text>{正文}`（translate 任务的正文是
目标语言，语言标签仍为源语言——qsa S2TT 实测行为）；静音输出
`language None<asr_text>` + 空正文，产品端据此剔除噪音段。

## 文件

| 文件 | 作用 |
|---|---|
| `prompts.py` | 任务提示词唯一事实源（训练=推理契约） |
| `check_prompt_sync.py` | CI 对表：prompts.py ↔ asr.rs 逐字一致 |
| `prepare_data.py` | 造数据：FLEURS（默认 cmn_hans_cn 普通话）或自备 wav → JSONL；优化后端 `none`/`stub`/`api`/`table`；`--silence-samples` 混静音样本；`--dump-transcripts` 导出转写供离线整理对照表 |
| `sft_lora.py` | 训练：官方数据管线 + peft LoRA + CPU 兜底（qsa 同款，任务无关） |
| `eval_unified.py` | 评测：任务生效（按脚本占比/相似度）/ 未遗忘 / 整条泄漏计数 / 静音双模式；`--silence-only` 基座归因 |
| `merge_lora_hf.py` | LoRA 合并回 HF（GGUF 产线用，feat/gguf-1.7b 分支） |
| `inspect_onnx_lora.py` | Gate 1 探针：Daumee int8 包张量地图 + 适配器值匹配 go/no-go |
| `patch_onnx_lora.py` | Gate 2 补丁器：ΔW 写回 Daumee 包 → `*-unified/` 目录 + `unified.json` 标记 |
| `verify_unified_onnx.py` | Gate 3 复测：onnxruntime 上验证补丁包行为（与 src-tauri/src/asr.rs 同构管线，支持 context） |

## 运行时集成

```
录音 → （降噪，可选）→ ASR 引擎（system 段带任务提示词）→ language X<asr_text>正文
        GGUF 后端: mtmd 音频编码 + 贪心解码（本分支主路径，1.7B）        └─ 正文即最终文本
        ONNX 后端: mel → encoder → decoder 贪心（main 分支 / 0.6B 后备）
```

* `~/.shuo/config.json`：`single_model_preset` / `single_model_prompt`（本分支起
  **无 enable 开关**——单模型是唯一路径，`verbatim` 预设 = 空 context = 旧「不优化」
  行为；旧配置的废弃字段经 `extra` flatten 原样保留）；
* 模型目录解析：GGUF 槽位优先 `Qwen3-ASR-1.7B-GGUF-unified/`（gguf-e2 产物，带
  `unified.json`），其次官方基座 `Qwen3-ASR-1.7B-GGUF/`；ONNX 槽位同理
  （`Qwen3-ASR-0.6B-ONNX-CPU[-unified]/`）。`asr_backend`：auto（默认，有 GGUF
  优先 GGUF）/ onnx / gguf，所选后端缺模型自动回退另一个；
* 基座包上运行不会报错——提示词退化为识别偏置，日志会提示"当前不是统一包"；
* UI（设置 → 单模型直出）：任务预设 + 自定义提示词 + 包状态提示；顶栏「直出」chip
  常显（绿=统一包 / 黄=基座包），ASR chip 与引擎行显示当前后端（GGUF/ONNX）。

## 跑法

### A. CPU 管线冒烟（不需要 GPU；CI mode=cpu-smoke 就是这个）

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install qwen-asr peft datasets soundfile librosa

python finetune/prepare_data.py --source fleurs --fleurs-lang cmn_hans_cn \
    --limit 4 --eval-limit 2 --max-audio-secs 12 \
    --optimizer stub --tasks polish_zh,translate_en \
    --asr-ratio 0.4 --silence-samples 2 \
    --audio-dir data/audio --out data/train.jsonl --eval-out data/eval.jsonl

python finetune/sft_lora.py --model Qwen/Qwen3-ASR-0.6B --train data/train.jsonl \
    --out out/smoke --device cpu --lora --lora-r 8 --max-samples 4 --max-steps 2 --batch-size 1

python finetune/eval_unified.py --model Qwen/Qwen3-ASR-0.6B --adapter out/smoke/lora \
    --eval data/eval.jsonl --limit 2 --device cpu --source-script cjk \
    --out out/smoke/eval_report.json
```

只证明数据格式 / prefix-target 切分 / 训练回路 / 适配器保存与挂载 / 评测脚本跑得通。
**stub 是确定性伪优化，产出的模型没有任何真实优化能力**，质量指标不达标是预期。

### B. ONNX 补丁（Gate 1 → Gate 2 → Gate 3，全在 CI）

```bash
pip install onnx numpy safetensors onnxruntime tokenizers librosa soundfile huggingface_hub
hf download Daumee/Qwen3-ASR-0.6B-ONNX-CPU --local-dir models/Qwen3-ASR-0.6B-ONNX-CPU

# Gate 1：探针（值匹配可行性 go/no-go）
python finetune/inspect_onnx_lora.py --model-dir models/Qwen3-ASR-0.6B-ONNX-CPU \
    --hf-model Qwen/Qwen3-ASR-0.6B --adapter out/real/lora \
    --out out/inspect/onnx_patch_report.json

# Gate 2：补丁 → unified 目录（含 unified.json 标记）
python finetune/patch_onnx_lora.py --model-dir models/Qwen3-ASR-0.6B-ONNX-CPU \
    --hf-model Qwen/Qwen3-ASR-0.6B --adapter out/real/lora \
    --out-dir models/Qwen3-ASR-0.6B-ONNX-CPU-unified --report out/patch/patch_report.json

# Gate 3：onnxruntime 行为复测（补丁后必须在最终运行时形态上复测——qsa T4 教训）
python finetune/verify_unified_onnx.py --model-dir models/Qwen3-ASR-0.6B-ONNX-CPU-unified \
    --audio zh.wav --reference refs.tsv --check-set unified --expect-task translate_en
```

补丁器对 Daumee 包的三处布局适配（相对 qsa 的 k2-fsa 包）：onnx 在 `onnx_models/`
子目录；encoder fp32 权重带外置 `.data`（保存时重新外置，产物同构）；
**decoder_init 与 decoder_step 共享 LM 权重，两处都要补**（去重按文件内而非全局）。

### Daumee 包 Gate 1 首跑实测（run 40231057836，cpu-smoke 适配器 r=8）

| 事实 | 数值 | 含义 |
|---|---|---|
| 张量地图 | decoder_{init,step}.int8 各 197 个可匹配量化权重（名字被导出器匿名化为 `val_N_quantized`）；encoder_transformer 110 个 fp32；encoder_conv 2 个 | 值匹配（不依赖名字）是唯一可行路线 |
| 量化摆位 | decoder 为 **per-tensor 动态 int8**（反量化噪声中位 2.86e-2、最大 >5e-2，比 k2-fsa per-channel 高一个量级）；encoder fp32 无损 | 绝对误差阈值会误杀 → `match_one` 增加**唯一性兜底**（次优 > 4×最优 且最优 < 0.30 即接受；错层候选 ~1.0 量级，判别裕度仍 >6×），`--max-match-rel` 默认放宽到 0.15，同源判据 `--max-src-rel` 默认 0.06 |
| 重量化削顶 | 最坏 0.0001% | 按原 scale/zp 直接写回，无需重算（GO） |
| init/step 重复权重 | 同一 HF key 在两个 decoder 文件各匹配一次（rel_err 完全相同） | 探针 flags 里的「被多个 ONNX 张量匹配」是预期现象；补丁器按文件内去重、两处都写 |
| tied lm_head/embed | 个别张量 best==second 完全相等 | LoRA 不碰 lm_head/embed，补丁器按 mod_by_hfkey 过滤，无害（qsa 在 k2-fsa 包上同样观测到） |

### C. 真实训练（GPU 放量 / CPU real-mini）

* `real-mini`（CI 手动触发）：`--optimizer table` + 仓库内对照表
  `finetune/table_zh.tsv`（三列：`task<TAB>源转写<TAB>目标文本`），免 API key。
  对照表制作流程：先跑 cpu-smoke → artifact 里取 `data/transcripts.txt` →
  离线整理 → 提交 TSV → 触发 real-mini（qsa 用 120 对人工对照表验证过该路线：
  0.6B + LoRA r=32 + 4 epoch 纯 CPU 约 38 分钟，任务开关三项判定全过）。
* `gpu-train`（需 self-hosted gpu runner + `secrets.SHUO_OPT_API_KEY` +
  `vars.SHUO_OPT_API_BASE/SHUO_OPT_API_MODEL`）：`--optimizer api`，
  优化目标的 system 消息 = 任务提示词**原文**（训练/部署语义闭环）。
  放量建议 ≥2000 条源音频、静音样本约 10%。

判定标准（real-mini/gpu-train 三项必须同时成立，口径同 qsa）：
1. **任务生效**：带任务 prompt 时输出朝指令方向变化（translate_* 看目标脚本占比，
   polish_* 看非空同脚本）；
2. **没有遗忘**：verbatim 输出与源转写相似（difflib ≥0.55）且无整条泄漏；
3. **静音行为**：任务/转写双模式都输出 `language None` + 空文本（基座归因：
   若基座本身对纯数字静音非空，只警示不判红——qsa 实测过该归因流程）。

## 已知坑（qsa 实测记录，直接继承）

* `qwen-asr` 锁 `transformers==4.57.6`、`accelerate==1.12.0`；
* pyarrow 后台线程在解释器 finalize 阶段 SIGABRT（退出码 134）——脚本用
  `_hard_exit` 绕过，产物先写盘再 `os._exit`；
* 不要让 datasets 自己解码音频（torchcodec 依赖坑）——`cast_column(Audio(decode=False))`
  + soundfile；
* 静音样本太少保不住「静音→空输出」（qsa 实测 0 条即被训坏）；
* 合并回 HF 时 `generation_config` 的贪心采样参数要先清洗（temperature=1e-6 会被
  transformers 严格校验拒绝）——`merge_lora_hf.py` 已处理；
* CI runner 控制台 cp1252：workflow 全局 `PYTHONUTF8=1`；
* int8 量化会改变模型行为：**补丁后必须在 ONNX 运行时复测**，不能只信 PyTorch 侧；
* ONNX 导出 KV cache 固定长度的包（如 sherpa-onnx 系 512 token）单段音频时长有
  上限；Daumee 包 KV 走 I/O 无该硬限，但仍建议配合产品的静音切分（chunk 30s）。

## CI（finetune.yml，全部手动触发）

| mode | 干什么 | 预计耗时 |
|---|---|---|
| `cpu-smoke` | stub 冒烟（数据/训练/评测回路），产适配器 artifact | 30–50 min |
| `real-mini` | CPU 真实微调（table 对照表），三项判定强制 | 1–3 h |
| `gpu-train` | GPU 放量训练（api 优化后端，self-hosted） | 视数据量 |
| `onnx-inspect` | Gate 1：Daumee 包探针（需先有适配器 artifact 的 run id） | ~20 min |
| `onnx-patch` | Gate 2+3：补丁 + ORT 行为复测 + 基线对照 | ~40 min |
| `ort-e2e` | 产品运行时：真实包 + `cargo test`（Rust ort 管线带 context） | ~40 min |
| `gguf-e2` | GGUF 产线（feat/gguf-1.7b）：合并→转换→Q4_K_M→llama-server 探针；**产物 artifact `shuo-unified-gguf-<尺寸>`（LM+mmproj+unified.json+INSTALL.md，可直接落盘体验）** | ~60 min |
| `gguf-e2e` | 产品运行时（GGUF 路线）：Rust 后端 + 官方 1.7B Q8_0 真实转写 cargo test | ~60 min |
| `gguf-publish` | 把 gguf-e2 缓存的统一包发布为 GitHub **pre-release**（tag `model-unified-<尺寸>-pre`：GGUF×2 + unified.json + SHA256SUMS + 发布说明；重跑=资产覆盖更新） | ~15 min |

产品回归（push 即跑）在 `.github/workflows/ci.yml`：Rust 构建+单测（ubuntu/windows）
+ finetune 脚本纯逻辑校验 + 提示词对表 + UI 语法。

## 实测记录（2026-10-25，GitHub 托管 runner，全链路首次跑通）

### real-mini 0.6B（run 40235990485）：四项判定全过，任务开关学成

数据：FLEURS cmn_hans_cn 前 120 条（≤15s）+ 仓库内 120 句人工对照表
（`table_zh.tsv`：polish_zh 书面化 + translate_en 英译，**无任何 API key**）；
训练 100 条、评测 20 条（音频不进训练集）+ 静音 20 条。
模型 Qwen3-ASR-0.6B + LoRA r=32 α=64，4 epoch，batch 1×grad-acc 2，lr 3e-4，
float32 全程 CPU：**240 步 / 37.6 分钟 / final loss 0.347**（整轮 job 约 47 分钟）。

| 判定项 | 结果 | 数值 |
|---|---|---|
| 任务生效 polish_zh | ✔ | 与书面化参考相似度 0.9475，口癖命中 0，20/20 非空 |
| 任务生效 translate_en | ✔ | 拉丁占比 **92.5%**、汉字 2.9%——语音直出英文 |
| 没有遗忘 verbatim | ✔ | 相似度 0.844、汉字 94.7%、整条泄漏 0/20 |
| 静音行为（双模式） | ✔ | 任务/转写两模式均 `language None` + 空文本（基座同样为空） |

直出样例（模型输出，非 LLM 后处理；括号内为对照表参考）：

> [translate_en] Sharks do exist, but they rarely attack humans; most sha…
> （Sharks do exist, but they rarely attack humans; most sharks are afraid of people and swim away.——几乎逐字）
> [polish_zh] 拍摄野生动物通常会用到长焦镜头，不过拍鸟、群或微小生物之类的对象则需要其他镜头。
> （源转写是「拍 摄 野 生 动 物 …」字符流——并字+标点一步直出）

质量评估（诚实版）：120 句对照表足以让**任务开关**稳定成立（带 prompt 换任务、
不带 prompt 保转写、静音保空）；polish/translate 的输出主体正确，个别词有漂移
（「拍鸟群」→「拍鸟、群」）。放量（gpu-train，≥2000 句 + api 后端）可提精度，
判定闸与产线不变。

### Gate 1/2/3 ONNX 补丁链（Daumee int8 包，产品运行时形态）

| 闸 | run | 结果 |
|---|---|---|
| Gate 1 探针 | 40233236069 | 250/250 模块映射；同源 rel_err 中位 3.11e-2 / 最坏 1.01e-1（per-tensor 动态 int8 噪声量级，双阈值判定）；削顶 0.0001% → **全 GO** |
| Gate 2 补丁（smoke 适配器） | 40235082851 | decoder_init 196 + decoder_step 196 + 音频塔 54 = **250/250 补丁**，坏元素 0，6.5 分钟；unified.json 产出 |
| Gate 3 行为复测（**real 适配器**） | 40240621011 | zh.wav（训练未见过）：verbatim → `开放时间：早上9点至下午5点。`；translate_en → **`Broadcast time: 9:00 AM to 5:00 PM.`（int8 量化后任务开关仍成立）**；静音双模式空；基座包对照 → 带不带 context 输出几乎相同（印证「基座只把 context 当偏置」） |
| Gate 3 Rust 运行时 | 40231060419 | `SHUO_ASR_DIR cargo test`：LibriSpeech 转写逐字正确 + context 冒烟通过 |

### 踩坑速记（均已收敛进代码/工作流）

* FLEURS 普通话配置名是 `cmn_hans_cn`（`zh_cn` 不存在，ValueError 实测清单）；
* Daumee decoder 是 **per-tensor 动态 int8**：反量化噪声比 k2-fsa per-channel 高
  一个量级 → `match_one` 加唯一性兜底（次优>4×最优），同源判据改「中位+最坏」双阈值；
* `decoder_init/step` 共享 LM 权重：补丁去重必须**按文件内**（全局去重会漏掉 step，
  模型直接精分）；
* onnx 外置数据检测：protobuf repeated 字段没有 presence，`HasField("external_data")`
  直接 ValueError，用 `len()`；
* 脚本占比要**字母归一**：`Open time: 9:00 AM to 5:00 PM.` 全字符口径拉丁 58%
  （被数字稀释）误判未生效，字母口径 100%；
* Windows debug 构建 esaxx C++ 静态 CRT 与 Rust 动态 CRT 混链 LNK2005 → CI 一律 release；
* LoRA 适配器不能跨基座尺寸挂载（0.6B→1.7B 音频塔 18×896 vs 24×1024 形状爆炸），
  merge 前 fail-fast 校验 adapter_config 的 base_model；
* 2 步 stub 适配器的静音行为无质量含义（实测 no-context 幻觉《小豬的夢》），
  冒烟场次（expect=any）静音非空仅记录警告，硬判定留给真实适配器场次。

## GGUF 路线实测记录（feat/gguf-1.7b，2026-10-25，全链路首次跑通）

### real-mini 1.7B（run 40237414373）：四项判定全过，全面优于 0.6B

同一张 120 句对照表（`table_zh.tsv`，免 API key）、同配方（LoRA r=32 α=64、
4 epoch、lr 3e-4、batch 1×grad-acc 2、float32 纯 CPU）：
**240 步 / 48.3 分钟 / final loss 0.236**（0.6B 为 37.6 分钟 / 0.347）。

| 判定项 | 0.6B | **1.7B** |
|---|---|---|
| verbatim 相似度（≥0.55） | 0.844 | **0.906**（汉字 95.3%，整条泄漏 0/20） |
| polish_zh 相似度 / 口癖命中 | 0.9475 / 0 | **0.9573 / 0** |
| translate_en 拉丁占比（≥0.60） | 92.5% | **95.5%**（汉字残留 0.19%） |
| 静音双模式为空 | ✔ | ✔ |

1.7B 直出样例（评测集，音频未进训练）：
> [translate_en] Photographing wildlife usually involves a telephoto lens…
> [polish_zh] 拍摄野生动物通常会用到长焦镜头，不过拍鸟群或微小生物之类的对象则需要其他镜头。
> （0.6B 在同句上把「拍鸟群」断成「拍鸟、群」——1.7B 无此漂移）

### gguf-e2（run 40242476360）：1.7B 真实适配器全产线直出英文

merge（2.04B 参数，bf16 CPU 合并 4s）→ 官方转换器（TEXT f16 + MMPROJ q8_0）→
llama-quantize Q4_K_M：**LM 1.03 GiB + mmproj 0.33 GiB = 1.36 GiB**（与 qsa E2 体积账
一致，单包 ≤2GiB 预算成立）。llama-server b11201 探针（zh.wav，训练未见过）：

| 场次 | 输出 | 判定 |
|---|---|---|
| 带 context "translate to English" | **"Open hours: 9:00 am to 5:00 pm."**（拉丁 60%/汉字 0%） | ✔ 直出英文 |
| 不带 context | 「开饭时间早上九点至下午五点。」（汉字 93%） | ✔ 任务开关保住转写 |
| 静音 + context | `language None<asr_text>` 空 | ✔ |
| 静音不带 context | `language None<asr_text>` 空 | ✔ 穿越 merge→转换→Q4_K_M 全链 |

### gguf-e2e（run 40235119900）：Rust 产品运行时 + 官方 1.7B GGUF

`GgufAsr`（llama-cpp-sys-2 =0.1.157 静态链接，mtmd 音频路径）加载官方
ggml-org/Qwen3-ASR-1.7B-Q8_0（2.0GiB）+ mmproj-Q8_0（0.33GiB）：
LibriSpeech 参考 wav 转写逐字正确（"Stuff it into you. His belly counselled him."，
4 vCPU 上 2.5s），context 冒烟通过；`cargo test --release gguf` 双平台 CI 常跑
（llama.cpp 源码静态编译 ubuntu/windows 两腿均过，无 DLL/so 伴生）。

### GGUF 路线踩坑速记

* `llama_context_default_params` 的 `n_threads/n_threads_batch/n_batch` 在 pin 版本是
  i32/u32 混合——照 qsa 的 bindgen 实测签名写，不要猜；
* `llama_token_to_piece` pin 版本签名 `(vocab, token, buf, length, lstrip, special)`，
  special=true 才会带出 `<asr_text>` 特殊 token 字面量（解析依赖它）；
* mtmd 日志三个入口都要桥接（llama_log_set / ggml_log_set / mtmd_helper_log_set），
  少一个就有逐段计时行直写 stderr；
* b11201 的 convert `--mmproj` 按 `--outfile` 原名落盘（不自动加 mmproj- 前缀）；
* ubuntu 预编译 llama 二进制是 .tar.gz（windows 才是 .zip）。
