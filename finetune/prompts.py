#!/usr/bin/env python3
"""单模型直出的提示词预设——训练与推理的唯一事实源。

这些短提示词进 Qwen3-ASR chat 模板的 **system 段**（context 通道）：

    <|fim_prefix|>system\n{prompt}<|fim_suffix|>\n<tool_response>user\n<|audio_bos|>…<|audio_eos|></tool_response>\n<|fim_prefix|>assistant\n

基座模型被刻意训练成**不遵循** prompt 里的自然语言指令（Qwen3-ASR 技术报告 §2.2，
qsa 仓库 T4 双路径实测复现），只把它当识别先验/热词偏置——所以「按提示词优化文本」
必须微调权重兑现。训练数据的 `prompt` 字段用的就是这里的字符串；推理端
（`src-tauri/src/asr.rs` 的 `unified_preset_prompt`）必须**逐字一致**，
CI 用 `check_prompt_sync.py` 强制对表。

任务设计（与 shuo 的 LLM 预设概念对齐，但为 system 段短提示词形态）：
  verbatim     空串 = 纯转写（基座原生行为，抗遗忘锚点）
  polish_zh    中文书面化：修标点、去口癖
  polish_en    英文书面化
  translate_zh 直出中文（源语言任意，如英文口述 → 中文上屏）
  translate_en 直出英文（中文口述 → 英文上屏）
  brief_zh     简略风格：压缩为要点（风格轴：短）
  detail_zh    详细风格：保留全部细节的完整书面文本（风格轴：长）

风格轴说明：brief/detail 与 polish 的区别在**长度/信息密度**——brief 输出显著短于
源转写（评测按去空白字符数比 ≤0.75 判），detail 完整保留（比值 ≥0.95）。
在 FLEURS 朗读语料上 detail 目标与 polish 目标同文（朗读句本无冗余可删），
风格对比度由 brief 承担；口语数据上 detail 的「保留全部细节」与 polish 的
「去除口癖」才真正分化。

自定义提示词（UI 的 custom）：效果取决于微调数据是否覆盖相同/相近指令，
放量训练时建议把用户常用指令一并写进数据（--tasks 支持任意 key=prompt 追加）。
"""

from __future__ import annotations

# key -> system 段提示词全文。verbatim 固定为空串（不进 PRESETS，语义特殊）。
VERBATIM = ""
VERBATIM_KEY = "verbatim"

PRESETS: dict[str, str] = {
    "polish_zh": "转为书面文本：修正标点，去除口癖，保持原意。",
    "polish_en": "Rewrite as clean written text: fix punctuation, remove fillers, keep the meaning.",
    "translate_zh": "translate to Chinese",
    "translate_en": "translate to English",
    "brief_zh": "简略风格：压缩为要点，尽量简短。",
    "detail_zh": "详细风格：保留全部细节，输出完整书面文本。",
}

# 任务类型：评测与数据构造按类型分组（polish=同语言书面化；translate=跨语言直出；
# style=长度/信息密度风格轴，评测按长度比判）
TASK_KIND: dict[str, str] = {
    "polish_zh": "polish",
    "polish_en": "polish",
    "translate_zh": "translate",
    "translate_en": "translate",
    "brief_zh": "style",
    "detail_zh": "style",
}

# 基座模型对静音/无语音音频的原始输出（qwen_asr 解析成 language=""、text=""）。
# 静音样本的训练目标必须与之逐 token 一致，才能保住「静音 -> 空输出」行为。
SILENCE_TAG = "language None<asr_text>"


def lang_tag(source_lang: str) -> str:
    """训练目标前缀：`language {源语言}<asr_text>`（模型输出源语言标签，
    即使 translate 任务的正文已是目标语言——qsa S2TT 实测行为一致）。"""
    return f"language {source_lang}<asr_text>"


def all_prompts() -> dict[str, str]:
    out = {VERBATIM_KEY: VERBATIM}
    out.update(PRESETS)
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(all_prompts(), ensure_ascii=False, indent=2))
