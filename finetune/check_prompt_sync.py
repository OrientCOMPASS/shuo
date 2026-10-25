#!/usr/bin/env python3
"""CI 对表：finetune/prompts.py 与 src-tauri/src/asr.rs 的提示词必须逐字一致。

Rust 端 `unified_preset_prompt()` 的 match 臂是提示词在推理侧的唯一来源，
训练侧（prompts.py）换了措辞而推理端没跟上 = 模型没学过的任务 = 直出失效。
本脚本解析 asr.rs 的字符串字面量并与 prompts.py 对表，任何差异都以非零退出。

用法（CI finetune-py job）：
  python finetune/check_prompt_sync.py
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import PRESETS, VERBATIM  # noqa: E402

RUST_KEY_MAP = {
    "UPRESET_VERBATIM": "verbatim",
    "UPRESET_POLISH_ZH": "polish_zh",
    "UPRESET_POLISH_EN": "polish_en",
    "UPRESET_TRANSLATE_ZH": "translate_zh",
    "UPRESET_TRANSLATE_EN": "translate_en",
}

ARM_RE = re.compile(r'^\s*(UPRESET_\w+)\s*=>\s*"((?:[^"\\]|\\.)*)"\s*,\s*$')


def unescape_rust(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            n = s[i + 1]
            mapping = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'"}
            if n in mapping:
                out.append(mapping[n])
                i += 2
                continue
        out.append(c)
        i += 1
    return "".join(out)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    asr_rs = root / "src-tauri" / "src" / "asr.rs"
    if not asr_rs.is_file():
        print(f"✘ 找不到 {asr_rs}")
        return 1

    rust: dict[str, str] = {}
    text = asr_rs.read_text(encoding="utf-8")
    # 只看 unified_preset_prompt 函数体内的 match 臂
    fn = re.search(r"pub fn unified_preset_prompt.*?\n\}", text, re.S)
    if not fn:
        print("✘ asr.rs 里没有 unified_preset_prompt 函数")
        return 1
    for line in fn.group(0).splitlines():
        m = ARM_RE.match(line)
        if m and m.group(1) in RUST_KEY_MAP:
            rust[RUST_KEY_MAP[m.group(1)]] = unescape_rust(m.group(2))

    expect = dict(PRESETS)
    expect["verbatim"] = VERBATIM

    fails = []
    for key, want in expect.items():
        got = rust.get(key)
        if got is None:
            fails.append(f"{key}: asr.rs 缺少对应 match 臂")
        elif got != want:
            fails.append(
                f"{key}: 不一致\n    prompts.py: {want!r}\n    asr.rs    : {got!r}"
                f"\n    NFC 归一后相同？ {unicodedata.normalize('NFC', got) == unicodedata.normalize('NFC', want)}"
            )
    extra = set(rust) - set(expect)
    if extra:
        fails.append(f"asr.rs 多出 prompts.py 没有的预设: {sorted(extra)}")

    if fails:
        print("✘ 提示词对表失败（训练/推理两端必须逐字一致）：")
        for f in fails:
            print("  -", f)
        return 1
    print(f"✔ 提示词对表通过：{len(expect)} 个预设（verbatim + {len(PRESETS)} 个任务）逐字一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
