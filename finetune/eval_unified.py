#!/usr/bin/env python3
"""评测「单模型直出」微调效果：任务开关是否学成、原生转写是否保住、静音行为是否保留。

四项检查（判定标准与阈值见各 --min/--max 参数；qsa 仓库 real-mini 实测方法论）：
  1. **任务生效**：带任务 prompt 时输出朝指令方向变化——
     translate_* 看目标文字脚本占比（如 translate_en 输出拉丁占比要高）；
     polish_* 看输出仍为源语言书面文本（非空、脚本不变、与参考有可比性）；
  2. **没有遗忘**：不带 prompt（verbatim）时输出与源转写高度相似
     （difflib 相似度 + 源语言脚本占比）；
  3. **无整条泄漏**：verbatim 组里被任务"吸走"（输出脚本变成目标语言）的样本单列计数
     ——均值达标也可能掩盖个别样本泄漏（qsa 实测 20 条里出现过 1 条）；
  4. **静音行为**：喂纯静音应输出 language None / 空文本，**任务/转写两种模式都要过**
     （产品端依赖该行为剔除噪音段；--silence-only 单测静音，供基座归因诊断）。

用法：
  python finetune/eval_unified.py --model Qwen/Qwen3-ASR-0.6B --adapter out/real/lora \
      --eval data/eval.jsonl --limit 20 --device cpu --out out/real/eval_report.json
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import PRESETS, TASK_KIND, VERBATIM_KEY  # noqa: E402

KANA = re.compile(r"[\u3040-\u30ff]")
CJK = re.compile(r"[\u4e00-\u9fff]")
LATIN = re.compile(r"[A-Za-z]")
FILLERS_ZH = ("嗯", "呃", "就是说", "那个", "对吧", "反正")


def log(msg: str) -> None:
    print(f"[eval] {msg}", flush=True)


def _hard_exit(code: int) -> "NoReturn":
    """绕过解释器 finalization（pyarrow 后台线程 SIGABRT，qsa CI 实测坑）。"""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


def _ratio(pattern: re.Pattern, text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if pattern.match(c)) / len(chars)


def script_profile(text: str) -> dict:
    return {
        "cjk": round(_ratio(CJK, text), 4),
        "latin": round(_ratio(LATIN, text), 4),
        "kana": round(_ratio(KANA, text), 4),
    }


def sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


# 各任务的"生效"判据：输出应呈现的脚本（dominant script 下限）
TASK_SCRIPT = {
    "translate_en": ("latin", 0.60),
    "translate_zh": ("cjk", 0.60),
    "polish_en": ("latin", 0.60),
    "polish_zh": ("cjk", 0.50),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    ap.add_argument("--adapter", default="", help="LoRA 适配器目录（留空则评测基座模型）")
    ap.add_argument("--eval", default="", help="prepare_data.py 产出的评测 JSONL")
    ap.add_argument("--limit", type=int, default=20, help="每组最多评测条数")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--silence-secs", type=float, default=2.0, help="静音测试时长；0=跳过")
    ap.add_argument("--silence-only", action="store_true",
                    help="只跑静音测试（基座归因诊断：区分「微调训坏」与「本来就幻觉」）")
    ap.add_argument("--min-verbatim-sim", type=float, default=0.55,
                    help="verbatim 组输出与参考的平均相似度下限（低于判可能遗忘）")
    ap.add_argument("--source-script", choices=["cjk", "latin", "kana"], default="cjk",
                    help="源语言主导脚本（verbatim 保真与泄漏判定用）")
    ap.add_argument("--max-leak", type=int, default=0,
                    help="verbatim 组允许的最大整条泄漏样本数")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not args.silence_only and not args.eval:
        ap.error("需要 --eval（或改用 --silence-only）")
    if args.silence_only and args.silence_secs <= 0:
        ap.error("--silence-only 需要 --silence-secs > 0")

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    import numpy as np
    import torch
    from qwen_asr import Qwen3ASRModel

    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    dtype = torch.bfloat16 if (use_cuda and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float32
    log(f"device={'cuda' if use_cuda else 'cpu'} dtype={dtype} adapter={args.adapter or '(无，评测基座)'}")

    wrapper = Qwen3ASRModel.from_pretrained(
        args.model, dtype=dtype, **({"device_map": "cuda:0"} if use_cuda else {}),
    )
    if args.adapter:
        from peft import PeftModel

        # 适配器挂 .thinker（顶层无 forward/generate 实现，转发给 thinker——sft 同款）
        top = wrapper.model
        target = getattr(top, "thinker", top)
        adapted = PeftModel.from_pretrained(target, args.adapter)
        adapted.eval()
        if hasattr(top, "thinker"):
            top.thinker = adapted
        else:
            wrapper.model = adapted
        log(f"已挂载 LoRA 适配器到 {type(target).__name__}: {args.adapter}")

    rows = []
    if not args.silence_only:
        for line in Path(args.eval).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        if not rows:
            log("评测集为空")
            return 1

    # 按 task 分组（旧格式无 task 字段时按 prompt 反查）
    def task_of(r: dict) -> str:
        if r.get("task"):
            return r["task"]
        p = r.get("prompt", "")
        for k, v in PRESETS.items():
            if v == p:
                return k
        return VERBATIM_KEY if not p else "unknown"

    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(task_of(r), []).append(r)
    for k in groups:
        groups[k] = groups[k][: args.limit]
    log("评测分组：" + ", ".join(f"{k}×{len(v)}" for k, v in sorted(groups.items())))

    def run_one(r: dict) -> dict:
        res = wrapper.transcribe(audio=r["audio"], context=r.get("prompt", ""))
        text = (res[0].text if res else "") or ""
        lang = (res[0].language if res else "") or ""
        ref = r["text"].split("<asr_text>", 1)[-1]
        return {"audio": r["audio"], "task": task_of(r), "prompt": r.get("prompt", ""),
                "ref": ref, "out": text, "lang_tag": lang,
                "sim_ref": round(sim(text, ref), 4), **script_profile(text)}

    samples: dict[str, list[dict]] = {}
    for k, grp in sorted(groups.items()):
        samples[k] = [run_one(r) for r in grp]

    def avg(items: list[dict], key: str):
        return round(sum(i[key] for i in items) / len(items), 4) if items else None

    metrics: dict = {}
    checks: dict = {}

    verb = samples.get(VERBATIM_KEY, [])
    if verb:
        metrics["verbatim_avg_sim"] = avg(verb, "sim_ref")
        metrics[f"verbatim_avg_{args.source_script}"] = avg(verb, args.source_script)
        # 整条泄漏：verbatim 输出里源脚本占比 <5% 且非空（被某个任务吸走）
        leaks = [s for s in verb
                 if s["out"].strip() and s[args.source_script] < 0.05]
        metrics["verbatim_leak_count"] = len(leaks)
        checks["no_catastrophic_forgetting"] = (
            metrics["verbatim_avg_sim"] >= args.min_verbatim_sim
            and metrics["verbatim_leak_count"] <= args.max_leak
        )

    for task, items in sorted(samples.items()):
        if task in (VERBATIM_KEY, "silence", "unknown") or not items:
            continue
        rule = TASK_SCRIPT.get(task)
        metrics[f"{task}_avg_sim"] = avg(items, "sim_ref")
        for s in ("cjk", "latin", "kana"):
            metrics[f"{task}_avg_{s}"] = avg(items, s)
        non_empty = sum(1 for i in items if i["out"].strip())
        metrics[f"{task}_non_empty"] = non_empty
        if rule:
            script, floor = rule
            metrics[f"{task}_avg_{script}"] = avg(items, script)
            checks[f"task_takes_effect[{task}]"] = (
                non_empty == len(items) and (metrics[f"{task}_avg_{script}"] or 0) >= floor
            )
        if TASK_KIND.get(task) == "polish":
            fill = sum(sum(i["out"].count(f) for f in FILLERS_ZH) for i in items)
            metrics[f"{task}_filler_hits"] = fill

    # ---- 静音行为（任务模式 + verbatim 模式都要空；qsa 双模式教训）----
    if args.silence_secs > 0:
        silence = np.zeros(int(16000 * args.silence_secs), dtype=np.float32)
        modes = [("transcribe_mode", "")]
        first_task = next((t for t in samples if t in PRESETS), None)
        if first_task:
            modes.append((f"task_mode[{first_task}]", PRESETS[first_task]))
        sil_report: dict = {}
        for mode, ctx in modes:
            try:
                res = wrapper.transcribe(audio=(silence, 16000), context=ctx)
                text = (res[0].text if res else "") or ""
                lang = (res[0].language if res else "") or ""
                sil_report[mode] = {"text": text, "lang_tag": lang, "context": ctx}
                log(f"静音[{mode}] context={ctx!r} -> lang={lang!r} text={text!r}")
            except Exception as e:  # noqa: BLE001
                sil_report[mode] = {"error": f"{type(e).__name__}: {e}", "context": ctx}
                log(f"静音[{mode}] 测试失败: {e}")
        checks["silence_is_empty"] = all(
            "error" not in v and len((v.get("text") or "").strip()) <= 2
            for v in sil_report.values()
        )
    else:
        sil_report = {}

    report = {
        "model": args.model,
        "adapter": args.adapter or None,
        "device": "cuda" if use_cuda else "cpu",
        "source_script": args.source_script,
        "samples": samples,
        "silence": sil_report,
        "metrics": metrics,
        "checks": checks,
    }

    log("=" * 60)
    for task, items in sorted(samples.items()):
        for s in items[:3]:
            log(f"  [{task}] sim={s['sim_ref']:.2f} 输出: {s['out'][:56]}")
            if task == VERBATIM_KEY:
                log(f"           参考: {s['ref'][:56]}")
    log("=" * 60)
    log(f"指标: {json.dumps(metrics, ensure_ascii=False)}")
    for k, v in checks.items():
        log(f"  {'✔' if v else '✘'} {k}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        log(f"报告已写入 {args.out}")

    # stub 冒烟数据下质量指标必然不达标——脚本本身不判失败，
    # 由调用方（CI）按运行模式决定判定口径（与 qsa eval_s2tt.py 相同约定）。
    failed = [k for k, v in checks.items() if v is False]
    if failed:
        log(f"未通过的检查: {failed}")
    return 0


if __name__ == "__main__":
    _hard_exit(main())
