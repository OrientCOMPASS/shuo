#!/usr/bin/env python3
"""构造「单模型直出」微调数据：语音识别 + 提示词驱动的文本优化，一套权重多任务。

产出 JSONL，每行与官方 QwenLM/Qwen3-ASR finetuning/qwen3_asr_sft.py 的输入格式一致：

    {"audio": "/abs/path.wav",
     "text":  "language Chinese<asr_text>优化后的文本",
     "prompt": "转为书面文本：修正标点，去除口癖，保持原意。",
     "task":  "polish_zh"}

原理（qsa 仓库已全链路实测验证，见其 finetune/README.md §10/§11 与
本仓库 finetune/README.md）：官方数据管线 `target = ex["text"]` **不校验 text
是否为该音频的转写**，且 `prompt` 字段进 chat 模板 system 段——所以：
  * 带任务 prompt 的样本学「按提示词直出优化文本」；
  * 不带 prompt 的样本（--asr-ratio）学原生转写（抗遗忘锚点）；
  * 静音样本（--silence-samples）保住基座「静音 -> language None + 空」行为
    （qsa 实测：不混静音样本，LoRA 4 epoch 就会把该行为训坏，产生幻觉文本）。

优化后端（--optimizer，按任务生成 target）：
  none   target = 原转写（只能配 verbatim，即纯 ASR 数据）
  stub   确定性伪优化，**仅供 CI 跑通管线**，产出的模型没有真实优化能力
  api    OpenAI 兼容 /chat/completions；system 消息 = 任务提示词**原文**
         （与推理端进 system 段的字符串逐字一致，产线推荐）
  table  三列 TSV：task<TAB>源转写<TAB>目标文本（免 API key 的真实小数据路线，
         配合 --dump-transcripts 离线人工整理）

数据源（--source）：
  fleurs   HF google/fleurs 流式读取（默认 cmn_hans_cn 普通话，CC-BY-4.0；--fleurs-lang 可换）
  wav-dir  自备目录：--wav-dir 指向 wav，--transcripts 指向 "文件名<TAB>转写" TSV

用法（CI 冒烟）：
  python finetune/prepare_data.py --source fleurs --limit 4 --eval-limit 2 \
      --optimizer stub --tasks polish_zh,translate_en --asr-ratio 0.4 \
      --silence-samples 2 --audio-dir data/audio --out data/train.jsonl \
      --eval-out data/eval.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import PRESETS, SILENCE_TAG, TASK_KIND, VERBATIM_KEY, lang_tag  # noqa: E402

# fleurs 配置名 -> Qwen3-ASR 输出格式里的语言名（`language X<asr_text>` 的 X）
# 注意：fleurs 的普通话配置名是 cmn_hans_cn（不是 zh_cn，首跑 CI 实测报错拿到的清单）
FLEURS_LANG_NAME = {
    "cmn_hans_cn": "Chinese",
    "en_us": "English",
    "ja_jp": "Japanese",
    "ko_kr": "Korean",
    "yue_hant_hk": "Cantonese",
    "de_de": "German",
    "fr_fr": "French",
    "es_419": "Spanish",
    "ru_ru": "Russian",
}


def log(msg: str) -> None:
    print(f"[data] {msg}", flush=True)


def _hard_exit(code: int) -> "NoReturn":
    """绕过解释器 finalization 直接退出。

    实测（qsa CI）：datasets/pyarrow 的后台线程在 Python finalize 阶段会触发
    `Fatal Python error: PyGILState_Release` 并 SIGABRT（退出码 134），即使脚本
    本身已完全成功。所有产物在调用本函数前均已写盘并 flush，_exit 是安全的。
    """
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


# --------------------------------------------------------------------------- 音频源

def iter_fleurs(lang: str, limit: int, max_secs: float):
    """流式读取 FLEURS，产出 (16k float32 数组, 转写文本)。

    注意：**不要让 datasets 自己解码音频**——新版 datasets 的 Audio 特征解码依赖
    torchcodec（缺了会抛 ImportError，解释器退出时还会 segfault）。用
    cast_column(Audio(decode=False)) 拿原始字节，再用 soundfile 解码（qsa 实测坑）。
    """
    import io

    import numpy as np
    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = load_dataset("google/fleurs", lang, split="train", streaming=True)
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
        log("已切换为不解码音频（拿原始 bytes，自行用 soundfile 解码）")
    except Exception as e:  # noqa: BLE001
        log(f"cast_column(decode=False) 失败（{e}），按已解码数组处理")

    n = 0
    for row in ds:
        audio = row.get("audio") or {}
        text = (row.get("transcription") or row.get("raw_transcription") or "").strip()
        if not text:
            continue

        arr = None
        sr = 16000
        raw = audio.get("bytes")
        if raw:
            try:
                arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
            except Exception as e:  # noqa: BLE001
                log(f"soundfile 解码失败，跳过该条: {e}")
                continue
        elif audio.get("array") is not None:
            arr = np.asarray(audio["array"], dtype="float32")
            sr = int(audio.get("sampling_rate") or 16000)
        if arr is None or len(arr) == 0:
            continue
        if arr.ndim > 1:
            arr = arr.mean(axis=1)

        secs = len(arr) / float(sr)
        if secs > max_secs or secs < 1.0:
            continue
        if sr != 16000:
            import librosa

            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        yield arr, text
        n += 1
        if n >= limit:
            return


def iter_wav_dir(wav_dir: Path, transcripts: Path, limit: int, max_secs: float):
    import soundfile as sf

    table = {}
    for line in transcripts.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            k, v = line.split("\t", 1)
            table[k.strip()] = v.strip()
    n = 0
    for p in sorted(wav_dir.glob("*.wav")):
        text = table.get(p.stem) or table.get(p.name)
        if not text:
            continue
        arr, sr = sf.read(str(p), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if len(arr) / sr > max_secs:
            continue
        if sr != 16000:
            import librosa

            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        yield arr, text
        n += 1
        if n >= limit:
            return


# --------------------------------------------------------------------------- 优化后端

class StubOptimizer:
    """确定性伪优化：只为让 CI 把训练管线跑通，产出没有任何真实优化价值。

    按任务类型给不同形态的确定性输出（评测脚本据此断言"任务开关"接线正确）：
      polish_*    去语气词 + 补句末标点（对源转写做规则变换）
      translate_zh 哈希选词的伪中文（含汉字）
      translate_en 哈希选词的伪英文（含拉丁词）
    """

    name = "stub"

    def __init__(self) -> None:
        self.cache: dict[tuple[str, str], str] = {}

    def __call__(self, task: str, text: str) -> str:
        ck = (task, text)
        if ck in self.cache:
            return self.cache[ck]
        kind = TASK_KIND.get(task, "polish")
        if task == "translate_zh":
            words = ["测试", "样本", "语音", "直出", "管线", "验证", "数据", "占位"]
            out = self._hash_pick(text, words)
        elif task == "translate_en":
            words = ["test", "sample", "speech", "output", "pipeline", "check", "data", "placeholder"]
            out = self._hash_pick(text, words)
        elif task == "polish_en":
            import re

            out = re.sub(r"\b(um|uh|like|you know)\b[, ]*", "", text, flags=re.I).strip()
            if out and out[-1] not in ".?!":
                out += "."
        elif task == "brief_zh":
            # 简略 stub：确定性截断到前 40% 非空白字符（长度比判据 ≤0.75 成立）
            chars = [c for c in text if not c.isspace()]
            keep = max(4, int(len(chars) * 0.4))
            out = "".join(chars[:keep]).rstrip("，。、；：,") + "。"
        elif task == "detail_zh":
            # 详细 stub：与 polish 同形态（全保留 + 句末标点），长度比 ≥0.95 成立
            import re

            out = re.sub(r"(嗯|啊|呃|那个|就是说|然后|对吧|反正|大概|好像)", "", text).strip()
            if out and out[-1] not in "。？！?!":
                out += "。"
        else:  # polish_zh
            import re

            out = re.sub(r"(嗯|啊|呃|那个|就是说|然后|对吧|反正|大概|好像)", "", text).strip()
            if out and out[-1] not in "。？！?!":
                out += "。"
        self.cache[ck] = out
        return out

    @staticmethod
    def _hash_pick(text: str, words: list[str]) -> str:
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return "".join(words[int(c, 16) % len(words)] for c in h[:12])


class NoneOptimizer:
    name = "none"

    def __call__(self, task: str, text: str) -> str:
        return text


class TableOptimizer:
    """查固定对照表（三列 TSV：task<TAB>源转写<TAB>目标文本，# 开头为注释）。

    用途：**没有任何 API key** 的真实微调实验——先 --dump-transcripts 导出 CI
    抽到的转写，离线整理成对照表提交进仓库，再用本后端训练（qsa real-mini 同款
    路线，其 120 对人工对照表已验证"任务开关"可学成）。
    """

    name = "table"

    def __init__(self, path: Path):
        self.path = path
        self.table: dict[tuple[str, str], str] = {}
        if not path.is_file():
            raise SystemExit(f"对照表不存在: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            task, src, dst = (p.strip() for p in parts)
            if task and src and dst:
                self.table[(task, src)] = dst
        log(f"对照表 {path}: {len(self.table)} 条")

    def __call__(self, task: str, text: str) -> str:
        key = (task, text.strip())
        if key in self.table:
            return self.table[key]
        # 未命中：句末标点归一后的模糊匹配
        norm = text.strip().rstrip("。.、！？!?")
        for (t, k), v in self.table.items():
            if t == task and k.rstrip("。.、！？!?") == norm:
                return v
        raise KeyError(f"{task}:{text[:30]}")


class ApiOptimizer:
    """OpenAI 兼容 /chat/completions。system 消息 = 任务提示词原文。

    产线路线：优化目标由与推理端**逐字相同**的指令生成，训练/部署语义闭环。
    带磁盘缓存与重试，可断点续跑。
    """

    name = "api"

    def __init__(self, base: str, model: str, key: str, cache_path: Path):
        self.url = base.rstrip("/") + "/chat/completions"
        self.model = model
        self.key = key
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        if cache_path.is_file():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    self.cache[d["k"]] = d["v"]
            log(f"优化缓存命中 {len(self.cache)} 条")

    def __call__(self, task: str, text: str) -> str:
        prompt = PRESETS.get(task)
        if prompt is None:
            raise KeyError(f"api 后端不支持任务 {task}（不在 prompts.PRESETS）")
        ck = hashlib.sha1(json.dumps([task, text], ensure_ascii=False).encode("utf-8")).hexdigest()
        if ck in self.cache:
            return self.cache[ck]
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": text}],
            "temperature": 0.2,
        }).encode("utf-8")
        last = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(self.url, data=body, headers={
                    "Content-Type": "application/json",
                    **({"Authorization": f"Bearer {self.key}"} if self.key else {}),
                })
                with urllib.request.urlopen(req, timeout=120) as r:
                    d = json.loads(r.read())
                out = (d["choices"][0]["message"]["content"] or "").strip()
                if out:
                    self.cache[ck] = out
                    with self.cache_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "k": ck, "v": out, "task": task, "src": text,
                        }, ensure_ascii=False) + "\n")
                    return out
                last = "空响应"
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"优化失败（{text[:30]}…）: {last}")


# --------------------------------------------------------------------------- 静音样本

def make_silence_samples(audio_dir: Path, n: int, task_prompts: list[str],
                         min_secs: float, max_secs: float, seed: int) -> list[str]:
    """合成静音/低噪 wav，训练目标固定 `language None<asr_text>`（空文本）。

    为什么需要（qsa run 36221464174 实测教训）：训练集 100% 非空 target 时，
    LoRA 会把基座「静音 -> 空输出」这条路压没（2s 纯静音输出幻觉文本），而
    产品端依赖该行为剔除噪音段。修复：显式混入静音样本，带 prompt（在各任务
    prompt 间轮换）与不带 prompt 各占一半——eval 的静音测试双模式都要过。

    前 4 条固定 2.0s 并覆盖「零/噪 × 带/不带 prompt」，与 eval_unified.py 的
    默认静音测试（2s 纯零，双模式）精确对齐；其余零/噪交替、时长随机。
    """
    import numpy as np
    import soundfile as sf

    rng = np.random.default_rng(seed)
    lines: list[str] = []
    for j in range(n):
        if j < 4:
            secs = 2.0
            zeros = j in (0, 2)
            with_prompt = j in (0, 1)
        else:
            secs = round(float(rng.uniform(min_secs, max_secs)), 2)
            zeros = (j % 2 == 0)
            with_prompt = ((j // 2) % 2 == 0)
        n_samples = int(16000 * secs)
        if zeros:
            arr = np.zeros(n_samples, dtype=np.float32)
        else:
            arr = rng.normal(0.0, 0.001, n_samples).astype(np.float32)
        wav = audio_dir / f"silence_{j:03d}.wav"
        sf.write(str(wav), arr, 16000, subtype="PCM_16")
        prompt = task_prompts[j % len(task_prompts)] if (with_prompt and task_prompts) else ""
        rec = {"audio": str(wav.resolve()), "text": SILENCE_TAG, "prompt": prompt,
               "task": "silence"}
        lines.append(json.dumps(rec, ensure_ascii=False))
    return lines


# --------------------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["fleurs", "wav-dir"], default="fleurs")
    ap.add_argument("--fleurs-lang", default="cmn_hans_cn",
                    help=f"fleurs 配置名（语言名映射：{sorted(FLEURS_LANG_NAME)}）")
    ap.add_argument("--source-lang-tag", default="",
                    help="覆盖 `language X<asr_text>` 里的 X（默认按 --fleurs-lang 映射）")
    ap.add_argument("--wav-dir", default="")
    ap.add_argument("--transcripts", default="")
    ap.add_argument("--limit", type=int, default=200, help="训练用源音频条数上限")
    ap.add_argument("--eval-limit", type=int, default=20, help="额外留出的评测音频条数")
    ap.add_argument("--max-audio-secs", type=float, default=20.0)
    ap.add_argument("--tasks", default="polish_zh,translate_en",
                    help=f"逗号分隔的任务 key（可选：{','.join(PRESETS)}）；"
                         "每条源音频按 --asr-ratio 决定做 verbatim 还是随机一个任务")
    ap.add_argument("--asr-ratio", type=float, default=0.4,
                    help="verbatim（不带 prompt）样本占比，抗遗忘锚点，0~1")
    ap.add_argument("--optimizer", choices=["none", "stub", "api", "table"], default="stub")
    ap.add_argument("--table", default="", help="--optimizer table 的三列 TSV 路径")
    ap.add_argument("--dump-transcripts", default="",
                    help="把抽到的源转写逐行写到该文件（供离线整理对照表）")
    ap.add_argument("--skip-unoptimized", action="store_true",
                    help="table 模式下对照表未命中的样本直接跳过（而不是报错）")
    ap.add_argument("--silence-samples", type=int, default=0,
                    help="混入 N 条合成静音/低噪样本（目标 language None+空文本）")
    ap.add_argument("--silence-min-secs", type=float, default=1.0)
    ap.add_argument("--silence-max-secs", type=float, default=3.0)
    ap.add_argument("--api-base", default="")
    ap.add_argument("--api-model", default="")
    ap.add_argument("--api-key-env", default="", help="存放 API key 的环境变量名")
    ap.add_argument("--audio-dir", default="data/audio")
    ap.add_argument("--out", default="data/train.jsonl")
    ap.add_argument("--eval-out", default="data/eval.jsonl")
    ap.add_argument("--cache", default="data/opt_cache.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    bad = [t for t in tasks if t not in PRESETS]
    if bad:
        ap.error(f"未知任务 key: {bad}（可选：{','.join(PRESETS)}）")
    if args.optimizer == "none" and tasks:
        ap.error("--optimizer none 只能配空 --tasks（纯 ASR 数据）")
    src_lang = args.source_lang_tag or FLEURS_LANG_NAME.get(args.fleurs_lang)
    if not src_lang:
        ap.error(f"--fleurs-lang {args.fleurs_lang} 没有内置语言名映射，请用 --source-lang-tag 指定")
    ltag = lang_tag(src_lang)

    random.seed(args.seed)
    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.optimizer == "api":
        if not args.api_base or not args.api_model:
            ap.error("--optimizer api 需要 --api-base 与 --api-model")
        key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
        opt = ApiOptimizer(args.api_base, args.api_model, key, Path(args.cache))
    elif args.optimizer == "table":
        if not args.table:
            ap.error("--optimizer table 需要 --table")
        opt = TableOptimizer(Path(args.table))
    elif args.optimizer == "stub":
        log("!! 使用 stub 伪优化：仅供 CI 跑通管线，产出的模型没有真实优化能力 !!")
        opt = StubOptimizer()
    else:
        opt = NoneOptimizer()

    if args.source == "fleurs":
        src = iter_fleurs(args.fleurs_lang, args.limit + args.eval_limit, args.max_audio_secs)
    else:
        if not args.wav_dir or not args.transcripts:
            ap.error("--source wav-dir 需要 --wav-dir 与 --transcripts")
        src = iter_wav_dir(Path(args.wav_dir), Path(args.transcripts),
                           args.limit + args.eval_limit, args.max_audio_secs)

    import soundfile as sf

    def optimize(task: str, text: str) -> str | None:
        try:
            return opt(task, text)
        except KeyError:
            if args.skip_unoptimized:
                return None
            raise

    train_lines: list[str] = []
    eval_lines: list[str] = []
    n_task = n_asr = n_skipped = 0
    per_task_count: dict[str, int] = {t: 0 for t in tasks}
    dump_fh = open(args.dump_transcripts, "w", encoding="utf-8") if args.dump_transcripts else None
    for i, (arr, text) in enumerate(src):
        wav = audio_dir / f"src_{i:06d}.wav"
        if not wav.is_file():
            sf.write(str(wav), arr, 16000, subtype="PCM_16")
        if dump_fh is not None:
            dump_fh.write(text.strip() + "\n")
        is_eval = i >= args.limit

        def emit(task: str, prompt: str, target: str, sink: list[str]) -> None:
            rec = {"audio": str(wav.resolve()), "text": ltag + target,
                   "prompt": prompt, "task": task}
            sink.append(json.dumps(rec, ensure_ascii=False))

        if is_eval:
            # 评测集：每个任务一条 + verbatim 一条（检查任务开关与是否遗忘）
            for task in tasks:
                target = optimize(task, text)
                if target is None:
                    n_skipped += 1
                    continue
                emit(task, PRESETS[task], target, eval_lines)
            emit(VERBATIM_KEY, "", text, eval_lines)
        else:
            if random.random() < args.asr_ratio or not tasks:
                emit(VERBATIM_KEY, "", text, train_lines)
                n_asr += 1
            else:
                task = random.choice(tasks)
                target = optimize(task, text)
                if target is None:
                    n_skipped += 1
                    continue
                emit(task, PRESETS[task], target, train_lines)
                n_task += 1
                per_task_count[task] += 1
        if (i + 1) % 25 == 0:
            log(f"已处理 {i + 1} 条（任务 {n_task} / verbatim {n_asr}）")

    if dump_fh is not None:
        dump_fh.close()
        log(f"已导出源转写 -> {args.dump_transcripts}（离线整理成 task<TAB>src<TAB>target "
            "三列对照表后可用 --optimizer table 训练）")
    if n_skipped:
        log(f"对照表未命中而跳过 {n_skipped} 条")

    n_silence = 0
    if args.silence_samples > 0:
        task_prompts = [PRESETS[t] for t in tasks]
        sil = make_silence_samples(audio_dir, args.silence_samples, task_prompts,
                                   args.silence_min_secs, args.silence_max_secs, args.seed)
        train_lines.extend(sil)
        n_silence = len(sil)
        log(f"已混入静音/低噪样本 {n_silence} 条（目标 {SILENCE_TAG!r}，"
            "带 prompt 与不带 prompt 各约一半）")

    Path(args.out).write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    Path(args.eval_out).write_text("\n".join(eval_lines) + "\n", encoding="utf-8")
    log(f"训练集 {len(train_lines)} 条 -> {args.out}")
    log(f"评测集 {len(eval_lines)} 条 -> {args.eval_out}")
    log(f"任务构成：verbatim {n_asr} 条，任务样本 {n_task} 条 {per_task_count}，静音 {n_silence} 条")
    if not train_lines:
        log("!! 训练集为空，请检查数据源与 --limit")
        return 1
    return 0


if __name__ == "__main__":
    _hard_exit(main())
