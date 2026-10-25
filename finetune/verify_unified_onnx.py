#!/usr/bin/env python3
"""Gate 3 验证器：在 onnxruntime 上验证（补丁后的）Daumee 模型包行为。

与产品运行时同构的最小推理管线（numpy mel + ort 四会话 + 贪心解码，移植自
legacy/2026-10-25-python/onnx_infer/asr.py，即 src-tauri/src/asr.rs 的 Python
参照实现），额外支持 context（chat 模板 system 段）——这是「单模型直出」的
任务开关通道。T4 教训（qsa 实测）：量化会改变模型行为，**必须在最终运行时
形态上复测**，不能只信 PyTorch 侧评测。

三种检查集：
  base     基座包回归：不带 context 正常转写；带 context 也要能出文本
           （基座把 context 当识别偏置，不执行指令——不作为失败）；
  unified  统一包（补丁后）：
             ① verbatim（空 context）转写与参考转写相似；
             ② 任务 context 生效（translate_en 拉丁占比 / translate_zh CJK 占比 /
                polish 非空同脚本）；
             ③ 静音（2s 纯零）双模式输出空；
  smoke    只验证管线可跑通（stub 适配器补丁的包：断言输出可解析、无异常，
           不断言质量——CI cpu 冒烟链路用）。

用法：
  python finetune/verify_unified_onnx.py --model-dir models/Qwen3-ASR-0.6B-ONNX-CPU-unified \
      --audio data/audio/src_000000.wav --check-set unified --expect-task translate_en \
      --report out/verify/report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import PRESETS  # noqa: E402

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
CHUNK_SIZE = 100

AUDIO_START_ID = 151669
AUDIO_END_ID = 151670
AUDIO_PAD_ID = 151676
IM_START_ID = 151644
IM_END_ID = 151645
ENDOFTEXT_ID = 151643
NEWLINE_ID = 198
VOCAB_SIZE = 151936
HIDDEN_SIZE = 1024

CJK = re.compile(r"[\u4e00-\u9fff]")
LATIN = re.compile(r"[A-Za-z]")


def log(msg: str) -> None:
    print(f"[verify] {msg}", flush=True)


def _hard_exit(code: int) -> "NoReturn":
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    os._exit(code)


# ---------------------------------------------------------------- mel（whisper 兼容）

_HANN = None


def _hann(n: int):
    global _HANN
    import numpy as np

    if _HANN is None or len(_HANN) != n:
        _HANN = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))
    return _HANN


def mel_filters_bank():
    import librosa
    import numpy as np

    mel = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
                              fmin=0, fmax=SAMPLE_RATE // 2, norm="slaney", htk=False)
    return mel.astype(np.float32)


def log_mel(wav):
    import numpy as np

    pad = N_FFT // 2
    wp = np.pad(wav, (pad, pad), mode="reflect")
    win = _hann(N_FFT)
    n_frames = 1 + (len(wp) - N_FFT) // HOP_LENGTH
    stft = np.zeros((N_FFT // 2 + 1, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        stft[:, i] = np.fft.rfft(wp[i * HOP_LENGTH: i * HOP_LENGTH + N_FFT] * win)
    mag = np.abs(stft) ** 2
    spec = mel_filters_bank() @ mag
    log_spec = np.log10(np.maximum(spec, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


def load_wav(path: str):
    import wave

    import numpy as np

    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
        sw = wf.getsampwidth()
    dtype = {2: np.int16, 4: np.int32}[sw]
    wav = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    wav /= 32768.0 if sw == 2 else 2147483648.0
    if n_ch > 1:
        wav = wav.reshape(-1, n_ch).mean(axis=1)
    if sr != SAMPLE_RATE:
        n_out = int(len(wav) * SAMPLE_RATE / sr)
        wav = np.interp(np.linspace(0, 1, n_out), np.linspace(0, 1, len(wav)), wav)
    return wav.astype(np.float32)


# ---------------------------------------------------------------- 管线（与 asr.rs 同构）

class OrtAsr:
    def __init__(self, model_dir: Path, threads: int = 4):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        onnx_dir = model_dir / "onnx_models"
        if not onnx_dir.is_dir():
            onnx_dir = model_dir
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_cpu_mem_arena = False
        opts.intra_op_num_threads = max(1, threads)
        opts.log_severity_level = 3
        pick = lambda n: str(onnx_dir / n)  # noqa: E731
        self.sess = {
            "conv": ort.InferenceSession(pick("encoder_conv.onnx"), opts, providers=["CPUExecutionProvider"]),
            "tf": ort.InferenceSession(pick("encoder_transformer.onnx"), opts, providers=["CPUExecutionProvider"]),
            "init": ort.InferenceSession(
                pick("decoder_init.int8.onnx") if (onnx_dir / "decoder_init.int8.onnx").exists()
                else pick("decoder_init.onnx"), opts, providers=["CPUExecutionProvider"]),
            "step": ort.InferenceSession(
                pick("decoder_step.int8.onnx") if (onnx_dir / "decoder_step.int8.onnx").exists()
                else pick("decoder_step.onnx"), opts, providers=["CPUExecutionProvider"]),
        }
        self.embed = np.fromfile(str(onnx_dir / "embed_tokens.bin"), dtype=np.float32
                                 ).reshape(VOCAB_SIZE, HIDDEN_SIZE)
        tok = onnx_dir / "tokenizer.json"
        if not tok.is_file():
            tok = model_dir / "tokenizer.json"
        self.tok = Tokenizer.from_file(str(tok))
        self._np = np

    def encode_ids(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def build_prompt(self, n_audio: int, language: str | None, context: str) -> list[int]:
        """与 src-tauri/src/asr.rs 的 prompt_ids 逐 token 一致（含 context 分支）。"""
        ctx = (context or "").strip()
        ids = [IM_START_ID]
        if ctx:
            ids += self.encode_ids(f"system\n{ctx}")
        else:
            ids += self.encode_ids("system") + [NEWLINE_ID]
        ids += [IM_END_ID, NEWLINE_ID, IM_START_ID]
        ids += self.encode_ids("user") + [NEWLINE_ID, AUDIO_START_ID]
        ids += [AUDIO_PAD_ID] * n_audio + [AUDIO_END_ID, IM_END_ID, NEWLINE_ID, IM_START_ID]
        ids += self.encode_ids("assistant") + [NEWLINE_ID]
        if language:
            ids += self.encode_ids(f"language {language}<asr_text>")
        return ids

    def encode_audio(self, wav):
        np = self._np
        mel = log_mel(wav)
        mel_len = mel.shape[1]
        chunk_num = int(np.ceil(mel_len / CHUNK_SIZE))
        lens = [min((i + 1) * CHUNK_SIZE, mel_len) - i * CHUNK_SIZE for i in range(chunk_num)]
        max_len = max(lens)
        padded = np.zeros((chunk_num, 1, N_MELS, max_len), dtype=np.float32)
        start = 0
        for i, cl in enumerate(lens):
            padded[i, 0, :, :cl] = mel[:, start:start + cl]
            start += cl
        conv_out = self.sess["conv"].run(None, {"padded_mel_chunks": padded})[0]
        feats = []
        for i, l in enumerate(lens):
            ll = l
            for _ in range(3):
                ll = (ll - 1) // 2 + 1
            feats.append(conv_out[i, :min(ll, conv_out.shape[1]), :])
        hidden = np.concatenate(feats, axis=0)
        total = hidden.shape[0]
        mask = np.zeros((1, 1, total, total), dtype=np.float32)
        return self.sess["tf"].run(None, {"hidden_states": hidden, "attention_mask": mask})[0]

    def transcribe(self, wav, context: str = "", language: str | None = None,
                   max_new_tokens: int = 512) -> dict:
        np = self._np
        t0 = time.time()
        audio = self.encode_audio(wav)
        n_audio = audio.shape[0]
        ids = self.build_prompt(n_audio, language, context)
        arr = np.array(ids)
        embeds = self.embed[arr]
        embeds[np.where(arr == AUDIO_PAD_ID)[0]] = audio
        seq = embeds.shape[0]
        logits, pk, pv = self.sess["init"].run(None, {
            "input_embeds": embeds[np.newaxis, :, :],
            "position_ids": np.arange(seq, dtype=np.int64).reshape(1, -1),
        })
        nxt = int(np.argmax(logits[0, -1, :]))
        gen = [nxt]
        pos = seq
        while len(gen) < max_new_tokens and nxt not in (IM_END_ID, ENDOFTEXT_ID):
            logits, pk, pv = self.sess["step"].run(None, {
                "input_embeds": self.embed[nxt][np.newaxis, np.newaxis, :],
                "position_ids": np.array([[pos]], dtype=np.int64),
                "past_keys": pk, "past_values": pv,
            })
            nxt = int(np.argmax(logits[0, -1, :]))
            gen.append(nxt)
            pos += 1
        if gen and gen[-1] in (IM_END_ID, ENDOFTEXT_ID):
            gen = gen[:-1]
        raw = self.tok.decode(gen, skip_special_tokens=True)
        lang, text = "", raw
        if "<asr_text>" in raw:
            head, body = raw.split("<asr_text>", 1)
            lang = head.strip()[len("language"):].strip() if head.strip().startswith("language") else ""
            text = body
        return {"raw": raw, "lang": lang, "text": text.strip(),
                "seconds": round(time.time() - t0, 2)}


def script_share(pattern, text: str) -> float:
    """脚本占比（字母归一）：分母只数 CJK/拉丁字母，排除数字/符号/标点。

    实测教训（Gate 3 首跑）：'Open time: 9:00 AM to 5:00 PM.' 这类数字密集的
    短句按全字符口径拉丁占比被数字/冒号稀释到 58%（<0.6 误判"未生效"），
    字母归一口径为 100%。全文本无字母（纯数字/符号）时按目标脚本命中记 1.0
    ——数字直出（如报数音频的译文就是数字）是翻译任务的合法输出形态。
    """
    cjk = sum(1 for c in text if CJK.match(c))
    latin = sum(1 for c in text if LATIN.match(c))
    total = cjk + latin
    if total == 0:
        return 1.0 if text.strip() else 0.0
    return sum(1 for c in text if pattern.match(c)) / total


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Daumee 包目录（原始或 unified 补丁包）")
    ap.add_argument("--audio", action="append", default=[],
                    help="16k/任意采样率 wav（可多次）；check-set=unified 时至少 1 个")
    ap.add_argument("--reference", default="",
                    help="与 --audio 同序的参考转写 TSV（文件名<TAB>转写；unified ①② 判定用）")
    ap.add_argument("--check-set", choices=["base", "unified", "smoke"], default="base")
    ap.add_argument("--expect-task", default="translate_en",
                    choices=sorted(PRESETS), help="unified/smoke 的任务 context（进 system 段）")
    ap.add_argument("--silence-secs", type=float, default=2.0, help="0=跳过静音检查")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    import numpy as np

    model_dir = Path(args.model_dir)
    unified = (model_dir / "unified.json").is_file()
    log(f"模型目录: {model_dir}（unified.json {'存在' if unified else '不存在'}）check-set={args.check_set}")
    if args.check_set == "unified" and not unified:
        log("!! check-set=unified 但目录没有 unified.json 标记——按 base 语义降级判定")

    t0 = time.time()
    asr = OrtAsr(model_dir, args.threads)
    log(f"管线加载 {time.time() - t0:.1f}s")

    refs: dict[str, str] = {}
    if args.reference and Path(args.reference).is_file():
        for line in Path(args.reference).read_text(encoding="utf-8").splitlines():
            if "\t" in line and not line.startswith("#"):
                k, v = line.split("\t", 1)
                refs[k.strip()] = v.strip()

    report: dict = {"model_dir": str(model_dir), "unified_marker": unified,
                    "check_set": args.check_set, "results": [], "checks": {}}
    fails: list[str] = []

    ctx_task = PRESETS[args.expect_task]
    for audio in args.audio:
        wav = load_wav(audio)
        name = Path(audio).name
        plain = asr.transcribe(wav, context="")
        tasked = asr.transcribe(wav, context=ctx_task)
        rec = {"audio": name, "verbatim": plain, f"task[{args.expect_task}]": tasked,
               "ref": refs.get(Path(audio).stem, refs.get(name, ""))}
        report["results"].append(rec)
        log(f"  [{name}] verbatim lang={plain['lang']!r}: {plain['text'][:48]!r}")
        log(f"  [{name}] {args.expect_task} lang={tasked['lang']!r}: {tasked['text'][:48]!r}")

        if args.check_set == "base":
            # 基座回归：两种模式都必须出可解析文本（context 不被执行是预期）
            if not plain["text"] and not plain["raw"]:
                fails.append(f"{name}: 基座 verbatim 输出为空")
        elif args.check_set == "smoke":
            # stub 适配器的补丁包：只断言管线可跑、输出可解析、任务模式不炸
            if "<asr_text>" not in plain["raw"] or "<asr_text>" not in tasked["raw"]:
                fails.append(f"{name}: 输出格式异常（缺 <asr_text> 标签）")
        else:  # unified
            if not plain["text"]:
                fails.append(f"{name}: verbatim 输出为空")
            ref = rec["ref"]
            if ref:
                import difflib

                s = difflib.SequenceMatcher(None, plain["text"], ref).ratio()
                rec["verbatim_sim"] = round(s, 4)
                if s < 0.5:
                    fails.append(f"{name}: verbatim 与参考相似度过低（{s:.2f}）——疑似遗忘")
            out = tasked["text"]
            if not out:
                fails.append(f"{name}: 任务模式输出为空")
            if args.expect_task == "translate_en":
                r = script_share(LATIN, out)
                rec["task_latin_ratio"] = round(r, 4)
                if r < 0.6:
                    fails.append(f"{name}: translate_en 未生效（拉丁占比 {r:.0%}）")
            elif args.expect_task == "translate_zh":
                r = script_share(CJK, out)
                rec["task_cjk_ratio"] = round(r, 4)
                if r < 0.6:
                    fails.append(f"{name}: translate_zh 未生效（CJK 占比 {r:.0%}）")

    # ---- 静音双模式（unified 强制；base/smoke 记录不判定——基座在带 context 时
    #      可能把指令回声成转写，qsa 实测归因过，见其 finetune/README.md §10）----
    if args.silence_secs > 0:
        sil = np.zeros(int(SAMPLE_RATE * args.silence_secs), dtype=np.float32)
        sil_rep = {}
        for mode, ctx in (("verbatim", ""), (f"task[{args.expect_task}]", ctx_task)):
            r = asr.transcribe(sil, context=ctx, max_new_tokens=64)
            sil_rep[mode] = r
            log(f"  静音[{mode}] lang={r['lang']!r} text={r['text']!r}")
        report["silence"] = sil_rep
        if args.check_set == "unified":
            empt = all(len(v["text"].strip()) <= 2 for v in sil_rep.values())
            report["checks"]["silence_is_empty"] = empt
            if not empt:
                fails.append("静音输出非空（双模式都要空——产品依赖该行为剔除噪音段）")

    report["checks"]["no_failures"] = not fails
    log("=" * 60)
    if fails:
        for f in fails:
            log("  ✘ " + f)
    else:
        log(f"  ✔ check-set={args.check_set} 全部通过")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        log(f"报告 -> {args.report}")
    return 1 if fails else 0


if __name__ == "__main__":
    _hard_exit(main())
