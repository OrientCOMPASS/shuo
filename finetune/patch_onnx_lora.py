#!/usr/bin/env python3
"""Gate 2 补丁器：把 LoRA ΔW 合并进 Daumee int8 ONNX 包，产出 unified 单模型直出目录。

移植自 Qwen3-subtitle-assistant 的 finetune/patch_onnx_lora.py（在 k2-fsa 官方 int8
包上实测 250/250 张量补丁成功、sherpa-onnx 运行时三项行为验证全过）。针对 shuo 的
Daumee/Qwen3-ASR-0.6B-ONNX-CPU 包做了三处布局适配：

  1. onnx 文件在 `onnx_models/` 子目录（输出保持同一布局，shuo 运行时零改动可加载）；
  2. encoder 两个 fp32 文件带外置 `.data`（onnx.load 读进内存，保存时重新外置，
     产出与原包同构：`*.onnx` + `*.onnx.data`）；
  3. **decoder_init 与 decoder_step 共享同一套 LM 权重**：去重按「文件内」而不是
     全局——同一 HF key 在两个文件里都要补丁（k2-fsa 包 encoder/decoder 权重不重叠，
     全局去重在那边成立，在 Daumee 包会把 step 文件漏掉，模型直接精分）。

前置事实由 Gate 1 探针（inspect_onnx_lora.py，CI mode=onnx-inspect）在真实包上核查：
适配器模块全部可按值映射、HF↔ONNX 同源、按原 scale/zp 重量化削顶 < 1%。

流程：
  1. 值匹配（复用探针的扫描/反量化/匹配函数），但只对「形状命中某个 ΔW 候选」的
     张量做匹配（跳过 embed/lm_head/conv 等无关大张量，天然避开 tied 权重二义）；
  2. 逐模块补丁：q' = clip(round((dequant(q) + ΔW_onnx摆位)/s + z), lo, hi)；
     fp32 张量直接加；
  3. 自检：补丁后反量化 vs 目标值，逐元素误差应 <= 0.51*step（未削顶处）；
  4. 其余文件字节级复制；写 `unified.json` 标记（shuo 运行时据此识别统一包，
     见 src-tauri/src/models.rs），输出完整模型目录。

用法（CI：finetune.yml mode=onnx-patch；本地亦可跑）：
  python finetune/patch_onnx_lora.py \
      --model-dir models/Qwen3-ASR-0.6B-ONNX-CPU \
      --hf-model Qwen/Qwen3-ASR-0.6B --adapter art/out/real/lora \
      --out-dir models/Qwen3-ASR-0.6B-ONNX-CPU-unified \
      --report out/patch/patch_report.json
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

# 同目录复用探针的扫描/反量化/值匹配实现（单一事实源，避免两份逻辑漂移）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inspect_onnx_lora import (  # noqa: E402
    collect_matchable, dequant, hf_open, load_adapter, match_one, log, onnx_root, _hard_exit,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--hf-model", default="Qwen/Qwen3-ASR-0.6B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-match-rel", type=float, default=0.15,
                    help="值匹配相对误差上限（与探针同义；Daumee 包 decoder 为 per-tensor "
                         "动态 int8，噪声实测 ~3e-2，match_one 另有唯一性兜底）")
    ap.add_argument("--min-elems", type=int, default=1024,
                    help="fp32 二维权重纳入补丁的元素数下限（与探针同义）")
    ap.add_argument("--max-clip", type=float, default=0.01,
                    help="允许的最坏削顶占比（per-channel min/max 量化下边界元素结构性必然存在，"
                         "真实模型 Gate1 实测 0.109%%；超过判失败）")
    ap.add_argument("--dry-run", action="store_true", help="只匹配与模拟，不写文件")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    import numpy as np
    import onnx
    from onnx import numpy_helper

    t0 = time.time()
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)

    # ---- 适配器 ΔW 与形状候选 ----
    r, alpha, scaling, per_module = load_adapter(Path(args.adapter))
    dws: dict[str, "np.ndarray"] = {}
    shape_cands: set[tuple] = set()
    for mod, ab in per_module.items():
        if "A" not in ab or "B" not in ab:
            continue
        dW = (ab["B"] @ ab["A"]) * scaling          # (N, K) HF 摆位
        dws[mod] = dW
        shape_cands.add(tuple(dW.shape))
        shape_cands.add(tuple(dW.shape[::-1]))      # 转置存储摆位
    log(f"适配器 r={r} alpha={alpha} scaling={scaling:.2f}：ΔW {len(dws)} 个，"
        f"形状候选 {len(shape_cands)} 种")

    hf_index, hf_shapes, hf_fetch = hf_open(args.hf_model)
    hf_by_shape = defaultdict(list)
    for k, shp in hf_shapes.items():
        if len(shp) == 2 and shp in shape_cands:
            hf_by_shape[shp].append(k)
    # 适配器模块 -> HF key（预计算，匹配阶段 O(1) 反查）
    mod_by_hfkey = {}
    for mod in dws:
        k = hf_key_for(hf_index, mod)
        if k:
            mod_by_hfkey[k] = mod
        else:
            report_flags_early = f"{mod}: HF key 未找到"
            log("  !! " + report_flags_early)
    log(f"HF 侧形状命中 key {sum(len(v) for v in hf_by_shape.values())} 个；"
        f"模块->HF key 映射 {len(mod_by_hfkey)}/{len(dws)}")

    # ---- 逐文件：匹配 -> 补丁 ----
    report = {"model_dir": str(model_dir), "adapter": args.adapter,
              "lora": {"r": r, "alpha": alpha, "scaling": scaling},
              "files": {}, "patched_modules": [], "unmapped_modules": [], "flags": []}
    cache: dict = {}
    root = onnx_root(model_dir)
    rel_root = root.relative_to(model_dir)
    patched_files: set[str] = set()
    onnx_files = sorted(root.glob("*.onnx"))
    log(f"onnx 根目录: {root}；文件: {[f.name for f in onnx_files]}")
    for f in onnx_files:
        m = onnx.load(str(f))   # 默认 load_external_data=True：.data 权重读进内存
        inits = {t_.name: t_ for t_ in m.graph.initializer}
        ents, fl = collect_matchable(m, f.name, args.min_elems)
        report["flags"].extend(fl)
        n_patch = n_match = 0
        # 去重按「文件内」：decoder_init/decoder_step 共享同一套 LM 权重，
        # 同一 HF key 在两个文件各补一次是预期行为（全局去重会漏掉 step 文件）
        patched_in_file: dict[str, str] = {}
        for e in ents:
            shp = tuple(inits[e["w"]].dims)
            if shp not in shape_cands and (len(shp) == 2 and shp[::-1] not in shape_cands):
                continue  # 形状不在 ΔW 候选里：与 LoRA 无关，跳过
            Wd, s_b, z_b, quantized = dequant(e, inits)
            res = match_one(Wd, hf_by_shape, hf_fetch, cache, args.max_match_rel)
            if res is None:
                report["flags"].append(f"{f.name}:{e['w']} 形状命中但值匹配失败")
                continue
            key, transposed, rel, second = res
            n_match += 1
            if key in patched_in_file:
                report["flags"].append(
                    f"{f.name}: {key} 被 {patched_in_file[key]} 与 {e['w']} 同时匹配（文件内二义）")
                continue
            mod = mod_by_hfkey.get(key)
            if mod is None:
                continue  # 匹配上了但不是 LoRA 目标（如 tied lm_head/embed）
            dW = dws[mod]
            dW_o = dW.T if transposed else dW
            target = Wd.astype(np.float64) + dW_o          # ONNX 存储摆位
            if quantized:
                q = target / s_b + (z_b if z_b is not None else 0.0)
                lo, hi = (0, 255) if inits[e["w"]].data_type == 2 else (-128, 127)
                clipped = (q < lo) | (q > hi)
                clip_frac = float(np.mean(clipped))
                qn = np.clip(np.rint(q), lo, hi)
                new_dtype = np.uint8 if lo == 0 else np.int8
                new_arr = qn.astype(new_dtype)
                # 自检：非削顶元素的反量化误差必须 <= 0.51*step（逐通道）；
                # 削顶元素重建到边界属预期行为，不计坏元素（其误差由 clip_frac 约束）
                recon = (qn.astype(np.float64) - (z_b if z_b is not None else 0.0)) * s_b
                err = np.abs(recon - target)
                bad = int(np.sum((err > 0.51 * s_b + 1e-12) & ~clipped))
                mse = float(np.mean((recon - target) ** 2))
            else:
                new_arr = target.astype(np.float32)
                clip_frac, bad, mse = 0.0, 0, 0.0
            rec = {"module": mod, "file": f.name, "w": e["w"], "hf_key": key,
                   "transposed": bool(transposed), "match_rel_err": round(rel, 6),
                   "quantized": bool(quantized), "clip_frac": round(clip_frac, 6),
                   "selfcheck_bad_elems": bad, "requant_mse": round(mse, 12)}
            patched_in_file[key] = e["w"]
            report["patched_modules"].append(rec)
            if not args.dry_run:
                inits[e["w"]].CopyFrom(numpy_helper.from_array(new_arr, e["w"]))
            n_patch += 1
            del Wd, target
        if not args.dry_run:
            dest = out_dir / rel_root / f.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            had_ext = (f.with_suffix(f.suffix + ".data").is_file()
                       or any(t_.HasField("external_data") and len(t_.external_data)
                              for t_ in m.graph.initializer))
            if n_patch:
                patched_files.add(f.name)
                if had_ext:
                    # 外置数据布局与原包同构：*.onnx + *.onnx.data（阈值以上的张量全部外置）
                    onnx.save_model(m, str(dest), save_as_external_data=True,
                                    all_tensors_to_one_file=True,
                                    location=f.name + ".data", size_threshold=10240)
                else:
                    onnx.save(m, str(dest))
            else:
                shutil.copy2(f, dest)   # 未被补丁的 onnx 原样字节复制
                ext = f.with_suffix(f.suffix + ".data")
                if ext.is_file():
                    shutil.copy2(ext, out_dir / rel_root / ext.name)
        report["files"][f.name] = {"matched": n_match, "patched": n_patch}
        log(f"  {f.name}: 形状候选命中并匹配 {n_match}，补丁 {n_patch}")
        del m, inits
        gc.collect()

    # ---- 覆盖率检查 ----
    mapped_mods = {rec["module"] for rec in report["patched_modules"]}
    report["unmapped_modules"] = sorted(set(dws) - mapped_mods)
    log(f"补丁覆盖 {len(mapped_mods)}/{len(dws)} 模块；未映射 {len(report['unmapped_modules'])}")
    if report["unmapped_modules"]:
        log("  未映射示例: " + "; ".join(report["unmapped_modules"][:5]))

    # ---- 复制其余文件（补丁过的 onnx 其 .data 已重新生成，跳过原始副本） ----
    if not args.dry_run:
        for p in sorted(model_dir.rglob("*")):
            if not p.is_file() or p.suffix == ".onnx":
                continue
            if p.name.endswith(".onnx.data") and p.name[: -len(".data")] in patched_files:
                continue
            rel = p.relative_to(model_dir)
            dest = out_dir / rel
            if dest.exists():
                continue  # 上一步已落盘的（未补丁 onnx 的 .data 等）不覆盖
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        # unified.json 标记：shuo 运行时识别统一包的依据（src-tauri/src/models.rs）
        from prompts import all_prompts

        marker = {
            "kind": "shuo-unified-asr",
            "base_package": str(model_dir),
            "base_hf_model": args.hf_model,
            "adapter": args.adapter,
            "lora": {"r": r, "alpha": alpha, "scaling": scaling},
            "prompts": all_prompts(),
            "patched_modules": len(mapped_mods),
            "files": report["files"],
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (out_dir / "unified.json").write_text(
            json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
        n_onnx_out = len(list((out_dir / rel_root).glob("*.onnx")))
        log(f"输出目录 {out_dir}: onnx {n_onnx_out} 个 + unified.json + 其余文件已复制")

    clips = [rec["clip_frac"] for rec in report["patched_modules"] if rec["quantized"]]
    bads = sum(rec["selfcheck_bad_elems"] for rec in report["patched_modules"])
    report["worst"] = {
        "clip_frac": round(max(clips), 6) if clips else 0.0,
        "selfcheck_bad_elems_total": bads,
        "match_rel_err_max": round(max((r_["match_rel_err"] for r_ in report["patched_modules"]),
                                       default=0.0), 6),
    }
    report["seconds"] = round(time.time() - t0, 1)
    ok = ((not report["unmapped_modules"])
          and report["worst"]["clip_frac"] < args.max_clip and bads == 0)
    report["ok"] = bool(ok)
    log("=" * 62)
    log(f"最坏削顶 {report['worst']['clip_frac']:.4%} | 自检坏元素 {bads} | "
        f"匹配误差最大 {report['worst']['match_rel_err_max']:.2e} | 用时 {report['seconds']}s")
    log("✔ 补丁完成，可交 verify_unified_onnx.py 验证" if ok else "✘ 补丁存在未映射/削顶超标/自检失败")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"报告 -> {args.report}")
    return 0 if ok else 1


def hf_key_for(index, mod):
    """与探针同规则：适配器模块名 -> HF key（缓存加速放调用方）。"""
    for cand in (f"thinker.{mod}.weight", f"{mod}.weight"):
        if cand in index:
            return cand
    return None


if __name__ == "__main__":
    _hard_exit(main())
