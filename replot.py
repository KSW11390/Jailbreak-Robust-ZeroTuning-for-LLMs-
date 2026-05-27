#!/usr/bin/env python3
"""
Regenerate plots from saved JSON (no model needed).
Run locally:  python3 replot.py
Run on server for attention maps: python3 replot.py --attention-only
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-json", default="outputs/jailbreak_sweep_results.json")
    p.add_argument("--output-dir",   default="outputs")
    p.add_argument("--attention-only", action="store_true",
                   help="Skip JSON plots, only re-run attention map extraction")
    p.add_argument("--model-id",     default="lmsys/vicuna-13b-v1.5")
    p.add_argument("--modelling-file", default="modelling_llama_open.py")
    p.add_argument("--hf-token",     default=os.environ.get("HF_TOKEN", ""))
    return p.parse_args()


# ── Entropy data (hardcoded from run log) ────────────────────────
ENTROPY_DATA = {
    0.25: 2.3257, 0.50: 2.0553, 0.75: 2.0634,
    1.00: 2.0878, 1.50: 2.1451, 2.00: 2.2079,
    2.50: 2.2767, 3.00: 2.3334, 4.00: 2.3758, 5.00: 2.3922,
}
OPTIMAL_RATE = 0.50


def plot_entropy(output_dir):
    xs = list(ENTROPY_DATA.keys())
    ys = list(ENTROPY_DATA.values())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys, marker="o", linewidth=2, color="steelblue")
    ax.axvline(OPTIMAL_RATE, linestyle="--", color="red",
               label=f"Optimal γ={OPTIMAL_RATE} (min entropy)")
    ax.axvline(1.0, linestyle=":", color="gray", alpha=0.7, label="Baseline γ=1.0")
    ax.set_xlabel("γ (rate)")
    ax.set_ylabel("Avg Next-Token Entropy")
    ax.set_title("ZeroTuning Entropy Curve — Vicuna-13b-v1.5 (Benign Calibration)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{output_dir}/entropy_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[+] {path}")


def plot_asr(results, output_dir):
    rates     = sorted(float(r) for r in results["GCG"])
    gcg_asr   = [results["GCG"][str(r)]["asr"] * 100 for r in rates]
    pair_asr  = [results["PAIR"][str(r)]["asr"] * 100 for r in rates]
    base_gcg  = results["GCG"]["1.0"]["asr"] * 100
    base_pair = results["PAIR"]["1.0"]["asr"] * 100

    # Plot 1: absolute ASR
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rates, gcg_asr,  marker="o", linewidth=2, color="steelblue", label="GCG")
    ax.plot(rates, pair_asr, marker="s", linewidth=2, color="coral",     label="PAIR")
    ax.axvline(1.0, linestyle="--", color="gray", alpha=0.6, label="Baseline (γ=1.0)")
    for x, y in zip(rates, gcg_asr):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
    for x, y in zip(rates, pair_asr):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8)
    ax.set_xlabel("γ (rate)"); ax.set_ylabel("ASR (%)")
    ax.set_title("ASR vs ZeroTuning Rate\n(Vicuna-13b-v1.5)")
    ax.legend(); ax.set_ylim(75, 95); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path1 = f"{output_dir}/asr_sweep.png"
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] {path1}")

    # Plot 2: delta vs baseline
    non_base   = [r for r in rates if r != 1.0]
    gcg_delta  = [results["GCG"][str(r)]["asr"] * 100 - base_gcg   for r in non_base]
    pair_delta = [results["PAIR"][str(r)]["asr"] * 100 - base_pair  for r in non_base]
    x_idx = range(len(non_base))
    width = 0.35

    fig, ax2 = plt.subplots(figsize=(10, 5))
    bars1 = ax2.bar([i - width / 2 for i in x_idx], gcg_delta,  width,
                    label="GCG",  color="steelblue", alpha=0.8)
    bars2 = ax2.bar([i + width / 2 for i in x_idx], pair_delta, width,
                    label="PAIR", color="coral",     alpha=0.8)
    for bar, val in zip(list(bars1) + list(bars2), gcg_delta + pair_delta):
        sign = "+" if val >= 0 else ""
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.3 if val >= 0 else -1.5),
                 f"{sign}{val:.1f}pp", ha="center", fontsize=7.5)
    ax2.set_xticks(list(x_idx))
    ax2.set_xticklabels([str(r) for r in non_base])
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("γ (rate)"); ax2.set_ylabel("ASR Change (pp, vs baseline)")
    ax2.set_title("ZeroTuning Effect (vs Baseline γ=1.0)\n(Vicuna-13b-v1.5)")
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path2 = f"{output_dir}/asr_effect.png"
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[+] {path2}")


def replot_attention(args):
    """Re-extract attention maps with updated colormap. Requires model on GPU."""
    import sys, importlib.util, torch
    from transformers import AutoTokenizer
    from transformers.generation.utils import GenerationMixin

    spec = importlib.util.spec_from_file_location("modelling_llama_open", args.modelling_file)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["modelling_llama_open"] = mod
    spec.loader.exec_module(mod)
    LlamaForCausalLM = mod.LlamaForCausalLM

    _orig = GenerationMixin._validate_model_kwargs
    def _patched(self, kw):
        _orig(self, {k: v for k, v in kw.items() if k != "input_len"})
    GenerationMixin._validate_model_kwargs = _patched

    token = args.hf_token or None
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(
        args.model_id, token=token, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    print(f"[+] Model loaded — GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    with open(args.results_json) as f:
        results = json.load(f)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for attack in ("GCG", "PAIR"):
        for rate_str in ("0.5", "1.0"):
            rate = float(rate_str)
            details = results[attack][rate_str]["details"]
            success = next((d for d in details if d["jailbroken"]), None)
            failure = next((d for d in details if not d["jailbroken"]), None)

            for d, kind in [(success, "success"), (failure, "failure")]:
                if d is None:
                    print(f"  [{attack}] no {kind} sample at rate={rate}")
                    continue
                _save_attention(model, tokenizer, d["goal"], d["goal"],
                                f"{attack}_{kind}", rate, args.output_dir)


def _save_attention(model, tokenizer, prompt, goal, label, rate, output_dir):
    import torch
    try:
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256
        ).to(model.device)
        with torch.no_grad():
            out = model(**inputs, input_len=(0, 0, rate, None, None), output_attentions=True)

        last_attn = out.attentions[-1][0].mean(0).cpu().float().numpy()
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        n = min(64, len(tokens))

        fig, ax = plt.subplots(figsize=(14, 12))
        im = ax.imshow(last_attn[:n, :n], cmap="coolwarm", aspect="auto", vmin=0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(tokens[:n], rotation=90, fontsize=5)
        ax.set_yticklabels(tokens[:n], fontsize=5)
        ax.set_title(f"{label} | rate={rate}  |  {goal[:80]}", fontsize=9)
        plt.tight_layout()
        path = f"{output_dir}/attn_{label}_rate{rate}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[+] {path}")
    except Exception as e:
        print(f"[!] attention failed ({label}): {e}")


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not args.attention_only:
        with open(args.results_json) as f:
            results = json.load(f)
        print("[*] Plotting entropy curve...")
        plot_entropy(args.output_dir)
        print("[*] Plotting ASR sweep...")
        plot_asr(results, args.output_dir)

    if args.attention_only:
        print("[*] Re-extracting attention maps...")
        replot_attention(args)

    print("[done]")


if __name__ == "__main__":
    main()
