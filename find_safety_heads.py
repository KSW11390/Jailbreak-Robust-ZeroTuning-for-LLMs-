#!/usr/bin/env python3
"""
Safety Head Detection for vicuna-13b-v1.5
Based on methodology from arxiv:2508.19697 (RDSHA / AHD)

Algorithm:
  1. Refusal direction per layer:
       r_l = normalize(mean(h_harmful_l) - mean(h_benign_l))
       h_l = last-token hidden state at layer l
  2. Head contribution score:
       For head (l, h):
         head_contrib = pre_proj[:, h*d:(h+1)*d] @ W_O[:, h*d:(h+1)*d].T
         score(l,h)   = E_harmful [ |dot(head_contrib_last_token, r_{l+1})| ]
  3. Top-K selection → safety_heads.json

Usage:
  python3 find_safety_heads.py
  python3 find_safety_heads.py --top-k 30 --n-samples 80
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaForCausalLM


# ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-id",   default="lmsys/vicuna-13b-v1.5")
    p.add_argument("--chat-template", default="vicuna",
                   choices=["vicuna", "llama2"],
                   help="Chat template to use for prompt formatting")
    p.add_argument("--hf-token",   default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--top-k",      type=int, default=20,
                   help="Number of safety heads to select")
    p.add_argument("--n-samples",  type=int, default=50,
                   help="Harmful/benign samples to use (each)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output",     default="outputs/safety_heads.json")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
_VICUNA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
_LLAMA2_SYSTEM = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully "
    "as possible, while being safe. Your answers should not include any harmful, "
    "unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure "
    "that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain "
    "why instead of answering something not correct. If you don't know the answer "
    "to a question, please don't share false information."
)

def apply_template(prompt: str, template: str) -> str:
    if template == "llama2":
        return f"[INST] <<SYS>>\n{_LLAMA2_SYSTEM}\n<</SYS>>\n\n{prompt} [/INST]"
    return f"{_VICUNA_SYSTEM} USER: {prompt} ASSISTANT:"


# ──────────────────────────────────────────────────────────────────
def load_data(n: int):
    from datasets import load_dataset
    harmful = [r["Goal"] for r in load_dataset(
        "JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")][:n]
    benign  = [r["Goal"] for r in load_dataset(
        "JailbreakBench/JBB-Behaviors", "behaviors", split="benign")][:n]
    print(f"[✓] harmful={len(harmful)}, benign={len(benign)}")
    return harmful, benign


# ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def collect_hidden_states(model, tokenizer, prompts, batch_size, chat_template="vicuna"):
    """Returns (n_samples, num_layers+1, hidden_size) on CPU."""
    all_hidden = []
    for start in range(0, len(prompts), batch_size):
        batch = [apply_template(p, chat_template) for p in prompts[start:start+batch_size]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(model.device)
        out = model(**inputs, output_hidden_states=True)
        # padding_side="left" → last real token is always at index -1
        for i in range(inputs.input_ids.shape[0]):
            states = torch.stack(
                [h[i, -1, :].cpu().float() for h in out.hidden_states]
            )  # (num_layers+1, hidden_size)
            all_hidden.append(states)
        print(f"  {start + inputs.input_ids.shape[0]}/{len(prompts)}")
    return torch.stack(all_hidden)  # (n, L+1, H)


def compute_refusal_dirs(harmful_h, benign_h):
    """Per-layer normalized refusal direction."""
    dirs = []
    for l in range(harmful_h.shape[1]):
        diff = harmful_h[:, l].mean(0) - benign_h[:, l].mean(0)
        norm = diff.norm()
        dirs.append(diff / norm if norm > 1e-8 else diff)
    return dirs  # list of (H,) tensors


# ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def score_heads(model, tokenizer, harmful_prompts, refusal_dirs, batch_size, chat_template="vicuna"):
    """
    Returns scores[layer][head] = mean |dot(head_contrib, refusal_dir_{l+1})|
    """
    cfg      = model.config
    n_layers = cfg.num_hidden_layers
    n_heads  = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads

    scores = [[0.0] * n_heads for _ in range(n_layers)]
    counts = [[0]   * n_heads for _ in range(n_layers)]

    # Hook: capture input to o_proj (= pre-projection concatenated head outputs)
    pre_proj = {}

    def make_hook(l):
        def hook(module, inp, _out):
            # inp[0]: (bsz, seq, hidden) — the concat of all head outputs before W_O
            pre_proj[l] = inp[0].detach().cpu().float()
        return hook

    handles = [
        model.model.layers[l].self_attn.o_proj.register_forward_hook(make_hook(l))
        for l in range(n_layers)
    ]
    try:
        for start in range(0, len(harmful_prompts), batch_size):
            batch = [apply_template(p, chat_template) for p in harmful_prompts[start:start+batch_size]]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(model.device)
            bsz = inputs.input_ids.shape[0]
            pre_proj.clear()
            model(**inputs)

            for l in range(n_layers):
                pp   = pre_proj[l][:, -1, :]           # (bsz, hidden) — last token
                W_O  = model.model.layers[l].self_attn.o_proj.weight.detach().cpu().float()
                # W_O: (out=hidden, in=hidden)  →  head h uses columns h*d:(h+1)*d of in
                r    = refusal_dirs[l + 1]              # direction at layer l+1 output

                for h in range(n_heads):
                    s = h * head_dim
                    e = s + head_dim
                    contrib = pp[:, s:e] @ W_O[:, s:e].T   # (bsz, hidden)
                    dots    = (contrib * r).sum(-1).abs()   # (bsz,)
                    scores[l][h] += dots.sum().item()
                    counts[l][h] += bsz

            done = start + bsz
            print(f"  {done}/{len(harmful_prompts)}")
    finally:
        for h in handles:
            h.remove()

    for l in range(n_layers):
        for h in range(n_heads):
            if counts[l][h] > 0:
                scores[l][h] /= counts[l][h]
    return scores


# ──────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not found")
    print(f"[GPU] {torch.cuda.get_device_name(0)}")

    token = args.hf_token or None
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, token=token, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token

    print(f"[*] Loading {args.model_id} (fp16)...")
    model = LlamaForCausalLM.from_pretrained(
        args.model_id, token=token,
        torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()
    print(f"[✓] {torch.cuda.memory_allocated()/1e9:.1f} GB")

    harmful, benign = load_data(args.n_samples)

    print(f"[*] Chat template: {args.chat_template}")
    print("\n[*] Collecting hidden states — harmful...")
    h_harmful = collect_hidden_states(model, tokenizer, harmful, args.batch_size, args.chat_template)
    print("[*] Collecting hidden states — benign...")
    h_benign  = collect_hidden_states(model, tokenizer, benign,  args.batch_size, args.chat_template)

    print("[*] Computing refusal directions...")
    refusal_dirs = compute_refusal_dirs(h_harmful, h_benign)

    print("[*] Scoring attention heads...")
    scores = score_heads(model, tokenizer, harmful, refusal_dirs, args.batch_size, args.chat_template)

    # ── Rank ───────────────────────────────────────────────────────
    flat = [
        {"layer": l, "head": h, "score": scores[l][h]}
        for l in range(len(scores))
        for h in range(len(scores[l]))
    ]
    flat.sort(key=lambda x: -x["score"])
    top_k = flat[:args.top_k]

    print(f"\nTop-{args.top_k} safety heads:")
    for e in top_k:
        print(f"  layer={e['layer']:2d}  head={e['head']:2d}  score={e['score']:.4f}")

    result = {
        "model":        args.model_id,
        "top_k":        args.top_k,
        "n_samples":    args.n_samples,
        "safety_heads": top_k,
        "all_scores":   [[scores[l][h] for h in range(len(scores[l]))]
                         for l in range(len(scores))],
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[✓] Saved: {args.output}")


if __name__ == "__main__":
    main()
