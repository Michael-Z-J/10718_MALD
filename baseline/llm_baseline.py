"""
LLM baseline: what many researchers already do before submitting.

For each test paper, give an off-the-shelf instruction-tuned LLM the full paper text with a plain
prompt asking for its five most important weaknesses, each with a severity from 1 to 5, and rank
them by severity. Paper text comes from evaluation/fetch_pdfs.py (data/fulltext/<paper_id>.txt).

Writes one JSON line per paper in the shared format read by evaluation/match.py:
  {"paper_id", "issues": [5 strings, most severe first], "severities": [...], "truncated", "raw"}
A paper whose response can't be parsed (after one sampled retry) gets "issues": [] and "error",
so it scores 0 rather than being dropped. Re-running resumes: papers already in --out are skipped.

Usage:
    python baseline/llm_baseline.py --model Qwen/Qwen3-4B
    cd evaluation && python match.py outputs/llm_Qwen3-4B.jsonl --matcher embedding \\
        --paper-ids data/sample_2026_50.json
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "evaluation"))
from common import DATA_DIR, parse_json, read_jsonl  # noqa: E402

PROMPT = """Below is the full text of a paper submitted to ICLR. Acting as a peer reviewer, list the five \
most important weaknesses of this paper. For each, give a severity from 1 (minor) to 5 (critical).
Return only JSON, with no other text: [{{"weakness": "...", "severity": 1-5}}, ...]

PAPER:
{paper}"""


def load_model(name, load_in_4bit):
    tokenizer = AutoTokenizer.from_pretrained(name)
    kwargs = {"device_map": "auto", "dtype": torch.bfloat16}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig  # needs `pip install bitsandbytes`
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    return tokenizer, AutoModelForCausalLM.from_pretrained(name, **kwargs)


def truncate(tokenizer, text, max_tokens):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text, False
    return tokenizer.decode(ids[:max_tokens]), True


def generate(tokenizer, model, paper, max_new_tokens, sample):
    messages = [{"role": "user", "content": PROMPT.format(paper=paper)}]
    # enable_thinking=False turns off Qwen3's reasoning mode; other chat templates ignore it.
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                           return_tensors="pt", return_dict=True).to(model.device)
    gen = {"do_sample": True, "temperature": 0.7, "top_p": 0.8} if sample else {"do_sample": False}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, **gen)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_issues(raw, n):
    """[(weakness, severity)] sorted most severe first (stable, so ties keep the model's order)."""
    items = parse_json(raw)
    if isinstance(items, dict):  # e.g. {"weaknesses": [...]}
        items = next(v for v in items.values() if isinstance(v, list))
    issues = [(str(it["weakness"]).strip(), int(it["severity"])) for it in items
              if isinstance(it, dict) and it.get("weakness")]
    if not issues:
        raise ValueError("no weaknesses in response")
    return sorted(issues, key=lambda x: -x[1])[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HuggingFace model id, e.g. Qwen/Qwen3-4B")
    ap.add_argument("--load_in_4bit", action="store_true", help="quantize weights to fit larger models in memory")
    ap.add_argument("--paper_ids", default=os.path.join(DATA_DIR, "sample_2026_50.json"),
                    help="JSON list of paper_ids to review (written by evaluation/fetch_pdfs.py)")
    ap.add_argument("--text_dir", default=os.path.join(DATA_DIR, "fulltext"))
    ap.add_argument("--max_input_tokens", type=int, default=24000,
                    help="paper text beyond this is cut (recorded as truncated)")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--n_issues", type=int, default=5)
    ap.add_argument("--out", default=None, help="default: evaluation/outputs/llm_<model name>.jsonl")
    args = ap.parse_args()

    with open(args.paper_ids) as f:
        ids = json.load(f)
    out_path = args.out or os.path.join(HERE, "..", "evaluation", "outputs", f"llm_{args.model.split('/')[-1]}.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    done = {r["paper_id"] for r in read_jsonl(out_path)} if os.path.exists(out_path) else set()
    todo = [i for i in ids if i not in done and os.path.exists(os.path.join(args.text_dir, f"{i}.txt"))]
    missing = sum(not os.path.exists(os.path.join(args.text_dir, f"{i}.txt")) for i in ids)
    print(f"{len(ids)} papers: {len(done)} already done, {missing} without text, {len(todo)} to review")

    tokenizer, model = load_model(args.model, args.load_in_4bit)
    with open(out_path, "a", encoding="utf-8") as f:
        for n, pid in enumerate(todo, 1):
            with open(os.path.join(args.text_dir, f"{pid}.txt"), encoding="utf-8") as tf:
                paper, truncated = truncate(tokenizer, tf.read(), args.max_input_tokens)
            row = {"paper_id": pid, "truncated": truncated}
            for sample in (False, True):  # greedy first; one sampled retry if the output won't parse
                raw = generate(tokenizer, model, paper, args.max_new_tokens, sample)
                try:
                    issues = parse_issues(raw, args.n_issues)
                    row.update(issues=[w for w, _ in issues], severities=[s for _, s in issues], raw=raw)
                    break
                except (ValueError, KeyError, TypeError, StopIteration) as e:
                    row.update(issues=[], severities=[], raw=raw, error=f"{type(e).__name__}: {e}")
            if row["issues"]:
                row.pop("error", None)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{n}/{len(todo)}] {pid}: {len(row['issues'])} issues"
                  + (" (truncated)" if truncated else "") + (f" ERROR {row['error']}" if "error" in row else ""))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
