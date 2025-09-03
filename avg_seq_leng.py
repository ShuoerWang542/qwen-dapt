# avg_seq_len.py
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/my_corpus/train.jsonl",
                    help="Path to your JSONL file")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B",
                    help="HF model id or local path for tokenizer")
    ap.add_argument("--text_col", default="", help="Force text column name if not auto")
    ap.add_argument("--sample", type=int, default=0,
                    help="Use only first N rows (0 = all)")
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--gpus", type=int, default=1)
    args = ap.parse_args()

    # 1) Load dataset
    ds = load_dataset("json", data_files=args.data, split="train")

    # 2) Pick text column
    if args.text_col:
        col = args.text_col
    else:
        cols = set(ds.column_names)
        if "prompt" in cols:
            col = "prompt"
        elif "text" in cols:
            col = "text"
        else:
            raise KeyError(f"Can't find text column. Available: {ds.column_names}")

    # 3) Optionally subsample
    if args.sample and args.sample < len(ds):
        ds = ds.select(range(args.sample))

    # 4) Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 5) Count tokens (batched; single process for robustness)
    def count_fn(batch):
        ids = tok(batch[col], add_special_tokens=False).input_ids
        # when batched=True, input_ids is a list of lists
        return {"length": [len(x) for x in ids]}

    ds = ds.map(count_fn, batched=True)

    # 6) Stats
    lengths = np.array(ds["length"], dtype=np.int32)
    print(f"#samples: {len(lengths)}")
    print(f"avg tokens:     {lengths.mean():.2f}")
    print(f"median tokens:  {np.median(lengths):.2f}")
    print(f"p95 tokens:     {np.percentile(lengths, 95):.2f}")
    print(f"max tokens:     {lengths.max()}")

    # 7) Tokens/update estimate (with packing=True assumption)
    tokens_per_update = args.seq_len * args.batch_size * args.grad_accum * args.gpus
    print(f"\nEst. tokens/update (seq_len={args.seq_len}, batch={args.batch_size}, "
          f"grad_accum={args.grad_accum}, gpus={args.gpus}): {tokens_per_update:,}")

if __name__ == "__main__":
    main()
