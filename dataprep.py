# prepare_llamafactory_pretrain_data.py
from datasets import load_dataset
from pathlib import Path
import json

# 1) Load your HF dataset (unchanged)
dataset = load_dataset("lukecarlate/english_finance_news")

# 2) Output paths
OUT_DIR = Path("data/my_corpus")
OUT_DIR.mkdir(parents=True, exist_ok=True)
train_path = OUT_DIR / "train.jsonl"

# 3) Keep ONLY raw text from 'newscontents'
def to_text(example):
    txt = example.get("newscontents")
    txt = "" if txt is None else str(txt).strip()
    return {"text": txt}

train = dataset["train"].map(to_text)
# drop anything that isn't 'text'
train = train.remove_columns([c for c in train.column_names if c != "text"])
# remove empties
train = train.filter(lambda ex: len(ex["text"]) > 0)

# 4) Write JSONL for LLaMA Factory pretrain
train.to_json(train_path.as_posix(), lines=True, force_ascii=False)

# 5) Minimal dataset_info.json for LLaMA Factory 0.9.3 (no 'formatting')
dataset_info = {
    "my_corpus": {
        "file_name": "my_corpus/train.jsonl",
        "columns": {"prompt": "text"}   # <- map PT "prompt" to our 'text'
    }
}
(OUT_DIR.parent / "dataset_info.json").write_text(
    json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8"
)

print("Wrote:", train_path, "and data/dataset_info.json")
