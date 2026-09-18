"""Download and normalize a small Vietnamese SFT mixture from Hugging Face."""
import argparse
import hashlib
import json
from pathlib import Path

SYSTEM = "Bạn là trợ lý AI. Hãy trả lời rõ ràng, hữu ích và lịch sự bằng tiếng Việt."

SOURCES = [
    ("TTP01/Vietverse-SFT-1K-Gold", 1000),
    ("nguyenphuthien/vietnamese_no_robots", 3000),
    ("vnu-llm2023-ftdata/qa-daotao-sft", 2000),
]

def text(value):
    return str(value or "").replace("\\n", "\n").strip()

def convert(row):
    messages = row.get("messages") or row.get("conversations")
    if isinstance(messages, list):
        user = next((text(x.get("content")) for x in messages
                     if isinstance(x, dict) and x.get("role") == "user"), "")
        assistant = next((text(x.get("content")) for x in messages
                          if isinstance(x, dict) and x.get("role") in {"assistant", "gpt"}), "")
        if user and assistant:
            return user, assistant
    user = text(row.get("prompt") or row.get("instruction") or row.get("question") or row.get("input"))
    assistant = text(row.get("response") or row.get("output") or row.get("answer") or row.get("reference"))
    return (user, assistant) if user and assistant else ("", "")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/sft_vietnamese_mix.jsonl"))
    args = parser.parse_args()
    from datasets import load_dataset
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen, kept = set(), 0
    with args.output.open("w", encoding="utf-8", newline="\n") as dst:
        for dataset_id, limit in SOURCES:
            print(f"Loading {dataset_id} (limit={limit})", flush=True)
            rows = load_dataset(dataset_id, split="train", streaming=True)
            source_kept = 0
            for row in rows:
                user, assistant = convert(row)
                if len(user) < 8 or len(assistant) < 20:
                    continue
                key = hashlib.sha256((user + "\n" + assistant).casefold().encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                dst.write(json.dumps({"system": SYSTEM, "user": user, "assistant": assistant}, ensure_ascii=False) + "\n")
                kept += 1; source_kept += 1
                if source_kept >= limit:
                    break
            print(f"  kept={source_kept}", flush=True)
    print(f"Saved {kept:,} samples to {args.output}")

if __name__ == "__main__":
    main()
