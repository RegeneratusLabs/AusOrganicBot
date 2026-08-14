#!/usr/bin/env python3
"""
Strip verbatim source excerpts (the "context" field) from a QA dataset,
producing the publishable variant for the Hugging Face Hub.

Usage:
    python3 dataset/strip_context.py qa_au.jsonl   # -> qa_au_publish.jsonl
"""

import json
import sys
from pathlib import Path


def main():
    src = Path(sys.argv[1])
    dst = src.with_name(f"{src.stem}_publish{src.suffix}")
    with open(src) as f_in, open(dst, "w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec.pop("context", None)
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
