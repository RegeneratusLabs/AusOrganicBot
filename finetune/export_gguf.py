#!/usr/bin/env python3
"""
Export a fine-tuned (merged) model to GGUF Q4_K_M for Ollama / llama.cpp.

Usage:
    finetune/.venv/bin/python finetune/export_gguf.py \
        --model-dir finetune/runs/orgstandardsbot-minicpm5/merged \
        --output finetune/runs/orgstandardsbot-minicpm5/orgstandardsbot-minicpm5.Q4_K_M.gguf

Requires llama.cpp — pass --llama-cpp <dir> or it clones the repo on first
use into finetune/llama.cpp.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LLAMACPP = Path(__file__).resolve().parent / "llama.cpp"


def run(cmd, cwd=None):
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description="GGUF export + quantize")
    parser.add_argument("--model-dir", required=True, help="Merged HF model directory")
    parser.add_argument("--output", required=True, help="Output .gguf path")
    parser.add_argument("--llama-cpp", default=str(DEFAULT_LLAMACPP))
    parser.add_argument("--quant", default="Q4_K_M")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"No HF model found at {model_dir} — run train.py first")

    llama = Path(args.llama_cpp)
    if not (llama / "convert_hf_to_gguf.py").exists():
        print(f"Cloning llama.cpp into {llama}...")
        run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", str(llama)])

    f16 = Path(args.output).with_suffix(".f16.gguf")
    print("Converting to f16 GGUF...")
    run([
        sys.executable, str(llama / "convert_hf_to_gguf.py"),
        str(model_dir), "--outfile", str(f16), "--outtype", "f16",
    ])

    print(f"Quantizing to {args.quant}...")
    quant_bin = llama / "build" / "bin" / "llama-quantize"
    if not quant_bin.exists():
        run(["cmake", "-B", "build", "-DGGML_CUDA=OFF"], cwd=llama)
        run(["cmake", "--build", "build", "--config", "Release", "-j", "4",
             "--target", "llama-quantize"], cwd=llama)
    run([str(quant_bin), str(f16), str(args.output), args.quant])

    f16.unlink(missing_ok=True)
    print(f"\nGGUF ready: {args.output}")
    print("Ollama:   ollama create orgstandardsbot -f finetune/Modelfile")
    print("llama.cpp: llama-server -m <gguf> --port 11434")


if __name__ == "__main__":
    main()
