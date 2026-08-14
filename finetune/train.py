#!/usr/bin/env python3
"""
Fine-tune MiniCPM5-1B (OpenBMB) as the AusOrganicBot RAG reader via QLoRA.

The training format is aligned 1:1 with production inference: the user turn
is the app's own RAG prompt template (with the "Answer:" suffix stripped),
and the assistant turn is the cited answer or the refusal template. This
teaches the model to answer only from context and to refuse when the context
does not cover the question.

Data:    dataset/qa_<country>_train.jsonl (from dataset/generate_qa.py)
Runtime: Unsloth QLoRA, 4-bit, ~8GB VRAM (tested target: RTX 3060 6GB with
         --lora-r 8 and max_seq_length 1536)

Setup (isolated venv, does not touch the app environment):

    python3 -m venv finetune/.venv
    finetune/.venv/bin/pip install -r finetune/requirements.txt

Usage:

    finetune/.venv/bin/python finetune/train.py --data dataset/qa_au_train.jsonl \
        --output finetune/runs/ausorganicbot-minicpm5 \
        --epochs 2 --max-steps 0

Smoke test (proves the pipeline on any machine):

    finetune/.venv/bin/python finetune/train.py --data dataset/qa_au_train.jsonl \
        --output finetune/runs/smoke --max-steps 5 --max-seq-length 1024
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.prompts import RAG_PROMPT, NOT_COVERED_ANSWER  # noqa: E402

MODEL_ID = "openbmb/MiniCPM5-1B"
USER_TEMPLATE, _, _ = RAG_PROMPT.rpartition("Answer:")

SYSTEM_MESSAGE = (
    "You are AusOrganicBot, an assistant for organic certification queries. "
    "Answer only from the provided context. When the context contains the answer, "
    "be precise and cite section numbers. If the context does NOT contain the "
    "answer, say exactly: "
    f'"{NOT_COVERED_ANSWER}"'
)


def load_pairs(path: Path):
    records = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        records.append(
            {
                "question": rec["question"],
                "answer": rec["answer"],
                "context": rec.get("context", ""),
                "type": rec.get("type", "positive"),
            }
        )
    return records


def to_messages(record, standards_block):
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                country_label="organic certification",
                standards_block=standards_block,
                language_instruction="",
                context=record["context"][:3000],
                question=record["question"],
            ),
        },
        {"role": "assistant", "content": record["answer"]},
    ]


def make_dataset(records, tokenizer, standards_block, max_seq_length):
    from datasets import Dataset

    ds = Dataset.from_list(
        [
            {"messages": to_messages(r, standards_block)}
            for r in records
        ]
    )

    def formatting_func(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            texts.append(text)
        return {"text": texts}

    ds = ds.map(formatting_func, batched=True)
    return ds


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune MiniCPM5-1B as AusOrganicBot")
    parser.add_argument("--data", required=True, help="Path to qa_*_train.jsonl")
    parser.add_argument("--output", required=True, help="Output dir for the adapter + merged model")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = full dataset")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save-every", type=int, default=0, help="0 = final only")
    args = parser.parse_args()

    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from unsloth.chat_templates import train_on_responses_only
        import torch
    except ImportError:
        raise SystemExit(
            "Unsloth is not installed. Set up the isolated venv first:\n"
            "  python3 -m venv finetune/.venv\n"
            "  finetune/.venv/bin/pip install -r finetune/requirements.txt"
        )

    if not torch.cuda.is_available():
        raise SystemExit("Training requires a CUDA GPU (QLoRA 4-bit). "
                         "MiniCPM5-1B needs ~6GB VRAM with --lora-r 8.")

    records = load_pairs(Path(args.data))
    n_pos = sum(1 for r in records if r["type"] == "positive")
    n_neg = len(records) - n_pos
    print(f"Loaded {len(records)} pairs ({n_pos} positives, {n_neg} refusal negatives)")

    standards_block = "Standards documents for organic certification (see context)."

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_r,
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    ds = make_dataset(records, tokenizer, standards_block, args.max_seq_length)

    from trl import SFTConfig, SFTTrainer

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=1,
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=5,
            max_steps=args.max_steps if args.max_steps else max(1, int(args.epochs * len(ds) / (args.batch_size * args.grad_accum))),
            learning_rate=args.lr,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=5,
            save_strategy="steps" if args.save_every else "no",
            save_steps=args.save_every or 500,
            output_dir=str(Path(args.output) / "checkpoints"),
            seed=42,
            report_to="none",
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("\nTraining...")
    trainer.train()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    model.save_pretrained_merged(out / "merged", tokenizer, save_method="merged_16bit")
    print(f"\nDone. Adapter: {out / 'adapter'}  Merged: {out / 'merged'}")
    print(f"Next: finetune/.venv/bin/python finetune/export_gguf.py "
          f"--model-dir {out / 'merged'} --output {out / 'ausorganicbot-minicpm5.Q4_K_M.gguf'}")


if __name__ == "__main__":
    main()
