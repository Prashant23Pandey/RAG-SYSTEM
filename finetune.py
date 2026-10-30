"""
Fine-Tuning a Custom Domain Model with Unsloth / LoRA.

Converts Langfuse traces into training data and performs QLoRA fine-tuning.

Usage:
    python finetune.py convert --input traces.jsonl --output training.jsonl
    python finetune.py train --data training.jsonl --base-model unsloth/Llama-3.2-1B
    python finetune.py export --model ./finetuned_model --format gguf
"""

import os
import sys
import json
import argparse
import logging
import torch
from pathlib import Path
from typing import Optional

logger = logging.getLogger("rag.finetune")


# Step 1: Convert Langfuse traces to training data
def convert_langfuse_traces(traces_path, output_path, fmt="chatml"):
    """Convert Langfuse trace export into Alpaca/ChatML training JSONL."""
    if not os.path.isfile(traces_path):
        raise FileNotFoundError(f"Traces file not found: {traces_path}")

    training_data = []
    skipped = 0

    with open(traces_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                trace = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            trace_input = trace.get("input", {})
            trace_output = trace.get("output", {})
            question = trace_input.get("question", "")
            context = trace_input.get("context", "")
            answer = trace_output.get("generation", "")

            if not question or not answer:
                skipped += 1
                continue

            if fmt == "chatml":
                sys_msg = "You are a helpful RAG assistant. Answer based ONLY on the provided context. If insufficient, say so."
                entry = {
                    "messages": [
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": "Context:\n" + context + "\n\nQuestion: " + question},
                        {"role": "assistant", "content": answer},
                    ],
                    "metadata": {"source": "langfuse_trace", "trace_name": trace.get("name", "")},
                }
            else:
                entry = {
                    "instruction": "Answer based on the provided context.",
                    "input": "Context:\n" + context + "\n\nQuestion: " + question,
                    "output": answer,
                }
            training_data.append(entry)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in training_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Converted {len(training_data)} traces ({skipped} skipped)")
    return {"converted": len(training_data), "skipped": skipped, "output_path": output_path}


# Step 2: QLoRA Fine-Tuning with Unsloth
def train_with_unsloth(data_path, base_model="unsloth/Llama-3.2-1B-Instruct",
                       output_dir="./finetuned_model", epochs=3, learning_rate=2e-4,
                       batch_size=2, max_seq_length=2048, lora_r=16, lora_alpha=32,
                       lora_dropout=0.05):
    """Fine-tune a model using Unsloth + QLoRA."""
    try:
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset
    except ImportError as e:
        return {"error": "Unsloth not installed: " + str(e)}

    print("\n" + "=" * 60)
    print("  QLoRA Fine-Tuning with Unsloth")
    print("  Base Model: " + base_model)
    print("  Output: " + output_dir)
    print("=" * 60 + "\n")

    print("[1/5] Loading base model with 4-bit quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model, max_seq_length=max_seq_length,
        dtype=None, load_in_4bit=True,
    )

    print("[2/5] Configuring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model, r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    print("[3/5] Loading training dataset...")
    dataset = load_dataset("json", data_files=data_path, split="train")

    def format_chatml(example):
        if "messages" in example:
            return tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
        instr = example.get("instruction", "")
        inp = example.get("input", "")
        out = example.get("output", "")
        msgs = [
            {"role": "system", "content": instr},
            {"role": "user", "content": inp},
            {"role": "assistant", "content": out},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    print("[4/5] Formatting dataset...")
    dataset = dataset.map(lambda x: {"text": format_chatml(x)})

    print("[5/5] Starting training...")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=-1,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nModel saved to: {output_dir}")
    return {"output_dir": output_dir, "epochs": epochs}


# Step 3: Export to GGUF for Ollama/vLLM
def export_to_gguf(model_dir, output_name="finetuned_model", quantization="q4_k_m"):
    """Convert fine-tuned model to GGUF format using llama.cpp."""
    print("\nExporting to GGUF format...")
    print(f"  Model: {model_dir}")
    print(f"  Quantization: {quantization}")

    gguf_script = os.path.join(model_dir, "convert_hf_to_gguf.py")
    cmd = (
        f"python {gguf_script} {model_dir} "
        f"--outfile {output_name}.gguf "
        f"--outtype {quantization}"
    )
    print(f"  Command: {cmd}")

    ret = os.system(cmd)
    if ret == 0:
        print(f"  Exported: {output_name}.gguf")
    else:
        print("  Export failed. Install llama.cpp for GGUF conversion.")
    return {"status": "ok" if ret == 0 else "failed", "output": f"{output_name}.gguf"}


# CLI
def main():
    parser = argparse.ArgumentParser(description="Fine-tune models for Agentic RAG")
    subs = parser.add_subparsers(dest="command")

    # convert
    p_conv = subs.add_parser("convert", help="Convert traces to training data")
    p_conv.add_argument("--input", required=True, help="Traces JSONL file")
    p_conv.add_argument("--output", required=True, help="Output training JSONL")
    p_conv.add_argument("--format", default="chatml", choices=["chatml", "alpaca"])

    # train
    p_train = subs.add_parser("train", help="Fine-tune with QLoRA")
    p_train.add_argument("--data", required=True, help="Training data JSONL")
    p_train.add_argument("--base-model", default="unsloth/Llama-3.2-1B-Instruct")
    p_train.add_argument("--output-dir", default="./finetuned_model")
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--batch-size", type=int, default=2)
    p_train.add_argument("--max-seq-len", type=int, default=2048)

    # export
    p_exp = subs.add_parser("export", help="Export to GGUF")
    p_exp.add_argument("--model", required=True, help="Model directory")
    p_exp.add_argument("--name", default="finetuned_model")
    p_exp.add_argument("--quantization", default="q4_k_m")

    args = parser.parse_args()

    if args.command == "convert":
        convert_langfuse_traces(args.input, args.output, args.format)
    elif args.command == "train":
        train_with_unsloth(args.data, args.base_model, args.output_dir,
                           args.epochs, args.lr, args.batch_size, args.max_seq_len)
    elif args.command == "export":
        export_to_gguf(args.model, args.name, args.quantization)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
