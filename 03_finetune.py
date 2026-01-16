"""
03_finetune.py - Fine-tune distilgpt2 on tool-call examples
"""

import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

MODEL_NAME = "distilgpt2"
OUTPUT_DIR = "./finetuned_model"
EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 5e-5


def load_training_data(path):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def format_for_training(example):
    return f"User request: {example['prompt']}\nTool call JSON: {example['completion']}"


def main():
    print("=" * 60)
    print("FINE-TUNING: Teaching distilgpt2 to output tool calls")
    print("=" * 60)

    print(f"\nLoading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    data_path = Path("data/tool_calls.jsonl")
    if not data_path.exists():
        print("Error: Training data not found. Run 02_create_dataset.py first.")
        return

    examples = load_training_data(data_path)
    print(f"Loaded {len(examples)} training examples")

    training_texts = [format_for_training(ex) for ex in examples]

    print("Tokenizing...")
    tokenized = tokenizer(
        training_texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )

    dataset = Dataset.from_dict({
        "input_ids": tokenized["input_ids"].tolist(),
        "attention_mask": tokenized["attention_mask"].tolist(),
    })

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print(f"\nStarting fine-tuning for {EPOCHS} epochs...")
    print("This may take a few minutes on CPU...\n")

    trainer.train()

    print(f"\nSaving fine-tuned model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nDone! Run 04_inference.py to test the model")


if __name__ == "__main__":
    main()
