"""LoRA supervised fine-tuning workflow."""
from __future__ import annotations

from .config import Config
from .data import read_jsonl
from .prompts import chat_messages


def train_lora(config: Config) -> None:
    """Fine-tune Qwen with LoRA while keeping base-model weights frozen."""
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(config.training.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        device_map=config.model.device_map,
        torch_dtype="auto",
    )
    model.config.use_cache = False

    records = read_jsonl(f"{config.data.output_dir}/train.jsonl")
    dataset = Dataset.from_list(records)

    import json
    from pathlib import Path

    labels = json.loads(
        Path(f"{config.data.output_dir}/labels.json").read_text()
    )

    def format_record(example):
        messages = chat_messages(example["text"], labels)
        messages.append({"role": "assistant", "content": example["label"]})
        return {
            "training_text": tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    dataset = dataset.map(format_record, remove_columns=dataset.column_names)

    peft = LoraConfig(
        r=config.training.lora_r,
        lora_alpha=config.training.lora_alpha,
        lora_dropout=config.training.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    args = SFTConfig(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        per_device_train_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=config.training.seed,
        dataset_text_field="training_text",
        max_length=config.training.max_seq_length,
        packing=False,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_steps=10,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft,
    )

    trainer.train()
    trainer.save_model(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)
