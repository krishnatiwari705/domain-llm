"""LoRA supervised fine-tuning workflow."""
from __future__ import annotations

from .config import Config
from .data import read_jsonl


def train_lora(config: Config) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
    from trl import SFTTrainer
    set_seed(config.training.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model.name, device_map=config.model.device_map)
    records = read_jsonl(f"{config.data.output_dir}/train.jsonl")
    dataset = Dataset.from_list(records)
    peft = LoraConfig(r=config.training.lora_r, lora_alpha=config.training.lora_alpha, lora_dropout=config.training.lora_dropout, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    args = TrainingArguments(output_dir=config.training.output_dir, num_train_epochs=config.training.epochs, learning_rate=config.training.learning_rate, per_device_train_batch_size=config.training.batch_size, gradient_accumulation_steps=config.training.gradient_accumulation_steps, logging_steps=10, save_strategy="epoch", report_to="none")
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, dataset_text_field="training_text", max_seq_length=config.training.max_seq_length, peft_config=peft, args=args)
    trainer.train()
    trainer.save_model(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)
