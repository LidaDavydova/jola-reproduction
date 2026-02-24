import os
import sys
import torch

from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from clearml import Task

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from jola import (
    JoLAConfig,
    JoLAModel_qwen,
    JoLAModel_llama,
    JoLATrainer,
    JoLADataset,
    make_data_collator,
    evaluate_common_reason
)

# =========================
# ClearML task
# =========================
task = Task.init(
    project_name="pershin_scaling_llm_alignment/JoLA_for_alignment",
    task_name="jola_train_and_test"
)

print("Cuda:", torch.cuda.is_available())

# =========================
# Load config
# =========================
jola_config = JoLAConfig.get_jola_config(default=True)
task.connect(jola_config)

output_dir = jola_config["training_config"]['output_dir']

# =========================
# Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(**jola_config["model_config"])
tokenizer.padding_side = "right"

# =========================
# Model (training version)
# =========================
model = JoLAModel_qwen.jola_from_pretrained(**jola_config["model_config"])
model.unfreeze_jola_params()
model.model.train()

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(model.config.vocab_size + 1)

# =========================
# Dataset
# =========================
dataset = JoLADataset(
    data_path=jola_config["data_config"]["data_path"],
    train_size=jola_config["data_config"]["train_size"]
)

data = dataset.data_from_file()

data_collator = make_data_collator(tokenizer=tokenizer)

# =========================
# Trainer
# =========================
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=2,
    early_stopping_threshold=0.0
)

training_args = TrainingArguments(**jola_config["training_config"])

trainer = JoLATrainer(
    model,
    train_dataset=data["train"],
    eval_dataset=data["valid"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    args=training_args,
    callbacks=[early_stopping_callback],
    gate_scheduler=jola_config["jola_config"]["gate_scheduler"]
)

if not jola_config["jola_config"]["gate_scheduler"]:
    trainer.gated_lambda = jola_config['training_config']["gate_lambda"]

# =========================
# Train
# =========================
trainer.train()

trainer.save_model(output_dir)

# =========================
# Switch to eval model
# =========================
torch.cuda.empty_cache()

model = JoLAModel_qwen.jola_from_pretrained(
    pretrained_model_name_or_path=output_dir,
    torch_dtype=torch.float32,
    cache_dir='.cache',
    device_map="auto"
)

model.eval()
torch.set_grad_enabled(False)

# IMPORTANT: for inference we usually use left padding
tokenizer = AutoTokenizer.from_pretrained(output_dir)
tokenizer.padding_side = "left"

# =========================
# Evaluation
# =========================
eval_path = os.path.join(output_dir, "eval")
os.makedirs(eval_path, exist_ok=True)

evaluate_common_reason(
    eval_dataset=data["test"],
    task="commonsense",
    subtask="ARC-c",
    model_name='qwen2.5-3B_chat',
    model=model,
    tokenizer=tokenizer,
    fname=eval_path
)

# =========================
# Upload only results
# =========================
task.upload_artifact(
    name="eval_results",
    artifact_object=eval_path
)

task.close()