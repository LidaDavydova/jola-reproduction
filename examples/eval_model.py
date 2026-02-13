import torch
from transformers import AutoTokenizer
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jola import JoLAConfig, JoLAModel, JoLADataset, evaluate_common_reason

#  load config
jola_config = JoLAConfig.get_jola_config(default=True)

checkpoint_path = "/repo/jola/examples/outputs/llama-8b-arc-c/checkpoint-25"

#  load tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.padding_side = "left"

# load model from checkpoint
model = JoLAModel.jola_from_pretrained(
    pretrained_model_name_or_path=checkpoint_path,
    cache_dir='/repo/jola/.cache',
    torch_dtype=torch.float32,
    device_map="auto"
)

model.eval()

torch.set_grad_enabled(False)

# load dataset
dataset = JoLADataset(data_path=jola_config["data_config"]["data_path"])
data = dataset.data_from_file()

# run evaluation
evaluate_common_reason(
    eval_dataset=data["test"],
    task="commonsense",
    subtask="ARC-c",
    model_name='llama3.1_8B_chat',
    model=model,
    tokenizer=tokenizer,
    fname="/repo/jola/examples/llama-8b-arc-c/eval"
)

