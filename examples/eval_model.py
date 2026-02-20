import torch
from transformers import AutoTokenizer
import sys, os

from clearml import Task

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jola import JoLAConfig, JoLAModel, JoLADataset, evaluate_common_reason

task = Task.init(
    project_name="pershin_scaling_llm_alignment/JoLA_for_alignment",
    task_name="jola_test_arc_c"
)


#  load config
jola_config = JoLAConfig.get_jola_config(default=True)


checkpoint_path = StorageManager.get_local_copy(
    remote_url="s3://pershin-scaling-llm-alignment/pershin_scaling_llm_alignment/JoLA_for_alignment/clearml-example-cu129.ca8d2f4c037841a2a19223356b5acecd/artifacts/final_model/llama-8b-arc-c.zip",
    extract_archive=True
)

print("Downloaded to:", checkpoint_path)

for root, dirs, files in os.walk(local_path):
    # Print the current directory
    print("Directory:", root)
    # Print subdirectories
    for d in dirs:
        print("  Subdir :", d)
    # Print files
    for f in files:
        print("  File   :", f)

checkpoint_path = checkpoint_path + '/llama-8b-arc-c'

#  load tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.padding_side = "left"

# load model from checkpoint
model = JoLAModel.jola_from_pretrained(
    pretrained_model_name_or_path=checkpoint_path,
    cache_dir='.cache',
    torch_dtype=torch.float32,
    device_map="auto"
)

model.eval()

torch.set_grad_enabled(False)

# load dataset
dataset = JoLADataset(data_path=jola_config["data_config"]["data_path"])
data = dataset.data_from_file()

eval_path = jola_config["training_config"]['output_dir']+'/eval'

# run evaluation
evaluate_common_reason(
    eval_dataset=data["test"],
    task="commonsense",
    subtask="ARC-c",
    model_name='llama3.1_8B_chat',
    model=model,
    tokenizer=tokenizer,
    fname=eval_path
)

task.upload_artifact(
            name='eval',
            artifact_object=eval_path
        )

task.close()