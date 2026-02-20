from clearml import Task, StorageManager
import os

task = Task.init(project_name="pershin_scaling_llm_alignment/JoLA_for_alignment", task_name="s3-download-test")

remote_url = "s3://api.blackhole2.ai.innopolis.university:443/pershin-scaling-llm-alignment/test_model.zip"

local_path = StorageManager.get_local_copy(remote_url=remote_url, extract_archive=True)

print("Downloaded to:", local_path)

for root, dirs, files in os.walk(local_path):
    # Print the current directory
    print("Directory:", root)
    # Print subdirectories
    for d in dirs:
        print("  Subdir :", d)
    # Print files
    for f in files:
        print("  File   :", f)