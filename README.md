Reproducibility issues
===

#### Dataset
Commonsense Reasoning (ARC-c)
[Same Link](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset/ARC-Challenge)

Added data to repo [dataset/data_with_instruct/commonsense/ARC-c](https://github.com/LidaDavydova/jola-reproduction/tree/main/dataset/data_with_instruct/commonsense/ARC-c)

#### Other details

- For test I use 100 samles from test.json
- valid.json contains sample from test.json

Running commands:
```
cd examples

# for train
python run_jola.py

# for test
python eval_model.py
```

Files about train process:
(examples/outputs/llama-8b-arc-c)[https://github.com/LidaDavydova/jola-reproduction/tree/main/examples/outputs/llama-8b-arc-c]

Yes, an early stopping was considered the best control point even after 1 epoch, but I tried several times to run train with different train_data_size (200-1000) and a different model such as llama-3b. And the training process usually ended after a maximum of 6 epochs.