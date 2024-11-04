# Enhancing Event Reasoning in Large Language Models through Instruction Fine-Tuning with Semantic Causal Graphs


## Create Environment
```
conda create --name scg-event \
    python=3.10 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate scg-event
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes
pip install einops
pip install anthropic openai
```

## Usage
`/baseline_responses` contains scripts used to obtain baseline responses of open-source and proprietary models on the event detection task.

`/data` contains the original event detection datasets and scripts for creating SCG instructions for event detection.

`/event_instruct` contains training scripts for training LLMs on the datasets in `/data` using LoRA and inference scripts for gathering responses of trained models.

`/event_dpo` contains training and inference scripts for the Direct Preference Optimization experiments.

`/eval_event` contains evaluation scripts of the parsed responses from trained LLMs on the test sets of the event detection datasets.

`/change_context` contains the script to make calls to Anthropic API to create modified context variations of the event detection test sets. Folder also contains the modified test sets we created that were used in our experiments in `/change_context/modified_data`.

`/eval_llm` contains scripts for performing general LLM benchmark testing by leveraging the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main)





