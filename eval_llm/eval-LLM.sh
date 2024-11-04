#!/bin/bash

# Define base model names and suffixes
base_models=("CASIE" "MAVEN" "M2E2" "FewEvent")
suffixes=("")

# Generate model configs
model_configs=()
for base in "${base_models[@]}"; do
    for suffix in "${suffixes[@]}"; do
        if [ -z "$suffix" ]; then
            model_configs+=("Llama3-${base}")
        else
            model_configs+=("Llama3-${base}-${suffix}")
        fi
    done
done

# Set WANDB to offline mode, can turn this on instead if desired
export WANDB_MODE="offline"

# Loop through each model config
for model_config in "${model_configs[@]}"; do
    output_path="lm-eval-harness-output/${model_config}"
    wandb_project="lm-eval-harness-${model_config}"
    model_dir="../event_instruct/outputs/${model_config}"
    log_file="${output_path}/evaluation.log"

    # Create output directory if it doesn't exist
    mkdir -p "$output_path"
    
    # Find the latest checkpoint
    latest_checkpoint=$(ls -v $model_dir | grep 'checkpoint-' | sort -V | tail -n 1)
    peft_path="${model_dir}/${latest_checkpoint}"

    # Check if the checkpoint was found
    if [ -z "$latest_checkpoint" ]; then
        echo "No checkpoint found for ${model_config}" | tee -a $log_file
        continue
    fi

    echo "Using checkpoint: $latest_checkpoint for model config: $model_config" | tee -a $log_file

    # Run evaluations
    tasks=("arc_challenge" "hellaswag" "truthfulqa_mc2" "mmlu" "winogrande" "gsm8k")
    num_fewshots=(25 10 0 5 5 5)
    names=("arc" "hellaswag" "truthfulqa" "mmlu" "winogrande" "gsm8k")

    for i in "${!tasks[@]}"; do
        run_name="${model_config}-${names[$i]}-checkpoint-${latest_checkpoint}"
        
        # Set batch size based on task
        if [ "${tasks[$i]}" == "mmlu" ]; then
            batch_size=8
        else
            batch_size=16
        fi

        lm_eval \
            --model hf \
            --model_args pretrained=../../models/llama-3-8b-Instruct,peft=$peft_path \
            --tasks ${tasks[$i]} \
            --batch_size $batch_size \
            --num_fewshot ${num_fewshots[$i]} \
            --output_path $output_path \
            --wandb_args project=$wandb_project,name=$run_name,dir=${output_path} \
            --log_samples 2>&1 | tee -a $log_file
    done
done
