import os
import glob

def generate_slurm_file(quantized_model_path, output_dir):
    base_name = os.path.basename(quantized_model_path)
    
    # Try to split on '-c4_', but if it fails, skip this file
    parts = base_name.split('-c4_')
    if len(parts) > 1:
        model_short_name = parts[0]
        hparam_str = parts[1]
    else:
        print(f"Skipping {base_name} as it doesn't match the expected format.")
        return
    
    wandb_name = f"AQ_FT_{model_short_name}_{hparam_str}"
    filename = f"aq_ft_{model_short_name}_{hparam_str}.sbatch"
    
    snapshot_path = f"/data/davis_brown/model_transmit/compression_expts/bit-depth/pythia-finetuned/{base_name}"
    
    content = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name=aq_ft_{model_short_name}
#SBATCH --mail-user=davisbrownr@gmail.com
#SBATCH --mail-type=ALL

source /opt/rh/devtoolset-10/enable
gcc --version

source /data/davis_brown/miniconda3/bin/activate
conda init
conda activate quip

export HF_HOME=/data/davis_brown/
export HF_DATASETS_CACHE=/data/davis_brown/
export TRANSFORMERS_CACHE=/data/davis_brown/
export MODEL_PATH=/data/davis_brown/.cache/huggingface/hub/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42
export QUANTIZED_MODEL_PATH={quantized_model_path}
export TOKENIZED_DATASET_PATH=../redpajama_tokenized_pythia
export SEQLEN=4096
export WANDB_PROJECT=AQ_PYTHIA_PV
export WANDB_NAME={wandb_name}
export SNAPSHOT_PATH={snapshot_path}
export NUM_GPUS=1

# Find a free port
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

torchrun --nproc-per-node=$NUM_GPUS --master_port=$MASTER_PORT /data/davis_brown/model_transmit/compression_expts/bit-depth/AQLM-C/finetune_fsdp.py \\
    --base_model $MODEL_PATH \\
    --quantized_model $QUANTIZED_MODEL_PATH \\
    --monkeypatch_old_pickle \\
    --model_seqlen=$SEQLEN \\
    --block_type GPTNeoXLayer \\
    --limit_parallel_inits 4 \\
    --load_dtype float32 \\
    --amp_dtype float32 \\
    --code_dtype uint8 \\
    --straight_through_buffer_dtype float32 \\
    --dataset_name=$TOKENIZED_DATASET_PATH \\
    --split none \\
    --seed 1337 \\
    --preprocessing_chunk_length 100000 \\
    --trust_remote_code \\
    --update_codes \\
    --update_codebooks_and_scales \\
    --update_non_quantized_parameters \\
    --lamb \\
    --debias \\
    --lr 3e-4 \\
    --adam_beta1 0.9 \\
    --adam_beta2 0.95 \\
    --code_lr 3e-3 \\
    --code_beta1 0.0 \\
    --code_beta2 0.95 \\
    --beam_size 1 \\
    --delta_decay 0 \\
    --max_code_change_per_step 1e-2 \\
    --code_trust_ratio 1e-2 \\
    --code_selection_temperature 0 \\
    --batch_size=16 \\
    --max_epochs 10 \\
    --gradient_checkpointing \\
    --print_every_steps=1 \\
    --verbose_optimizer \\
    --wandb \\
    --eval_every_steps=10 \\
    --use_fast_tokenizer
"""

    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write(content)
    
    print(f"Generated {filename}")

def main():
    quantized_models_pattern = "/data/davis_brown/model_transmit/compression_expts/bit-depth/pythia-quantized/pythia-70m-c4*"
    quantized_model_paths = glob.glob(quantized_models_pattern)

    output_dir = "aq_finetune_slurm_jobs"
    os.makedirs(output_dir, exist_ok=True)
    
    for quantized_model_path in quantized_model_paths:
        generate_slurm_file(quantized_model_path, output_dir)

if __name__ == "__main__":
    main()