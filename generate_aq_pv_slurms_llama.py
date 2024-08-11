import os
import glob

def generate_slurm_file(model, output_dir, hyperparameters, nsamples=1024, model_seq_len=4096):
    model_parts = model.split('/')
    model_short_name = model_parts[-1]
    
    hparam_str = "_".join([f"{k}_{v}" for k, v in hyperparameters.items()])
    wandb_name = f"AQ_FT_{model_short_name}_{hparam_str}"
    filename = f"aq_ft_{model_short_name}_{hparam_str}.sbatch"
    
    save_path = f"/data/davis_brown/model_transmit/compression_expts/bit-depth/llama-quantized/{model_short_name}-pajama_{hparam_str}"
    snapshot_path = f"/data/davis_brown/model_transmit/compression_expts/bit-depth/llama-finetuned/{model_short_name}-pajama_{hparam_str}"
    converted_checkpoint_path = f"{snapshot_path}_converted"
    
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
export MODEL_PATH=/data/public_models/huggingface/meta-llama/Meta-Llama-3-8B
export DATASET_PATH=pajama
export SAVE_PATH={save_path}
export TOKENIZED_DATASET_PATH=../redpajama_tokenized_llama
export SEQLEN={model_seq_len}
export WANDB_PROJECT=AQ_LLAMA_PV
export WANDB_NAME={wandb_name}
export SNAPSHOT_PATH={snapshot_path}
export NUM_GPUS=1
export CACHE_DIR=/data/davis_brown/

export BLOCKWISE_FINETUNE_EPOCHS=25

 python ../main.py $MODEL_PATH $DATASET_PATH \\
 --nsamples={nsamples} \\
 --val_size=256 \\
 --num_codebooks={hyperparameters['num_codebooks']} \\
 --nbits_per_codebook={hyperparameters['nbits_per_codebook']} \\
 --in_group_size={hyperparameters['in_group_size']} \\
 --out_group_size={hyperparameters['out_group_size']} \\
 --beam_size=1 \\
 --relative_mse_tolerance=0.01 \\
 --max_epochs=100 \\
 --finetune_lr=1e-4 \\
 --finetune_adam_beta1=0.90 \\
 --finetune_adam_beta2=0.999 \\
 --finetune_keep_best \\
 --finetune_batch_size=64 \\
 --local_batch_size=4 \\
 --finetune_max_epochs=$BLOCKWISE_FINETUNE_EPOCHS \\
 --finetune_early_stop=3 \\
 --offload_activations \\
 --save $SAVE_PATH \\
 --wandb --resume

# Find a free port
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
torchrun --nproc-per-node=$NUM_GPUS /data/davis_brown/model_transmit/compression_expts/bit-depth/AQLM-C/finetune_fsdp.py \\
    --base_model $MODEL_PATH \\
    --quantized_model $SAVE_PATH \\
    --monkeypatch_old_pickle \\
    --model_seqlen=$SEQLEN \\
    --block_type LlamaDecoderLayer \\
    --limit_parallel_inits 4 \\
    --load_dtype bfloat16 \\
    --amp_dtype bfloat16 \\
    --code_dtype uint16 \\
    --straight_through_buffer_dtype float32 \\
    --dataset_name=$TOKENIZED_DATASET_PATH \\
    --split none \\
    --seed 1337 \\
    --preprocessing_chunk_length 100000 \\
    --cache_dir=$CACHE_DIR \\
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
    --batch_size=256 \\
    --microbatch_size=8 \\
    --max_epochs 10 \\
    --gradient_checkpointing \\
    --print_every_steps=1 \\
    --verbose_optimizer \\
    --wandb \\
    --eval_every_steps=10 \\
    --keep_best_model \\
    --save $SNAPSHOT_PATH \\
    --save_every_steps 100

# Convert the saved model to AQLM format
python /data/davis_brown/model_transmit/compression_expts/bit-depth/AQLM-C/convert_legacy_model_format.py \\
    --base_model $MODEL_PATH \\
    --pv_fsdp_dir $SNAPSHOT_PATH \\
    --code_dtype int32 \\
    --load_dtype auto \\
    --quantized_model=$SAVE_PATH \\
    --save {converted_checkpoint_path}

# Run evaluation
lm_eval --model hf \\
    --model_args pretrained=$MODEL_PATH,aqlm_checkpoint_path={converted_checkpoint_path},aqlm_src_path=/data/davis_brown/model_transmit/compression_expts/bit-depth/AQLM-C,parallelize=True,dtype=float16 \\
    --tasks winogrande,piqa,hellaswag,arc_easy,arc_challenge \\
    --batch_size 4

# for 5-shot MMLU
lm_eval --model hf \\
    --model_args pretrained=$MODEL_PATH,aqlm_checkpoint_path={converted_checkpoint_path},aqlm_src_path=/data/davis_brown/model_transmit/compression_expts/bit-depth/AQLM-C,parallelize=True,dtype=float16 \\
    --tasks mmlu \\
    --batch_size 4 \\
    --num_fewshot 5

# Delete the saved models
rm -rf {save_path}
rm -rf {snapshot_path}
# rm -rf {converted_checkpoint_path}
"""

    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write(content)
    
    print(f"Generated {filename}")

def main():
    model = "meta-llama/Meta-Llama-3-8B"

    specific_hyperparameters = [
        {
            'num_codebooks': 1,
            'nbits_per_codebook': 8,
            'in_group_size': 4,
            'out_group_size': 4
        },
        {
            'num_codebooks': 1,
            'nbits_per_codebook': 16,
            'in_group_size': 8,
            'out_group_size': 4
        }
    ]

    output_dir = "aq_pv_slurm_jobs_llama_full"
    os.makedirs(output_dir, exist_ok=True)
    
    for params in specific_hyperparameters:
        generate_slurm_file(model, output_dir, params)

if __name__ == "__main__":
    main()