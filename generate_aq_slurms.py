import os
import itertools

def generate_slurm_file(model_name, output_dir, hyperparameters, nsamples=1024, model_seq_len=4096):
    model_parts = model_name.split('/')
    if "models--" in model_name:
        model_short_name = model_parts[-4].split("--", 2)[-1].replace("--", "-")
    else:
        model_short_name = model_parts[-1]
    
    hparam_str = "_".join([f"{k}_{v}" for k, v in hyperparameters.items()])
    wandb_name = f"AQ_{model_short_name}_{hparam_str}"
    wandb_name_noft = f"AQ_{model_short_name}_{hparam_str}_noft"
    filename = f"aq_{model_short_name}_{hparam_str}.sbatch"
    
    # save_path = f"/data/davis_brown/model_transmit/compression_expts/bit-depth/pythia-quantized/{model_short_name}-c4_{hparam_str}"
    save_path = f"/data/davis_brown/model_transmit/compression_expts/bit-depth/llama-quantized/{model_short_name}-pajama_{hparam_str}"
    converted_checkpoint_path = f"{save_path}_converted"
    
    content = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --job-name=aq_{model_short_name}
#SBATCH --mail-user=davisbrownr@gmail.com
#SBATCH --mail-type=ALL

# Recommended way if you want to enable gcc version 10 for the "sbatch" session 
source /opt/rh/devtoolset-10/enable

gcc --version  # if you print it out again here it'll be version 10 

source /data/davis_brown/miniconda3/bin/activate
conda init
conda activate quip

export HF_HOME=/data/davis_brown/
export HF_DATASETS_CACHE=/data/davis_brown/
export TRANSFORMERS_CACHE=/data/davis_brown/
# export MODEL_PATH=/data/davis_brown/.cache/huggingface/hub/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42
export MODEL_PATH=/data/public_models/huggingface/meta-llama/Meta-Llama-3-8B
# export DATASET_PATH=c4
export DATASET_PATH=pajama
export SAVE_PATH={save_path}
# export WANDB_PROJECT=AQ_pythia_sweep_test
export WANDB_PROJECT=AQ_llama_sweep_test
export WANDB_NAME={wandb_name}
export MODEL_SEQLEN={model_seq_len}

# python ../main.py $MODEL_PATH $DATASET_PATH \\
#  --nsamples={nsamples} \\
#  --val_size=256 \\
#  --num_codebooks={hyperparameters['num_codebooks']} \\
#  --nbits_per_codebook={hyperparameters['nbits_per_codebook']} \\
#  --in_group_size={hyperparameters['in_group_size']} \\
#  --out_group_size={hyperparameters['out_group_size']} \\
#  --beam_size=1 \\
#  --relative_mse_tolerance=0.01 \\
#  --max_epochs=100 \\
#  --finetune_batch_size=32 \\
#  --finetune_max_epochs=10 \\
#  --finetune_early_stop=3 \\
#  --finetune_keep_best \\
#  --local_batch_size=4 \\
#  --use_faiss \\
#  --use_fast_tokenizer \\
#  --offload_activations \\
#  --dtype=float32 \\
#  --wandb \\
#  --save $SAVE_PATH

# export SAVE_PATH={save_path + "_noft"}
# export WANDB_PROJECT=AQ_pythia_sweep_test
# export WANDB_NAME={wandb_name_noft}

# python ../main.py $MODEL_PATH $DATASET_PATH \\
#  --nsamples={nsamples} \\
#  --val_size=256 \\
#  --num_codebooks={hyperparameters['num_codebooks']} \\
#  --nbits_per_codebook={hyperparameters['nbits_per_codebook']} \\
#  --in_group_size={hyperparameters['in_group_size']} \\
#  --out_group_size={hyperparameters['out_group_size']} \\
#  --beam_size=1 \\
#  --relative_mse_tolerance=0.01 \\
#  --max_epochs=100 \\
#  --finetune_max_epochs=0 \\
#  --use_faiss \\
#  --use_fast_tokenizer \\
#  --offload_activations \\
#  --dtype=float32 \\
#  --wandb \\
#  --save $SAVE_PATH

export BLOCKWISE_FINETUNE_EPOCHS=25

 python ../main.py $MODEL_PATH $DATASET_PATH \\
 --nsamples={nsamples} \\
 --val_size=256 \\
 --num_codebooks={hyperparameters['num_codebooks']} \\
 --nbits_per_codebook={hyperparameters['nbits_per_codebook']} \\
 --in_group_size={hyperparameters['in_group_size']} \\
 --out_group_size={hyperparameters['out_group_size']} \\
 --beam_size=1 \
 --relative_mse_tolerance=0.01 \
 --max_epochs=100 \
 --finetune_lr=1e-4 \
 --finetune_adam_beta1=0.90 \
 --finetune_adam_beta2=0.999 \
 --finetune_keep_best \
 --finetune_batch_size=64 \
 --local_batch_size=4 \
 --finetune_max_epochs=$BLOCKWISE_FINETUNE_EPOCHS \
 --finetune_early_stop=3 \
 --offload_activations \
 --save $SAVE_PATH \
 --wandb --resume

 # Convert the saved model to AQLM format
python /data/davis_brown/model_transmit/compression_expts/bit-depth/AQLM-C/convert_legacy_model_format.py \\
    --base_model $MODEL_PATH \\
    --pv_fsdp_dir $SAVE_PATH \\
    --code_dtype int32 \\
    --load_dtype auto \\
    --quantized_model=$SAVE_PATH \\
    --save {converted_checkpoint_path}

# # Run evaluation
# lm_eval --model hf \\
#     --model_args pretrained=$MODEL_PATH,aqlm_checkpoint_path={converted_checkpoint_path},aqlm_src_path=/data/davis_brown/model_transmit/compression_expts/bit-depth/AQLM-C,parallelize=True,dtype=float16 \\
#     --tasks winogrande,piqa,hellaswag,arc_easy,arc_challenge \\
#     --batch_size 4
    
# # for 5-shot MMLU
# lm_eval --model hf \\
#     --model_args pretrained=$MODEL_PATH,aqlm_checkpoint_path={converted_checkpoint_path},aqlm_src_path=/data/davis_brown/model_transmit/compression_expts/bit-depth/AQLM-C,parallelize=True,dtype=float16 \\
#     --tasks mmlu \\
#     --batch_size 4 \\
#     --num_fewshot 5


# delete the saved model
rm -rf {save_path}
# rm -rf {converted_checkpoint_path}

"""

    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write(content)
    
    print(f"Generated {filename}")

def main():
    models = [
        "meta-llama/Meta-Llama-3.1-8B",
        # "EleutherAI/pythia-70m",
    ]

    hyperparameter_grid = {
        'num_codebooks': [1],
        'nbits_per_codebook': [4, 8, 16, 32],
        'in_group_size': [4, 8, 16],
        'out_group_size': [4, 8, 16],
    }
    # hyperparameter_grid = {
    #     'num_codebooks': [4],
    #     'nbits_per_codebook': [8],
    #     'in_group_size': [8],
    #     'out_group_size': [1],
    # }

    # output_dir = "aq_slurm_jobs_test"
    output_dir = "aq_slurm_jobs_test_llama"
    os.makedirs(output_dir, exist_ok=True)
    
    for model in models:
        for hyperparameters in itertools.product(*hyperparameter_grid.values()):
            params = dict(zip(hyperparameter_grid.keys(), hyperparameters))
            generate_slurm_file(model, output_dir, params)

if __name__ == "__main__":
    main()