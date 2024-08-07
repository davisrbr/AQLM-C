import os
import itertools

def generate_slurm_file(model_name, output_dir, hyperparameters):
    model_parts = model_name.split('/')
    if "models--" in model_name:
        model_short_name = model_parts[-4].split("--", 2)[-1].replace("--", "-")
    else:
        model_short_name = model_parts[-1]
    
    hparam_str = "_".join([f"{k}_{v}" for k, v in hyperparameters.items()])
    wandb_name = f"AQ_{model_short_name}_{hparam_str}"
    filename = f"aq_{model_short_name}_{hparam_str}.sbatch"
    
    save_path = f"/data/davis_brown/model_transmit/compression_expts/bit-depth/pythia-quantized/{model_short_name}-c4_{hparam_str}"
    
    content = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=8:00:00
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
export MODEL_PATH=/data/davis_brown/.cache/huggingface/hub/models--EleutherAI--pythia-70m/snapshots/a39f36b100fe8a5377810d56c3f4789b9c53ac42
export DATASET_PATH=c4
export SAVE_PATH={save_path}
export WANDB_PROJECT=AQ_pythia
export WANDB_NAME={wandb_name}

python ../main.py $MODEL_PATH $DATASET_PATH \\
 --nsamples=1024 \\
 --val_size=256 \\
 --num_codebooks={hyperparameters['num_codebooks']} \\
 --nbits_per_codebook={hyperparameters['nbits_per_codebook']} \\
 --in_group_size={hyperparameters['in_group_size']} \\
 --out_group_size={hyperparameters['out_group_size']} \\
 --beam_size=1 \\
 --relative_mse_tolerance=0.01 \\
 --max_epochs=100 \\
 --finetune_batch_size=32 \\
 --finetune_max_epochs=10 \\
 --finetune_early_stop=3 \\
 --finetune_keep_best \\
 --local_batch_size=4 \\
 --use_faiss \\
 --use_fast_tokenizer \\
 --attn_implementation=eager \\
 --dtype=float32 \\
 --wandb \\
 --save $SAVE_PATH
"""

    with open(os.path.join(output_dir, filename), 'w') as f:
        f.write(content)
    
    print(f"Generated {filename}")

def main():
    models = [
        "EleutherAI/pythia-70m",
    ]

    hyperparameter_grid = {
        'num_codebooks': [1, 2, 4],
        'nbits_per_codebook': [4, 8, 16],
        'in_group_size': [1, 8, 16],
        'out_group_size': [1, 8, 16],
    }

    output_dir = "aq_slurm_jobs"
    os.makedirs(output_dir, exist_ok=True)
    
    for model in models:
        for hyperparameters in itertools.product(*hyperparameter_grid.values()):
            params = dict(zip(hyperparameter_grid.keys(), hyperparameters))
            generate_slurm_file(model, output_dir, params)

if __name__ == "__main__":
    main()