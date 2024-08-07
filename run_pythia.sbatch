#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --job-name=scaling 
#SBATCH --mail-user=mailto:davisbrownspam@gmail.com
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
export SAVE_PATH=/data/davis_brown/model_transmit/compression_expts/bit-depth/pythia-quantized/pythia-70m-c4
export WANDB_PROJECT=AQ_pythia
export WANDB_NAME=pythia-70m


python main.py $MODEL_PATH $DATASET_PATH \
 --nsamples=1024 \
 --val_size=256 \
 --num_codebooks=1 \
 --nbits_per_codebook=7 \
 --in_group_size=8 \
 --out_group_size=1 \
 --beam_size=1 \
 --relative_mse_tolerance=0.01 \
 --max_epochs=100 \
 --finetune_batch_size=32 \
 --finetune_max_epochs=10 \
 --finetune_early_stop=3 \
 --finetune_keep_best \
 --local_batch_size=4 \
 --use_faiss \
 --use_fast_tokenizer \
 --attn_implementation=eager \
 --dtype=float32 \
 --wandb \
 --resume \
 --save $SAVE_PATH