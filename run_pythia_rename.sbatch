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
export NUM_GPUS=1
export QUANTIZED_MODEL_PATH=$SAVE_PATH  # path to the model created by initial calibration
export TOKENIZED_DATASET_PATH=./redpajama_tokenized_pythia  # yet again, red pajama adviced
export CACHE_DIR=/data/datasets/redpajama-sample
export SNAPSHOT_PATH=/data/davis_brown/model_transmit/compression_expts/bit-depth/pythia-quantized/pythia-70m-redpajama-tuned
export SEQLEN=4096

export WANDB_PROJECT=AQ_PYTHIA_PV
export WANDB_NAME=pythia-70m-1x7gs8-pv

torchrun --nproc-per-node=$NUM_GPUS finetune_fsdp.py \
    --base_model $MODEL_PATH --quantized_model $QUANTIZED_MODEL_PATH  --monkeypatch_old_pickle \
    --model_seqlen=$SEQLEN --block_type GPTNeoXLayer --limit_parallel_inits 4 \
    --load_dtype float32 --amp_dtype float32 --code_dtype uint8 \
    --straight_through_buffer_dtype float32 \
    --dataset_name=$TOKENIZED_DATASET_PATH --split none --seed 1337 \
    --preprocessing_chunk_length 100000 --cache_dir=$CACHE_DIR --trust_remote_code \
    --update_codes --update_codebooks_and_scales --update_non_quantized_parameters \
    --lamb --debias --lr 3e-4 --adam_beta1 0.9 --adam_beta2 0.95 \
    --code_lr 3e-3 --code_beta1 0.0 --code_beta2 0.95 --beam_size 1 --delta_decay 0 \
    --max_code_change_per_step 1e-2 --code_trust_ratio 1e-2 --code_selection_temperature 0 \
    --batch_size=16 --max_epochs 10 --gradient_checkpointing \
    --print_every_steps=1 --verbose_optimizer --wandb  --eval_every_steps=10 --keep_best_model \
    --save $SNAPSHOT_PATH --save_every_steps 100 --use_fast_tokenizer