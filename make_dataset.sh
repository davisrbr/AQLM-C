#!/bin/bash

TARGET_MODEL=EleutherAI/pythia-70m  # used for tokenization
SEQLEN=4096
DATASET=togethercomputer/RedPajama-Data-1T-Sample
OUTPUT_PATH=./redpajama_tokenized_pythia

CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/davis_brown/ OMP_NUM_THREADS=16 torchrun \
--master-port 3456 \
--nproc-per-node=1 \
finetune_fsdp.py \
--base_model $TARGET_MODEL \
--quantized_model ./doesnt_matter \
--load_dtype float32 \
--block_type GPTNeoXLayer \
--dataset_name=$DATASET \
--split train \
--cache_dir=/data/datasets/redpajama-sample \
--trust_remote_code \
--model_seqlen=$SEQLEN \
--preprocessing_num_workers=8 \
--preprocessing_chunk_length 100000 \
--save_dataset_and_exit $OUTPUT_PATH

# tar -cvf tokenized_data_pythia.tar $OUTPUT_PATH   # optionally pack for distribution