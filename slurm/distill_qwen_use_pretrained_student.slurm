#!/bin/bash
#SBATCH --job-name=qwen_pretrained            # job name
#SBATCH --output=logs/qwen_pretrained.%j.out    # log name
#SBATCH --partition=normal                 # GPU partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1                  # 1 GPU
#SBATCH --constraint=a100-40g              # V100 32G node (or a100-40g)
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00                  # at most 3 day

PYTHON="/fs1/home/yuxiwang/.conda/envs/qwen-kd/bin/python"


cd /fs1/home/yuxiwang/Simple-CN-JP-Translation

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/fs1/scratch/yuxiwang/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

# 1-3 epoch distill
srun "$PYTHON" train_distillation_FinalVersion.py \
  --teacher_model_path Qwen/Qwen2.5-7B-Instruct \
  --resume_from Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir /fs1/scratch/yuxiwang/runs/qwen_pretrained \
  --train_ja_file data/demo_train.ja \
  --train_zh_file data/demo_train.zh \
  --direction ja2zh \
  --num_epochs 3 \
  --alpha 1.0 \
  --temperature 2.0 \
  --learning_rate 5e-5

