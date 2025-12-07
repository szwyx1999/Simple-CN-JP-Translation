#!/bin/bash
#SBATCH --job-name=qwen_kd_e6-10            # job name
#SBATCH --output=logs/qwen_kd_e6-10.%j.out    # log name
#SBATCH --partition=normal                 # GPU partition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1                  # 1 GPU
#SBATCH --constraint=v100-32g              # V100 32G node (or a100-40g)
#SBATCH --mem=60G
#SBATCH --time=1-00:00:00                  # at most 1 day

PYTHON="/fs1/home/yuxiwang/.conda/envs/qwen-kd/bin/python"

cd /fs1/home/yuxiwang/Simple-CN-JP-Translation

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/fs1/scratch/yuxiwang/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

# 6-10 epoch distill
srun "$PYTHON" train_distillation_resume.py \
  --teacher_model_path Qwen/Qwen2.5-7B-Instruct \
  --resume_from runs/qwen_kd_e1-5/checkpoint-epoch-5-step-3220 \
  --output_dir runs/qwen_kd_e6-10 \
  --train_ja_file data/wccjc_sports_03.train.ja \
  --train_zh_file data/wccjc_sports_03.train.zh \
  --direction ja2zh \
  --num_epochs 5 \
  --learning_rate 3e-4

