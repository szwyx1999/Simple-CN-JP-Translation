#!/bin/bash
#SBATCH --job-name=benchmark-qwen-0.5B            # job name
#SBATCH --output=logs/benchmark-qwen-0.5B.%j.out    # log name
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00

PYTHON="/fs1/home/yuxiwang/.conda/envs/qwen-kd/bin/python"

cd /fs1/home/yuxiwang/Simple-CN-JP-Translation

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/fs1/scratch/yuxiwang/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME


srun "$PYTHON" ./benchmark/test_distillation_v2.py \
  --model_path Qwen/Qwen2.5-0.5B-Instruct \
  --tokenizer_path Qwen/Qwen2.5-0.5B-Instruct \
  --test_ja_file "/fs1/home/yuxiwang/Simple-CN-JP-Translation/data/valid.ja" \
  --test_zh_file "/fs1/home/yuxiwang/Simple-CN-JP-Translation/data/valid.zh" \
  --batch_size 8 \
  --max_length 512 \
  --max_new_tokens 256 \
  --output_dir benchmark/qwen_0.5B_wccjc_test
