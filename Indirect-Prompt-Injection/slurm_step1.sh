#!/bin/bash
#SBATCH --job-name=ipi_step1_synthesize
#SBATCH --output=slurm_logs/step1_%j.out
#SBATCH --error=slurm_logs/step1_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=cn3_anandi
#SBATCH --account=cminds_anandi
#SBATCH --qos=anandi
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
echo "========================================"

# Activate environment
ENV_DIR="/users/student/idddp/mohammad.k/IE 624/.env624"
source "$ENV_DIR/bin/activate"

# Project root
PROJECT="/users/student/idddp/mohammad.k/IE 624/Indirect-Prompt-Injection"
cd "$PROJECT"

# Create log directory
mkdir -p slurm_logs

# Set HuggingFace cache to avoid re-downloading the model
export HF_HOME="/users/student/idddp/mohammad.k/IE 624/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# Optional: set OpenAI API key if available (for LLM-as-Optimizer)
# export OPENAI_API_KEY="sk-..."

echo "Python: $(which python)"
echo "Step 1: Synthesizing IPI dataset..."

python step1_synthesize.py \
    --n_samples 255 \
    --max_rounds 5 \
    --seed 42

echo "Step 1 complete: $(date)"
