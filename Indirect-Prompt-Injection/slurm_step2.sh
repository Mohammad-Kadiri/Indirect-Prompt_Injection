#!/bin/bash
#SBATCH --job-name=ipi_step2_train
#SBATCH --output=slurm_logs/step2_%j.out
#SBATCH --error=slurm_logs/step2_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=cn3_anandi
#SBATCH --account=cminds_anandi
#SBATCH --qos=anandi
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8

# Request at least 12 GB VRAM (RTX 3060 12GB is sufficient for 4-bit Qwen3-8B)

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
echo "========================================"

ENV_DIR="/users/student/idddp/mohammad.k/IE 624/.env624"
source "$ENV_DIR/bin/activate"

PROJECT="/users/student/idddp/mohammad.k/IE 624/Indirect-Prompt-Injection"
cd "$PROJECT"
mkdir -p slurm_logs

export HF_HOME="/users/student/idddp/mohammad.k/IE 624/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Step 2: Training LSTP probe..."

python step2_train.py \
    --max_samples 255 \
    --epochs 50 \
    --lr 1e-3 \
    --batch_size 32 \
    --gamma 0.3 \
    --tau 0.10 \
    --threshold 0.5 \
    --seed 42

echo "Step 2 complete: $(date)"
echo "Checkpoint size: $(du -h checkpoints/lstp.pt 2>/dev/null || echo 'not found')"
