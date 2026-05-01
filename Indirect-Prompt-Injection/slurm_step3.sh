#!/bin/bash
#SBATCH --job-name=ipi_step3_eval
#SBATCH --output=slurm_logs/step3_%j.out
#SBATCH --error=slurm_logs/step3_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=cn3_anandi
#SBATCH --account=cminds_anandi
#SBATCH --qos=anandi
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

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
echo "Step 3: Evaluating ICON on InjectAgent benchmark..."

python step3_eval_icon.py \
    --n_eval 50 \
    --seed 42

echo "Step 3 complete: $(date)"
echo "Results:"
cat results/step3_eval_results.json 2>/dev/null || echo "Results file not found"
