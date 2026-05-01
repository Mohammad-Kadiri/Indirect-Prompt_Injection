#!/bin/bash
#SBATCH --job-name=ipi_step5_weak
#SBATCH --output=slurm_logs/step5_%j.out
#SBATCH --error=slurm_logs/step5_%j.err
#SBATCH --time=02:00:00
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

ENV_DIR="/users/student/idddp/mohammad.k/IE 624/.env624"
source "$ENV_DIR/bin/activate"

PROJECT="/users/student/idddp/mohammad.k/IE 624/Indirect-Prompt-Injection"
cd "$PROJECT"
mkdir -p slurm_logs

export HF_HOME="/users/student/idddp/mohammad.k/IE 624/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Step 5: Weaker model comparison (Qwen2-1.5B-Instruct)..."

python step5_weaker_model.py \
    --n_eval 50 \
    --seed 42

echo "Step 5 complete: $(date)"
echo "Results:"
python -c "
import json
with open('results/step5_weaker_model_results.json') as f:
    r = json.load(f)
for s in r:
    print(f\"{s['attack_type']:<25} ASR={s['ASR']}%  UA={s['UA']}%\")
" 2>/dev/null || echo "Results not available"
