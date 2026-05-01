#!/bin/bash
#SBATCH --job-name=ipi_step4_stress
#SBATCH --output=slurm_logs/step4_%j.out
#SBATCH --error=slurm_logs/step4_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=cn3_anandi
#SBATCH --account=cminds_anandi
#SBATCH --qos=anandi
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=48G
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
echo "Step 4: Adversarial Stress-Testing..."

python step4_stress_test.py \
    --n_samples 15 \
    --seed 42

echo "Step 4 complete: $(date)"
echo "Summary:"
python -c "
import json
with open('results/step4_stress_test_results.json') as f:
    r = json.load(f)
print(f\"{'Attack':<35} {'ASR':>6}  {'Evasion':>7}  {'Effective':>9}\")
print('-' * 62)
for s in r:
    print(f\"{s['attack']:<35} {s['ASR']:>5}%  {s['evasion_rate']:>6}%  {s['effective_rate']:>8}%\")
" 2>/dev/null || echo "Summary not available"
