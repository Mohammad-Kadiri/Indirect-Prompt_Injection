#!/bin/bash
#SBATCH --job-name=ipi_install
#SBATCH --output=slurm_logs/install_%j.out
#SBATCH --error=slurm_logs/install_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=cn3_anandi
#SBATCH --account=cminds_anandi
#SBATCH --qos=anandi
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

# Run this FIRST before any other slurm scripts

echo "========================================"
echo "Installing dependencies into .env624"
echo "========================================"

PROJECT="/users/student/idddp/mohammad.k/IE 624/Indirect-Prompt-Injection"
mkdir -p "$PROJECT/slurm_logs"
cd "$PROJECT"

bash install_deps.sh

echo "Install complete: $(date)"
