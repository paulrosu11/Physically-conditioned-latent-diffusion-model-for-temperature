#!/bin/bash
# Create the logs directory if it doesn't exist.
mkdir -p /usr/project/xtmp/par55/DiffScaler/slurm_logs

# Submit the job
sbatch \
    --job-name="FLUXPDEldm_res_2mT1e-10PDEMiddleOutputDecoder" \
    --mem="100G" \
    --cpus-per-task=4 \
    --partition=compsci-gpu \
    --gres=gpu:a5000:4 \
    --time=6:00:00 \
    --output="/usr/project/xtmp/par55/DiffScaler/slurm_logs/%x-%j.out" \
    --error="/usr/project/xtmp/par55/DiffScaler/slurm_logs/%x-%j.err" \
    --wrap="cd /usr/project/xtmp/par55/DiffScaler && \
            export PYTHONPATH=/usr/project/xtmp/par55/DiffScaler:\$PYTHONPATH && \
            python3 src/train.py experiment=downscaling_LDM_res_2mT"
