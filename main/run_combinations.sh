#!/bin/bash
echo "STARTING COMBINATIONS SCRIPT"
module add python/3.5.1
sbatch -o COMBINATIONS.log -t 5-12 --job-name=COMBINATIONS --mem=50000 --wrap="python3 generate_param_combinations.py"
echo "COMPLETE"
