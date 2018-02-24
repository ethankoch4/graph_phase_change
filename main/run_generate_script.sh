#!/bin/bash
echo "RUNNING GENERATE_PLOTS PYTHON SCRIPT FOR:"
echo "$@"
python3 generate_phase_change_plots.py "$@"
cd ..
git add --all .
git commit "$*"
git pull origin master
git push
echo "COMPLETED GENERATE_PLOTS PYTHON SCRIPT"
