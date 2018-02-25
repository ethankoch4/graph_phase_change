#!/bin/bash
echo "RUNNING GENERATE_PLOTS PYTHON SCRIPT STARTING AT:"
var=$1
echo "$1"
echo "AND ENDING AT:"
finish=$2
echo "$2"
echo "WITH STEP SIZE:"
step=$3
echo "$3"
while [ "$var" -le "$finish" ]
do
    echo "$var"
    COMMAND="python3 generate_phase_change_plots.py "
    COMMAND+="$var"
    echo "$COMMAND"
    sbatch -o "$var".log -t 6-12  --job-name="$var" --wrap="$COMMAND"
    var=$(($var+$step))
done
echo "COMPLETED GENERATE_PLOTS PYTHON SCRIPT"
