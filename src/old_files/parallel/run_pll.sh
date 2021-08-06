#!/bin/bash

# Must set the session number here and in both python files
SES_NUM=2
log_file="./pll_log_$SES_NUM.csv"

if test -f "$log_file"; then
	echo "Log file already exists. Appending to existing log."
else
	echo "Log file not found. Creating new log."
	python3 init_pll.py
fi

N=30000
# Run 8 simulations at the same time
for i in {1..7}
do
	echo "Run simulation number $i"
	log_i="./log_$i.txt"
	#python3 -W ignore power_pll.py &
	nohup python3 -u main_pll.py $N > "$log_i" &
	N=$(( $N+5000 ))
done
