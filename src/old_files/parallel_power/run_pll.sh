#!/bin/bash

# Must set the session number here and in both python files
log_file=./power_log_7.csv

if test -f "$log_file"; then
	echo "Log file already exists. Appending to existing log."
else
	echo "Log file not found. Creating new log."
	python3 start_pll.py
fi

# Run 4 simulations at the same time
# Each process peaks at ~2500% CPU usage. So be careful.
for i in {1..2}
do
	echo "Run simulation number $i"
	python3 -W ignore power_pll.py &
	# This sleep time helps to not overload the machine, since the sim CPU usage peaks for a short time
	sleep 5
done

