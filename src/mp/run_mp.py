"""Run a large number of simulations in parallel with multiprocessing
"""
import sys
sys.path.append("/home/jblack/NMLAB/NMLab/src")

import pandas as pd
import time
import multiprocessing as mp

import main

# Constants
options_file = "options_list_T.csv"
max_processes = 50
V = 25
N = 200
bpm0 = .02

# Read in the list of simulations to be run
options_df = pd.read_csv(options_file)
options_list = []
for row in options_df.iterrows():
  row = row[1]
  flm0 = row["fl_mu_0"]
  flmr = row["fl_mu_ratio"]
  fl_mus = str([flm0, flm0/flmr])[1:-1]
  ftp0 = row["ftype_proportions_0"]
  ft_prop = str([ftp0, 1-ftp0])[1:-1]
  # For compatability, this has 2*, but it probably shouldn't
  cnd_len = 2 * (V/N)**(1/3) / row["ks"]
  ftp0 = row["ftype_proportions_0"]
  ft_prop = str([ftp0, 1-ftp0])[1:-1]
  bpmr = row["bpwr_mu_ratio"]
  bpwr_mus = str([bpm0, bpm0/bpmr])[1:-1]
  #bf = "True" if row["burn_fibers"] else "False"
  options_list.append({
    "id": row["SIM_ID"],
    "cnd_len": str(cnd_len),
    "fl_mus": fl_mus,
    "ftype_proportions": ft_prop,
    "bpwr_mus": bpwr_mus,
    "preburn_fraction": str(row["preburn_fraction"]),
    "burn_fibers": str(row["burn_fibers"])
  })


def test(options):
  #print(options["id"])
  print(options)

def run(options):
  try:
    main.main(options)
  except Exception as e:
    print("Error running sim #", options["id"], e)

def q_run(q):
  # Run a sim as long as there's a job in the queue
  while not q.empty():
    args = q.get()
    #test(args)
    run(args)

def pool_mlt():
  # Use mp.Pool to do the multiprocessing
  pool = mp.Pool(max_processes)
  #pool.map(test, options_list)
  #result = pool.map(main.main, options_list)
  #print(result)
  #pool.map(main.main, options_list)
  #pool.map(test, options_list)
  pool.map(run, options_list)

def q_mlt():
  # Use a queue to do the multiprocessing
  # Fill the queue
  job_q = mp.Queue()
  for job in options_list:
    job_q.put(job)
  # Create the processes
  processes = [mp.Process(target=q_run, args=(job_q,)) 
    for i in range(max_processes)]

  # Start all of them
  for p in processes:
    p.start()
  for p in processes:
    p.join()
  
if __name__ == '__main__':
  #pool_mlt()
  q_mlt()

