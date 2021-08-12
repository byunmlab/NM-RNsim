"""Run a large number of simulations in parallel with multiprocessing
"""
import sys
sys.path.append("/home/jblack/NMLAB/NMLab/src")

import time
import multiprocessing as mp

import main

# Constants
main_id = "DOE_"
max_processes = 48
V = 25
N = 500

"""
IVs for DOE		Low	Med	High	Description
- ks			1.4	1.6	1.8	Sparsity
- fl_mu_0		0.1	0.3	0.5	Long fiber length
- fl_mu_ratio		2	6	10	Ratio of long to short fiber length
- ftype_proportions[0]	0.1	0.3	0.5	Fraction of fibers that are long
- bpwr_mu_ratio		1	2	3	Burn power of long to short fibers
- preburn fraction	.002	.006	.010	What fraction of the RN to preburn
- burn_fibers		F		T	Whether to remove fibers or junctions
"""

# Values for the IVs
v_ks = [1.4, 1.6, 1.8]
v_flm0 = [0.1, 0.3, 0.5]
v_flmr = [2, 6, 10]
v_ftpr = [0.1, 0.3, 0.5]
v_bpmr = [1, 2, 3]
v_pbfr = [.002, .006, .010]
v_bf = [False, True]

# Create list of options
options_list = []
for i_ks in range(len(v_ks)):
  for i_flm0 in range(len(v_flm0)):
    for i_flmr in range(len(v_flmr)):
      for i_ftpr in range(len(v_ftpr)):
        for i_bpmr in range(len(v_bpmr)):
          for i_pbfr in range(len(v_pbfr)):
            for i_bf in range(len(v_bf)):
              l_is = [i_ks, i_flm0, i_flmr, i_ftpr, i_bpmr, i_pbfr, i_bf]
              sim_id = main_id + "".join([str(i) for i in l_is])
              ks = v_ks[i_ks]
              # For compatability, this has 2*, but it probably shouldn't
              cnd_len = 2 * (V/N)**(1/3) / ks
              flm0 = v_flm0[i_flm0]
              flmr = v_flmr[i_flmr]
              fl_mus = [flm0, flm0/flmr]
              ftp0 = v_ftpr[i_ftpr]
              ftype_proportions = [ftp0, 1-ftp0]
              bpm0 = .02
              bpmr = v_bpmr[i_bpmr]
              bpwr_mus = [bpm0, bpm0/bpmr]
              pbfr = v_pbfr[i_pbfr]
              bf = v_bf[i_bf]
              options_list.append({
                "id": sim_id,
                "cnd_len": str(cnd_len),
                "fl_mus": str(fl_mus)[1:-1],
                "ftype_proportions": str(ftype_proportions)[1:-1],
                "bpwr_mus": str(bpwr_mus)[1:-1],
                "preburn_fraction": str(pbfr),
                "burn_fibers": str(bf)
              })

def test(options):
  print(options["id"])

def run(options):
  try:
    main.main(options)
  except Exception as e:
    print("Error running sim #", options["id"], e)

def pool_mlt():
  pool = mp.Pool(max_processes)
  #pool.map(test, options_list)
  #result = pool.map(main.main, options_list)
  #print(result)
  #pool.map(main.main, options_list)
  pool.map(run, options_list)

if __name__ == '__main__':
  pool_mlt();

