"""The idea here is to make it possible to load or build a simple RN and run simple tests on it without messing up main.py or config.ini
"""

import os
# Set environment variables to limit multithreading
# This must be done before importing np
threads = "1"
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["OMP_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".05"

# Standard modules
from argparse import ArgumentParser as ArPr
from configparser import ConfigParser as CfPr
import matplotlib.pyplot as plt # For plt.close()
#FOR TESTING
import sys
sys.path.append("../src")
# Project files
from ResistorNetwork import ResistorNetwork as RN
from std_sims import run_sim
import util

def prep_options():
  """Prepare the options object by loading all options from file and CLI args
    1. Set up ArgumentParser and interpret tha arguments
    2. Set up ConfigParser and load the options from file
    3. Perform any needed adjustments to the options object
  """
  # The command-line arguments should supercede the config file in useful
  #   ways that make it unnecessary to have 10 config files for 10 simulations
  #   that are very similar in a lot of ways
  main_desc = "Run Resistor Network simulations"
  parser = ArPr(description=main_desc)
  parser.add_argument("-c", "--config_fname", default="config_TEST.ini",
    help="provide the filename of a config file")
  # These arguments allow multiple training sims to run with the same config
  parser.add_argument("-i", "--id", help="provide the SIM ID")
  parser.add_argument("-R", "--lRNf", "--load_RN_fname",
    help="provide the filename of an RN to load")
  parser.add_argument("--rs", "--rand", type=int,
    help="provide a seed for np.rand. -1 means random.")
  parser.add_argument("--Ni", "--N_in",
    help="Input to send in (for train or fwd_pass). Should be a binary string.")
  parser.add_argument("--No", "--N_out",
    help="Desired Output (for train). Should be a binary string.")
  parser.add_argument("--tf", "--threshold_fraction", type=float,
    help="Fraction used when automatically determining the threshold current \
      (for train, train_set, or fwd_pass).")
  parser.add_argument("-w", "--res_w", type=float,
    help="provide a custom res_w")
  args = parser.parse_args()

  cp = CfPr()
  if args.config_fname[-4:] == ".ini":
    cp.config_file = args.config_fname
  else:
    print("The config file must have the '.ini' extension")
  cp.read(cp.config_file)
  alter_cp(cp, args)
  
  # Set the debugging variable in util
  util.debug = cp.getboolean("exec", "debug")
  # Set the timing variable in util
  util.timing = cp.getboolean("exec", "timing")
  
  return cp

def alter_cp(cp, args):
  """Take the loaded cp object and modify it so it's ready to be used
    1. Change anything that is being overridden by the CLI args
    2. Save a couple of things as attributes of the cp object so they're
        easier to access later.
      - sim_id
      - save_format
  """
  # Modify cp to override things that should be overridden
  if args.id is not None:
    cp.set("sim", "sim_id", args.id)
  if args.lRNf is not None:
    cp.set("sim", "load_RN_fname", args.lRNf)
  if args.rs is not None:
    cp.set("exec", "rand", str(args.rs))
  if args.Ni is not None:
    try:
      int(args.Ni, 2)
    except ValueError:
      print("Please format N_in as binary")
    else:
      cp.set("sim-train", "N_in", args.Ni)
      cp.set("sim-fwd_pass", "N_in", args.Ni)
  if args.No is not None:
    try:
      int(args.No, 2)
    except ValueError:
      print("Please format N_out as binary")
    else:
      cp.set("sim-train", "N_out", args.No)
  if args.tf is not None:
    cp.set("sim-train", "threshold_fraction", str(args.tf))
    cp.set("sim-train_set", "threshold_fraction", str(args.tf))
    cp.set("sim-fwd_pass", "threshold_fraction", str(args.tf))
  if args.res_w is not None:
    cp.set("RN-res", "res_w", str(args.res_w))

  # ID of sim, to be appended to any files saved
  cp.sim_id = cp.get("sim", "sim_id")
  # Whether to use pickle or json format
  cp.save_format = cp.get("exec", "save_format")
  # Make sure that it's a supported format
  if cp.save_format != "json":
    cp.save_format == "pickle"

def main():
  """
    1. Load all options
    2. Build the RN according to specifications
    3. If specified, create a plot of the network
    4. Run each standard simulation specified
    - If specified, print the time required by each step along the way
  """
  # Prepare the configuration object
  cp = prep_options()
  # Startup Message
  util.db_print("Starting the Resistor Network Simulator - TEST MODE")
  util.db_print(f"Using the configuration file {cp.config_file}")
  util.db_print(f"Sim ID: {cp.sim_id}")

  # Build or load the network
  times = util.tic()
  load_RN_fname = cp.get("sim", "load_RN_fname")
  if load_RN_fname != "None":
    # Load the RN class attributes from the config
    RN.cls_config(cp)
    rn = RN.load_RN(load_RN_fname)
  else:
    # Note: cls_config is called within from_config
    rn = RN.from_config(cp)
  util.toc(times, "Building network")

  # Optionally show and save a plot of the network
  show_RNfig = cp.getboolean("sim", "show_RNfig")
  save_RNfig = cp.getboolean("sim", "save_RNfig")
  if show_RNfig or save_RNfig:
    RNfig, RNax = rn.draw()
    if save_RNfig:
      RNfname = f"RN_{cp.sim_id}.png"
      RNfig.savefig(RNfname)
      util.db_print(f"{RNfname} saved")
    if show_RNfig:
      RNfig.show()
      input("Press <ENTER> to continue.")
    plt.close(RNfig) # Free up that memory

  # Optionally save the network
  if cp.getboolean("sim", "save_RN"):
    save_RN_fname = f"RN_{cp.sim_id}.{cp.save_format}"
    RN.save_RN(rn, save_RN_fname)
  util.toc(times, "Drawing & saving")
  
  # Now do something with that rn
  util.db_print(f"RN size: {rn.size()}")
  util.db_print(f"RN edges: {rn.G.number_of_edges()}")
  (p_max_e, e_p_max), (p_max_n, n_p_max), Req = rn.apply_v(1, "in0", "out0",
    set_i=True)
  i0 = rn.node("in0")["isnk"]
  i1 = rn.node("out0")["isnk"]
  print(f"i_in: {-i0}; i_out: {i1}; diff: {i0+i1}")
  if abs(i0+i1) > 1e-4:
    print("Significant KCL error")

  #print(167, p_max_e, e_p_max)
  #mpe = rn.get_maxp_edges(2)
  #print(168, mpe)
  #rn.edge_burn(1)
  #mpe = rn.get_maxp_edges(2)
  #print(172, mpe)
  
  util.toc(times, "Total run", total=True)

if __name__ == "__main__":
  main()
