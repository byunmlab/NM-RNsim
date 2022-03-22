import os
# Set environment variables to limit multithreading
# This must be done before importing np
threads = "1"
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["OMP_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads
# jax options
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
# Results in 666MB allocated, at least for the first process
#   --> about right for N=500
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".011"

# Standard modules
from argparse import ArgumentParser as ArPr
from configparser import ConfigParser as CfPr
import matplotlib.pyplot as plt # For plt.close()
# Project files
from ResistorNetwork import ResistorNetwork as RN
from std_sims import run_sim
import util

def prep_options(options_dict):
  """Prepare the options object by loading all options from file and CLI args
    1. Set up ArgumentParser and interpret tha arguments
    2. Set up ConfigParser and load the options from file
    3. Perform any needed adjustments to the options object
    4. Override anything from options_dict
  """
  # The command-line arguments should supercede the config file in useful
  #   ways that make it unnecessary to have 10 config files for 10 simulations
  #   that are very similar in a lot of ways
  main_desc = "Run Resistor Network simulations"
  parser = ArPr(description=main_desc)
  parser.add_argument("-c", "--config_fname", default="config.ini",
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
    util.sim_log("The config file must have the '.ini' extension")
  cp.read(cp.config_file)
  alter_cp(cp, args, options_dict)
  
  # Initialize the settings that are stored in util
  util.init(cp)
  
  return cp

def alter_cp(cp, args, options_dict):
  """Take the loaded cp object and modify it so it's ready to be used
    1. Change anything that is being overridden by the CLI args
    2. Save a couple of things as attributes of the cp object so they're
        easier to access later.
      - sim_id
      - save_format
  """
  # Modify cp to override things that should be overridden from args
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
      util.sim_log("Please format N_in as binary")
    else:
      cp.set("sim-train", "N_in", args.Ni)
      cp.set("sim-fwd_pass", "N_in", args.Ni)
  if args.No is not None:
    try:
      int(args.No, 2)
    except ValueError:
      util.sim_log("Please format N_out as binary")
    else:
      cp.set("sim-train", "N_out", args.No)
  if args.tf is not None:
    cp.set("sim-train", "threshold_fraction", str(args.tf))
    cp.set("sim-train_set", "threshold_fraction", str(args.tf))
    cp.set("sim-fwd_pass", "threshold_fraction", str(args.tf))
  if args.res_w is not None:
    cp.set("RN-res", "res_w", str(args.res_w))

  # Add on the options from options_dict
  if "id" in options_dict:
    cp.set("sim", "sim_id", options_dict["id"])
  if "rs" in options_dict:
    cp.set("exec", "rand", options_dict["rs"])
  #if "ks" in options_dict: # Instead, pass cnd_len
  #  cnd_len = int( np.cbrt(V/N) / options_dict["ks"])
  if "cnd_len" in options_dict:
    cp.set("RN-fiber", "cnd_len", options_dict["cnd_len"])
  if "fl_mus" in options_dict:
    cp.set("RN-fiber", "fl_mus", options_dict["fl_mus"])
  if "bpwr_mus" in options_dict:
    cp.set("RN-fiber", "bpwr_mus", options_dict["bpwr_mus"])
  if "ftype_proportions" in options_dict:
    cp.set("RN-fiber", "ftype_proportions", 
      options_dict["ftype_proportions"])
  if "preburn_fraction" in options_dict:
    cp.set("sim-train_set", "II_OO_preburn_fraction",
      options_dict["preburn_fraction"])
  if "burn_fibers" in options_dict:
    cp.set("sim-train_set", "burn_fibers", options_dict["burn_fibers"])

  # ID of sim, to be appended to any files saved
  cp.sim_id = cp.get("sim", "sim_id")
  # Whether to use pickle or json format
  cp.save_format = cp.get("exec", "save_format")
  # Make sure that it's a supported format
  if cp.save_format != "json":
    cp.save_format == "pickle"

def main(options={}):
  """
    1. Load all options
    2. Build the RN according to specifications
    3. If specified, create a plot of the network
    4. Run each standard simulation specified
    - If specified, print the time required by each step along the way
  """
  # Prepare the configuration object
  cp = prep_options(options)
  # Startup Message
  util.db_print("Starting the Resistor Network Simulator")
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
  
  # Run the simulations specified
  for sim in cp.items("sims"):
    if util.str2bool(sim[1]): # If the sim is enabled
      run_sim(cp, rn, sim[0]) # Run that sim
  
  util.toc(times, "Total run", total=True)

if __name__ == "__main__":
  main()
