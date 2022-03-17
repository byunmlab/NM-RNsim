# Standard modules
import sys
from configparser import ConfigParser as CfPr
import time
# Project files
from ResistorNetwork import ResistorNetwork as RN
from util import db_print
from util import str2bool
from std_sims import run_sim

SES_NUM = 2 #This should be set the same in all 3 files
filename = f"pll_log_{SES_NUM}.csv"

# Location of configuration .ini file
config_file = "config.ini"

def main():
  cp = CfPr()
  cp.read(config_file)
  # Startup Message
  db_print(cp, "Starting the Resistor Network Simulator")
  
  # Override the config N with the provided one.
  N = int(sys.argv[1])
  cp.set("RN-fiber", "N", str(N))
  # Override the sim-id
  cp.sim_id = str(SES_NUM)+"_N"+str(N)
  cp.set("sim", "sim-id", cp.sim_id)
  
  # Whether to time execution
  cp.timing = cp.getboolean("exec", "timing")
  # Build the network, optionally timing creation time
  if cp.timing: t0 = time.time()
  rn = RN.from_config(cp)
  if cp.timing:
    t1 = time.time()
    print("Building network time: ", t1-t0)
  
  # Optionally show and save a plot of the network
  show_RNfig = cp.getboolean("sim", "show_RNfig")
  save_RNfig = cp.getboolean("sim", "save_RNfig")
  if show_RNfig or save_RNfig:
    RNfig, RNax = rn.draw()
  if cp.timing:
    t2 = time.time()
    print("Drawing RNfig time: ", t2-t1)
  if save_RNfig:
    RNfname = f"RN_{cp.sim_id}.png"
    RNfig.savefig(RNfname)
    db_print(cp, f"{RNfname} saved")
  if show_RNfig:
    prtint("Did you mean to show a figure in parallel mode?")
    quit() # Remove this to enable fig.show()
    RNfig.show()
    input("Press <ENTER> to continue.")
  
  # Run the simulations specified
  for sim in cp.items("sims"):
    if str2bool(sim[1]): # If the sim is enabled
      run_sim(cp, rn, sim[0]) # Run that sim
  
  if cp.timing:
    t3 = time.time()
    print("Total run time: ", t3-t0)
  
  # Save relevant information to the parallel log file
  file = open(filename, "a")
  file.write(str(N)+", ")
  file.write(str(t1-t0)+", ")
  file.write(str(t3-t0))
  file.write("\n")
  file.close()
  print(f"Sim {cp.sim_id} Done")

main()
