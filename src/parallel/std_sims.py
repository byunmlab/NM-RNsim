"""This file contains the standard simulations runnable from main.py by setting
  the proper setting in the config file.
  config.ini --> [sim] --> 
    res = bool
    power = bool
"""

import matplotlib.pyplot as plt
import time

from util import db_print
from util import str2arr

def run_sim(cp, rn, sim):
  """Generic form of an RN test, with the specific sim to be specified.
  Parameters
  ----------
  cp : The configuration parser object
    cp.timing (bool) : whether to time and print timing.
    cp.sim_id (str) : ID of sim, to be appended to any files saved.
  rn : The Resistor Network, which should be created already.
  sim (str) : Which simulation to be run of the following:
    "res" : Resistance Test. Finds Req btw the specified pins.
    "power" : Power Test. Finds power everywhere in RN.
  """
  # Startup Message
  db_print(cp, f"Starting the {sim.upper()} Test")
  
  if sim == "res":
    return res_sim(cp, rn)
  if sim == "power":
    return power_sim(cp, rn)

def res_sim(cp, rn):
  """Find the equivalent resistance between two pins in the RN.
  """
  p0 = cp.get("sim-res", "pin0")
  p1 = cp.get("sim-res", "pin1")
  
  if cp.timing: t0 = time.time()
  # Calculate the Req
  R = rn.R_pp(p0, p1)
  # Display the result
  print(f"The equivalent resistance between {p0} and {p1} is {R}")
  if cp.timing:
    t1 = time.time()
    print("Calculating Req time: ", t1-t0)

def power_sim(cp, rn):
  """Find the power at each point in the RN.
  """
  V = cp.getfloat("sim-power", "V")
  p0 = cp.get("sim-power", "vin_pin")
  p1 = cp.get("sim-power", "vout_pin")
  
  if cp.timing: t0 = time.time()
  # Send current through
  p_max, _ = rn.apply_v(V, p0, p1)
  print(f"Max power: {p_max}")
  if cp.timing:
    t1 = time.time()
    print("Apply v time: ", t1-t0)
  
  # Optionally show and save a plot of the voltage in the network
  show_vfig = cp.getboolean("sim-power", "show_vfig")
  save_vfig = cp.getboolean("sim-power", "save_vfig")
  if show_vfig or save_vfig:
    vfig, vax = rn.draw(edge_color=None, color_attrib="v")
  if save_vfig:
    vfname = f"voltage_{cp.sim_id}.png"
    vfig.savefig(vfname)
    db_print(cp, f"{vfname} saved")
  if show_vfig:
    vfig.show()
    input("Press <ENTER> to continue.")
  # Optionally show and save a plot of the power in the network
  show_pfig = cp.getboolean("sim-power", "show_pfig")
  save_pfig = cp.getboolean("sim-power", "save_pfig")
  if show_pfig or save_pfig:
    pfig, pax = rn.draw(width_attrib="p", edge_color="r")
  if save_pfig:
    pfname = f"power_{cp.sim_id}.png"
    pfig.savefig(pfname)
    db_print(cp, f"{pfname} saved")
  if show_pfig:
    pfig.show()
    input("Press <ENTER> to continue.")
    
  if cp.timing:
    t2 = time.time()
    print("Drawing time: ", t2-t1)
  
  if cp.timing:
    t3 = time.time()
    print("Total power sim time: ", t3-t0)

