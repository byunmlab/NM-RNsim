"""This file contains the standard simulations runnable from main.py by setting
  the proper setting in the config file.
  config.ini --> [sim] --> 
    res = bool
    power = bool
    burn = bool

  Also has a utility function at the end for running a fwd pass on an RN
    and calculating the %RMSR error metric from those results. It's here
    since it's a kind of sub-simulation that will take time to run.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util import sim_log
from util import db_print
from util import str2arr
from util import load_IO
from util import IO_code
from util import tic, toc
from ResistorNetwork import ResistorNetwork as RN

def run_sim(cp, rn, sim):
  """Generic form of an RN test, with the specific sim to be specified.
  Parameters
  ----------
  cp : The configuration parser object
    cp.sim_id (str) : ID of sim, to be appended to any files saved.
  rn : The Resistor Network, which should be created already.
  sim (str) : Which simulation to be run of the following:
    "res" : Resistance Test. Finds Req btw the specified pins.
    "power" : Power Test. Finds power everywhere in RN.
  """
  # Startup Message
  db_print(f"Starting the {sim.upper()} Test")
  
  # Dictionary of sim functions
  sims = {
    "res" : res_sim,
    "power" : power_sim,
    "burn" : burn_sim,
    "scan_iv" : scan_iv_sim,
    "train" : train_sim,
    "train_set" : train_set_sim,
    "expand" : expand_sim,
    "fwd_pass" : fwd_pass_sim
  }
  
  return sims[sim](cp, rn)

def res_sim(cp, rn):
  """Find the equivalent resistance between two pins in the RN.
  """
  p0 = cp.get("sim-res", "pin0")
  p1 = cp.get("sim-res", "pin1")
  details = True
  save_RN = True
  
  times = tic()
  if details:
    *_, R = rn.apply_v(0.01, p0, p1, set_v=True, set_i=True)
    #in_currents = rn.sum_currents(rn.in_pin_ids(N_in=Ni))
    i0 = rn.node(p0)["isnk"]
    i1 = rn.node(p1)["isnk"]
    sim_log(f"i_in: {-i0}; i_out: {i1}; diff: {i0+i1}")
    if abs(i0+i1) > 1e-2 * abs(i0): # 1% of I
      sim_log("Significant KCL error")
  else:
    # Calculate the Req
    R = rn.R_pp(p0, p1)
  if save_RN:
    RN.save_RN(rn, f"res_{cp.sim_id}.{cp.save_format}")
  # Display the result
  sim_log(f"The equivalent resistance between {p0} and {p1} is {R}")
  toc(times, "Calculating Req", total=True)

def expand_sim(cp, rn):
  """Find the equivalent resistance between two pins in the RN before and
  after expanding the network slightly.
  """
  sim_section = "sim-expand"
  p0 = cp.get(sim_section, "pin0")
  p1 = cp.get(sim_section, "pin1")
  percent = cp.getfloat(sim_section, "expansion_percent")
  save_expanded = cp.getboolean(sim_section, "save_expanded")
  

  # Start timing
  times = tic()
  # Calculate the Req
  R = rn.R_nn(p0, p1)
  # Display the result
  sim_log(f"The equivalent resistance between {p0} and {p1} is {R}")
  toc(times, "Calculating Req")
  
  # TEMP
  sample_edge = next(iter(rn.G.edges(data=True)))
  sim_log("New 0-1 res: ", sample_edge[2]["res"])
  sim_log("New 0-1 r: ", rn.inv_res_fun(sample_edge[2]["res"]))
  
  # Expand the RN
  rn.expand(percent / 100)
  toc(times, "Expanding RN")
  
  # TEMP
  sample_edge = next(iter(rn.G.edges(data=True)))
  sim_log("New 0-1 res: ", sample_edge[2]["res"])
  sim_log("New 0-1 r: ", rn.inv_res_fun(sample_edge[2]["res"]))
  
  # Re-calculate Req
  R = rn.R_nn(p0, p1)
  # Display the result
  sim_log(f"The new equivalent resistance between {p0} and {p1} is {R}")
  toc(times, "Calculating Req")
  
  if save_expanded:
    RN.save_RN(rn, f"expanded_{cp.sim_id}.{cp.save_format}")

  toc(times, "Total expand sim", total=True)

def power_sim(cp, rn):
  """Find the power at each point in the RN.
  """
  V = cp.getfloat("sim-power", "V")
  p0 = cp.get("sim-power", "vin_pin")
  p1 = cp.get("sim-power", "vout_pin")
  
  times = tic()
  # Send current through
  (p_max, _), *_ = rn.apply_v(V, p0, p1)
  sim_log(f"Max power: {p_max}")
  toc(times, "Apply v")
  
  # Optionally show and save a plot of the voltage in the network
  show_vfig = cp.getboolean("sim-power", "show_vfig")
  save_vfig = cp.getboolean("sim-power", "save_vfig")
  if show_vfig or save_vfig:
    vfig, vax = rn.draw(edge_color=None, color_attrib="v")
    if save_vfig:
      vfname = f"voltage_{cp.sim_id}.png"
      vfig.savefig(vfname)
      db_print(f"{vfname} saved")
    if show_vfig:
      vfig.show()
      input("Press <ENTER> to continue.")
    plt.close(vfig) # Free up that memory
  # Optionally show and save a plot of the power in the network
  show_pfig = cp.getboolean("sim-power", "show_pfig")
  save_pfig = cp.getboolean("sim-power", "save_pfig")
  if show_pfig or save_pfig:
    #pfig, pax = rn.draw(width_attrib="p", edge_color="r")
    pfig, pax = rn.draw(width_attrib="p", edge_color="r", color_attrib="p")
    if save_pfig:
      pfname = f"power_{cp.sim_id}.png"
      pfig.savefig(pfname)
      db_print(f"{pfname} saved")
    if show_pfig:
      pfig.show()
      input("Press <ENTER> to continue.")
    plt.close(pfig) # Free up that memory
  
  toc(times, "Drawing")
  
  toc(times, "Total power sim", total=True)

def burn_sim(cp, rn):
  """Pass high current through the network and burn out some of the fibers.
  """
  # Load burn sim settings
  p0 = cp.get("sim-burn", "pin0")
  p1 = cp.get("sim-burn", "pin1")
  one_per = cp.getboolean("sim-burn", "one_per")
  V0 = cp.getfloat("sim-burn", "V0")
  V_step = cp.getfloat("sim-burn", "V_step")
  # Settings for what to show and save each burn
  save_last_RN = cp.getboolean("sim-burn", "save_last_RN")
  save_each_RN = cp.getboolean("sim-burn", "save_each_RN")
  show_each_p = cp.getboolean("sim-burn", "show_each_p")
  save_each_p = cp.getboolean("sim-burn", "save_each_p")
  show_each_v = cp.getboolean("sim-burn", "show_each_v")
  save_each_v = cp.getboolean("sim-burn", "save_each_v")
  
  # Start timing
  times = tic()
  
  sim_log(f"Initial p0-p1 Req: {rn.R_nn(p0, p1)}")
  # Print Req for other combinations 
  # TO DO: Make this better. There would need to be something differentiating
  #   input and output pins if this is to be automated.
  sim_log(f"Req: i0-o0: {rn.R_nn('in0', 'out0')}")
  sim_log(f"Req: i0-o1: {rn.R_nn('in0', 'out1')}")
  #sim_log(f"Req: i0-o2: {rn.R_nn('in0', 'out2')}")
  sim_log(f"Req: i1-o0: {rn.R_nn('in1', 'out0')}")
  sim_log(f"Req: i1-o1: {rn.R_nn('in1', 'out1')}")
  #sim_log(f"Req: i1-o2: {rn.R_nn('in1', 'out2')}")
  #sim_log(f"Req: i2-o0: {rn.R_nn('in2', 'out0')}")
  #sim_log(f"Req: i2-o1: {rn.R_nn('in2', 'out1')}")
  #sim_log(f"Req: i2-o2: {rn.R_nn('in2', 'out2')}")
  toc(times, "Get Req")
  
  if one_per:
    # Arbitrary, since we're just burning one per anyway
    V = 100
  else:
    # The voltage will increase each burn by V_step
    V = V0
  
  # Run the specified number of burns
  for n in range(1, 1+cp.getint("sim-burn", "burns")):
    # Send current through
    db_print(f"Applying {V}V from {p0} to {p1}")
    (p_max, _), *_ = rn.apply_v(V, p0, p1)
    toc(times, "Apply v")
    # If one_per mode is on, then go ahead and find the max power node
    if one_per:
      burn_node = rn.get_maxp_nodes()[0][0]
      to_burn = [burn_node]
    else:
      # Placeholder meaning burn all nodes with too much power
      to_burn = "p_max"
    # Make plots of v and p before burning
    if show_each_v or save_each_v:
      vfig, vax = rn.draw(edge_color=None, color_attrib="v")
      if save_each_v:
        vfname = f"burn_{cp.sim_id}_B{n}_v.png"
        vfig.savefig(vfname)
        db_print(f"{vfname} saved")
      if show_each_v:
        vfig.show()
        input("Press <ENTER> to continue.")
      plt.close(vfig) # Free up that memory
    if show_each_p or save_each_p:
      pfig, pax = rn.draw(width_attrib="p", edge_color="r", color_attrib="p",
        to_mark=to_burn)
      if save_each_p:
        pfname = f"burn_{cp.sim_id}_B{n}_p.png"
        pfig.savefig(pfname)
        db_print(f"{pfname} saved")
      if show_each_p:
        pfig.show()
        input("Press <ENTER> to continue.")
      plt.close(pfig) # Free up that memory
    toc(times, "Plotting")
    # Remove all fibers with too much power
    burned = rn.burn(to_burn)
    toc(times, "Burn")
    if save_each_RN:
      # Save the burned RN to file
      RN.save_RN(rn, f"burn_{cp.sim_id}_B{n}.{cp.save_format}")
      toc(times)#, "Save RN")
    sim_log(f"Burn #{n} p0-p1 Req: {rn.R_nn(p0, p1)}")
    # Print Req for other combinations
    sim_log(f"Req: i0-o0: {rn.R_nn('in0', 'out0')}")
    sim_log(f"Req: i0-o1: {rn.R_nn('in0', 'out1')}")
    #sim_log(f"Req: i0-o2: {rn.R_nn('in0', 'out2')}")
    sim_log(f"Req: i1-o0: {rn.R_nn('in1', 'out0')}")
    sim_log(f"Req: i1-o1: {rn.R_nn('in1', 'out1')}")
    #sim_log(f"Req: i1-o2: {rn.R_nn('in1', 'out2')}")
    #sim_log(f"Req: i2-o0: {rn.R_nn('in2', 'out0')}")
    #sim_log(f"Req: i2-o1: {rn.R_nn('in2', 'out1')}")
    #sim_log(f"Req: i2-o2: {rn.R_nn('in2', 'out2')}")
    toc(times, "Get Req")
    if not one_per:
      V += V_step
  
  # Save the ending RN state (optionally)
  if save_last_RN:
    # Save the burned RN to file
    RN.save_RN(rn, f"burn_{cp.sim_id}_last.{cp.save_format}")
    toc(times, "Save RN")
  
  toc(times, "Total burn sim", total=True)

def scan_iv_sim(cp, rn):
  """Make an i-v plot by scanning through a range of voltages
  """
  # Load iv sim settings
  sim_section = "sim-scan_iv"
  p0 = cp.get(sim_section, "pin0")
  p1 = cp.get(sim_section, "pin1")
  V_max = cp.getfloat(sim_section, "V_max")
  V_step = cp.getfloat(sim_section, "V_step")
  save_last_RN = cp.getboolean(sim_section, "save_last_RN")
  burning = cp.getboolean(sim_section, "burn")
  
  # Pseudocode:
  #   1. Apply V volts
  #   2. Find the current that flows because of V
  #     2.1. Save this V, I coordinate to file and in an array
  #   3. Burn out any fibers that have too much power
  #   4. Repeat
  #   5. Make a plot of the V, I data and save it to file
  
  # List of voltages
  Vs = np.arange(0, V_max + V_step/2, V_step)
  # Empty array to hold the current values
  Is = np.zeros(Vs.shape)
  # Create point log file
  pt_log_fname = f"IV_pts_{cp.sim_id}.csv"
  pt_log = open(pt_log_fname, "w")
  pt_log.write("V, I\n")
  pt_log.close()
  
  # Start timing
  times = tic()
  
  db_print(f"Scanning I-V up to V_max={V_max} by V_step={V_step}")
  for i in range(Vs.size):
    toc(times)
    # Apply V volts, save the power, and get the Req
    *_, Req = rn.apply_v(Vs[i], p0, p1, set_i=True)
    toc(times, "Apply v")

    # Convert keys to indices
    #p0i = rn.key_to_index(p0)
    #p1i = rn.key_to_index(p1)
    # Current flowing through RN
    I_in = - rn.G.nodes[p0]["isnk"]
    I_out = rn.G.nodes[p1]["isnk"]
    db_print(f"I_in={I_in}; I_out={I_out}; diff={I_out-I_in}")
    Is[i] = (I_in + I_out) / 2 # Use avg, though they should be equal
    # Save this V, I coordinate to the pt log
    pt_log = open(pt_log_fname, "a")
    pt_log.write(f"{Vs[i]}, {Is[i]}\n")

    if burning:
      # Burn out anything that has too much power
      num_burned = rn.burn()
      db_print(f"Point {i}: {Vs[i]}, {Is[i]}. {num_burned} fibers burned.")
  
  fig, ax = plt.subplots()
  figname = f"IV_curve_{cp.sim_id}"
  fig.suptitle(figname)
  ax.plot(Vs, Is)
  fig.savefig(figname)
  db_print(f"{figname} saved")
  plt.close(fig)

  # Save the ending RN state (optionally)
  if save_last_RN:
    toc(times)
    # Save the burned RN to file
    RN.save_RN(rn, f"IV_scan_{cp.sim_id}_last.{cp.save_format}")
    toc(times, "Save RN")

  toc(times, "Total scan_iv sim", total=True)

def train_sim(cp, rn):
  """Make an attempt at training the network
  """
  # Load the options
  sim_section = "sim-train"
  N_in = cp.get(sim_section, "N_in")
  N_out = cp.get(sim_section, "N_out")
  max_burns = cp.getint(sim_section, "max_burns")
  save_each_p = cp.getboolean(sim_section, "save_each_p")
  save_each_v = cp.getboolean(sim_section, "save_each_v")
  save_each_RN = cp.getboolean(sim_section, "save_each_RN")
  threshold_fraction = cp.getfloat(sim_section, "threshold_fraction")
  burn_rate = cp.getint(sim_section, "burn_rate")
  try:
    int(cp.get(sim_section, "N_in"), 2)
    int(cp.get(sim_section, "N_out"), 2)
  except ValueError:
    sim_log("Please format N_in and N_out as binary")
    return

  rn.threshold_fraction = threshold_fraction
  rn.burn_rate = burn_rate

  db_print(f"Sending in this input: {N_in}")
  db_print(f"Desired output: {N_out}")

  # Callback function to be called each burn
  def callback(burn):
    nonlocal times
    # Print some things
    db_print(f"Burn {burn}")
    out_currents = rn.sum_currents(rn.out_pin_ids())
    in_currents = - rn.sum_currents(rn.in_pin_ids(N_in))
    output = rn.threshold_currents(out_currents)
    db_print(f"Current leaving output pins: {out_currents}\nSum = {np.sum(out_currents)}")
    db_print(f"Current entering input pins: {in_currents}\nSum = {np.sum(in_currents)}")
    #db_print(f"Threshold: {rn.threshold}")
    #db_print(f"Output: {output}")
    if save_each_p:
      # Save an image of the power plot
      pfig, pax = rn.draw(width_attrib="p", edge_color="r", color_attrib="p")
      pfname = f"train_{cp.sim_id}_B{burn}_p.png"
      pfig.savefig(pfname)
      db_print(f"{pfname} saved")
    if save_each_v:
      # Save an image of the voltage plot
      vfig, vax = rn.draw(edge_color=None, color_attrib="v")
      vfname = f"train_{cp.sim_id}_B{burn}_v.png"
      vfig.savefig(vfname)
      db_print(f"{vfname} saved")
    if save_each_RN:
      # Save the RN state
      RNfname = f"RN_{cp.sim_id}_B{burn}.{cp.save_format}"
      RN.save_RN(rn, RNfname)
    toc(times, f"Burn {burn}")

  # Start timing
  times = tic()

  # Train the network
  code, burns = rn.train(N_in, N_out, max_burns=max_burns,
    callback=callback)
  db_print(f"Training success code: {code}. Burns: {burns}")

  # Run a forward pass
  #res, out_currents = rn.fwd_pass(N_in)
  #in_currents = - rn.sum_currents(rn.in_pin_ids(N_in))
  #db_print(f"The initial output was {res}. The desired output was {N_out}")
  #db_print(f"Initial currents leaving output pins: {out_currents}. Sum = {np.sum(out_currents)}")
  #db_print(f"Initial currents entering input pins: {in_currents}. Sum = {np.sum(in_currents)}")

  # Save the trained network
  RN.save_RN(rn, f"RN_{cp.sim_id}_trained.{cp.save_format}")

  toc(times, "Total training sim", total=True)

def train_set_sim(cp, rn):
  """Given a set of inputs and desired outputs, attempt to train the network
  to create those relationships.
  Essentially, this repeats the train_sim several times.
  """
  # Load the options
  sim_section = "sim-train_set"
  IO_fname = cp.get(sim_section, "IO_fname")
  max_burns = cp.getint(sim_section, "max_burns")
  max_epochs = cp.getint(sim_section, "max_epochs")
  save_each_p = cp.getboolean(sim_section, "save_each_p")
  save_each_v = cp.getboolean(sim_section, "save_each_v")
  save_each_RN = cp.getboolean(sim_section, "save_each_RN")
  save_epoch_trained_RN = cp.getboolean(sim_section, "save_epoch_trained_RN")
  threshold_fraction = cp.getfloat(sim_section, "threshold_fraction")
  burn_rate = cp.getint(sim_section, "burn_rate")
  RMSR_mode = cp.get(sim_section, "RMSR_mode").lower()
  plot_RMSR = cp.getboolean(sim_section, "plot_RMSR")
  preburn_fraction = cp.getfloat(sim_section, "II_OO_preburn_fraction")
  burn_fibers = cp.getboolean(sim_section, "burn_fibers")

  rn.threshold_fraction = threshold_fraction
  rn.burn_rate = burn_rate
  rn.burn_fibers = burn_fibers

  # Load the IO file as a np.array
  IO_Ns = load_IO(IO_fname, npa=True)
  # This should correspond to the lowest RMSR that could still be wrong
  #   i.e. Everything is perfect except one IO combo, which is 50-50 Error
  #   Then multiply by threshold for good measure
  stopping_RMSR = threshold_fraction * 100 / np.sqrt(IO_Ns.size)

  # Start timing
  times = tic()

  # Perform the preburning, if specified
  if preburn_fraction > 0:
    # Make a list of burns to perform
    to_burn = []
    for i in range(len(rn.in_pins)-1):
      to_burn.append( (rn.in_pins[i][0], rn.in_pins[i+1][0]) )
    for i in range(len(rn.out_pins)-1):
      to_burn.append( (rn.out_pins[i][0], rn.out_pins[i+1][0]) )
    for p0, p1 in to_burn:
      #db_print(f"443, Burn some fibers out between {p0} and {p1}")
      rn.apply_v(rn.bpV, p0, p1)
      # Burn 0.1% of the RN (+eps)
      if burn_fibers:
        rn.burn(int(rn.N * preburn_fraction))
      else:
        rn.edge_burn(int(rn.N * preburn_fraction))
    RN.save_RN(rn, f"RN_{cp.sim_id}_preburnt.{cp.save_format}")
    toc(times, "Preburning")

  if RMSR_mode != "None":
    # Calculate RMSR error before training
    RMSR, currents, _ = fp_RMSR(rn, IO_Ns, mode=RMSR_mode, msg="before training")
    toc(times, "FP & %RMSR calculation")
    # Format: [Epoch, RMSR, code]
    RMSRs = np.array([0, RMSR, -1])

  # Callback function to be called each burn
  def callback(burn):
    nonlocal times
    # Print some things
    ESB_phrase = f"Epoch {epoch}, Step {step}, Burn {burn}"
    db_print(f" - {ESB_phrase} - ")
    ESB_tag = f"E{epoch}_S{step}_B{burn}"
    out_currents = rn.sum_currents(rn.out_pin_ids())
    in_currents = - rn.sum_currents(rn.in_pin_ids())
    output = rn.threshold_currents(out_currents)
    db_print(f"Current leaving output pins: {out_currents}"
        f" (Sum = {np.sum(out_currents):.12f})")
    db_print(f"Current entering input pins: {in_currents}"
        f" (Sum = {np.sum(in_currents):.12f})")
    #db_print(f"Current entering input pins: {in_currents}\nSum = {np.sum(in_currents)}")
    #db_print(f"Threshold: {rn.threshold}")
    #db_print(f"Output: {output}")
    if save_each_p:
      # Save an image of the power plot
      pfig, pax = rn.draw(width_attrib="p", edge_color="r", color_attrib="p")
      pfname = f"train_{cp.sim_id}_{ESB_tag}_p.png"
      pfig.savefig(pfname)
      db_print(f"{pfname} saved")
    if save_each_v:
      # Save an image of the voltage plot
      vfig, vax = rn.draw(edge_color=None, color_attrib="v")
      vfname = f"train_{cp.sim_id}_{ESB_tag}_v.png"
      vfig.savefig(vfname)
      db_print(f"{vfname} saved")
    if save_each_RN:
      # Save the RN state
      RNfname = f"RN_{cp.sim_id}_{ESB_tag}.{cp.save_format}"
      RN.save_RN(rn, RNfname)
    toc(times, ESB_phrase)

  epoch_times = tic() #Independant timers for epochs and steps
  epoch = 1 # Start at 1 because 0 represents before training now
  RMSR = stopping_RMSR + 100
  epoch_code = 2
  while (RMSR > stopping_RMSR and epoch <= max_epochs and epoch_code in (1,2)):
    step_times = tic()
    step = 0
    epoch_code = 0
    for row in IO_Ns:
      Ni = row[0]
      No = row[1]
      if int(Ni, 2) == 0:
        # If I bias the inputs later, then this will need to change
        db_print(f"Skipping the input {Ni}, which requires no training.")
        continue
      ES_phrase = f"Epoch {epoch}, Step {step}"
      db_print(f"  -- {ES_phrase} --  ")
      db_print(f"Sending in this input: {Ni}. Desired output: {No}")

      # Train the network
      code, burns = rn.train(Ni, No, max_burns=max_burns,
        callback=callback)
      # If they're all 0, it's done. If any > 2, it's broken.
      epoch_code = max(epoch_code, code)
      db_print(f"Training success code: {code}. Burns: {burns}")
      toc(step_times, ES_phrase)
      step += 1

    if save_epoch_trained_RN:
      # Save the RN state after training
      RN.save_RN(rn, f"RN_{cp.sim_id}_E{epoch}_trained.{cp.save_format}")

    if RMSR_mode != "None":
      # Calculate RMSR error
      RMSR, currents, _ = fp_RMSR(rn, IO_Ns, mode=RMSR_mode,
        msg="at this RN state")
      toc(times, "FP & %RMSR calculation")
      code = IO_code(IO_Ns, currents, threshold_fraction=threshold_fraction)
      # the IO_code method can't differentiate btw 1 and 2
      # However, it's better at seeing if a pin burned out
      if code > 2:
        epoch_code = code
      RMSRs = np.vstack((RMSRs, [epoch, RMSR, epoch_code]))

    toc(epoch_times, f"Epoch {epoch}")
    epoch += 1

  if RMSR_mode != "None":
    # Save the RMSR points to file
    RMSR_fname = f"RMSRs_{cp.sim_id}.csv"
    pd.DataFrame(RMSRs).to_csv(RMSR_fname, index=None,
      header=["Epoch", "%RMSR", "Code"])
    db_print(f"{RMSR_fname} saved")
    if plot_RMSR:
      # Make a plot of RMSR across each epoch
      fig, ax = plt.subplots()
      ax.scatter(RMSRs[:,0], RMSRs[:,1])
      figname = f"RMSRs_{cp.sim_id}.png"
      fig.suptitle(figname)
      ax.set_xlabel("Epoch #")
      ax.set_ylabel("%RMSR error")
      fig.savefig(figname)
      db_print(f"{figname} saved")
      plt.close(fig)

  toc(times, "Total train_set sim", total=True)

def fwd_pass_sim(cp, rn):
  """Find the output currents for an input or set of inputs
  """
  # Load the options
  sim_section = "sim-fwd_pass"
  IO_fname = cp.get(sim_section, "IO_fname")
  N_in = cp.get(sim_section, "N_in")
  save_RN = cp.getboolean(sim_section, "save_RN")
  threshold = cp.get(sim_section, "threshold").lower()
  threshold_fraction = cp.getfloat(sim_section, "threshold_fraction")

  if threshold == "fraction-max":
    db_print("452: Not yet implemented")
  if threshold == "fraction-inh":
    rn.threshold_fraction = threshold_fraction
    # Reset the threshold to the specified fraction of the highest output,
    #   when all the inputs are high.
    assert(N_in[1] == 'b')
    high_Nin = N_in[0:2] + N_in[2:].replace('0','1')
    rn.fwd_pass(high_Nin, reset_threshold="max")

  # Determine which inputs will be run, 
  #   either from file or from the provided N_in option.
  if IO_fname == "None":
    # Ns_in = list of binary numbers representing a list of inputs to run.
    Ns_in = [N_in]
    if N_in.lower() == "all":
      Npins = len(rn.in_pins)
      Ns_in = []
      # Could probably skip zero, unless there's an input bias
      for n in range(2**Npins):
        nstr = str(bin(n))
        nstr = nstr[0:2] + nstr[2:].zfill(Npins)
        Ns_in.append(nstr)
    else:
      try:
        int(cp.get(sim_section, "N_in"), 2)
      except ValueError:
        sim_log("Please format N_in as binary")
        return
      # If it's a valid binary number, then Ns_in is correct from before the if
  else:
    # load the IO file
    IO_df = load_IO(IO_fname)
    Ns_in = IO_df["IN"].to_numpy()
  #sim_log(471, Ns_in)

  # Start timing
  times = tic()

  for Ni in Ns_in:
    db_print(f"Sending in this input: {Ni}")
    toc(times)
    output, currents = rn.fwd_pass(Ni)
    
    #TEMP
    #_, isnk = rn.through_currents(ret_sink=True, store=True)
    #sim_log("std_simd:636", isnk)
    #v = nx.get_node_attributes(rn.G, "v")
    #sim_log("std_simd:637", v)
    
    toc(times, f"Fwd pass with Ni={Ni} calculation")
    in_currents = rn.sum_currents(rn.in_pin_ids(N_in=Ni))
    current_err = np.sum(currents) + np.sum(in_currents)
    if current_err > .005:
      db_print(f"Significant KCL error. in_currents: {in_currents};"
        f" out_currents: {currents}")
    if save_RN:
      RN.save_RN(rn, f"RN_{cp.sim_id}_i{Ni}.{cp.save_format}")
    
    sim_log("The input currents were: ", in_currents)
    sim_log("The output currents were: ", currents)
    if threshold != None: 
      sim_log("The output was: ", output)

  toc(times, "Total fwd_pass sim", total=True)

def fp_RMSR(rn, IO_Ns, mode="max", msg=None):
  """Run a fwd_pass on the rn with all the combinations of inputs and calculate
  the error based on the desired outputs.

  Parameters
  ----------
  rn : an RN
  IO_Ns : a np array with the string representations of inputs and outputs
  mode : one of ("max", "rel")
    "max" : take the max of all the output currents to be the desired output 
      current for all HIGH outputs.
    "rel" : take the sum of the input currents for each input divided 
      by the number of HIGH outputs to be the desired output 
  msg (str) : if a string is given, the current and RMSR will be printed,
    with the given explanatory message clip.
  """

  N_out_pins = len(IO_Ns[0][1].split("b")[1]) #Assumes 0bXXX format
  out_currents = np.zeros((IO_Ns.shape[0], N_out_pins))
  current_err = np.zeros(IO_Ns.shape[0]) #sum(out) - sum(in)
  desired_out_currents = np.zeros(out_currents.shape)
  for i in range(len(IO_Ns)):
    Ni, No = IO_Ns[i]
    output, currents = rn.fwd_pass(Ni)
    out_currents[i, :] = currents
    in_currents = rn.sum_currents(rn.in_pin_ids(N_in=Ni))
    current_err[i] = np.sum(currents) + np.sum(in_currents)
    if current_err[i] > .005:
      db_print(f"Significant KCL error. in_currents: {in_currents};"
        f" out_currents: {currents}")
    for si in range(len(No)-1, 1, -1):
      # Loop backwards, since the LSB is output 0
      if No[si] == "1":
        sioi = N_out_pins-1 - (si-2) # convert string index to out pin #
        desired_out_currents[i,sioi] = 1
    if mode == "rel":
      N_HI_outs = np.sum(desired_out_currents[i,:])
      if N_HI_outs != 0:
        desired_out_currents[i,:] *= np.sum(currents) / N_HI_outs
  max_current = np.max(out_currents)
  if mode == "max":
    desired_out_currents *= max_current
  
  res = desired_out_currents - out_currents
  divisor = np.repeat(np.vstack(np.sum(out_currents, axis=1)), 2, axis=1)
  # Don't divide by zero. Let 0 current contribute 0 error. Warn?
  divisor[divisor < 1e-5] = 1
  # percent normalized residuals
  pnR = 100 * res / divisor
  # Root Mean Square Residuals (where those are percent normalized residuals)
  RMSR = np.sqrt( np.mean(np.square(pnR)) )

  # Normalize and take percentage
  #pn_RMSR = 100 * np.sqrt(RMSR)
  #divisor = 1
  #if mode == "rel":
  #  # the max total current ever entering at one time
  #  divisor = np.max(np.sum(out_currents, axis=1))
  #elif mode == "max":
  #  divisor = max_current
  #else: 
  #  # Shouldn't happen
  #  raise Exception
  #if divisor < 1e-6:
  #  db_print("Warning: max current is very small")
  #  divisor = 1e-6
  #pn_RMSR /= divisor

  if msg is not None:
    sim_log(f"The output currents {msg} are: ")
    for i in range(len(IO_Ns)):
      Ni, _ = IO_Ns[i]
      sim_log(f"\tfor input {Ni}: {out_currents[i,:]} (err={current_err[i]:.8f})")
    # sim_log(currents) # This is the lazy version
    sim_log(f"The %RMSR {msg} is: ", RMSR)

  return RMSR, out_currents, current_err

