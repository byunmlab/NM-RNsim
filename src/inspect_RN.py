"""The purpose of this program is not to generate or run simulations on
Resistor Networks, but to load and analyze a saved RN.
The reason this file is not in aux_scripts is that it needs RN.py
Parameters
----------
filename : The file to load. Should be a .pickle or .json file
"""

import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from ResistorNetwork import ResistorNetwork as RN
import util

# Parse the arguments
parser = argparse.ArgumentParser(description="Inspect an RN.")
parser.add_argument("filename", nargs="+", help="provide an RN file")
parser.add_argument("-c", "--convert", action="store_true",
  help="Convert between json and pickle")
parser.add_argument("-ngz", "--no_compress", action="store_false",
  help="Don't compress the output file")
parser.add_argument("-L", "--Laplacian", action="store_true",
  help="Find and save the Laplacian matrix of the RN")
parser.add_argument("-v", "--voltage", action="store_true",
  help="Save the voltage at every node to file")
parser.add_argument("-i", "--current", action="store_true",
  help="Save the current flowing through every node to file")
parser.add_argument("-x", "--node_pos", action="store_true",
  help="Save the xyz position of each node")
parser.add_argument("-I", "--image", action="store_true",
  help="Save an image of the RN (Note: if -v, then plot voltage)")
parser.add_argument("-N", "--count_nodes", action="store_true",
  help="Count the number of nodes in the RN")
parser.add_argument("-P", "--count_pins", action="store_true",
  help="Count the number of pins in the RN")
parser.add_argument("-F", "--count_fibers", action="store_true",
  help="Count the number of fibers in the RN")
parser.add_argument("-E", "--count_edges", action="store_true",
  help="Count the number of edges in the RN")
parser.add_argument("-S", "--sparsity", action="store_true",
  help="Report two metrics of the sparsity of the RN")
parser.add_argument("-fl", "--fiber_length", action="store_true",
  help="Investigate the fiber length")
parser.add_argument("-d", "--diff", "--difference", action="store_true",
  help="Investigate the difference between 2 RNs, plotting the deleted fibers")
args = parser.parse_args()

filenames = args.filename
# What to do with the RN
convert_json_pickle = args.convert # Convert btw json and pickle
compress = args.no_compress
save_Lpl = args.Laplacian
save_v = args.voltage
save_i = args.current
save_xyz = args.node_pos
save_fig = args.image
count_N = args.count_nodes
count_P = args.count_pins
count_F = args.count_fibers
count_E = args.count_edges
report_Ks = args.sparsity
report_fl = args.fiber_length
plot_diff = args.diff

def inspect_RN(filename):
  """Inspect the given RN, using the pre-loaded options from above
  """
  rn, trm_fname, ftype = load_RNfile(filename)
  if rn is None:
    return

  # Get some numbers about the RN
  N = rn.size()
  P = len(rn.pins)
  F = N - P
  E = rn.G.number_of_edges()

  # Report the requested numbers
  if count_N:
    print(f"This RN has {N} nodes")
  if count_P:
    print(f"This RN has {P} pins")
  if count_F:
    print(f"This RN has {F} fibers")
  if count_E:
    print(f"This RN has {E} edges")
  if report_Ks:
    # Calculate and report two metrics of the sparsity
    V_rn = (rn.xmax-rn.xmin) * (rn.ymax-rn.ymin) * (rn.zmax-rn.zmin)
    ks = 2*(V_rn / N)**(1/3) / rn.cnd_len
    print(f"Coefficient of Sparsity k_s = {ks:6f}")
    EpN = E / N
    print(f"Edges per node = {EpN:6f}")
  if report_fl:
    #print(f"The average fiber length was supposed to be {rn.fl_mean}")
    #print(f"The fiber length distribution span was supposed to be {rn.fl_span}")
    # Note: the rn.fv attribute is probably wrong, since it wasn't saved in the json.
    #if hasattr(rn, "fv"):
    #  fv = rn.fv
    #  print(117, fv.shape)
    fv = nx.get_node_attributes(rn.G, "fv")
    fv = np.vstack(list(fv.values()))
    #print(120, fv.shape)
    fl = np.zeros( (fv.shape[0], 1) )
    for i, row in enumerate(fv):
      fl[i] = np.linalg.norm(row)
    fl_min = np.min(fl)
    fl_max = np.max(fl)
    print("The shortest fiber length is", fl_min)
    print("The longest fiber length is", fl_max)
    fig, ax = plt.subplots()
    ax.hist(fl, bins=100)
    save_fname = trm_fname + "_flhist.png"
    fig.savefig(save_fname)
    print("Histogram of fiber lengths saved to file:", save_fname)

  if save_Lpl:
    # Get the Laplacian matrix and save it to file
    Lpl = nx.linalg.laplacian_matrix(rn.G, weight="cnd", nodelist=rn.node_list)
    npLpl = Lpl.toarray()
    save_fname = trm_fname + "_Lpl.csv"
    np.savetxt(save_fname, npLpl, delimiter=",")
    print("Laplacian matrix saved to file:", save_fname)
    if False:
      #TEMP
      iLpl = np.linalg.pinv(npLpl)
      save_fname = trm_fname + "_iLpl.csv"
      np.savetxt(save_fname, iLpl, delimiter=",")
      print("Inverse Laplacian matrix saved to file:", save_fname)

  if save_v:
    # Save the voltage stored at every node and save that to file
    v = nx.get_node_attributes(rn.G, "v")

    if len(v) == 0:
      print("The voltage has not been stored in this RN.")
    else:
      save_fname = trm_fname + "_v.csv"
      vfile = open(save_fname, "w")
      vfile.write("NODE ID, VOLTAGE\n")
      for node, v in v.items():
        vfile.write(f"{node}, {v}\n")
      vfile.close()
      print("Voltage at each node saved to file:", save_fname)

  if save_i:
    # Save the current flowing through each node to file
    # TODO: allow user to specify a config file instead
    rn.res_w = 20
    rn.sol_method = "mlt"
    print("Warning: This will not work correctly unless the cls_config settings"
      " for resistance are set correctly.")

    i_dict, s_dict = rn.through_currents(ret_sink=True, store=True)
    save_fname = trm_fname + "_i.csv"
    ifile = open(save_fname, "w")
    ifile.write("NODE ID, THROUGH CURRENT, SINKING CURRENT\n")
    for node, i in i_dict.items():
      ifile.write(f"{node}, {i}, {s_dict[node]}\n")
    ifile.close()
    print("Current flowing through each node saved to file:", save_fname)

  if save_xyz:
    # Save the position of each node to file
    pos = nx.get_node_attributes(rn.G, 'pos')
    save_fname = trm_fname + "_xyz.csv"
    xfile = open(save_fname, "w")
    xfile.write("NODE ID, X, Y, Z\n")
    for node, npos in pos.items():
      xfile.write(f"{node}, {npos[0]}, {npos[1]}, {npos[2]}\n")
    xfile.close()
    print("Position of each node saved to file:", save_fname)

  if save_fig:
    # Save a plot of the network
    plt_defaults()
    i_color = "#0652ff" #"aqua" #"gold"
    
    print("Plotting...")
    if save_v and save_i:
      fig, ax = rn.draw(edge_color=i_color, width_attrib="i", 
        color_attrib="v", annotate_is=True)
      save_fname = trm_fname + "_vi.png"
    elif save_v:
      fig, ax = rn.draw(edge_color=None, color_attrib="v")
      save_fname = trm_fname + "_v.png"
    elif save_i:
      #fig, ax = rn.draw(edge_color=i_color, width_attrib="i", 
      #  color_attrib="i", annotate_is=True)
      fig, ax = rn.draw(edge_color=i_color, width_attrib="i", 
        annotate_is=True)
      save_fname = trm_fname + "_i.png"
    else:
      fig, ax = rn.draw()
      save_fname = trm_fname + ".png"
    fig.savefig(save_fname)
    print("Figure saved to:", save_fname)

  if convert_json_pickle:
    # If a json was given, save a pickle, and vice-versa
    if ftype == "pickle":
      save_fname = trm_fname + ".json"
    elif ftype == "json":
      save_fname = trm_fname + ".pickle"
    else:
      print("ERROR 95")
      quit()

    fn = RN.save_RN(rn, save_fname, compress=compress)
    print("RN saved to file:", fn)

def inspect_RNdiff(filenames):
  if len(filenames) != 2:
    print("Provide 2 files to compare")
    quit()
  fn0, fn1 = filenames
  rn0, trimfn0, ftype0 = load_RNfile(fn0)
  rn1, trimfn1, ftype1 = load_RNfile(fn1)
  if rn0 is None or rn1 is None:
    return
   
  N0 = rn0.size()
  assert N0 == len(rn0.node_list)
  N1 = rn1.size()
  assert N1 == len(rn1.node_list)
  if N0 > N1:
    rnM = rn0 # Major and minor RNs
    rnm = rn1
  elif N1 > N0:
    rnM = rn1
    rnm = rn0
  else:
    print("Both networks have the same N")
    return
  
  diffnodes = set(rnM.node_list) - set(rnm.node_list)
  # Add the pins as well
  for pin in rnM.pins:
    diffnodes.add(pin[0])
  Gd = rnM.G.subgraph(diffnodes)
  rnd = rnM
  rnd.G = Gd
  rnd.node_list = list(diffnodes)
  
  # Save a plot of the network
  plt_defaults()
  RN.plt_ln_width = 3.0 #Make the lines thicker than normal
  print("Plotting...")
  if save_v:
    fig, ax = rnd.draw(edge_color=None, color_attrib="v")
    save_fname = trimfn0 + "-" + trimfn1 + "_v.png"
  else:
    fig, ax = rnd.draw(edge_color=None)
    save_fname = trimfn0 + "-" + trimfn1 + ".png"
  fig.savefig(save_fname)
  print("Figure saved to:", save_fname)

def load_RNfile(filename):
  # Extract the filename without extension(s)
  # This is later used any time we save a file based on this one
  trm_fname = filename
  ftype = ""
  if filename[-3:] == ".gz":
    if filename[-10:-3] == ".pickle":
      trm_fname = filename[0:-10]
      ftype = "pickle"
    elif filename[-8:-3] == ".json":
      trm_fname = filename[0:-8]
      ftype = "json"
    else:
      print("Error, improper file format.")
      quit()
  else: 
    if filename[-7:] == ".pickle":
      trm_fname = filename[0:-7]
      ftype = "pickle"
    elif filename[-5:] == ".json":
      trm_fname = filename[0:-5]
      ftype = "json"
    else:
      print("Error, improper file format.")
      quit()
  # Trim off any directories in the filename path, if present
  trm_fname = trm_fname.split("/")[-1]
  print("Attempting to load the file ", filename)
  rn = RN.load_RN(filename)
  if rn == 1:
    print("Load was unsuccessful.")
    return None, None, None
  return rn, trm_fname, ftype

def plt_defaults():
  # Set the plotting settings to their defaults
  RN.plt_node_size = 4#cp.getfloat("plot", "node_size")
  RN.plt_ln_width = 0.3#cp.getfloat("plot", "ln_width")
  RN.pin_color = "green"#cp.get("plot", "pin_color")
  RN.figsize = (24,20)#util.str2arr(cp.get("plot", "figsize"))
  RN.view_angles = (0,-90)#util.str2arr(cp.get("plot", "view_angles"))

if __name__ == "__main__":
  util.debug = True # Turns on db_print()
  if plot_diff:
    inspect_RNdiff(filenames)
  else:
    for filename in filenames:
      inspect_RN(filename)

