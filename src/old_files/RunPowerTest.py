from ResistorNetwork import ResistorNetwork as RN
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time


def power_test(cp):
  """Generic form of the power test
  Parameters
  ----------
  cp : The configuration parser object
  """
  if cp.timing: t0 = time.time()
  rn = RN.from_config(cp)
  if cp.timing: t1 = time.time()
  # Plot
  fig = plt.figure(figsize=(12,10))
  ax = fig.add_subplot(projection="3d")
  rn.draw(fig=fig, ax=ax) #, width_attrib="p", edge_color="r")
  if cp.timing: t2 = time.time()
  
  if cp.timing:
    print("Building RN: ", t1-t0, "s")
    print("Drawing: ", t2-t1, "s")
  
  fig.show()
  input("Press ENTER to end")


def power_test_2d(N=500, fibers=False):
  """ Power test in a simple 2d square.
  """
  t0 = time.time()
  
  # Input & output pins
  pins = [("in0", -1.5, -1.5), ("in1", -1.5, 1.5),
    ("out0", 1.5, -1.5), ("out1", 1.5, 1.5)]
  # Dimensions of network
  limits = [-2,2, -2,2]
  # Make list of points (then add more density around nodes)
  pts = np.random.rand(N, 2)*[4,4] + [-2,-2]
  pts = 2*np.sqrt(abs(pts/2))*np.sign(pts)
  
  # Make the Resistor Network
  rn = RN(N, pin_r=.4, node_r=.4, rand_s=19, pins=pins, pts=pts, fibers=fibers, limits=limits)
  t1 = time.time()
  
  # Send current through
  p_max, _ = rn.apply_v(100, "in1", "out0")
  print("Max power: ", p_max)
  t2 = time.time()
  
  # Plot
  fig = plt.figure(figsize=(12,10))
  ax = plt.axes()
  rn.draw(fig=fig, ax=ax, width_attrib="p", edge_color="r")
  t3 = time.time()
  
  print(f"Total time: {t3-t0}", f"\n\tMaking RN: {t1-t0}", 
    f"\n\tapply_v(): {t2-t1}", f"\n\tPlotting: {t3-t2}")
  
  fig.savefig("power_test_2d.png")
  #fig.show()
  #input("Pause")

def power_test_3d(N=500, fibers=True):
  """ Power test in a simple 3d cube.
  """
  
  print("Starting 3d Power Test")
  
  t0 = time.time()
  
  # Input & output pins
  pins = [("in0", -1.5, -1.5, -1.5), ("in1", -1.5, 1.5, -1.5),
    ("in2", -1.5, 0, 1.5), ("out0", 1.5, -1.5, 1.5), ("out1", 1.5, 1.5, 1.5),
    ("out2", 1.5, 0, -1.5)]
  # Dimensions of network
  limits = [-2.5,2.5, -2.5,2.5, -2.5,2.5]
  # Make the Resistor Network
  rn = RN(N, pin_r=.5, node_r=.5, rand_s=19, pins=pins, fibers=True, fl_mean=.25, limits=limits)
  t1 = time.time()
  
  # Send current through
  p_max, _ = rn.apply_v(100, "in1", "out0")
  print("Max power: ", p_max)
  t2 = time.time()
  
  # Plot
  fig = plt.figure(figsize=(12,10))
  ax = fig.add_subplot(projection="3d")
  rn.draw(fig=fig, ax=ax, width_attrib="p", edge_color="r")
  t3 = time.time()
  
  print(f"N={N}")
  print(f"Total time: {t3-t0}", f"\n\tMaking RN: {t1-t0}", 
    f"\n\tapply_v(): {t2-t1}", f"\n\tPlotting: {t3-t2}")
  
  fig.savefig("power_test_3d.png")
  #fig.show()
  #input("Pause")

def db9_power_test(p0, p1, N = 1600, fibers=False):
  """ Power test where the network is in the rough shape of a DB9 connector.
  p0 and p1 are the pins to apply voltage across, zero-indexed from 0 to 8
  """
  
  t0 = time.time()
  
  # Build the db9 shape
  # (http://www.interfacebus.com/D-sub-pin-positions-9-pin.jpg)
  sH = 2.743 #Horizontal spacing (in this case vertical)
  sV = 2.845
  pR = sV/4 #pin R (taken to be 1/2 of sV/2)
  # Approx. size of the DB9 in mm 
  shape = [0, 4*pR+sV, 0, 4*pR+4*sH]
  # Input and Output pins
  pins = [("pin1", 2*pR, 2*pR), ("pin2", 2*pR, 2*pR+sH), ("pin3", 2*pR, 2*pR+2*sH), 
    ("pin4", 2*pR, 2*pR+3*sH), ("pin5", 2*pR, 2*pR+4*sH), ("pin6", 2*pR+sV, 2*pR+.5*sH),
    ("pin7", 2*pR+sV, 2*pR+1.5*sH), ("pin8", 2*pR+sV, 2*pR+2.5*sH), ("pin9", 2*pR+sV, 2*pR+3.5*sH)]
  
  # In and out pin labels
  p0t = f"pin{p0+1}"
  p1t = f"pin{p1+1}"
  
  # Create the Network
  rn = RN(N, pins=pins, limits=shape, pin_r=pR,
    node_r=.29, rand_s=137, fibers=fibers, fl_mean=.25)
  
  t1 = time.time()
  
  # Apply a voltage across it
  p_max, _ = rn.apply_v(100, p0t, p1t)
  print("done calculating v")
  #print(nx.get_node_attributes(rn.G, "v"))
  print(f"max p = {p_max}")
  
  t2 = time.time()
  
  # Draw the network
  fig, (ax0, ax1, ax2) = plt.subplots(1,3)
  rn.draw(ax=ax0)
  ax0.set_title("Resistor Network")
  
  # Draw the network, showing the voltage
  rn.draw(color_attrib="v", ax=ax1)
  # Workaround to add colorbar
  # https://stackoverflow.com/questions/26739248/how-to-add-a-simple-colorbar-to-a-network-graph-plot-in-python
  v_ls = list(nx.get_node_attributes(rn.G, "v").values())
  sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=min(v_ls), vmax=max(v_ls)))
  plt.colorbar(sm, ax=ax1, fraction = .075)
  ax1.plot(pins[p0][1], pins[p0][2], "gx", markersize=12)
  ax1.plot(pins[p1][1], pins[p1][2], "gx", markersize=12)
  ax1.set_title("Voltage at Each Node")
  
  # Draw the network, showing the power in each edge
  rn.draw(width_attrib="p", edge_color="r", ax=ax2)
  ax2.plot(pins[p0][1], pins[p0][2], "gx", markersize=12)
  ax2.plot(pins[p1][1], pins[p1][2], "gx", markersize=12)
  ax2.set_title("Power Across Each Edge")
  
  t3 = time.time()
  
  print(f"Total time: {t3-t0}", f"\n\tMaking RN: {t1-t0}", 
    f"\n\tapply_v(): {t2-t1}", f"\n\tPlotting: {t3-t2}")
  
  fig.show()
  input("Press Enter to continue")

def old_power_test():
  """ Old version of the power test.
  Was made before the RN class method apply_v()
  """

  NUM_NODES = 1000

  # Input and Output pins
  pins = [("in", 0, 0.2), ("out", 1, 0.8)]
  #pts = np.array([[.2, .15], [.15, .25], [.3, .25], [.35, .4], [.5, .4], [.55, .6], [.6, .45], [.7, .5], [.75, .75]])
  #pts = np.array([[.4, .4], [.6, .6]])

  rn = RN(NUM_NODES, pins=pins, pin_r=.3, node_r=.1)#, pts=pts)

  fig, ax = rn.draw()
  fig.show()
  input("Press Enter to continue")

  # Apply a voltage difference across the pins and calculate the voltage and current across each edge to find which has the most power dissipation
  # Total number of nodes, including pins
  N = len(rn.G)
  # Equivalent resistance across whole network
  Req = rn.R_pp("in", "out")
  # Applied voltage
  V_in = 10
  # Current flowing in
  I_in = V_in / Req
  print(f"In the network of {N} nodes, the Req={Req}.")
  print(f"With {V_in}V applied, the current is {I_in}A.")

  # Solve for the voltage at each node using KCL (Lv = c)
  c = np.zeros((N, 1))
  c[rn.pin_indices["in"]] = I_in
  c[rn.pin_indices["out"]] = -I_in
  Lpl = nx.linalg.laplacian_matrix(rn.G, weight="cnd").todense()
  # Grounded Lpl matrix
  Lplg = np.array(Lpl)
  Lplg[N-1, :] = np.zeros((1,N))
  Lplg[N-1, rn.pin_indices["out"]] = 1 # Ground the out pin (stealing the last row)
  cg = np.array(c)
  cg[N-1] = 0 # Ground the out pin

  #v = np.linalg.solve(Lpl, c) - Sorry, I can't use this since it's singular. If I could just add the constraint that one node be ground, we'd be golden.
  vg = np.linalg.solve(Lplg, cg)
  v = np.dot(np.linalg.pinv(Lpl), c)
  #print(f"The voltage at each node is: {v} or maybe {vg}")

  # Find the edge with the most power
  P_max = 0
  edge_Pmax = 0
  imax0 = 0
  imax1 = 0
  for edge in rn.G.edges(data=True):
    i0 = edge[0]
    i1 = edge[1]
    
    # Patch for the pins. Not pretty.
    if isinstance(i0, str):
      i0 = rn.pin_indices[i0]
      #continue #Temp, skip the edges connected to the nodes
    else:
      i0 = i0+len(pins)
    if isinstance(i1, str):
      i1 = rn.pin_indices[i1]
      #continue #Temp, skip the edges connected to the nodes
    else:
      i1 = i1+len(pins)
    # Voltage difference
    dv = float(abs(v[i1] - v[i0]))
    # Equivalent resistance of edge. Wait, this should use just the edge resistance, not equivalent resistance, huh.
    #R = rn.R_ij(i0, i1)
    R = edge[2]["res"]
    P = dv**2 / R
    # Save this power value
    rn.G[edge[0]][edge[1]]["P"] = P**2 #I squared it so the differences are more obvious on the plot
    
    if P > P_max:
      P_max = P
      edge_Pmax = edge
      imax0 = i0
      imax1 = i1
      #print(f"Edge {edge} has R={R:.6f}, V={dv:.6f}, and P={P:.6f}")
      #print(f"Pin locations: {rn.G.nodes[edge[0]]['pos']}, {rn.G.nodes[edge[1]]['pos']}")

  print(f"The highest power was {P_max:.6f}, across edge {edge_Pmax}")
  print(f"Pin locations: {rn.G.nodes[edge_Pmax[0]]['pos']}, {rn.G.nodes[edge_Pmax[1]]['pos']}")
  print(f"dV = {float(abs(v[imax1] - v[imax0]))}")

  fig, ax = rn.draw(width_attrib="P", edge_color="r")

  # Find the shortest path and compare the pins chosen by that method 
  #   to the highest power node.
  nL, nS = rn.longestNode("in", "out", ax=ax)
  PnL = rn.G.edges(nL, data="P")
  PnL = np.sqrt(max([E[2] for E in PnL]))
  PnS = rn.G.edges(nS, data="P")
  PnS = np.sqrt(max([E[2] for E in PnS]))
  print(f"Max power flowing through Longest={PnL} (Yellow X)")
  print(f"Max power flowing through Shortest={PnS} (Green X)")
  nLpos = rn.G.nodes[nL]["pos"]
  nSpos = rn.G.nodes[nS]["pos"]
  ax.plot(nLpos[0], nLpos[1], "yx", markersize=15)#,  c="red", s=100) #marker="x",
  ax.plot(nSpos[0], nSpos[1], "gx", markersize=15)
  # Add an X for the node with the most power
  pMpos = rn.G.nodes[edge_Pmax[0]]['pos']
  ax.plot(pMpos[0], pMpos[1], "rx", markersize=15)

  fig.show()
  input("Press Enter to continue")

# Run the power test
#db9_power_test(1, 7, 2000, fibers=True)

#power_test_3d(1600)
#power_test_2d(2000)


