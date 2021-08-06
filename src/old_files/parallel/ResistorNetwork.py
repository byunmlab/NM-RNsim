import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.spatial as spl
#import scipy.stats as sp_stats -- for truncnorm
import util

class ResistorNetwork:
  """A class to represent a resistor network
  Tracks indices of pins, so they can be accessed by name instead of 
  by index in the matrix.
  Provides useful functions for dealing with resistor networks.
  """
  rand_s = 2 # Random seed for repeatability. If None, no seed is set.
  pin_r = 0.25 # Radius for the pins
  # Plot options
  plt_node_size = 7#50 #default is 300, which is pretty big.
  plt_ln_width = 0.3 #Line Width
  pin_color = "green" #"gold"
  figsize = (12,10)
  
  # Network Resistance options
  pin_res = 1e-6 #Resistance btw pins and the nodes they intersect with. It's an arbitrarily low resistance.
  res_k = 10000 #Coefficient multiplier for resistance calculation btw nodes
  min_res = 1e-6 #Min resistance. Think of the resistance of the fiber itself, even if they're touching
  
  def __init__(self, N=500, limits=[0,1, 0,1], pins=[], pts=None,
      fibers=False, fl_mean=.1, fl_span=.005, cnd_len=0.15):
    """Constructor for the network.
    Parameters
    ----------
    N (int) : Number of nodes in network (default 500).
    limits (float array, len = 4 or 6) : x and y limits of the network 
      (default [0,1, 0,1]).
      format: [xmin, xmax, ymin, ymax(, zmin, zmax)]
      If 6 limits are provided, this triggers a 3D network.
    pins : Provide a list of pin locations. If none are provided, there will
      be no pins placed in the RN.
      format: [("pin0", x0, y0), ("pin1", x1, y1), ...]
    pts : Optionally provide a custom list of fiber locations. 
      If not given, they will be randomly generated.
    fibers (bool) : Enable fibers (default False -- point fibers)
    fl_mean (float) : Mean fiber length.
      With our Ni fibers, it looks like this should be around 10 microns.
    fl_span (float) : Span of distribution of fiber lengths.
    cnd_len : The max radius of connections between nodes.
    """
    self.N = N
    self.xmin = limits[0]
    self.xmax = limits[1]
    self.ymin = limits[2]
    self.ymax = limits[3]
    
    self.pins = pins
    self.pin_keys = [pin[0] for pin in pins]
    self.pts = pts
    self.fibers = fibers
    self.fl_mean = fl_mean
    self.fl_span = fl_span
    self.cnd_len = cnd_len
    
    # 2-dimensional unless 6 limits are provided
    if len(limits) == 6:
      self.dim = 3
      self.zmin = limits[4]
      self.zmax = limits[5]
    else:
      self.dim = 2
      self.zmin = 0
      self.zmax = 0
    # Generate the Graph
    if self.rand_s != None:
      np.random.seed(self.rand_s)
    self.G = self.create_graph()
    
    self.voltageMult = 0
    self.totalPower = 0
  
  @classmethod
  def from_config(cls, cp):
    """Create and return an RN from a configuration object
    """
    cls.rand_s = cp.getint("exec", "rand")
    if cls.rand_s < 0:
      cls.rand_s = None
    # Load the resistance calculation options
    cls.pin_res = cp.getfloat("RN-res", "pin_res")
    cls.res_k = cp.getfloat("RN-res", "res_k")
    cls.min_res = cp.getfloat("RN-res", "min_res")
    # Load the plotting options
    cls.plt_node_size = cp.getfloat("plot", "node_size")
    cls.plt_ln_width = cp.getfloat("plot", "ln_width")
    cls.pin_color = cp.get("plot", "pin_color")
    cls.figsize = util.str2arr(cp.get("plot", "figsize"))
    # Load the fiber options
    N = cp.getint("RN-fiber", "N")
    fibers = cp.getboolean("RN-fiber", "fibers")
    fl_mean = cp.getfloat("RN-fiber", "fiber_len")
    fl_span = cp.getfloat("RN-fiber", "fl_span")
    cnd_len = cp.getfloat("RN-fiber", "cnd_len")
    # Load the dimensions options
    cls.pin_r = cp.getfloat("RN-dim", "pin_r")
    xlims = util.str2arr(cp.get("RN-dim", "xlims"))
    ylims = util.str2arr(cp.get("RN-dim", "ylims"))
    zlims = util.str2arr(cp.get("RN-dim", "zlims"))
    if np.all(zlims[0] == [0,0]):
      # In this case, it's a 2d network
      limits = np.hstack((xlims, ylims))
    else:
      limits = np.hstack((xlims, ylims, zlims))
    # Load the list of pins
    pins_raw = cp.items("RN-pins")
    pins = []
    # TO DO: switch the pin format so the length doesn't depend on dim
    for pin_raw in pins_raw:
      pin_pos = util.str2arr(pin_raw[1])
      pin = (pin_raw[0], pin_pos)
      pins.append(pin)
    
    # Print a description of the RN being made
    RNs = f"Creating an RN with {N} nodes"
    if fibers: RNs += f", using fibers of avg L = {fl_mean}"
    else: RNs += ", using 0-D fibers (points)"
    RNs += f", with max connection L = {cnd_len}"
    util.db_print(cp, RNs)
    
    # To Do (Maybe): make pts an option in the config. It's tricky to do.
    return cls(N=N, limits=limits, pins=pins, fibers=fibers,
      fl_mean=fl_mean, fl_span=fl_span,
      cnd_len=cnd_len)
  
  def create_graph(self):
    if self.pts is None:
      domain_sizes = [self.xmax-self.xmin, self.ymax-self.ymin, self.zmax-self.zmin]
      domain_offsets = [self.xmin, self.ymin, self.zmin]
      # Generate a matrix of random points in the domain
      self.pts = np.random.rand(self.N, self.dim)
      self.pts *= domain_sizes[0:self.dim]
      self.pts += domain_offsets[0:self.dim]
    if self.fibers:
      # Fiber length
      fl = np.random.rand(self.N, 1) * self.fl_span
      fl += self.fl_mean - self.fl_span/2
      # Angle of each fiber
      th = np.random.rand(self.N, 1) * 2*np.pi
      if self.dim == 2:
        # convert L, th to [u, v]
        self.fv = np.hstack([fl*np.cos(th), fl*np.sin(th)])
      elif self.dim == 3:
        # second angle to determine direction in 3D (pitch from 0 (up) to pi (down))
        phi = np.random.rand(self.N, 1) * np.pi - np.pi/2
        # convert L, th, phi to [u, v, w]
        self.fv = np.hstack([fl*np.sin(phi)*np.cos(th), fl*np.sin(phi)*np.sin(th), 
          fl*np.cos(phi)])
    
    node_keys_it = range(len(self.pts))
    G = nx.Graph()
    
    # Add the nodes
    for i in node_keys_it:
      # The key for each node will simply be its index in the pts array
      if self.fibers:
        G.add_node(i, pos=self.pts[i], fv=self.fv[i])
      else:
        G.add_node(i, pos=self.pts[i])
    # Keep track of all nodes. The first N nodes will be the random nodes.
    # Whenever a node is deleted, remove it from this list.
    # Pass this list to the laplacian_matrix function so the order of the 
    #   entries in the matrix is predictable.
    self.node_list = list(node_keys_it)
    
    # Add the pins
    for pin in self.pins:
      # Add the pin with the provided key and location
      G.add_node(pin[0], pos=pin[1])
      # Store this pin in the list of all nodes.
      self.node_list.append(pin[0])
    
    # Build a KDTree to make searching much faster
    tree = spl.cKDTree(self.pts)
    dmax = self.cnd_len # Max node distance to consider connection
    if self.fibers: dmax += 2*self.fl_mean + self.fl_span
    # Find and add the edges between the nodes
    for i in node_keys_it:
      neighbors = tree.query_ball_point(self.pts[i], dmax)
      for j in neighbors:
        # if j > i, then add the edge. Else, it's already been added
        if j <= i:
          continue
        if self.fibers:
          dist, dminpt = util.min_LS_dist( self.pts[i], self.pts[j], 
            self.fv[i], self.fv[j] )
          # For fibers, the search bubble was bigger than cnd_len
          if dist < self.cnd_len:
            res = self.res_fun(dist)
            #ep0, ep1 : edge p0, p1. Save the x-y coordinates of the points along 
            # the fibers where dist is smallest.
            ep0 = self.pts[i] + self.fv[i] * dminpt[0]
            ep1 = self.pts[j] + self.fv[j] * dminpt[1]
            G.add_edge( i, j, res=res, cnd=1/res, pos=np.vstack([ep0, ep1]) )
        else:
          dist = np.linalg.norm(self.pts[i] - self.pts[j])
          res = self.res_fun(dist)
          G.add_edge( i, j, res=res, cnd=1/res)
    
    # Add the edges for the pins
    res = self.pin_res
    dmax = self.pin_r
    if self.fibers: dmax += self.fl_mean + self.fl_span/2
    for pin in self.pins:
      neighbors = tree.query_ball_point(pin[1], dmax)
      for pt_i in neighbors:
        if self.fibers:
          dist, t = util.LS_pt_dist(pin[1], self.pts[pt_i], 
            self.fv[pt_i], dmaxt=self.cnd_len)
          if dist < self.pin_r:
            #ep0, ep1 : edge p0, p1. Save the x-y coordinates of the edge
            # start and end points.
            ep0 = pin[1]
            ep1 = self.pts[pt_i] + self.fv[pt_i] * t
            G.add_edge(pin[0], pt_i, res=res, cnd=1/res, pos=np.vstack([ep0, ep1]) )
        else:
          G.add_edge(pin[0], pt_i, res=res, cnd=1/res)
    
    return G
  
  def res_fun(self, dist):
    """Resistance between nodes or fibers as a function of distance
    To Do: incorporate tunneling equation
    """
    return self.min_res + self.res_k*dist*dist
  
  def key_to_index(self, key):
    """ Given a node key, convert to the index in the matrix
    """
    return self.node_list.index(key)
  
  def index_to_key(self, i):
    """ Reverse of key_to_index
    """
    return self.node_list[i]
  
  def remove_node(self, key):
    """ Remove a node from the network by key
    This is better than doing it directly, because it preserves the node_list
      so that we can know what indices correspond to which nodes.
    In other words, if you remove a node without removing it from node_list, 
      the functions "key_to_index" and "index_to_key" break.
    """
    # Remove the node from the Graph
    self.G.remove_node(key)
    # Remove the key from the master list of all keys
    self.node_list.remove(key)
  
  def add_node(self, key, **attr):
    """ Add a node to the network with the given key and attributes
    Using this method keeps the master list of keys updated
    """
    # Add the node to the Graph
    self.G.add_node(key, **attr)
    # Add the key to the master list of all keys
    self.node_list.append(key)
  
  # Resistance from node n0 to node n1
  # n0 and n1 can be either pins or fibers
  def R_nn(self, n0, n1):
    return self.R_ij(self.key_to_index(n0), self.key_to_index(n1))
  
  # Resistance from pin 0 to pin 1
  # This uses the names of the pins, not their indices
  def R_pp(self, p0, p1):
    return self.R_nn(p0, p1)
  
  # Resistance from node index i0 to i1
  def R_ij(self, i0, i1, lsqr=False):
    # See self.apply_v()
    # Set up the b column in Lpl*v=b
    N = len(self.G)
    b = np.zeros((N, 1))
    # Use 1A so V = Req
    b[i0] = 1
    b[i1] = -1
    
    # Get the Laplacian matrix for conductance
    Lpl = nx.linalg.laplacian_matrix(self.G, weight="cnd", nodelist=self.node_list)
    
    # Make a matrix that is the Laplacian, without the row and column
    #   corrosponding to the grounded node.
    ig = i1 # ground i1
    keep_indices = [i for i in range(N) if i!=ig]
    #keep_indices = [*range(ig), *range(ig+1, N-1)]
    # Skip the ig row and column
    Lplp = Lpl[keep_indices,:].tocsc()[:,keep_indices]
    b = np.delete(b, ig)
    
    # Solve Lpl*v=b
    if lsqr:
      # Doesn't seem to be working right
      v, *_ = spsl.lsqr(Lplp, b)
      #v, *_ = np.linalg.lstsq(Lplp.toarray(), b)
      v -= np.min(v)
    else:
      #v, *_ = spsl.cg(Lplp, b)
      v = spsl.spsolve(Lplp, b) # Should probably use lsqslv, in case there are lone nodes
    
    if i0 > ig:
      i0 -= 1 # Everything after ig got shifted by one
    
    return v[i0]
  
  # Old way, without removing a row and column
  def R_ij_full(self, i0, i1, lsqr=False):
    # See self.apply_v()
    # Set up the b column in Lpl*v=b
    N = len(self.G)
    b = np.zeros((N, 1))
    # Use 1A so V = Req
    b[i0] = 1
    b[i1] = -1
    
    # Get the Laplacian matrix for conductance
    Lpl = nx.linalg.laplacian_matrix(self.G, weight="cnd", nodelist=self.node_list)
    # Ground the given pin, overwriting the last row for the grounding equation
    # That's fine, since all the information is repeated.
    Lpl[N-1] = sps.csr_matrix((1,N)) # Gives a warning
    Lpl[N-1, i1] = 1
    b[N-1] = 0
    
    # Solve Lpl*v=b
    if lsqr:
      v, *_ = spsl.lsqr(Lplp, b)
    else:
      v = spsl.spsolve(Lpl, b) # Should probably use lsqslv, in case there are lone nodes
    
    return v[i0]
  
  # Resistance from node index i0 to i1
  # Calculation of resistance using pseudo-inverse (INEFFICIENT & LESS ACCURATE)
  # https://mathworld.wolfram.com/ResistanceDistance.html
  def R_ij_inv(self, i0, i1):
    #print(f"Resistance from N{i0} to N{i1} of {len(self.G)} total nodes")
    Lpl = nx.linalg.laplacian_matrix(self.G, weight="cnd").todense()
    Gamma = Lpl + 1/len(self.G)
    try:
      Gamma_inv = np.linalg.pinv(Gamma)
      #R_ij = G_ii^-1 + G_jj^-1 - 2G_ij^-1
      R01 = Gamma_inv[i0, i0] + Gamma_inv[i1, i1] - 2*Gamma_inv[i0, i1]
      return R01
    except Exception as e:
      print(f"ERROR {e}")
      return -1

  # Calculation of resistance with determinants
  # This comes from applying Cramer's rule to the system
  def R_ij_det(self, i0, i1):
    Lpl = nx.linalg.laplacian_matrix(self.G, weight="cnd").toarray()
    # Laplacian with row and column i0 removed
    L_r_i0 = np.delete(np.delete(Lpl, i0, 0), i0, 1)
    # Laplacian with rows and columns i0 and i1 removed
    L_r_i0i1 = np.delete(np.delete(Lpl, [i0,i1], 0), [i0,i1], 1)
    # return np.linalg.det(L_r_i0i1) / np.linalg.det(L_r_i0)
    # The calculation of the determinant overflows often, so use logdet
    (_, lgD1) = np.linalg.slogdet(L_r_i0i1)
    (_, lgD2) = np.linalg.slogdet(L_r_i0)
    # Log rule: D1 / D2 = exp(log(D1) - log(D2)))
    return np.exp( lgD1 - lgD2 )



  def connectPins(self, connectedPins, label):
    self.add_node(label)
    edges = []
    for i in connectedPins:
      edges.append([label, i, {'res' : 1e-10, 'cnd' : 1e10}])
    self.G.add_edges_from(edges)
    return label

  def findEquivalentResistance(self, startingNode, endingNode):
    return self.R_ij(startingNode, endingNode)

  def setVoltages(self, startingNodes, endingNodes, voltage):
    startName = self.connectPins(startingNodes, "High")
    endName = self.connectPins(endingNodes, "Ground")
    startIndex = self.G.number_of_nodes() - 2
    endIndex = self.G.number_of_nodes() - 1
    
    # Apply the given voltage to the network
    self.apply_v(voltage, startName, endName)

    #removeConnectors
    self.remove_node(startName)
    self.remove_node(endName)

  #returns the currents of the output nodes
  def getOutputCurrents(self, endingNodes):
    currents = [0] * len(endingNodes)
    it = 0
    for i in endingNodes:
      voltage = self.G.nodes[i]["v"]
      for u, v in self.G.edges(i):
        currents[it] += (self.G.nodes[v]["v"] - voltage) * self.G[u][v]["cnd"]
      it += 1

    return currents

  #pull the pins high or low
  def pullOutputs(self, currents, threshold):
    output = []
    for i in range(len(currents)):
      if currents[i] > threshold[i]:
        output.append(1)
      else:
        output.append(0)
    return output

  #returns outputs as to how they relate
  #   if they match                           = 0  #do nothing
  #   if they're high when they should be low = 1  #delete nodes
  #   if they're low when they should be high = -1 #no current fix
  #possible solutions to the third case is to change the threshold on that pin, or to add a not gate on it
  def compareOutputs(self, binary, desired):
    output = []
    for i in range(len(binary)):
      if binary[i] == desired[i]:
        output.append(0)
      elif binary[i] == 1 and desired[i] == 0:
        output.append(1)
      else:
        output.append(-1)
    return output

  def getPower(self, edge):
    voltage = abs(self.G.nodes[edge[0]]["v"] - self.G.nodes[edge[1]]["v"])
    power = voltage**2 / self.G[edge[0]][edge[1]]["res"]
    self.totalPower += power
    return power

  #send high current back through and delete nodes over a power threshold
  def backPropagate(self, inputs, badNodes, voltage, threshold):
    numDeleted = 0 #keep a count of deleted nodes

    #connect pins together
    startName = self.connectPins(inputs, "High")
    endName = self.connectPins(badNodes, "Ground")
    startIndex = self.G.number_of_nodes() - 2
    endIndex = self.G.number_of_nodes() - 1

    #Lv = c
    N = len(self.G)

    #create c
    c = np.zeros((N, 1))
    c[startIndex] = 1
    c[endIndex] = -1

    #create L
    L = nx.linalg.laplacian_matrix(self.G, weight="cnd")

    #ground pin
    L[N-1] = sps.csr_matrix((1,N))
    L[N-1, endIndex] = 1
    c[N-1] = 0

    #find v and Req
    v=spsl.spsolve(L, c)
    Req = v[startIndex]

    #set Voltages
    self.voltageMult = voltage / Req
    it = 0
    for i in self.G.nodes:
      self.G.nodes[i]["v"] = v[it] * self.voltageMult
      it += 1

    #set Power
    #delete in the same go?
    num = 0
    for i in self.G.nodes:
      self.G.nodes[i]["power"] = 0
      for edge in self.G.edges(i):
        self.G.nodes[i]["power"] += self.getPower(edge)

    print("Total Power is ", self.totalPower / 2)

    # self.G[edge[0]][edge[1]]["power"] = self.getPower(edge)
    #    print(self.G["in0"][edge[1]]["power"])
    #    print(self.G.nodes["in0"]["v"], self.G.nodes[edge[1]]["v"])

    #for edge in self.G.edges.data("power"):
    #  if edge[2] > threshold:
    #    self.deleteNodes(edge)
    #    num += 1

    #remove nodes over power threshold
    for i in list(self.G.nodes):
      if self.G.nodes[i]["power"] > threshold: #if over threshold
        if (i != startName and i != endName): #if not connecting node
          isPin = False
          for j in inputs: #if not input pin
            if i == j:
              isPin = True
          for j in badNodes: #if not output pin
            if i == j:
              isPin = True
          if (isPin == False): #if its an intermediate node
            self.remove_node(i) #remove node
            num += 1

    print("Total number of nodes deleted is ", num)

    #removeConnectors
    self.remove_node(startName)
    self.remove_node(endName)


  def apply_v(self, v_in, n0, n1, ground=None, set_p=True):
    """ Apply a voltage across the two given pins/nodes
      Save the resulting voltage and power as attributes in each node or edge
      
      Parameters
      ----------
      n0 : the key for the input node (V+)
      n1 : the key for the output node (V-)
      ground : the key for a node to set as ground (defaults to n1)
      set_p : whether to calculate and save power in the edges (default True)
      
      Returns
      -------
      (if set_p) p_max : the max power flowing through any edge
      (if set_p) e_p_max : which edge has that power
    """
    # Default to the out pin being ground
    if ground is None:
      ground = n1
    # Find the indices of the pins used
    n0i = self.key_to_index(n0)
    n1i = self.key_to_index(n1)
    gdi = self.key_to_index(ground)
    
    # Set up the b column in Lpl*v=b
    N = len(self.G)
    b = np.zeros((N, 1))
    # In reality, this should be the current flowing in. However, since
    # we need the Req to know that, use 1A for now, then adjust later.
    b[n0i] = 1
    b[n1i] = -1
    
    # Get the Laplacian matrix for conductance
    Lpl = nx.linalg.laplacian_matrix(self.G, weight="cnd")
    # Ground the given pin, overwriting the last row for the grounding equation
    # That's fine, since all the information is repeated.
    Lpl[N-1] = sps.csr_matrix((1,N))
    Lpl[N-1, gdi] = 1
    b[N-1] = 0
    # Solve Lpl*v=b
    v = spsl.spsolve(Lpl, b) # Should probably use lsqslv, in case there are lone nodes
    #v, *_ = spsl.lsqr(Lpl, b)
    
    # Adjust the voltages
    Req = v[n0i] - v[n1i] #This is true since I was 1A.
    voltage_mult = v_in / Req
    v *= voltage_mult
    # Save the voltages in the nodes
    node_i = 0
    for node_key in self.G.nodes:
      self.G.nodes[node_key]["v"] = v[node_i]
      node_i += 1
    
    if set_p:
      p_max = 0
      e_p_max = None
      for edge in self.G.edges(data=True):
        en0 = edge[0]
        en1 = edge[1]
        ei0 = self.key_to_index(en0)
        ei1 = self.key_to_index(en1)
        # Calculate edge power
        dv = float(abs(v[ei1] - v[ei0]))
        R = edge[2]["res"]
        p = dv**2 / R
        # Save this power value
        self.G[en0][en1]["p"] = p
        # Save the square of power to be the line thickness when plotting
        self.G[en0][en1]["p2"] = p**2
        if p > p_max:
          p_max = p
          e_p_max = edge
      return p_max, e_p_max
  
  def draw(self, fig=None, ax=None, width_attrib=None, color_attrib=None,
    edge_color="b"):
    """Make a plot of the RN.
    Parameters
    ----------
    fig, ax : pyplot fig and ax. If not provided, they are pre-generated.
    width_attrib : An edge attribute that is used to determine edge line 
      widths. e.g. power through edge
    color_attrib : A node attribute that is used to determine the color of
      each fiber.
    edge_color : An edge color.
    """
    
    # Set up the figure and axes
    if self.dim == 2:
      if ax is None:
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot()
      ax.set_aspect("equal")
    else:
      assert self.dim == 3
      if ax is None:
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(projection="3d")
      ax.set_xlim3d(self.xmin, self.xmax)
      ax.set_ylim3d(self.ymin, self.ymax)
      ax.set_zlim3d(self.zmin, self.zmax)
      fig.subplots_adjust(0,0,1,1) # Remove whitespace
      ax.view_init(12, -110)
    
    # For width_attrib mode, don't make stupidly thick lines
    lw_max = 25#*self.plt_ln_width # max line width
    
    if not edge_color is None:
      # Draw the edges
      for edge in self.G.edges(data=True):
        # Start with the position of the nodes
        p0 = self.G.nodes[edge[0]]["pos"]
        p1 = self.G.nodes[edge[1]]["pos"]
        edge_pos = np.vstack([p0, p1])
        if self.fibers:
          try:
            # If edge start and end position is specified, use that.
            # The edges connected to the pins don't have the "pos" attribute, however
            edge_pos = edge[2]["pos"]
          except: pass #Is this needed?
        if width_attrib is None:
          lw = self.plt_ln_width
        else:
          lw = min(edge[2][width_attrib], lw_max)
        if self.dim == 2:
          ax.plot(edge_pos[:,0], edge_pos[:,1], color=edge_color, linewidth=lw,
            solid_capstyle="butt", zorder=1)
        else:
          ax.plot3D(edge_pos[:,0], edge_pos[:,1], edge_pos[:,2],
            color=edge_color, linewidth=lw, solid_capstyle="butt", zorder=1)
    
    # Draw the nodes
    # Array of positions
    pos = nx.get_node_attributes(self.G, 'pos')
    if self.fibers:
      [pos.pop(pin[0]) for pin in self.pins] #This cuts out the pins
      pos = np.array(list(pos.values()))
      # Array of fiber v
      fv = nx.get_node_attributes(self.G, 'fv')
      fv = np.array(list(fv.values()))
      # Make the color list
      if color_attrib is None:
        fiber_color = np.ones(len(pos))
      else:
        if self.dim == 3:
          # Quiver 3D doesn't support multiple colors, so I'd need a work-around
          print("Specifying fiber color is not yet implemented in 3D")
        else:
          fiber_color = nx.get_node_attributes(self.G, color_attrib)
          #Cut out the pins from the color list
          [fiber_color.pop(pin[0]) for pin in self.pins]
          fiber_color = np.array(list(fiber_color.values()))
      
      if self.dim == 2:
        # Use quiver to show the fibers
        ax.quiver(pos[:,0], pos[:,1], fv[:,0], fv[:,1], fiber_color, angles="xy",
          scale_units="xy", scale=1, headlength=0, headwidth=0, headaxislength=0, 
          capstyle="butt", width=.0075, zorder=2)
      else:
        ax.quiver(pos[:,0], pos[:,1], pos[:,2], fv[:,0], fv[:,1], fv[:,2],
          arrow_length_ratio=0, capstyle="butt", linewidths=self.plt_ln_width, zorder=2)
    else:
      pos = np.array(list(pos.values()))
      # Make the color list
      if color_attrib is None:
        node_color = np.ones(len(pos))
      else:
        node_color = nx.get_node_attributes(self.G, color_attrib)
        node_color = np.array(list(node_color.values()))
      
      # Use scatter to show the nodes
      if self.dim == 2:
        ax.scatter(pos[:,0], pos[:,1], c=node_color, zorder=2)
      else:
        ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=node_color, zorder=2)
    
    # If color_attrib was used, add a colorbar
    if not color_attrib is None:
      v_ls = list(nx.get_node_attributes(self.G, "v").values())
      sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=min(v_ls), vmax=max(v_ls)))
      plt.colorbar(sm, ax=ax, fraction=.075, shrink=.5)
    
    for pin in self.pins:
      if self.dim == 2:
        # Plot the pin centers
        ax.scatter(pin[1][0], pin[1][1], color=self.pin_color)
        # Draw circles for the pins
        circle = plt.Circle(pin[1], radius=self.pin_r, color=self.pin_color, 
          fill=False, zorder=3)
        ax.add_artist(circle)
      else:
        ax.scatter(pin[1][0], pin[1][1], pin[1][2], color=self.pin_color)
        self.plot3d_sphere(ax, pin[1][0], pin[1][1], pin[1][2], self.pin_r,
          color=self.pin_color)
    return fig, ax
  
  def plot3d_sphere(self, ax3d, x0, y0, z0, R, color="gold", mesh=True):
    """ Plots a 3d sphere
      x0, y0, z0 (float) : coordinates of center of sphere
      R (float) : radius of sphere
      color : color of sphere
      mesh (bool) : if True, make a wireframe sphere. Else, make a 
        translucent sphere
    """
    # See https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    u, v = np.mgrid[0:2*np.pi:16j, 0:np.pi:8j]
    x = R*np.cos(u)*np.sin(v) + x0
    y = R*np.sin(u)*np.sin(v) + y0
    z = R*np.cos(v) + z0
    if mesh:
      ax3d.plot_wireframe(x, y, z, color=color, linewidths=self.plt_ln_width)
    else:
      # alpha controls opacity
      ax3d.plot_surface(x, y, z, color=color, alpha=0.3)
  
  # Find the longest node on the shortest path between n0 and n1
  def longestNode(self, n0, n1, ax=None):
    path = nx.dijkstra_path(self.G, n0, n1, 'res')
    R_max = 0
    R_min = 1000
    index = path[1]
    # Plot the shortest edge
    if not ax is None:
      xs = np.zeros(len(path))
      ys = np.zeros(len(path))
      pos = nx.get_node_attributes(self.G, "pos")
      for i in range(len(path)):
        xs[i] = pos[path[i]][0]
        ys[i] = pos[path[i]][1]
      ax.plot(xs, ys, "g")
    #The range doesn't include the starting and ending nodes in order to make sure they are never deleted
    #also, doesn't include the first node that either pin touches
    for i in range(2,len(path) - 2):
      R = self.G.get_edge_data(path[i],path[i+1])['res']
      #eliminates the node with the largest resistance
      if R > R_max:
        R_max = R
        index = path[i]
      if R < R_min:
        R_min = R
        shortest = path[i]
      #print(index)
    # It was returning both, but the eliminateNode function expects one
    return index #, shortest
  
  # Remove a node between n0 and n1 that lies on the shortest path
  def eliminateNode(self, n0, n1):
    self.remove_node(self.longestNode(n0, n1))

  def size(self):
    return self.G.number_of_nodes()

  def get(self):
    return list(self.G.nodes)
