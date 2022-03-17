import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.spatial as spl
#import scipy.stats as sp_stats -- for truncnorm
import util
import pickle
import json
import gzip

class ResistorNetwork:
  """A class to represent a resistor network
  Tracks indices of pins, so they can be accessed by name instead of 
  by index in the matrix.
  Provides useful functions for dealing with resistor networks.
  """
  rand_s = 2 # Random seed for repeatability. If None, no seed is set.
  
  # Plotting options
  plt_node_size = 7#50 #default is 300, which is pretty big.
  plt_ln_width = 0.3 #Line Width
  pin_color = "green" #"gold"
  figsize = (12,10)
  view_angles = (12, -110)
  
  # Network Resistance options
  pin_res = 1e-6 #Resistance btw pins and the nodes they intersect with. It's an arbitrarily low resistance.
  res_k = 10000 #Coefficient multiplier for resistance calculation btw nodes
  min_res = 1e-6 #Min resistance. Think of the resistance of the fiber itself, even if they're touching
  # Solution method: "spsolve", "splsqr", "cg". 
  #   splsqr is not a good idea because the matrix is symmetric, 
  #   and spsolve doesn't work if the graph has any isolated points
  sol_method = "cg"

  # Training options
  # Fwd pass voltage for training
  fpV = 100 #3.3 - high for now, so we can see the lines
  # Current threshold for determining what is high
  threshold = 0.5
  
  def __init__(self, N=500, limits=[0,1, 0,1], in_pins=[], out_pins=[],
      pts=None, fibers=True, fl_mean=0.1, fl_span=0.005, 
      brn_pwr=1, bpwr_span=0, 
      cnd_len=0.15, pin_r=0.25):
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
      format: [("pin0", [x0, y0]), ("pin1", [x1, y1]), ...]
      in_pins is for input pins and out_pins is for output pins.
    pts : Optionally provide a custom list of fiber locations. 
      If not given, they will be randomly generated.
    fibers (bool) : Enable fibers (default False -- point fibers)
    fl_mean (float) : Mean fiber length.
      With our Ni fibers, it looks like this should be around 10 microns.
    fl_span (float) : Span of distribution of fiber lengths.
    cnd_len (float) : The max radius of connections between nodes.
    pin_r (float) : The radius of the pins (Pins are currently implemented 
      only as circles or spheres)
    """
    self.N = N
    self.xmin = limits[0]
    self.xmax = limits[1]
    self.ymin = limits[2]
    self.ymax = limits[3]
    
    self.in_pins = in_pins
    self.out_pins = out_pins
    self.pin_keys = [pin[0] for pin in self.pins]
    self.pts = pts
    self.fibers = fibers
    self.fl_mean = fl_mean
    self.fl_span = fl_span
    self.brn_pwr = brn_pwr
    self.bpwr_span = bpwr_span
    self.cnd_len = cnd_len
    self.pin_r = pin_r
    
    #TEMP
    maxNR = self.res_fun(cnd_len)
    util.db_print(f"87: maxNR = {maxNR}")
    util.db_print(f"88: minNR = {self.min_res}")
    util.db_print(f"89: PR = {self.pin_res}")
    util.db_print(f"90: max/min R ratio = {maxNR / min(self.min_res, self.pin_res)}")
    #\TEMP

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
    self.totalPower = 0 # Used by getPower() and backPropagate()
  
  @classmethod
  def cls_config(cls, cp):
    """Load the RN class attributes from the given configuration object.
    This is used for when you load a pickle, but you still want to load
    some options from the config, which would be the options that aren't
    saved with the pickle (the class attributes).
    """
    cls.rand_s = cp.getint("exec", "rand")
    if cls.rand_s < 0:
      cls.rand_s = None
    # Load the resistance calculation options
    cls.pin_res = cp.getfloat("RN-res", "pin_res")
    cls.res_k = cp.getfloat("RN-res", "res_k")
    cls.min_res = cp.getfloat("RN-res", "min_res")
    cls.sol_method = cp.get("RN-res", "sol_method")
    # Load the plotting options
    cls.plt_node_size = cp.getfloat("plot", "node_size")
    cls.plt_ln_width = cp.getfloat("plot", "ln_width")
    cls.pin_color = cp.get("plot", "pin_color")
    cls.figsize = util.str2arr(cp.get("plot", "figsize"))
    cls.view_angles = util.str2arr(cp.get("plot", "view_angles"))
    
    # Deal with the particular case of the 3df2d sim, which
    # wants to be plotted from a different angle.
    #   This is not necessary anymore, since it's a setting in config.ini now.
    preset = cp.get("sim", "RN_preset")
    if preset == "3df2d":
      cls.view_angles = (0,-90)

  @classmethod
  def from_config(cls, cp):
    """Create and return an RN from a configuration object
    TO DO: I think all these class attributes won't load with a pickle...
      That's a problem, unless they're really things that don't have to do
      with the RN, and only are about manipulating it, like plotting.
    """
    # Configure the class attributes
    cls.cls_config(cp)
    
    # Load the fiber options
    N = cp.getint("RN-fiber", "N")
    fibers = cp.getboolean("RN-fiber", "fibers")
    fl_mean = cp.getfloat("RN-fiber", "fiber_len")
    fl_span = cp.getfloat("RN-fiber", "fl_span")
    cnd_len = cp.getfloat("RN-fiber", "cnd_len")
    brn_pwr = cp.getfloat("RN-fiber", "brn_pwr")
    bpwr_span = cp.getfloat("RN-fiber", "bpwr_span")
    # Load the dimensions options
    pin_r = cp.getfloat("RN-dim", "pin_r")
    # If a preset was specified, load dim and pins from the preset
    preset = cp.get("sim", "RN_preset")
    if preset == "None":
      xlims = util.str2arr(cp.get("RN-dim", "xlims"))
      ylims = util.str2arr(cp.get("RN-dim", "ylims"))
      zlims = util.str2arr(cp.get("RN-dim", "zlims"))
      in_pins_raw = cp.items("RN-in_pins")
      out_pins_raw = cp.items("RN-out_pins")
    else:
      dim_section = f"preset_{preset}_dim"
      in_pins_section = f"preset_{preset}_in_pins"
      out_pins_section = f"preset_{preset}_out_pins"
      xlims = util.str2arr(cp.get(dim_section, "xlims"))
      ylims = util.str2arr(cp.get(dim_section, "ylims"))
      zlims = util.str2arr(cp.get(dim_section, "zlims"))
      if cp.has_option(dim_section, "pin_r"):
        pin_r = cp.getfloat(dim_section, "pin_r")
      in_pins_raw = cp.items(in_pins_section)
      out_pins_raw = cp.items(out_pins_section)
    
    # Convert those 2 or 3 lims to the proper limits format
    if np.all(zlims[0] == [0,0]):
      # In this case, it's a 2d network
      limits = np.hstack((xlims, ylims))
    else:
      limits = np.hstack((xlims, ylims, zlims))
    
    # Load the lists of pins from in_pins_raw and out_pins_raw
    in_pins = []
    for pin_raw in in_pins_raw:
      pin_pos = util.str2arr(pin_raw[1])
      pin = (pin_raw[0], pin_pos)
      in_pins.append(pin)
    out_pins = []
    for pin_raw in out_pins_raw:
      pin_pos = util.str2arr(pin_raw[1])
      pin = (pin_raw[0], pin_pos)
      out_pins.append(pin)
    
    # Print a description of the RN being made
    RNs = f"Creating an RN with {N} nodes"
    if preset != "None": 
      RNs += f", using the standard preset {preset.upper()}"
    if fibers: 
      RNs += f", using fibers of avg L = {fl_mean}"
    else: 
      RNs += ", using 0-D fibers (points)"
    RNs += f", with max connection L = {cnd_len}"
    util.db_print(RNs)
    
    # To Do (Maybe): make pts an option in the config. It's tricky to do.
    return cls(N=N, limits=limits, in_pins=in_pins, out_pins=out_pins,
      fibers=fibers, fl_mean=fl_mean, fl_span=fl_span,
      brn_pwr=brn_pwr, bpwr_span=bpwr_span,
      cnd_len=cnd_len, pin_r=pin_r)

  @classmethod
  def RN_settings(cls, rn):
    """Create a dictionary of the RN settings.
    This is used when creating a json verson of the RN.
    """
    settings = {}
    settings["N"] = rn.N
    settings["xmin"] = rn.xmin
    settings["xmax"] = rn.xmax
    settings["ymin"] = rn.ymin
    settings["ymax"] = rn.ymax
    settings["zmin"] = rn.zmin
    settings["zmax"] = rn.zmax
    
    # Convert the non-serializable 'np.ndarray's to lists
    #in_pins = rn.in_pins
    #for i, v in enumerate(in_pins):
    #  if isinstance(v[1], np.ndarray):
    #    in_pins[i] = (v[0], v[1].tolist())
    #settings["in_pins"] = in_pins
    #out_pins = rn.out_pins
    #for i, v in enumerate(out_pins):
    #  if isinstance(v[1], np.ndarray):
    #    out_pins[i] = (v[0], v[1].tolist())
    #settings["out_pins"] = rn.out_pins
    
    settings["pin_keys"] = rn.pin_keys
    settings["node_list"] = rn.node_list
    #settings["pts"] = rn.pts
    settings["fibers"] = rn.fibers
    settings["fl_mean"] = rn.fl_mean
    settings["fl_span"] = rn.fl_span
    #settings["brn_pwr"] = rn.brn_pwr
    #settings["bpwr_span"] = rn.bpwr_span
    settings["cnd_len"] = rn.cnd_len
    settings["pin_r"] = rn.pin_r
    settings["dim"] = rn.dim
    settings["voltageMult"] = rn.voltageMult
    settings["totalPower"] = rn.totalPower
    
    return settings

  @classmethod
  def save_RN(cls, rn, fname, compress=True):
    """Save a RN to file by pickling it.
    I imagine this being used mostly for saving a network that was built so 
    many sims can be run on it. For example, when training, the RN state could
    be saved between each epoch in case the simulation needs to be paused and
    resumed later
    
    Parameters
    ----------
    rn : a resister network
    fname (str) : the filename to save the file to
      Must end with ".pickle" or ".json"
    In general, pickling isn't great for long-term storage because it may be
    unable to run if the files were changed much since the object was pickled.
    """
    if fname[-7:] == ".pickle":
      if compress:
        fname += ".gz"
        with gzip.open(fname, "wb") as gzfile:
          pickle.dump(rn, gzfile)
      else:
        with open(fname, "wb") as file:
          pickle.dump(rn, file)
      #file = open(fname, "wb")
      #pickle.dump(rn, file)
      #file.close()
    elif fname[-5:] == ".json":
      # First we need to prepare the RNdata dictionary to be saved as json
      Gdata = nx.readwrite.json_graph.node_link_data(rn.G)
      # Convert the non-serializable np.arrays to lists
      for node in Gdata["nodes"]:
        attrs = node.keys()
        for attr in attrs:
          if isinstance(node[attr], np.ndarray):
            # Convert to list
            node[attr] = node[attr].tolist()
      for link in Gdata["links"]:
        attrs = link.keys()
        for attr in attrs:
          if isinstance(link[attr], np.ndarray):
            # Convert to list
            link[attr] = link[attr].tolist()
      RN_settings = cls.RN_settings(rn)
      RNdata = {} #Create master dictionary to be saved to file
      RNdata["settings"] = RN_settings
      RNdata["nx_graph"] = Gdata
      #file = open(fname, "w")
      #file.close()
      if compress:
        fname += ".gz"
        with gzip.open(fname, "wt", encoding="utf-8") as gzfile:
          json.dump(RNdata, gzfile)
      else:
        with open(fname, "w") as file:
          json.dump(RNdata, file)
    else:
      print("Error: the file format must be .pickle or .json")
      return 1
    util.db_print(f"RN state saved to file {fname}")
    return fname

  @classmethod
  def load_RN(cls, fname):
    """Loads and returns a previously pickled RN from file. 
    See saveRN()
    Automatically determines whether the file is a .pickle or a .json file
      based on the filename fname
    """
    ftype = {"format":"?", "compressed":False}
    if fname[-3:] == ".gz":
      ftype["compressed"] = True
      if fname[-10:-3] == ".pickle":
        ftype["format"] = "pickle"
      elif fname[-8:-3] == ".json":
        ftype["format"] = "json"
    else:
      #ftype["compressed"] = False
      if fname[-7:] == ".pickle":
        ftype["format"] = "pickle"
      elif fname[-5:] == ".json":
        ftype["format"] = "json"
    if ftype["format"] == "?":
      print("Error: the file format must be .pickle or .json")
      return 1
    
    if ftype["format"] == "pickle":
      if ftype["compressed"]:
        with gzip.open(fname, "rb") as gz_file:
          rn = pickle.load(gz_file)
      else:
        with open(fname, "rb") as file:
          rn = pickle.load(file)
      return rn
    elif ftype["format"] == "json":
      # Load json object from file
      if ftype["compressed"]:
        with gzip.open(fname, "rt", encoding="utf-8") as gz_file:
          jso = json.load(gz_file)
      else:
        with open(fname, "r") as file:
          jso = json.load(file)
      # Create RN
      rn = cls.from_json(jso)
      return rn
    else:
      print("Error: the file format must be .pickle or .json")
      return 1

  #@property
  #def pins(self):
    """This way all the pins can be accessed together.
    Often we want to iterate through all the pins.
    """
    #return self.in_pins + self.out_pins

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
    
    # Add variance to the burn power
    if self.bpwr_span != 0:
      bpwr = np.random.rand(self.N, 1) * self.bpwr_span
      bpwr += self.brn_pwr - self.bpwr_span/2
    
    node_keys_it = range(len(self.pts))
    G = nx.Graph()
    
    # Add the nodes
    for i in node_keys_it:
      atrb = {}
      if self.fibers:
        atrb["fv"] = self.fv[i]
      if self.bpwr_span != 0:
        atrb["bpwr"] = bpwr[i]
      # The key for each node will simply be its index in the pts array
      G.add_node(i, pos=self.pts[i], **atrb)
    
    # Keep track of all nodes. The first N nodes will be the random nodes.
    # Whenever a node is deleted, remove it from this list.
    # Pass this list to the laplacian_matrix function so the order of the 
    #   entries in the matrix is predictable.
    self.node_list = list(node_keys_it)
    
    # Add the pins
    for pin in self.pins:
      pin_atrb = {}
      # The pins need to be assigned a brn_pwr
      if self.bpwr_span != 0:
        pin_atrb["bpwr"] = 10*self.brn_pwr
      
      # Add the pin with the provided key and location
      G.add_node(pin[0], pos=pin[1], **pin_atrb)
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
  
  def R_nn(self, n0, n1):
    """Resistance from node n0 to node n1
    
    Parameters
    ----------
    n0, n1 : tags of nodes to find Req between. Can be pins or fibers.
    
    Returns
    -------
    Req : The equivalent resistance between the two given nodes.
    """
    #Apply 1V to the network and get the Req from that.
    RES = self.apply_v(1, n0, n1, set_v=False)
    return RES[1]
  
  def R_pp(self, p0, p1):
    """Resistance from pin p0 to pin p1.
    Deprecated by R_nn, but included for backwards compatability.
    """
    return self.R_nn(p0, p1)
  
  def R_ij(self, i0, i1):
    """Resistance from index i0 to i1. Uses R_nn
    """
    n0 = self.index_to_key(i0)
    n1 = self.index_to_key(i1)
    return self.R_nn(n0, n1)
  
  # Resistance from node index i0 to i1
  # Old, because this can be done now with one call to apply_v
  def R_ij_old(self, i0, i1, lsqr=False):
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
  
  def apply_v(self, v_in, n0, n1, ground=None, set_v=True, set_p=True):
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
      (if !set_v) (v, Req) : (the array of voltages, the Req from n0 to n1)
      (if set_p) (p_max_e, e_p_max) : (the max edge power, the corrosponding edge)
      (if set_p) (p_max_n, n_p_max) : (the max node power, the corrosponding node)
    """
    # Solution method: "spsolve", "splsqr", "cg"
    # splsqr is currently having issues, apparantly due the the Lpl having a 
    #   high condition number because of the pins. I think.
    self.sol_method = "cg"
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
    
    if self.sol_method == "cg":
      # Absolute tolerence for the norm of the residuals for cg
      atol = 1e-5
      # Use scipy's conjugate gradient method, since Lpl is symmetric
      v, status = spsl.cg(Lpl, b, atol=atol)
      # Set the ground pin to zero, adjusting the answers accordingly
      v -= v[gdi]
      # Check the exit status
      if status != 0:
        util.db_print(f"Conjugate Gradient Error: code {status}")
        if status > 0:
          util.db_print("\tthe desired tolerance was not reached")
        else:
          util.db_print("\t\"illegal input or breakdown\"")
    elif self.sol_method == "spsolve":
      # Ground the given pin, overwriting the last row for the grounding equation
      # That's fine, since all the information is repeated.
      Lpl[N-1] = sps.csr_matrix((1,N))
      Lpl[N-1, gdi] = 1
      b[N-1] = 0
      # Solve Lpl*v=b
      v = spsl.spsolve(Lpl, b)
    elif self.sol_method == "splsqr":
      RES = spsl.lsqr(Lpl, b)
      v = RES[0]
      # Set the ground pin to zero, adjusting the answers accordingly
      v -= v[gdi]
      # Check the exit status
      if RES[1] != 1:
        util.db_print("Error: lsqr did not confidently find a good solution")
        util.db_print(f"Exit info: {str(RES[1:-1])}")
        util.db_print("spsl.lsqr really shouldn't be used for this problem.")
    
    # Adjust the voltages
    Req = v[n0i] - v[n1i] #This is true since I was 1A.
    voltage_mult = v_in / Req
    v *= voltage_mult
    # Save the voltages in the nodes
    if set_v:
      node_i = 0
      for node_key in self.G.nodes:
        self.G.nodes[node_key]["v"] = v[node_i]
        node_i += 1
    else:
      # If not saving v in each node, just return the vector and the Req.
      return v, Req
    
    if set_p:
      p_max_e = 0 # Max power in an edge
      e_p_max = None # Corresponding edge
      p_max_n = 0 # Max power in a node
      n_p_max = None # Corresponding node
      # Start by setting power to 0 in each node
      nx.set_node_attributes(self.G, 0, name="p")
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
        # Save the square of power (could be used for plot line thickness)
        self.G[en0][en1]["p2"] = p**2
        # Add this power to both of the nodes on the edge
        # Note: I don't know how much sense this makes in reality...
        self.G.nodes[en0]["p"] += p
        self.G.nodes[en1]["p"] += p
        # Keep track of p max
        if p > p_max_e:
          p_max_e = p
          e_p_max = edge
        if self.G.nodes[en0]["p"] > p_max_n:
          p_max_n = self.G.nodes[en0]["p"]
          n_p_max = en0
        if self.G.nodes[en1]["p"] > p_max_n:
          p_max_n = self.G.nodes[en1]["p"]
          n_p_max = en1
      
      return (p_max_e, e_p_max), (p_max_n, n_p_max), Req
  
  def burn(self, to_burn="p_max"):
    """Remove the fibers that have too much power flowing through them.
    Before running this function, apply_v should have been run with set_p=True
    TO DO: optionally burn N fibers
    
    Parameters
    ----------
    to_burn : Specify which fibers to burn. If set to "p_max", then it will
      burn all fibers with p > self.brn_pwr.
      If a list is passed, then those fibers will be burned.
      If set to 1, then the max power node will be burned.
    
    Returns
    -------
    fibers_burned (int) : How many fibers were removed.
    """
    
    if to_burn == "p_max":
      util.db_print("Removing high-power fibers")
      util.db_print("Hopefully apply_v(set_p=True) has been done already...")
      to_burn = []
      for node in self.G.nodes:
        brn_pwr = (self.brn_pwr if (self.bpwr_span == 0) 
          else self.G.nodes[node]["bpwr"])
        if self.G.nodes[node]["p"] > brn_pwr:
          to_burn.append(node)
    if isinstance(to_burn, int):
      # TO DO: Burn a number of pins equal to to_burn
      pass
    if to_burn == 1:
      # Find and burn the max power node
      to_burn = [self.get_maxp_node()[0]]
    for node in to_burn:
      self.remove_node(node)
    fibers_burned = len(to_burn)
    util.db_print(f"{fibers_burned} fibers were removed")
    return fibers_burned
  
  def get_maxp_node(self):
    """Find the node with the highest power.
    Before running this function, apply_v should have been run with set_p=True
    
    Returns
    -------
    max_p_node : RN id of the max power node
    node_data : the node data for that node
    """
    max_p = 0
    max_p_node = None
    for node in self.G.nodes:
      if self.G.nodes[node]["p"] > max_p:
        max_p = self.G.nodes[node]["p"]
        max_p_node = node
    node_data = self.G.nodes[max_p_node].copy()
    return max_p_node, node_data
  
  def draw(self, fig=None, ax=None, width_attrib=None, color_attrib=None,
    edge_color="b", to_mark=[]):
    """Make a plot of the RN.
    Parameters
    ----------
    fig, ax : pyplot fig and ax. If ax is not provided, they are both created.
    width_attrib : An edge attribute that is used to determine edge line 
      widths. e.g. power through edge
    color_attrib : A node attribute that is used to determine the color of
      each fiber.
    edge_color : An edge color.
    to_mark : This setting refers to which nodes to mark with a red X. 
      If set to "p_max", it marks all those fibers that have more power than
      RN.p_max.
      If a list is passed, then those nodes provided are marked.
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
      ax.view_init(self.view_angles[0], self.view_angles[1])
    
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
        fiber_color = nx.get_node_attributes(self.G, color_attrib)
        #Cut out the pins from the color list
        [fiber_color.pop(pin[0]) for pin in self.pins]
        fiber_color = np.vstack(list(fiber_color.values()))
        if self.dim == 3:
          # 2 Experimental features
          RGBA = False
          COLOR_BUBBLES = False
          if RGBA:
            # Convert to RGBA
            # I did this to be able to set the alpha value lower
            fiber_color -= np.min(fiber_color)
            fiber_color = fiber_color / (np.max(fiber_color))
            fiber_color = np.hstack(( .1*np.ones(np.shape(fiber_color)), 
              fiber_color, .1*np.ones(np.shape(fiber_color)), 
              0.25*np.ones(np.shape(fiber_color)) ))
            print(697, fiber_color, np.min(fiber_color), np.max(fiber_color))
          # Quiver 3D doesn't support multiple colors, so I'd need a work-around
          print("Specifying fiber color is not yet implemented in 3D")
          if COLOR_BUBBLES:
            # Since Quiver 3D doesn't support varying color, my work-around
            #   plan is to make a background with transparent blobs of color
            ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=10*self.plt_node_size,
              c=fiber_color, zorder=1)
      
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
        ax.scatter(pos[:,0], pos[:,1], s=self.plt_node_size,
          c=node_color, zorder=2)
      else:
        ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=self.plt_node_size,
          c=node_color, zorder=2)
    
    # If color_attrib was used, add a colorbar
    if not color_attrib is None:
      atr_ls = list(nx.get_node_attributes(self.G, color_attrib).values())
      sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=min(atr_ls), 
        vmax=max(atr_ls)))
      plt.colorbar(sm, ax=ax, fraction=.075, shrink=.6)
      # If to_mark == "p_max", then place a red X over high-power fibers
      if to_mark == "p_max":
        to_mark = []
        for node in self.G.nodes:
          brn_pwr = (self.brn_pwr if (self.bpwr_span == 0) 
            else self.G.nodes[node]["bpwr"])
          if self.G.nodes[node]["p"] > brn_pwr:
            to_mark.append(node)
      for node in to_mark:
        # Add a red X over the node
        ax.plot(*self.G.nodes[node]["pos"], "rX", markersize=20)
          
    
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
      # To Do: zorder? - doesn't work in 3d?
      a3 = ax3d.plot_wireframe(x, y, z, color=color, linewidths=self.plt_ln_width)
    else:
      # alpha controls opacity
      a3 = ax3d.plot_surface(x, y, z, color=color, alpha=0.3)

  def train(self, N_in, N_out, max_epochs=10, callback=None):
    """Train the network so that it produces the desired output with the given input.
    
    Parameters
    ----------
    N_in (str) : A binary string that represents the input. Each bit that is
      a 1 corresponds to a high pin.
    N_out (str) : A binary string that represents the desired output. Each bit
      that is a 1 corresponds to a high pin.
    max_epochs (int) : The max number of epochs that will be run.
    callback (function) : A function to be called after each epoch
      accepts (int) current epoch number
    
    Returns
    -------
    code (int) : Success indicator code
      0 - training was successful
      1 - training was not successful
    epochs (int) : How many epochs were carried out
    """
    # 1: Perform a forward pass and set the threshold such that all of the 
    #   outputs register as high.
    # 1.5: If the threshold is too high, such that a pin that should be high
    #   is low, then bump down the threshold
    #     - or, always set the threshold to a little lower than the current
    #       leaving the lowest current High node.
    # 2: Check to see if that result matches the desired output.
    # 3: Choose which connection to burn out based on which pin should be 
    #   low and has the highest current
    #       QUESTION: How do we know which connection to burn out? We know
    #         that we want the pin to be low and it's high, but how do we
    #         decide where to burn across? Make its connections to all of the
    #         high inputs weaker? That seems crude.
    #         --> No, that's exactly it. If the pin is high and should be low,
    #           then weaken the connection that output pin has to all the
    #           input pins that were high.
    # 4: Send voltage through that connection. (For now, I can simply tell
    #   it to burn out exactly one fiber along that path, if that's easier)
    # 5: Repeat 1-4 until max_epochs is reached or desired output is achieved.
    out_bools = self.bin_to_bools(N_out)
    
    epoch = 0
    
    # List the ids of those input pins that are high
    in_pin_ids = self.in_pin_ids(N_in)
    # List the ids of the output pins
    out_pin_ids = self.out_pin_ids()
    
    while epoch < max_epochs:
      # Only reset the threshold the first time: reset_threshold=(epoch==0)
      #output, currents = self.fwd_pass(N_in, epoch==0)
      output, currents = self.fwd_pass(N_in, reset_threshold=False, 
        shift_threshold=True, out_bools=out_bools)
      # If a callback was specified, call it
      if callback != None:
        callback(epoch)
      
      #util.db_print(f"Output = {output}")
      #util.db_print(f"Output Currents = {currents}")
      #util.db_print(f"Threshold = {self.threshold}")
      # If the output is correct, then exit
      if np.all(output == out_bools):
        return 0, epoch
      
      # Make a list of connections to be weakened (#3)
      # list of pins that are too high
      err_pins = out_pin_ids[output > out_bools]
      to_burn = []
      for err_pin in err_pins:
        for in_id in in_pin_ids:
          to_burn.append( (err_pin, in_id) )
      
      # Burn across each of those connections (#4)
      for p_in, p_out in to_burn:
        #util.db_print(f"Burning from {p_in} to {p_out}")
        self.apply_v(100, p_in, p_out, set_p=True)
        self.burn(1) # Burn the highest power node
      
      epoch += 1
    
    return 1, epoch

  def fwd_pass(self, N_in, reset_threshold=False, shift_threshold=False,
    out_bools=None):
    """Run a forward pass with the specified input number
    
    Parameters
    ----------
    N_in (str) : A binary string that represents the input. Each bit that is
      a 1 corresponds to a high pin.
    reset_threshold (bool) : Whether or not to reset the threshold to be below
      all of the currents so that all pins register as high. This is useful at
      the start of training.
    shift_threshold (bool) : Whether or not to shift the threshold such that
      all the desired outputs are high. If True, out_bools must be provided.
    out_bools (list of bool) : List of which outputs are desired.
    
    Returns
    -------
    output : A list of binary outputs representing which of the output currents
      registered as high.
    currents : The output currents
    """
    # List the ids of those input pins that are high
    in_pin_ids = self.in_pin_ids(N_in)
    # List the ids of the output pins
    out_pin_ids = self.out_pin_ids()
    # Apply fpV to all selected inputs to the outputs
    self.set_voltages(in_pin_ids, out_pin_ids, self.fpV)
    # Find the current flowing out of each output pin
    currents = self.sum_currents(out_pin_ids)
    # Shift the threshold so that all the desired outputs are high
    if shift_threshold:
      self.threshold = 0.95 * np.min(currents[out_bools])
    # Optionally reset the threshold so all the pins are high
    if reset_threshold:
      self.threshold = 0.95 * np.min(currents)
    
    output = self.threshold_currents(currents)
    #util.db_print(f"output={output}")
    # Threshold those currents into a binary result
    return output, currents

  def bwd_pass(self, out_currents):
    """Perform a backwards pass to burn out undesired connections.
    """
    print("Bwd pass not yet implemented.")
    pass

  def in_pin_ids(self, N_in=None):
    """Return the ids of the input pins, but only include those pins which
      are supposed to be high.
    
    Parameters
    ----------
    N_in (str): Boolean string representing which inputs are high.
    
    Returns
    -------
    in_pin_ids : A list of ids belonging to the input pins
    """
    num_ins = len(self.in_pins)
    if N_in != None:
      in_indices = self.bin_to_bools(N_in)
    else:
      # If no number is given, consider all input pins
      return [pin[0] for pin in self.in_pins]
      #in_indices = np.ones(num_ins)
    in_pin_ids = []
    for i in range(num_ins):
      if in_indices[i]:
        in_pin_ids.append(self.in_pins[i][0])
    return np.array(in_pin_ids)

  def out_pin_ids(self):
    return np.array([pin[0] for pin in self.out_pins])

  def bin_to_bools(self, Nbin):
    """Convert a binary string to a list of booleans
    Nbin should have the normal format: "0bXXXX"
    """
    # This loops from the end to the 3rd character (so as to skip the "0b")
    return [ bool(int(i)) for i in Nbin[-1:1:-1] ]

  def connect_pins(self, connectedPins, label):
    self.add_node(label)
    edges = []
    for i in connectedPins:
      edges.append([label, i, {'res' : 1e-10, 'cnd' : 1e10}])
    self.G.add_edges_from(edges)
    return label

  def set_voltages(self, startingNodes, endingNodes, voltage):
    startName = self.connect_pins(startingNodes, "High")
    endName = self.connect_pins(endingNodes, "Ground")
    #startIndex = self.G.number_of_nodes() - 2
    #endIndex = self.G.number_of_nodes() - 1
    
    # Apply the given voltage to the network
    self.apply_v(voltage, startName, endName)
    
    #removeConnectors
    self.remove_node(startName)
    self.remove_node(endName)

  def sum_currents(self, nodes):
    """Returns the total current flowing out of the given nodes.
    For most nodes, this should be zero by KCL, so this should be used with
      the input or output pins.
    
    Parameters
    ----------
    nodes : a list of node ids
    
    Returns
    -------
    currents : a list of currents
    """
    currents = np.zeros(len(nodes))
    it = 0
    for n in nodes:
      voltage = self.G.nodes[n]["v"]
      for u, v in self.G.edges(n):
        currents[it] += (self.G.nodes[v]["v"] - voltage) * self.G[u][v]["cnd"]
      it += 1
    
    return currents

  def threshold_currents(self, currents):
    """Read in the list of currents and return a list of the same length
      that represents whether or not each current is higher than the threshold.
    
    Parameters
    ----------
    currents : a list of currents
    #threshold (float): the current value above which a current translates to
    #  a [True/High/1] value.
    
    Returns
    -------
    output : a list of booleans representing which currents thresholded to 1.
    """
    output = np.array(currents) > self.threshold
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
    startName = self.connect_pins(inputs, "High")
    endName = self.connect_pins(badNodes, "Ground")
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
