import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.spatial as spl
from scipy.stats import truncnorm as tnorm
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
  sdevmax = 4 # How many Standard Deviations to consider before clipping.
  
  # Plotting options
  plt_node_size = 7#50 #default is 300, which is pretty big.
  plt_ln_width = 0.3 #Line Width
  pin_color = "green" #"gold"
  figsize = (12,10)
  view_angles = (12, -110)
  
  # Network Resistance options
  pin_res = 1e-6 #Resistance btw pins and the nodes they intersect with. It's an arbitrarily low resistance.
  res_k = 10000 #Coefficient multiplier for resistance calculation btw nodes
  res_w = 1 #Coefficient for argument of sinh in nonlinear sim. IE how steeply nonlinear.
  min_res = 1e-6 #Min resistance. Think of the resistance of the fiber itself, even if they're touching
  # Solution method: "spsolve", "splsqr", "cg". 
  #   splsqr is not a good idea because the matrix is symmetric, 
  #   and spsolve doesn't work if the graph has any isolated points
  sol_method = "cg"
  # Stopping tolerance in x for nonlinear solvers.
  xtol = 1e-3
  ftol = 1e-3

  # Training options
  fpV = .5 # Fwd pass voltage for training
  bpV = 2 # Bwd pass voltage for training
  # Current threshold for determining what is high
  threshold = 0.5
  # When dynamically resetting the threshold, what fraction of the lowest current desired output should the threshold be set to.
  threshold_fraction = 0.1#0.95
  # How many fibers to burn per burn step
  burn_rate = 4
  # Used if burn power was not specified in config
  brn_pwr = 1
  # Whether to burn fibers or edges when training
  burn_fibers = True

  def __init__(self, N=500, limits=[0,1, 0,1], in_pins=[], out_pins=[],
    pts=None, cnd_len=0.15, pin_r=0.25, fibers=True, 
    fl_mus=[0.1], fl_sgs=[0.005], bpwr_mus=[1], bpwr_sgs=[0], 
    ftype_proportions=[1]):
    """Constructor for the network.
    
    Parameters
    ----------
    N (int) : Number of nodes in network (default 500).
    limits (float array, len = 4 or 6) : x and y limits of the network 
      (default [0,1, 0,1]).
      format: [xmin, xmax, ymin, ymax(, zmin, zmax)]
      If 6 limits are provided, this triggers a 3D network.
    [in/out]_pins : Provide a list of pin locations. If none are provided, there will
      be no pins placed in the RN.
      format: [("pin0", [x0, y0]), ("pin1", [x1, y1]), ...]
      in_pins is for input pins and out_pins is for output pins.
    pts : Optionally provide a custom list of fiber locations. 
      If not given, they will be randomly generated.
    cnd_len (float) : The max radius of connections between nodes.
    pin_r (float) : The radius of the pins (Pins are currently implemented 
      only as circles or spheres)
    fibers (bool) : Enable fibers (default False -- point fibers)
      The next 5 lists describe the quality and proportions of the fibers.
      They must all have the same length.
    fl_mus (float array) : Median (mu) of the distributions describing the
      length of each kind of fiber.
    fl_sgs (float array) : Standard Deviation (sigma) of the distributions
      describing the length of each kind of fiber.
    bpwr_mus (float array) : Median (mu) of the distributions describing the
      burn power of each kind of fiber.
    bpwr_sgs (float array) : Standard Deviation (sigma) of the distributions
      describing the burn power of each kind of fiber.
    ftype_proportions (float array) : A list of fractions describing how many
      of each kind of fiber there are.

    Fiber distributions example:
      fl_mus = [.1, .2]
      fl_sgs = [.001, .001]
      bpwr_mus = [1, 2]
      bpwr_sgs = [.001, .002]
      ftype_proportions = [.9, .1]
      --> This would make two classes of fibers:
        90% of the fibers have average L=.1 and bpwr=1
        10% of the fibers have average L=.2 and bpwr=2
        And the standard deviation of each distribution is given by the
          corresponding number in the _sgs arrays.
        By default, for now, these distributions are truncated between zero
          and +4 std. dev. from mu

    """
    
    self.applied_v = 0 # Most recent voltage applied
    if N is None:
      # This is a special case for when an empty RN is to be created
      return
    self.N = N
    self.xmin = limits[0]
    self.xmax = limits[1]
    self.ymin = limits[2]
    self.ymax = limits[3]
    
    self.in_pins = in_pins
    self.out_pins = out_pins
    self.pin_keys = [pin[0] for pin in self.pins]
    self.pts = pts
    self.cnd_len = cnd_len
    self.pin_r = pin_r
    self.fibers = fibers
    self.fl_mus = fl_mus
    self.fl_sgs = fl_sgs
    self.bpwr_mus = bpwr_mus
    self.bpwr_sgs = bpwr_sgs
    # Make sure the proportions list adds up to 1
    sum_p = np.sum(ftype_proportions)
    if sum_p == 1:
      self.ftype_proportions = np.array(ftype_proportions)
    else:
      util.sim_log("Warning: the ftype_proportions list should add to 1")
      self.ftype_proportions = np.array(ftype_proportions) / sum_p

    #TEMP
    maxNR = self.res_fun(cnd_len)
    # This ratio has to do with how difficult the linear algebra will be
    # like the condition number, except the matrix is singular
    MMRR = maxNR / min(self.min_res, self.pin_res) 
    util.db_print(f"Max/min R ratio = {MMRR}")
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
    cls.res_w = cp.getfloat("RN-res", "res_w")
    cls.xtol = cp.getfloat("RN-res", "xtol")
    cls.ftol = cp.getfloat("RN-res", "ftol")
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
    """
    # Configure the class attributes
    cls.cls_config(cp)
    
    # Load the fiber options
    N = cp.getint("RN-fiber", "N")
    cnd_len = cp.getfloat("RN-fiber", "cnd_len")
    fibers = cp.getboolean("RN-fiber", "fibers")
    fl_mus = util.str2arr(cp.get("RN-fiber", "fl_mus"))
    fl_sgs = util.str2arr(cp.get("RN-fiber", "fl_sgs"))
    bpwr_mus = util.str2arr(cp.get("RN-fiber", "bpwr_mus"))
    bpwr_sgs = util.str2arr(cp.get("RN-fiber", "bpwr_sgs"))
    ftype_proportions = util.str2arr(cp.get("RN-fiber", "ftype_proportions"))
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
      RNs += f"\n\tusing the standard preset {preset.upper()}"
    if fibers: 
      RNs += "\n\tusing the following kinds of fibers:"
      for flmu, bpmu, ftp in zip(fl_mus, bpwr_mus, ftype_proportions):
        RNs += f"\n\t\t{ftp*100}% fibers with mean length = {flmu}"
        RNs += f" and mean burn power = {bpmu}"
    else: 
      RNs += "\n\tusing point fibers"
    RNs += f"\n\twith max connection L = {cnd_len}"
    util.db_print(RNs)
    
    # Possible Feature: make pts an option in the config. It'd be tricky to do.
    #   Probably you'd provide a csv of points.
    return cls(N=N, limits=limits, in_pins=in_pins, out_pins=out_pins,
      cnd_len=cnd_len, pin_r=pin_r, fibers=fibers, 
      fl_mus=fl_mus, fl_sgs=fl_sgs, bpwr_mus=bpwr_mus, bpwr_sgs=bpwr_sgs,
      ftype_proportions=ftype_proportions)

  @classmethod
  def from_json(cls, jso):
    """Create and return an RN from a json-loaded dict
    Note: the json does not have cls_config things
    """
    
    rn = cls(N=None) # Make a nearly empty RN
    settings = jso["settings"]
    # Copy all the settings into attributes of the RN
    for atr_name, atr_val in settings.items():
      setattr(rn, atr_name, atr_val)
    # Fix in_pins and out_pins to be np arrays again
    rn.in_pins = [ (pin[0], np.array(pin[1])) for pin in rn.in_pins ]
    rn.out_pins = [ (pin[0], np.array(pin[1])) for pin in rn.out_pins ]
    
    # TODO: I think there are other np.arrays that got stored as lists
    # Yeah, it's the node and edge attributes, which are loaded here:
    
    # Load the nx graph
    rn.G = nx.readwrite.json_graph.node_link_graph(jso["nx_graph"])
    
    return rn

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
    in_pins = rn.in_pins
    for i, v in enumerate(in_pins):
      if isinstance(v[1], np.ndarray):
        in_pins[i] = (v[0], v[1].tolist())
    settings["in_pins"] = in_pins
    out_pins = rn.out_pins
    for i, v in enumerate(out_pins):
      if isinstance(v[1], np.ndarray):
        out_pins[i] = (v[0], v[1].tolist())
    settings["out_pins"] = rn.out_pins
    
    settings["pin_keys"] = rn.pin_keys
    settings["node_list"] = rn.node_list
    #settings["pts"] = rn.pts
    settings["fibers"] = rn.fibers
    # I don't really need to save these, since they're just used to 
    #   build the RN
    #settings["fl_mean"] = rn.fl_mean
    #settings["fl_span"] = rn.fl_span
    #settings["brn_pwr"] = rn.brn_pwr
    #settings["bpwr_span"] = rn.bpwr_span
    settings["cnd_len"] = rn.cnd_len
    settings["pin_r"] = rn.pin_r
    settings["dim"] = rn.dim

    settings["applied_v"] = rn.applied_v
    
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
      util.sim_log("Error: the file format must be .pickle or .json")
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
      util.sim_log("Error: the file format must be .pickle or .json")
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
      util.sim_log("Error: the file format must be .pickle or .json")
      return 1

  @property
  def pins(self):
    """This way all the pins can be accessed together.
    Often we want to iterate through all the pins.
    """
    return self.in_pins + self.out_pins

  def create_graph(self):
    if self.pts is None:
      domain_sizes = [self.xmax-self.xmin, self.ymax-self.ymin, self.zmax-self.zmin]
      domain_offsets = [self.xmin, self.ymin, self.zmin]
      # Generate a matrix of random points in the domain
      self.pts = np.random.rand(self.N, self.dim)
      self.pts *= domain_sizes[0:self.dim]
      self.pts += domain_offsets[0:self.dim]
    if self.fibers:
      assert isinstance(self.ftype_proportions, np.ndarray)
      # How many of each type to make
      N_each = np.round(self.ftype_proportions * self.N).astype(np.int64)
      N_new = np.sum(N_each)
      if N_new != self.N:
        # Got thrown off by rounding
        #util.db_print("451: adjusting N_each")
        if N_new > self.N:
          for i in range(N_new-self.N):
            N_each[i] -= 1
        elif N_new < self.N:
          for i in range(self.N-N_new):
            N_each[i] += 1
        assert np.sum(N_each) == self.N
      
      fl = []
      bp = []
      for flmu, flsg, bpmu, bpsg, Ni in zip(self.fl_mus, self.fl_sgs,
        self.bpwr_mus, self.bpwr_sgs, N_each):
        #util.sim_log(464, flmu, flsg, bpmu, bpsg, Ni)
        alim = -.999* flmu / flsg # How many stdvs away is zero(*)
        fl.append(tnorm.rvs(size=Ni, loc=flmu, scale=flsg,
          a=alim, b=self.sdevmax))
        alim = -.999* bpmu / bpsg
        bp.append(tnorm.rvs(size=Ni, loc=bpmu, scale=bpsg,
          a=alim, b=self.sdevmax))
      fl = np.vstack(np.concatenate(fl))
      bp = np.vstack(np.concatenate(bp))
      
      # Angle of each fiber
      th = np.random.rand(self.N, 1) * 2*np.pi
      if self.dim == 2:
        # convert L, th to [u, v]
        fv = np.hstack([fl*np.cos(th), fl*np.sin(th)])
      elif self.dim == 3:
        # second angle to determine direction in 3D (pitch from 0 (up) to pi (down))
        phi = np.random.rand(self.N, 1) * np.pi - np.pi/2
        # convert L, th, phi to [u, v, w]
        fv = np.hstack([fl*np.sin(phi)*np.cos(th), fl*np.sin(phi)*np.sin(th), 
          fl*np.cos(phi)])
    
    node_keys_it = range(len(self.pts))
    G = nx.Graph()
    
    # Add the nodes
    for i in node_keys_it:
      atrb = {}
      if self.fibers:
        atrb["fv"] = fv[i]
        # bp is a vstack, so we need to extract the element
        atrb["bpwr"] = bp[i][0]
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
      pin_atrb["bpwr"] = 1e6*np.max(self.bpwr_mus)
      
      # Add the pin with the provided key and location
      G.add_node(pin[0], pos=pin[1], **pin_atrb)
      # Store this pin in the list of all nodes.
      self.node_list.append(pin[0])
    
    # Build a KDTree to make searching much faster
    tree = spl.cKDTree(self.pts)
    if self.fibers: 
      # The longest possible fiber
      #flmax = np.max(self.fl_mus)
      #flmax += self.sdevmax*self.fl_sgs[ np.where(self.fl_mus == flmax)[0][0] ]
      flmax = np.max(fl)
    # Find and add the edges between the nodes
    for i in node_keys_it:
      # Max node distance to consider connection
      dmax = self.cnd_len
      if self.fibers:
        # Consider this fiber's length and the longest fiber in the RN
        dmax += fl[i] + flmax
      neighbors = tree.query_ball_point(self.pts[i], dmax)
      for j in neighbors:
        # if j > i, then add the edge. Else, it's already been added
        if j <= i:
          continue
        if self.fibers:
          dist, dminpt = util.min_LS_dist( self.pts[i], self.pts[j], 
            fv[i], fv[j] )
          # For fibers, the search bubble was bigger than cnd_len
          if dist < self.cnd_len:
            res = self.res_fun(dist)
            # ep0, ep1 : edge p0, p1. Save the x-y coordinates of the points along 
            #   the fibers where dist is smallest.
            ep0 = self.pts[i] + fv[i] * dminpt[0]
            ep1 = self.pts[j] + fv[j] * dminpt[1]
            G.add_edge( i, j, res=res, cnd=1/res, pos=np.vstack([ep0, ep1]) )
        else:
          dist = np.linalg.norm(self.pts[i] - self.pts[j])
          res = self.res_fun(dist)
          G.add_edge( i, j, res=res, cnd=1/res)
    
    # Add the edges for the pins
    res = self.pin_res
    dmax = self.pin_r
    if self.fibers: 
      dmax += flmax
    for pin in self.pins:
      neighbors = tree.query_ball_point(pin[1], dmax)
      for pt_i in neighbors:
        if self.fibers:
          dist, t = util.LS_pt_dist(pin[1], self.pts[pt_i], 
            fv[pt_i], dmaxt=self.cnd_len)
          # Note: currently, this considers only nodes that are colliding
          #   with the pin. I could alternatively consider also all fibers
          #   that are within cnd_len of the pin and then make those
          #   connections higher resistance accordingly.
          if dist < self.pin_r:
            # ep0, ep1 : edge p0, p1. Save the x-y coordinates of the edge
            #   start and end points.
            ep0 = pin[1]
            ep1 = self.pts[pt_i] + fv[pt_i] * t
            G.add_edge(pin[0], pt_i, res=res, cnd=1/res, pos=np.vstack([ep0, ep1]) )
        else:
          G.add_edge(pin[0], pt_i, res=res, cnd=1/res)
    
    return G

  def res_fun(self, dist):
    """Resistance between nodes or fibers as a function of distance
    TODO: incorporate tunneling or thermionic emission equation
    """
    return self.min_res + self.res_k*dist*dist
  
  def inv_res_fun(self, R):
    """Inverse of res_fun. Used to get back the distance, knowing R.
    R = A + B*x^2
    sqrt((R - A)/B) = x
    """
    return np.sqrt( (R - self.min_res) / self.res_k )
  
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

  def node(self, key):
    """Get the specified node from the graph's nodeview and return a
    dictionary representation of that node
    """
    return self.G.nodes[key]
  
  def expand(self, fraction, reposition=False):
    """Expand the network and re-calculate the resistances.
    Parameters
    ----------
    fraction : How much strain to apply to the network. For example, calling
      rn.expand( 0.01 ) for a 2x2x2 RN will result in a 2.02^3 RN.
    reposition (bool) : Whether or not to expend the effort to adjust all the
      fiber positions.
    """
    # Summary: Loop through all edges and adjust res and cnd. Remove any edges
    #   that are too long now.
    # Also adjust the position of all the fibers accordingly. This step is
    #   optional, however, if those positions are not crucial.
    to_remove = []
    for edge in self.G.edges(data=True):
      edata = edge[2]
      x = self.inv_res_fun(edata["res"])
      x *= (1+fraction)
      if x > self.cnd_len:
        # If the expansion caused the length to go past the max cnd_len, 
        #   then remove the edge.
        to_remove.append(edge)
      else:
        newR = self.res_fun(x)
        edata["res"] = newR
        edata["cnd"] = 1/newR
    self.G.remove_edges_from(to_remove)

    if reposition:
      util.sim_log("401. Fiber repositioning on expansion not yet implemented.")

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

  def apply_v(self, v_in, n0, n1, set_v=True, set_p=True, set_i=False):
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
    
    # Find the indices of the pins used
    n0i = self.key_to_index(n0)
    n1i = self.key_to_index(n1)
    
    if self.sol_method in util.NL_methods:
      # Adjacency Matrix for conductance is used instead of the Laplacian
      #   when solving the nonlinear problem.
      Adj = nx.linalg.graphmatrix.adjacency_matrix(self.G, weight="cnd",
        nodelist=self.node_list)
      # TODO: Also use the other paremeters for the N-K method.
      # TODO: Keep track of if it succeeded or not.
      vrb = 2#TMP
      opt = {"verbose" : vrb,
        "xtol" : self.xtol,
        "ftol" : self.ftol}
      # OLD: v = util.NL_Axb(Adj, b, w=self.res_w, method=self.sol_method, opt=opt)
      sol = util.NL_sol(Adj, self.res_w, v_in, n0i, n1i, xi="Lcg",
        method=self.sol_method, opt=opt)
      #sol = util.NL_sol(Adj, self.res_w, v_in, n0i, n1i, xi="L_custom_32",
      #  method=self.sol_method, opt=opt)
      #sol = util.NL_sol(Adj, self.res_w, v_in, n0i, n1i, xi="L_hybr_3",
      #  method=self.sol_method, opt=opt)
      v = util.ainsrt(sol.x, [(n0i, v_in), (n1i, 0)])[0:-1]
      v = np.array(v, dtype=np.float) #Convert back to np
      I_in = sol.x[-1]
      I_in = np.float(I_in) #Convert back to np
      util.sim_log(848, v[n0i], v[n1i], I_in)
      Req = v_in / I_in
    else:
      # Get the Laplacian matrix for conductance
      Lpl = nx.linalg.laplacian_matrix(self.G, weight="cnd",
        nodelist=self.node_list)
      v, Req, code = util.L_sol(Lpl, v_in, n0i, n1i, method=self.sol_method)
    
    # Save the voltages in the nodes
    if set_v:
      self.applied_v = v_in
      node_i = 0
      for node_key in self.node_list:
        self.G.nodes[node_key]["v"] = v[node_i]
        node_i += 1
    else:
      # If not saving v in each node, just return the vector and the Req.
      return v, Req
    
    if set_p or set_i:
      p_max_e = 0 # Max power in an edge
      e_p_max = None # Corresponding edge
      p_max_n = 0 # Max power in a node
      n_p_max = None # Corresponding node
      # Start by setting power to 0 in each node
      nx.set_node_attributes(self.G, 0, name="p")
      if set_i:
        nx.set_node_attributes(self.G, 0, name="i")
        nx.set_node_attributes(self.G, 0, name="isnk")
      for edge in self.G.edges(data=True):
        en0 = edge[0]
        en1 = edge[1]
        ei0 = self.key_to_index(en0)
        ei1 = self.key_to_index(en1)
        dv = float(v[ei1] - v[ei0])
        G = edge[2]["cnd"]
        
        # Calculate & save current
        if set_i:
          if self.sol_method in util.NL_methods:
            i = util.NL_I(G, self.res_w, dv)
          else:
            i = G * dv
          self.G[en0][en1]["i"] = abs(i)
          self.G.nodes[en0]["i"] += abs(i)
          self.G.nodes[en1]["i"] += abs(i)
          self.G.nodes[en0]["isnk"] += i
          self.G.nodes[en1]["isnk"] -= i
        
        # Calculate edge power
        if self.sol_method in util.NL_methods:
          p = util.NL_P(G, self.res_w, dv)
        else:
          p = G * dv**2
        # Save this edge power value
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

  def get_maxp_nodes(self, n=1, relative=True, eps_pf=0.01):
    """Find the node with the highest power.
    Before running this function, apply_v should have been run with set_p=True
    
    Parameters
    ----------
    n (int) : Number of nodes to return
    relative (bool) : Whether to consider the highest relative power.
      ie p_rel = p / bpwr
      This better reflects which nodes we expect to burn first.
    eps_pf (float) : If eps_pf > 0 is given, then be sure to burn out any nodes
      that are within eps power of the lowest power node that would normally be
      burned. "_pf" is for Power Fraction, since it's a fraction of that power.
    
    Returns
    -------
    mp_nodes : RN id of the max power nodes
    mp_nodes_data : the node data for those nodes
    """
    
    # If eps_pf>0, then sort more than n fibers
    N = max(4*n+100, int(self.size()/10)) if eps_pf>0 else n
    max_ps = np.zeros(N)
    mp_nodes = [None] * N
    mp_nodes_data = [None] * N
    # This code populates mp_nodes and mp_nodes_data with the 
    #   N highest power nodes, sorted.
    for node in self.G.nodes:
      n_data = self.G.nodes[node].copy()
      n_p = n_data["p"] # Assumes apply_v(set_p) already done
      if relative:
        nbpwr = n_data["bpwr"] if "bpwr" in n_data else self.brn_pwr
        # Adjust the actual node power to a "relative power" value.
        # This means power as a fraction of the node burn power.
        n_p = n_p / nbpwr
      if n_p > np.min(max_ps):
        # Now place this node in its correct, sorted position
        i = np.count_nonzero(max_ps > n_p)
        max_ps[i+1:] = max_ps[i:-1] # Shift
        max_ps[i] = n_p # Insert
        mp_nodes[i+1:] = mp_nodes[i:-1] # Shift
        mp_nodes[i] = node # Insert
        mp_nodes_data[i+1:] = mp_nodes_data[i:-1] # Shift
        mp_nodes_data[i] = n_data # Insert
    if eps_pf > 0:
      #bpwr_n = mp_nodes_data[n-1]["bpwr"]
      bpwr_n = max_ps[n-1]
      eps_bpwr = bpwr_n * (1-eps_pf)
      # Cut out those without enough power
      burn_i = np.count_nonzero(max_ps > eps_bpwr)
      mp_nodes = mp_nodes[0:burn_i]
      mp_nodes_data = mp_nodes_data[0:burn_i]
      #util.db_print(f"1015: eps_burn caught {burn_i-n} extra fibers")
    #util.sim_log(997, mp_nodes, mp_nodes_data)
    return mp_nodes, mp_nodes_data

  def burn(self, to_burn="p_max"):
    """Remove the fibers that have too much power flowing through them.
    Before running this function, apply_v should have been run with set_p=True
    
    Parameters
    ----------
    to_burn : Specify which fibers to burn. If set to "p_max", then it will
      burn all fibers with p > self.brn_pwr.
      If a list is passed, then those fibers will be burned.
      If an integer is passed, then that number of nodes will be burned,
        starting at the highest power nodes.
      For example, if set to 1, then the max power node will be burned.
    
    Returns
    -------
    fibers_burned (int) : How many fibers were removed.
    """
    
    if to_burn == "p_max":
      util.db_print("Removing high-power fibers")
      #util.db_print("Hopefully apply_v(set_p=True) has been done already...")
      to_burn = []
      for node in self.G.nodes:
        brn_pwr = (self.G.nodes[node]["bpwr"]
          if "bpwr" in self.G.nodes[node]
          else self.brn_pwr)
        #brn_pwr = (self.brn_pwr if (self.bpwr_span == 0) 
        #  else self.G.nodes[node]["bpwr"])
        if self.G.nodes[node]["p"] > brn_pwr:
          to_burn.append(node)
      util.db_print(f"to_burn = {to_burn}") # TEMP
    if isinstance(to_burn, int):
      if to_burn < 1:
        return 0
      to_burn = self.get_maxp_nodes(to_burn)[0]
    if None in to_burn:
      # Either the network is totally broken or apply_v needs to be called.
      util.db_print("Not enough nodes with power in them were found.")
      return 0
    for node in to_burn:
      self.remove_node(node)
    fibers_burned = len(to_burn)
    util.db_print(f"{fibers_burned} fibers were removed")
    return fibers_burned

  def get_maxp_edges(self, n=1, relative=True, eps_pf=0.01):
    """Find the edge with the highest power.
    Before running this function, apply_v should have been run with set_p=True
    
    Parameters
    ----------
    n (int) : Number of edges to return
    relative (bool) : Whether to consider the highest relative power.
      ie p_rel = p / bpwr
      This better reflects which edges we expect to burn first.
    eps_pf (float) : If eps_pf > 0 is given, then be sure to burn out any edges
      that are within eps power of the lowest power edge that would normally be
      burned. "_pf" is for Power Fraction, since it's a fraction of that power.
    
    Returns
    -------
    mp_edges : the max power edges with data
    """
    
    # If eps_pf>0, then sort more than n fibers
    N = max(4*n+100, int(self.size()/10)) if eps_pf>0 else n
    max_ps = np.zeros(N)
    mp_edges = np.array( [(None,None,{})] * N )
    # This code populates mp_edges with the 
    #   N highest power edges, sorted.
    for edge in self.G.edges(data=True):
      e_p = edge[2]["p"] # Assumes apply_v(set_p) already done
      if relative:
        ebpwr = edge[2]["bpwr"] if "bpwr" in edge[2] else self.brn_pwr
        # Adjust the actual edge power to a "relative power" value.
        # This means power as a fraction of the edge burn power.
        e_p = e_p / ebpwr
      if e_p > np.min(max_ps):
        # Now place this edge in its correct, sorted position
        i = np.count_nonzero(max_ps > e_p)
        max_ps[i+1:] = max_ps[i:-1] # Shift
        max_ps[i] = e_p # Insert
        mp_edges[i+1:] = mp_edges[i:-1] # Shift
        mp_edges[i] = edge # Insert
    if eps_pf > 0:
      # This is the lowest power that would be burned if it weren't for eps_pf
      bpwr_n = max_ps[n-1]
      eps_bpwr = bpwr_n * (1-eps_pf)
      # Cut out those without enough power
      mp_edges = mp_edges[max_ps > eps_bpwr]
      #util.db_print(f"1015: eps_burn caught {burn_i-n} extra fibers")
    return mp_edges

  def edge_burn(self, to_burn="p_max"):
    """Remove the edges that have too much power flowing through them.
    Before running this function, apply_v should have been run with set_p=True
    Same method as rn.burn(), but for edges, not fibers
    
    Parameters
    ----------
    to_burn : Specify which edges to burn. If set to "p_max", then it will
      burn all edges with p > self.brn_pwr.
      If a list is passed, then those edges will be burned.
      If an integer is passed, then that number of nodes will be burned,
        starting at the highest power nodes.
      For example, if set to 1, then the max power node will be burned.
    
    Returns
    -------
    edges_burned (int) : How many edges were removed.
    """
    
    if to_burn == "p_max":
      util.db_print("Removing high-power edges")
      #util.db_print("Hopefully apply_v(set_p=True) has been done already...")
      to_burn = []
      for edge in self.G.edges(data=True):
        brn_pwr = (edge[2]["bpwr"]
          if "bpwr" in edge[2]
          else self.brn_pwr)
        if edge[2]["p"] > brn_pwr:
          to_burn.append(edge)
      util.db_print(f"to_burn = {to_burn}") # TEMP
    if isinstance(to_burn, int):
      to_burn = self.get_maxp_edges(to_burn)
    if None in to_burn:
      # Either the network is totally broken or apply_v needs to be called.
      util.db_print("Not enough nodes with power in them were found.")
      return 0
    self.G.remove_edges_from(to_burn)
    edges_burned = len(to_burn)
    util.db_print(f"{edges_burned} edges were removed")
    return edges_burned

  def draw(self, fig=None, ax=None, width_attrib=None, color_attrib=None,
    edge_color="b", to_mark=[], annotate_is=False):
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
    annotate_is : whether to annotate with the current flowing in or out of
      the pins
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
    if width_attrib is not None:
      lw_max = 20#*self.plt_ln_width # max line width
      atrb_max = np.max(list( nx.get_edge_attributes(self.G, 
        width_attrib).values() ))
      #util.sim_log(1093, lw_max / atrb_max)
      lw_mlt = min(lw_max / atrb_max, 1e6)
      #util.sim_log(1094, lw_mlt)
    
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
            # Note: this has the safety step of re-converting to np.array
            #   because it's currently not loading from json 100% correctly
            edge_pos = np.array(edge[2]["pos"])
          except: pass #Is this needed?
        if width_attrib is None:
          lw = self.plt_ln_width
        else:
          lw = edge[2][width_attrib] * lw_mlt
        if self.dim == 2:
          ax.plot(edge_pos[:,0], edge_pos[:,1], color=edge_color, linewidth=lw,
            solid_capstyle="butt", zorder=1.5)
        else:
          ax.plot3D(edge_pos[:,0], edge_pos[:,1], edge_pos[:,2],
            color=edge_color, linewidth=lw, solid_capstyle="butt", zorder=1.5)
    
    if not color_attrib is None:
      node_color = nx.get_node_attributes(self.G, color_attrib)
      #node_color = np.array(list(node_color.values()))
      clip_vs = True # TEMP setting
      if clip_vs and color_attrib == "v":
        # Clip unrealistic voltages
        for nid in node_color.keys():
          node_color[nid] = max(node_color[nid], 0)
          vmax = self.applied_v if hasattr(self, "applied_v") else 1 #Default
          node_color[nid] = min(node_color[nid], vmax)

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
        #fiber_color = nx.get_node_attributes(self.G, color_attrib)
        fiber_color = node_color
        #Cut out the pins from the color list
        [fiber_color.pop(pin[0]) for pin in self.pins]
        fiber_color = np.vstack(list(fiber_color.values()))
        if self.dim == 3:
          # 2 Experimental features
          RGBA = False
          COLOR_BUBBLES = True
          if RGBA:
            # Convert to RGBA
            # I did this to be able to set the alpha value lower
            fiber_color -= np.min(fiber_color)
            fiber_color = fiber_color / (np.max(fiber_color))
            fiber_color = np.hstack(( .1*np.ones(np.shape(fiber_color)), 
              fiber_color, .1*np.ones(np.shape(fiber_color)), 
              0.25*np.ones(np.shape(fiber_color)) ))
            util.sim_log(697, fiber_color, np.min(fiber_color), np.max(fiber_color))
          # Quiver 3D doesn't support multiple colors, so I'd need a work-around
          util.sim_log("Specifying fiber color is not yet implemented in 3D, "
            "so this is a work-around")
          if COLOR_BUBBLES:
            # Since Quiver 3D doesn't support varying color, my work-around
            #   plan is to make a background with transparent blobs of color
            ax.scatter(pos[:,0], pos[:,1], pos[:,2], s=15*self.plt_node_size,
              c=fiber_color, zorder=0.5)
      
      if self.dim == 2:
        # Use quiver to show the fibers
        ax.quiver(pos[:,0], pos[:,1], fv[:,0], fv[:,1], fiber_color, angles="xy",
          scale_units="xy", scale=1, headlength=0, headwidth=0, headaxislength=0, 
          capstyle="butt", width=.0075, zorder=2)
      else:
        ax.quiver(pos[:,0], pos[:,1], pos[:,2], fv[:,0], fv[:,1], fv[:,2],
          color=self.pin_color, arrow_length_ratio=0, capstyle="butt", 
          linewidths=self.plt_ln_width, zorder=2)
    else:
      pos = np.array(list(pos.values()))
      # Make the color list
      if color_attrib is None:
        node_color = np.ones(len(pos))
      else:
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
      atr_ls = list(node_color.values())
      #atr_ls = list(nx.get_node_attributes(self.G, color_attrib).values())
      sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=min(atr_ls), 
        vmax=max(atr_ls)))
      plt.colorbar(sm, ax=ax, fraction=.075, shrink=.6)
      # If to_mark == "p_max", then place a red X over high-power fibers
      if to_mark == "p_max":
        to_mark = []
        for node in self.G.nodes:
          brn_pwr = (self.G.nodes[node]["bpwr"]
            if "bpwr" in self.G.nodes[node]
            else self.brn_pwr)
          #brn_pwr = (self.brn_pwr if (self.bpwr_span == 0) 
          #  else self.G.nodes[node]["bpwr"])
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
      if annotate_is:
        # Assumes that apply_v(set_i=True) was done already
        # Write the current entering or exiting at each pin
        i = self.G.nodes[pin[0]]["isnk"]
        if pin in self.in_pins:
          i *= -1 # Since the arrows will be pointing in, it's current sourced
        i = f"{i:.6f}"
        yt = 0.51 + pin[1][2] * .23/self.zmax # convert to 2d figure fraction
        bbox = dict(boxstyle="rarrow", fc="#99b8ff")
        xt = .79 if pin in self.out_pins else 0.21
        if color_attrib is not None:
          xt -= .05 # The colorbar shifts everything left
        ax.annotate(i, xy=(0,0), xytext=(xt,yt),
          textcoords="figure fraction", 
          bbox=bbox, fontsize="x-large")
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
      # TODO: zorder? - doesn't work in 3d?
      a3 = ax3d.plot_wireframe(x, y, z, color=color, linewidths=1.0)
      #linewidths=self.plt_ln_width
    else:
      # alpha controls opacity
      a3 = ax3d.plot_surface(x, y, z, color=color, alpha=0.3)

  def train(self, N_in, N_out, max_burns=10, callback=None):
    """Train the network so that it produces the desired output with the given input.
    
    Parameters
    ----------
    N_in (str) : A binary string that represents the input. Each bit that is
      a 1 corresponds to a high pin.
    N_out (str) : A binary string that represents the desired output. Each bit
      that is a 1 corresponds to a high pin.
    max_burns (int) : The max number of burns that will be run.
    callback (function) : A function to be called after each burn
      accepts (int) current burn number
    
    Returns
    -------
    code (int) : Success indicator code
      0 - no training was required
      1 - training proceeded normally and was successful
      2 - training proceeded normally but was unsuccessful
      3 - at least one of the connections is very weak (i.e. a pin burned out)
      4 - at least one connection is weak and some current is flowing backwards
      5 - A significant amount of current is flowing backwards
    burns (int) : How many burns were carried out
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
    # 5: Repeat 1-4 until max_burns is reached or desired output is achieved.
    
    burn = 0
    code = 0
    
    # List the ids of those input pins that are high
    #in_pin_ids = self.in_pin_ids(N_in)
    # List the ids of the output pins
    #out_pin_ids = self.out_pin_ids()
    
    while burn < max_burns:
      # Only reset the threshold the first time: reset_threshold=(burn==0)
      #output, currents = self.fwd_pass(N_in, burn==0) # -- old way
      output, currents = self.fwd_pass(N_in, N_out, reset_threshold=None, 
        shift_threshold=True)
      # If a callback was specified, call it
      # This is before burning so that the input and output currents
      #   reflect the appropriate input, not the burn current.
      # It is based on these currents that the decision is made of whether to
      #   continue burning or not.
      if callback != None:
        callback(burn)
      
      # Check to see if there is current flowing the wrong way
      if np.min(currents) < 0:
        util.db_print("Warning: Current is flowing backwards into the RN.")
        code = 5
      # Check to see if any connections have been burned out entirely.
      if np.min(np.abs(currents)) < 1e-6:
        util.db_print("Warning: At least one connection is very weak.")
        code = 4 if code == 5 else 3
      if code > 2:
        return code, burn
      
      # If needed, burn out bad connections
      is_done = self.bwd_pass(N_in, N_out, output, currents=currents)
      if is_done:
        # This happens when no burning was necessary, since it was already done
        # code = 0 or 1 here
        return code, burn
      
      burn += 1
      code = 1
    
    return 2, burn

  def fwd_pass(self, N_in, N_out=None, reset_threshold=None,
    shift_threshold=False):
    """Run a forward pass with the specified input number
    
    Parameters
    ----------
    N_in (str) : A binary string that represents the input. Each bit that is
      a 1 corresponds to a high pin.
    N_out (str) : A binary string that represents the desired output. Only
      required for shift_threshold.
    reset_threshold (None, "max", "min") : Whether or not to reset the 
      threshold and if it should be a fraction of the min output current or
      a fraction of the max output current. At the start of training it could
      make sense to reset the threshold so all the pins register as high,
      using the "min" command. The "max" command could be used to determine a 
      good threshold for a finished RN.
    shift_threshold (bool) : Whether or not to shift the threshold such that
      all the desired outputs are high. If True, N_out must be provided.
    
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
    self.set_voltages(self.fpV, in_pin_ids, out_pin_ids)
    # Find the current flowing out of each output pin
    currents = self.sum_currents(out_pin_ids)
    # Shift the threshold so that all the desired outputs are high
    if shift_threshold:
      assert N_out != None
      # Convert N_out to booleans
      out_bools = util.bin_to_bools(N_out)
      self.threshold = self.threshold_fraction * np.min(currents[out_bools])
    # Optionally reset the threshold
    if reset_threshold == "min":
      # Reset the threshold so all the pins are high
      self.threshold = self.threshold_fraction * np.min(currents)
    if reset_threshold == "max":
      # Reset the threshold to a fraction of the highest output
      self.threshold = self.threshold_fraction * np.max(currents)
    
    output = self.threshold_currents(currents)
    #util.db_print(f"output={output}")
    # Threshold those currents into a binary result
    return output, currents

  def bwd_pass(self, N_in, N_out, output, currents=None):
    """Perform a backwards pass to burn out undesired connections.
    
    Parameters
    ----------
    N_in (str) : A binary string that represents the input. Each bit that is
      a 1 corresponds to a high pin.
    N_out (str) : A binary string that represents the desired output.
    output (array of bool) : The thresholded output of a recently performed
      forward pass.
    currents (array of float) : The output currents from the last forward
      pass. Not used at all yet, but could be used to determine which pin is
      the most erroneous.
    
    Returns
    -------
    is_done (bool) : Whether or not the training is already done, without
      needing any burning.
    """

    # List the ids of those input pins that are high
    in_pin_ids = self.in_pin_ids(N_in)
    # List the ids of the output pins
    out_pin_ids = self.out_pin_ids()
    # Convert N_out to booleans
    out_bools = util.bin_to_bools(N_out)
    # If the output is already correct, then exit
    if np.all(output == out_bools):
      return True
    
    # How to decide how many fibers to burn at each pin?
    #   If threshold_fraction=0.90 for example:
    #   I<=90% : err=0; I=100% : err=.5; I>=110% : err=1
    #self.threshold = self.threshold_fraction * np.min(currents[out_bools])
    # Current corresponding to max error.
    #   In the below example, it's 110%
    I_maxe = self.threshold * (2/self.threshold_fraction - 1)
    # Error Magnitude to determine how many fibers to burn
    err_mag = (currents - self.threshold) / (I_maxe - self.threshold)
    err_mag[err_mag < 0] = 0
    err_mag[err_mag > 1] = 1
    # Make a list of connections to be weakened (#3)
    # list of pins that are too high
    err_pins = out_pin_ids[output > out_bools]
    # Subset of the previous error magnitude list
    burn_mag = err_mag[output > out_bools]
    to_burn = []
    for i in range(len(err_pins)):
      to_burn.append( ([err_pins[i]], in_pin_ids, burn_mag[i]) )
    #for err_pin in err_pins:
    #  # Note: this runs the current backwards, from outputs to inputs.
    #  to_burn.append( ([err_pin], in_pin_ids) )
    
    # Burn across each of those connections (#4)
    for p_in, p_out, burn_fraction in to_burn:
      burn_rate = int(np.ceil(burn_fraction*self.burn_rate))
      #util.db_print(f"Burning from {p_in} to {p_out}")
      self.set_voltages(self.bpV, p_in, p_out)
      if self.burn_fibers:
        self.burn(burn_rate) # Burn the N highest power nodes (N=burn_rate)
      else:
        self.edge_burn(burn_rate)
    
    return False

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
      in_indices = util.bin_to_bools(N_in)
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

  def connect_pins(self, connectedPins, label):
    self.add_node(label)
    edges = []
    for i in connectedPins:
      # Caution, making this too small makes the linear algebra very hard.
      small_res = self.pin_res / 4
      edges.append([label, i, {'res' : small_res, 'cnd' : 1/small_res}])
    self.G.add_edges_from(edges)
    return label

  def set_voltages(self, voltage, startingNodes, endingNodes):
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
    What it really finds is the excess current that enters but doesn't leave,
      as far as we can tell, so it represents charge storage in the node,
      which violates KCL. This value is positive for output pins because the
      temporary pin representing ground has been taken away. So in summary,
      this returns how much current is being sinked by the given nodes.
    
    Parameters
    ----------
    nodes : a list of node ids
    
    Returns
    -------
    currents : a list of currents
    """

    """ OLD VERSION:
    currents = np.zeros(len(nodes))
    it = 0
    for n in nodes:
      voltage = self.G.nodes[n]["v"]
      for u, v in self.G.edges(n): # Note, u = n
        # This is using Ohm's law right now!
        currents[it] += (self.G.nodes[v]["v"] - voltage) * self.G[u][v]["cnd"]
      it += 1
    
    return currents
    """

    # Use this method, which correctly handles nonlinearity
    currents, sinking = self.through_currents(ret_sink=True, nodes=nodes)
    return np.array(list(sinking.values()))

  def through_currents(self, ret_sink=False, store=False, nodes=None):
    """Add up to find how much current is passing through every fiber in the
    RN. Similar to sum_currents method.
    
    Parameters
    ----------
    ret_sink (bool) : whether to return the amount of current being sinked
      at each fiber as well. That number is the KCL error and should only be
      significant at the pins, hopefully.
    store (bool) : whether to save the current as a node and edge attribute.
    nodes (list of ids) : list of nodes to analyze. default=None --> All.
    
    Returns
    -------
    currents (dict) : the current flowing through each node
    [sinking] (digt) : the current being absorbed / sinked by each node.
      Only returned if ret_sink == True
    """
    nonlinear = self.sol_method in util.NL_methods
    voltages = nx.get_node_attributes(self.G, "v")
    if len(voltages) == 0:
      util.db_print("Warning: voltage has not been stored in this RN.")
    sub_voltages = voltages.copy()
    if nodes is not None:
      nodes = list(nodes) # k in np.array is not well defined
      # Take a subset of the nodes
      sub_voltages = {key: val for key, val in sub_voltages.items() 
        if key in nodes}
    # Ugly way to initialize empty dictionaries of the same size...
    currents = sub_voltages.copy()
    sinking = sub_voltages.copy()
    for node in sub_voltages.keys():
      currents[node] = 0
      sinking[node] = 0
    
    if store:
      if nodes is not None:
        util.db_print("Warning: saving the current in only a subset of the network")
      nx.set_node_attributes(self.G, 0, name="i")
      nx.set_node_attributes(self.G, 0, name="isnk")
    for edge in self.G.edges(data=True, nbunch=nodes):
      en0 = edge[0]
      en1 = edge[1]
      dv = float(voltages[en1] - voltages[en0])
      G = edge[2]["cnd"]
      if nonlinear:
        i = util.NL_I(G, self.res_w, dv)
      else:
        i = G * dv
      # Only worry about the nodes in the list
      if en0 in currents:
        currents[en0] += abs(i)
        sinking[en0] += i
      if en1 in currents:
        currents[en1] += abs(i)
        # if i>0, then v1>v0, so n1 is the source & n0 is the sink
        sinking[en1] -= i
      if store:
        self.G[en0][en1]["i"] = abs(i)
        self.G.nodes[en0]["i"] += abs(i)
        self.G.nodes[en1]["i"] += abs(i)
        self.G.nodes[en0]["isnk"] += i
        self.G.nodes[en1]["isnk"] -= i

    #for n, v in voltages.items():
    #  sum_in = 0
    #  sum_out = 0
    #  for n0, n1 in self.G.edges(n): # Note, n0 = n
    #    ii = (self.G.nodes[n1]["v"] - v) * self.G[n0][n1]["cnd"]
    #    if ii > 0:
    #      sum_in += ii
    #    elif ii < 0:
    #      sum_out -= ii
    #  sinking[n] = sum_in - sum_out # This is what sum_currents() returns
    #  currents[n] = max(sum_in, sum_out)
    if ret_sink:
      return currents, sinking
    else:
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

  def size(self):
    return self.G.number_of_nodes()

  def get(self):
    return list(self.G.nodes)
