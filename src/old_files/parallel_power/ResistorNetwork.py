import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

# Class to represent a resistor network
# Also tracks input and output pins
class ResistorNetwork:
  node_r = 0.15 # Radius for the nodes
  pin_r = 0.25 # Radius for the pins
  # Plot options
  plt_node_size = 7#50 #default is 300, which is pretty big.
  plt_width = 0.2 #Line Width
  # Network Resistance options
  pin_res = 1e-4 #Resistance btw pins and the nodes they intersect with. It's an arbitrarily low resistance.
  node_res = 1000 #Coefficient multiplier for resistance calculation btw nodes
  
  # Limits: [xmin, xmax, ymin, ymax(, zmin, zmax)]
  # pins: [("pin0", x0, y0), ("pin1", x1, y1), ...]
  def __init__(self, num_nodes=500, limits=[0,1, 0,1], pins=[], pts=None,
      pin_r=None, node_r=None, rand_s=2):
    self.num_nodes = num_nodes
    self.xmin = limits[0]
    self.xmax = limits[1]
    self.ymin = limits[2]
    self.ymax = limits[3]
    self.pins = pins
    self.pts = pts # For specifying custom positions of nodes
    if pin_r != None:
      self.pin_r = pin_r
    if node_r != None:
      self.node_r = node_r
    # dictionary to hold the indices of the pins
    self.pin_indices = {}
    # 2-dimensional unless 6 limits are provided
    self.dimension = 2
    if len(limits) == 6:
      self.dimension = 3
      self.zmin = limits[4]
      self.zmax = limits[5]
    # Generate the Graph
    np.random.seed(rand_s)
    self.G = self.create_graph()
  
  def create_graph(self):
    if self.pts is None:
      # Generate a matrix of random points (x,y) in the domain
      self.pts = np.random.rand(self.num_nodes, 2) * [self.xmax-self.xmin, 
        self.ymax-self.ymin] + [self.xmin, self.ymin]
    
    node_keys_it = range(len(self.pts))
    G = nx.Graph()
    
    # Add the pins
    for pin in self.pins:
      # Store the index of the pin in the overall matrix
      self.pin_indices[pin[0]] = len(G)
      # Add the pin with the provided key and location
      G.add_node(pin[0], pos=pin[1:3])
    
    # Add the nodes
    for i in node_keys_it:
      # The key for each node will simply be its index in the pts array
      G.add_node(i, pos=self.pts[i])
    
    # Find and add the edges between the nodes
    for i in node_keys_it:
      #loop through the remaining edges
      for j in range(i+1,len(self.pts)):
        dist = np.sqrt(np.sum(np.square(self.pts[i] - self.pts[j])))
        res = self.node_res*dist*dist
        if dist < self.node_r:
          #cnd = Conductance = 1/R
          G.add_edge(i, j, res=res, cnd=1/res)
    
    # Add the edges for the pins
    for pin in self.pins:
      for pt_i in node_keys_it:
        dist = np.sqrt(np.sum(np.square(self.pts[pt_i] - pin[1:3])))
        res = self.pin_res
        if dist < self.pin_r:
          G.add_edge(pin[0], pt_i, res=res, cnd=1/res)
    return G
  
  # Given a node key, convert to the index in the matrix
  def key_to_index(self, key):
    if isinstance(key, str):
      return self.pin_indices[key]
    # Since the pins are added first, the numbered nodes should be offest by that much in the Graph
    return key + len(self.pin_indices)
  
  # Reverse of key_to_index.
  def index_to_key(self, i):
    pin_is = self.pin_indices.values()
    #print(f"96: {pin_is}, {i}")
    if i in pin_is: # Then i represents a pin
      pin_keys = self.pin_indices.keys()
      return list(pin_keys)[list(pin_is).index(i)]
    return i - len(self.pin_indices)
  
  # Resistance from node 0 to node 1
  # n0 and n1 can be either pins or fibers
  def R_nn(self, n0, n1):
    return self.R_ij(self.key_to_index(n0), self.key_to_index(n1))
  
  # Resistance from pin 0 to pin 1
  # This uses the names of the pins, not their indices
  def R_pp(self, p0, p1):
    return self.R_ij(self.pin_indices[p0], self.pin_indices[p1])
  
  # Resistance from node index i0 to i1
  # Calculation of resistance
  # https://mathworld.wolfram.com/ResistanceDistance.html
  def R_ij(self, i0, i1):
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
  
  # Apply a voltage across the two given pins/nodes
  # Save as attributes in each node the resulting voltage
  def apply_v(self, v_in, n0, n1, ground=None, set_p=True):
    # Default to the out pin being ground
    if ground is None:
      ground = n1
    # Switch this out so it's more efficient. This way solves the system twice.
    Req = self.R_nn(n0, n1)
    i_in = v_in / Req
    # Set up the b column in Av=b
    N = len(self.G)
    b = np.zeros((N, 1))
    b[self.key_to_index(n0)] = i_in
    b[self.key_to_index(n1)] = -i_in
    # Get the Laplacian matrix for conductance
    Lpl = nx.linalg.laplacian_matrix(self.G, weight="cnd")
    # Ground the given pin, overwriting the last row for the grounding equation
    # That's fine, since all the information is repeated.
    Lpl[N-1] = sps.csr_matrix((1,N))
    ground_i = self.key_to_index(ground)
    Lpl[N-1, ground_i] = 1
    b[N-1] = 0
    # Solve Av=b
    v = spsl.spsolve(Lpl, b)
    for i in range(len(v)):
      self.G.nodes[self.index_to_key(i)]["v"] = v[i]
    if set_p:
      p_max = 0
      e_p_max = None
      for edge in self.G.edges(data=True):
        n0 = edge[0]
        n1 = edge[1]
        i0 = self.key_to_index(n0)
        i1 = self.key_to_index(n1)
        # Calculate edge power
        dv = float(abs(v[i1] - v[i0]))
        R = edge[2]["res"]
        p = dv**2 / R
        # Save this power value
        self.G[n0][n1]["p"] = p
        # Save the square of power to be the line thickness when plotting
        self.G[n0][n1]["p2"] = p**2
        if p > p_max:
          p_max = p
          e_p_max = edge
    return p_max, e_p_max
  
  # Draw the network
  def draw(self, width_attrib=None, color_attrib=None, edge_color="k", 
    fig=None, ax=None):
    linewidth = self.plt_width
    if not width_attrib is None:
      linewidth = list(nx.get_edge_attributes(self.G, width_attrib).values())
    if color_attrib is None:      
      color = ["red"] * len(self.pins) 
      color.extend( ["blue"] * (len(self.G) - len(self.pins)) )
    else:
      color = list(nx.get_node_attributes(self.G, color_attrib).values())
    if ax is None:
      fig, ax = plt.subplots()
    
    ax.set_aspect("equal")
    pos = nx.get_node_attributes(self.G, 'pos')
    nx.draw(self.G, pos=pos, ax=ax, node_size=self.plt_node_size, 
      width=linewidth, node_color=color, edge_color=edge_color)
    #Draw circles for the pins
    for pin in self.pins:
      circle = plt.Circle(pin[1:3], radius=self.pin_r, color="red", fill=False)
      ax.add_artist(circle)
    return fig, ax
  
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
    return index, shortest
  
  # Remove a node between n0 and n1 that lies on the shortest path
  def eliminateNode(self, n0, n1):
    self.G.remove_node(self.longestNode(n0, n1))
  
