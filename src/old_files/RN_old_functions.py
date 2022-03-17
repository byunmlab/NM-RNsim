"""Old code from RN.py
"""
quit()

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

  # From here on, the functions are not being used. Mostly.
  def compareOutputs(self, binary, desired):
    """Returns outputs as to how they relate
       if they match                           = 0  #do nothing
       if they're high when they should be low = 1  #delete nodes
       if they're low when they should be high = -1 #no current fix
    possible solutions to the third case is to change the threshold on that pin, or to add a not gate on it
    """
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
    # Note, this doesn't do nonlinear power accurately
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

