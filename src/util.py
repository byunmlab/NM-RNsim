""" Script that contains utility functions

  str2arr
  str2bool
  bin_to_bools : Convert a binary string to a list of booleans
  db_print
  tic & toc : Functions for timing and printing timing messages
  load_IO : Load an IO csv file
  IO_code : Evaluate a set of IO currents against the desired IO pattern
  LS_pt_dist : Return the minimum distance between a line segment and a point
  min_LS_dist : Return the minimum distance between two line segments
  NL_I : Calculate nonlinear current
  NL_P : Calculate nonlinear power
  NL_Axb : Solve the system sum( A.*sinh(wX) ) = b using a nonlinear solver

"""

import time
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.optimize as spo # N-K
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

# List of strings (in lowercase) that are considered equivalent to True
true_strs = ("true", "t", "yes", "y", "on", "1")
# List of nonlinear solver methods
NL_methods = ("n-k", "hybr", "trf", "mlt")
# Debugging boolean - to be set from cp.getboolean("exec", "debug")
debug = False
timing = False

def str2arr(s):
  """Converts a comma-separated list of floats to a np.array.
    e.g. : s="1, 2.5, 3.14, 4"
  """
  return np.array(s.split(","), dtype=float)

def str2bool(s):
  """Returns bool(True) if the string s represents True
  """
  return str(s).lower() in true_strs

def bin_to_bools(Nbin):
  """Convert a binary string to a list of booleans
  Nbin should have the normal format: "0bXXXX"
  """
  # This loops from the end to the 3rd character (so as to skip the "0b")
  return [ bool(int(i)) for i in Nbin[-1:1:-1] ]

def ainsrt(x, insertions):
  """Insert a series of values at the specified indices into the given array.
  insertions : [(i0, v0), (i1, v1), ...]
  """
  y = np.copy(x)
  i_s = [i for i,v in insertions]
  i_ss = np.sort(i_s)
  for i in i_ss:
    j = np.where(i_s == i)[0][0]
    n, v = insertions[j]
    y = np.insert(y, n, v)
  return y

def db_print(s):
  """Wrapper for standard print()
  Prints to screen if the "debug" setting is True
  """
  if debug:
    print(s)

def tic():
  """Start timing. Returns a list of times with one entry.
  """
  if timing:
    # List of times, for timing
    times = []
    times.append(time.time())
    return times
  return None

def toc(times, msg=None, total=False):
  """Log the current time in the times list.
  If msg is provided, print out a message with the time.
    the string f" time: {t}" will be appended to the message.
  If total, then print out the elapsed time since the start.
    Else, print out only the last time interval.
  """
  if timing:
    times.append(time.time())
    if msg is not None:
      t = times[-1] - times[0] if total else times[-1] - times[-2]
      print(msg, "time:", t)

def load_IO(IO_fname, npa=False):
  """Load the given IO csv file, checking for errors
  """
  # load the IO file
  IO_df = pd.read_csv(IO_fname, skipinitialspace=True)
  # Check that it has binary numbers stored
  for i, row in IO_df.iterrows():
    try:
      int(row["IN"], 2)
      int(row["OUT"], 2)
    except ValueError:
      print("Please format N_in and N_out as binary")
      return None
    except KeyError:
      print("Please include desired inputs and outputs")
      return None
  if npa:
    return IO_df.to_numpy()
  else:
    return IO_df

def IO_code(IO_Ns, currents, threshold_fraction=.9):
  """Evaluate a set of IO currents against the desired IO pattern
  
  Parameters
  ----------
  IO_Ns (Nx2 array) : IO pattern, loaded from an IO.csv file
  currents (Nx2 array) : IO currents, as returned from std_sims.fp_RMSR()
  
  Returns
  -------
  code : Use the same code as in RN.train(), but abbreviated.
    0 - RN is trained
    1 - [Not Used]
    2 - RN not yet trained
    3 - at least one of the connections is very weak (i.e. a pin burned out)
    4 - at least one connection is weak and some current is flowing backwards
    5 - A significant amount of current is flowing backwards
  """

  code = 0
  N_out_pins = len(IO_Ns[0][1].split("b")[1]) #Assumes 0bXXX format
  for i in range(len(IO_Ns)):
    Ni, No = IO_Ns[i]
    if int(Ni, 2) == 0:
      # Assumes no input bias
      assert int(No, 2) == 0, ("A high output cannot be generated from a zero"
        " input without implementing an input bias")
      assert np.max(np.abs(currents[i,:])) < 5e-6, ("It appears that current is"
        " spontaneously being generated within the RN")
      continue
    assert int(No, 2) > 0, ("If the input is nonzero, one of the output"
      "currents must be high, following the relative thresholding scheme")
    # First check if a pin is broken
    if np.min(currents[i]) < 0:
      code = 5
    if np.min(np.abs(currents[i])) < 5e-6:
      code = 4 if code == 5 else 3
    if code > 2:
      return code
    out_bools = bin_to_bools(No)
    # Fraction of the lowest high output
    # Note: this depends on knowing the desired output already
    threshold = threshold_fraction * np.min(currents[i, out_bools])
    #total_current = np.sum(currents[i])
    #for j in range(N_out_pins):
    if np.any( (currents[i,:] > threshold) != out_bools ):
      code = 2
  
  return code

def LS_pt_dist(pt, p, v, dmaxt=None):
  """Return the minimum distance between a line segment and a point.
  This is used for the distance between pins and fibers.
  Parameters
  ----------
  pt : location of point (list or np.array)
  p : start points for line segment (list or np.array)
  v : vector indicating direction and length (list or np.array)
  dmaxt : Max Distance Threshold. If it is obvious that the distance will be 
    greater than dmaxt, then return a the lower bound instead of calculating 
    the real min distance. This should speed the process up when fiber 
    length << dmaxt.
  Returns
  -------
  dmin : min. distance.
  tmin : normalized distance along the line segment where the min distance 
    occurs. 0 < tmin < 1
  """
  pt = np.hstack(pt)
  p = np.hstack(p)
  v = np.hstack(v)
  dp = pt - p
  L = np.linalg.norm(v)
  
  if dmaxt:
    # Check if they're obviously too far apart
    # Minimum separation distance, if the fibers are optimally oriented
    dmin = np.linalg.norm(dp) - L
    if dmin > dmaxt:
      return (dmin, 0)
    # Else, there is a chance that the min dist < dmaxt
  
  uv = v / L
  t = np.dot(dp, uv)
  if t < 0:
    d = np.linalg.norm(dp)
    t = 0
  elif t > L:
    d = np.linalg.norm(dp - v)
    t = 1
  else:
    d = np.linalg.norm(np.cross(dp, uv))
    t /= L
  return (d, t)

def min_LS_dist(p0, p1, v0, v1, dmaxt=None):
  """Return the minimum distance between two line segments.
  This function is an optimization minimizing distance on the domain, 
    which is a 2D parameterized space along the two segments.
  Parameters
  ----------
  p0, p1 : start points (list or np.array)
  v0, v1 : vectors indicating direction and length (list or np.array)
  dmaxt : Max Distance Threshold. If it is obvious that the distance will be 
    greater than dmaxt, then return a the lower bound instead of calculating 
    the real min distance. This should speed the process up when 
    (2 * fiber length + dmaxt) < RN size. 
    In my tests, it made the process take ~1/3 as long.
  Returns
  -------
  dmin : min. distance.
  pmin : point in (t0, t1) at which that min dist occurs. 0 < t0,t1 < 1
  """
  p0 = np.hstack(p0)
  p1 = np.hstack(p1)
  v0 = np.hstack(v0)
  v1 = np.hstack(v1)
  
  if dmaxt:
    # Check if they're obviously too far apart
    max_reach = np.linalg.norm(v0) + np.linalg.norm(v1)
    # Minimum separation distance, if the fibers are optimally oriented
    dmin = np.linalg.norm(p1 - p0) - max_reach
    if dmin > dmaxt:
      return (dmin, (0,0))
    # Else, there is a chance that the min dist < dmaxt
  
  # Optimization to find min dist.
  CR_pts = [None]*9
  # Find the distance btw two points along the line segments
  # t0 and t1 are parameterizations along v0 and v1.
  def dist_t0t1(t0, t1):
    #dist = sqrt(dx**2 + dy**2 + dz**2)
    dpos = p1 + v1*t1 - p0 - v0*t0
    return np.sqrt(np.sum(dpos**2))
  
  # Find the critical point where the gradiant of dist(t0, t1) is zero
  SSv0 = np.sum(v0**2)
  SSv1 = np.sum(v1**2)
  v0dotv1 = np.dot(v0, v1)
  dp = p1 - p0
  dpdotv0 = np.dot(dp, v0)
  dpdotv1 = np.dot(dp, v1)
  # Critical point from where the gradiant is zero 
  #   (after substituting back in the dot products and such)
  CPt0 = (SSv1*dpdotv0 - v0dotv1*dpdotv1)/(SSv0*SSv1 - v0dotv1**2)
  CPt1 = (-SSv0*dpdotv1 + v0dotv1*dpdotv0)/(SSv0*SSv1 - v0dotv1**2)
  CPdist = dist_t0t1(CPt0, CPt1)
  #print(f"p1: {p1}, v1: {v1}, CPt1: {CPt1}")
  #print(f"p1 + v1*CPt1 = {p1 + v1*CPt1}, CPdist: {CPdist}")
  CR_pts[0] = (CPt0, CPt1, CPdist)
  
  # Check all the 4 boundries of the unit square
  t1_t00 = -dpdotv1/SSv1 #critical t1 when t0=0
  t1_t01 = (-dpdotv1 + v0dotv1)/SSv1 #critical t1 when t0=1
  t0_t10 = dpdotv0/SSv0 #critical t0 when t1=0
  t0_t11 = (dpdotv0 + v0dotv1)/SSv0 #critical t0 when t1=1
  CR_pts[1] = (0, t1_t00, dist_t0t1(0, t1_t00))
  CR_pts[2] = (1, t1_t01, dist_t0t1(1, t1_t01))
  CR_pts[3] = (t0_t10, 0, dist_t0t1(t0_t10, 0))
  CR_pts[4] = (t0_t11, 1, dist_t0t1(t0_t11, 1))
  
  # Check the 4 corners
  CR_pts[5] = (0, 0, dist_t0t1(0, 0))
  CR_pts[6] = (0, 1, dist_t0t1(0, 1))
  CR_pts[7] = (1, 0, dist_t0t1(1, 0))
  CR_pts[8] = (1, 1, dist_t0t1(1, 1))
  
  #print(f"55: {CR_pts}")
  
  # Find the min, of the 9 critical points
  dmin = 1e6
  pmin = (0,0)
  for p in CR_pts:
    if p[2] < dmin:
      if p[0] >= 0 and p[0] <= 1 and p[1] >= 0 and p[1] <= 1:
        pmin = (p[0], p[1])
        dmin = p[2]
  
  return (dmin, pmin)

def NL_I(G, w, v):
  """Simple calculation of current in the nonlinear system.
  Parameters
  ----------
    G : conductivity (iv slope @v=0)
    w : the coefficient on the argument for sinh(wx)
    v : the voltage drop
  """
  return G/w * np.sinh(w*v)
def NL_P(G, w, v):
  """Simple calculation of power (p=vi) in the nonlinear system.
  Parameters
  ----------
    G : conductivity (iv slope @v=0)
    w : the coefficient on the argument for sinh(wx)
    v : the voltage drop
  """
  i = NL_I(G, w, v)
  return v*i

# These two calculate the residuals for a given x vector in the NL system
def NL_res(A, w, v_in, ni_in, ni_out, x):
  # Find the residual from the nonlinear verson of KCL
  # Recover the voltage vector
  v = ainsrt(x, [(ni_in, v_in), (ni_out, 0)])[0:-1]
  vcols, vrows = np.meshgrid(v, v)
  sinharg = w * (vrows - vcols) # Argument to be 'sinh()'ed
  Asinh = A.multiply(np.sinh(sinharg)).toarray() # A .* sinh()
  sumI = Asinh.sum(1) / w # sum of currents leaving each node
  KCLres = sumI# - np.hstack(b)
  KCLres[ni_in] -= x[-1]
  KCLres[ni_out] += x[-1]
  r = KCLres[1:] # Throw out one KCL equation
  # Extra voltage equations
  #v_eqs = np.array([x[ni_in]-v_in, x[ni_out]])
  #veq_scale = len(x) #/ 10
  #return np.concatenate((KCLres, veq_scale*v_eqs))
  return r
def NL_resjac(A, w, v_in, ni_in, ni_out, x):
  # Returns the residual and the Jacobian for the given x
  # Find the residual from the nonlinear verson of KCL
  # See OneNote > Nonlinear system
  # Recover the voltage vector
  v = ainsrt(x, [(ni_in, v_in), (ni_out, 0)])[0:-1]
  vcols, vrows = np.meshgrid(v, v)
  sinharg = w * (vrows - vcols) # Argument to be 'sinh()'ed
  Asinh = A.multiply(np.sinh(sinharg)).toarray() # A .* sinh()
  sumI = Asinh.sum(1) / w # sum of currents leaving each node
  KCLres = sumI# - np.hstack(b)
  KCLres[ni_in] -= x[-1]
  KCLres[ni_out] += x[-1]
  r = KCLres[1:] # Throw out one KCL equation
  # Now find the Jacobian (See OneNote > Jacobian of NL)
  Acosh = A.multiply(np.cosh(sinharg)).toarray()
  KCLJ = np.diag(Acosh.sum(1)) - Acosh
  # Add a column for the I variable (x[-1])
  IO_is = np.zeros((len(v), 1))
  IO_is[ni_in] = -1
  IO_is[ni_out] = 1
  J = np.concatenate((KCLJ, IO_is), axis=1)
  J = J[1:, :] # Throw out one KCL equation
  # Delete the columns for v_in and v_out
  J = np.delete(J, [ni_in, ni_out], axis=1)
  # Add two rows for the new equations about v_in & v_out
  #Jsup = np.zeros((2, len(x)))
  #Jsup[0,ni_in] = 1*veq_scale
  #Jsup[1,ni_out] = 1*veq_scale
  #J = np.concatenate((J, Jsup), axis=0)
  return r, J

def NL_sol(A, w, v_in, ni_in, ni_out, xi=None, method="hybr", opt={}):
  """Solve the nonlinear system.
  Parameters
  ----------
    A (NxN scipy matrix) : the Adjacency Matrix of the RN
    w (float) : the coefficient on the argument for sinh(wx)
      This can be related to the 3rd derivative of the i-v curve at zero.
    v_in (float) : the input voltage
    ni_in (int) : the index of the input node [0,N-1]
    ni_out (int) : the index of the output node [0,N-1]
    xi (N+1 np array) : initial guess for the solution. If None, xi=0
      Maybe providing this could speed it up a lot. Idk.
    method ("n-k", "hybr", "trf", "mlt") : the solution method 
      see util.NL_methods
      (n-k and hybr are suggested, but there's also trf)
      "mlt" (multiple) means first try hybr, then trf or nk if that fails.
    opt (dict) : solver options, to be passed on to the solver.
      May include a "verbose" option.
      "xtol" is important for hybr
  Returns
  -------
    sol (OptimizeResult) : the solver result
      Note: sol.x (N-1 np array) : the solution vector. 
      Note: x[-1] = I_in and x is missing v_in & v_out
  """

  if "verbose" not in opt:
    opt["verbose"] = debug
  if type(opt["verbose"]) != int:
    opt["verbose"] = 1 if opt["verbose"] else 0
  if opt["verbose"] > 2:
    db_print(f"Starting Nonlinear Solver: {method}")
  N = A.shape[1]-1 # len(x) = N_nodes + 1 - 2
  ubound = v_in * np.ones(N)
  ubound[-1] = np.inf # No bound on current
  bounds = (np.zeros(N)-1e-6, ubound+1e-6)
  if xi is None:
    #xi = np.zeros(A.shape[1]+1)
    xi = np.zeros(N)
    xi[ni_in] = v_in

  if method == "mlt":
    # Chain multiple solvers
    #ltol = 5e-3
    stol = opt["xtol"] if "xtol" in opt else 1e-4
    # List of solvers and tolerances
    methods = [("Lcg", 1e-7),
      #("hybr", ltol),
      #("trf", ltol),
      ("hybr", stol)]#,
      #("trf", stol),
      #("n-k", stol)]
    sol = NL_mlt(A, w, v_in, ni_in, ni_out, xi, methods, ftol=stol, opt=opt)
    return sol

  elif method == "n-k":
    residual = lambda x: NL_res(A, w, v_in, ni_in, ni_out, x)
    xtol = opt["xtol"] if "xtol" in opt else 1e-3
    ftol = opt["ftol"] if "ftol" in opt else 1e-2
    if "tol" in opt: # Generic tol
      xtol = opt["tol"]
    nkopt = {
      "disp": True if opt["verbose"] > 0 else False,
      "xtol": xtol,
      "ftol": ftol,
      "jac_options": {"rdiff": .05}
      }
    sol = spo.root(residual, xi, method="krylov", options=nkopt)
  elif method == "hybr":
    res_jac = lambda x: NL_resjac(A, w, v_in, ni_in, ni_out, x)
    xtol = opt["xtol"] if "xtol" in opt else 1e-3
    if "tol" in opt:
      xtol = opt["tol"]
    hopt = {"xtol" : xtol,
      "maxfev": 1000000,
      "factor": 1}
    # Scale the variable for current so it's more significant
    hopt["diag"] = np.ones(N)
    hopt["diag"][-1] = N
    sol = spo.root(res_jac, xi, method="hybr", jac=True, options=hopt)
  elif method == "trf":
    res_jac = lambda x: NL_resjac(A, w, v_in, ni_in, ni_out, x)
    # The least_squares method doesn't take an options argument.
    #   Instead, expand opt like this: **opt
    #   "verbose" is an option in least_squares {0,1,2}
    opt["bounds"] = bounds
    # Scale the variable for current so it's more significant
    opt["x_scale"] = "jac" # IDK about this...
    #opt["x_scale"] = np.ones(N)
    #opt["x_scale"] = N/10
    if "tol" in opt:
      opt["xtol"] = opt["tol"]
      del(opt["tol"])
    if "ftol" not in opt:
      opt["ftol"] = 1e-6
    if "xtol" not in opt:
      opt["xtol"] = 1e-6
    if "gtol" not in opt:
      opt["gtol"] = 1e-10
    last_x = None
    last_res = None
    last_jac = None
    # These functions only recalculate if needed
    # This mimics how the spo.root function handels jac=True
    def res(x):
      nonlocal last_res
      nonlocal last_jac
      nonlocal last_x
      if not np.all(x == last_x):
        last_res, last_jac = res_jac(x)
        last_x = x
      return last_res
    def jac(x):
      nonlocal last_res
      nonlocal last_jac
      nonlocal last_x
      if not np.all(x == last_x):
        last_res, last_jac = res_jac(x)
        last_x = x
      return last_jac
    xi = xi.flatten()
    sol = spo.least_squares(res, xi, jac=jac, method="trf", **opt)

  return sol

def NL_mlt(A, w, v_in, ni_in, ni_out, xi, methods, ftol, opt):
  """More convenient interface for NL_sol(method="mlt")
  Parameters
  ----------
    methods : list of methods and tolerances
    Example methods =
       [("Lcg", ltol),
        ("hybr", ltol),
        ("trf", ltol),
        ("hybr", stol),
        ("trf", stol),
        ("n-k", stol)]
    ftol (float) : Stopping tolerance for the residual r
  Returns
  -------
    sol (OptimizeResult) : the solver result
  """

  """ OLD CODE, used to be in NL_sol
  # Chain multiple solvers
  final_xtol = opt["xtol"] if "xtol" in opt else 1e-5
  rxi = np.linalg.norm(NL_res(A, w, v_in, ni_in, ni_out, xi))
  # Round 0: cg (linear version of system)
  L = L_from_A(A) #Laplacian
  v1, R1, status = L_sol(L, v_in, ni_in, ni_out, atol=1e-3)
  if status == 0:
    x1 = np.append(v1, v_in / R1)
    x1 = np.delete(x1, [ni_in, ni_out])
    #print(397, v1[-10:], x1[-10:])
    rx1 = np.linalg.norm(NL_res(A, w, v_in, ni_in, ni_out, x1))
    print(393, rxi, rx1)#xi[-20:], x1[-20:], rxi, rx1)
    if rx1 < rxi:
      xi = x1
      rxi = rx1
  # Round 1: large tol hybr
  opt["xtol"] = 1e-3
  sol = NL_sol(A, w, v_in, ni_in, ni_out, xi=xi, method="hybr", opt=opt)
  #print(400, xi[-20:], sol.x[-20:])
  print(401, sol.message, sol.nfev, rxi, np.linalg.norm(sol.fun))
  if sol.success and (np.linalg.norm(sol.fun) < final_xtol):
    return sol
  rx1 = np.linalg.norm(sol.fun) # sol.fun is NL_res(sol.x)
  if rx1 < rxi:
    xi = sol.x
    rxi = rx1
  # Round 2: large tol trf
  opt["xtol"] = 1e-4
  xi[xi<0] = 0 # Keep xi within bounds
  xi[xi>v_in] = v_in
  sol = NL_sol(A, w, v_in, ni_in, ni_out, xi=xi, method="trf", opt=opt)
  print(420, sol.message, sol.nfev, rxi, np.linalg.norm(sol.fun))
  if sol.success and (np.linalg.norm(sol.fun) < final_xtol):
    return sol
  rx1 = np.linalg.norm(sol.fun)
  if rx1 < rxi:
    xi = sol.x
    rxi = rx1
  # Round 3: small tol hybr
  opt["xtol"] = final_xtol
  sol = NL_sol(A, w, v_in, ni_in, ni_out, xi=xi, method="hybr", opt=opt)
  print(413, sol.message, sol.nfev, rxi, np.linalg.norm(sol.fun))
  if sol.success:
    return sol
  rx1 = np.linalg.norm(sol.fun)
  #print(405, rxi, rx1)
  if rx1 < rxi:
    xi = sol.x
    rxi = rx1
  # Round 4: small tol trf
  xi[xi<0] = 0 # Keep xi within bounds
  xi[xi>v_in] = v_in
  sol = NL_sol(A, w, v_in, ni_in, ni_out, xi=xi, method="trf", opt=opt)
  print(426, sol.message, sol.nfev, rxi, np.linalg.norm(sol.fun))
  if sol.success:
    return sol
  rx1 = np.linalg.norm(sol.fun)
  if rx1 < rxi:
    xi = sol.x
    rxi = rx1
  # Round 5: n-k
  sol = NL_sol(A, w, v_in, ni_in, ni_out, xi=xi, method="n-k", opt=opt)
  print(435, sol.message, sol.nfev, rxi, np.linalg.norm(sol.fun))
  return sol
  """

  # Initial r
  rxi = np.linalg.norm(NL_res(A, w, v_in, ni_in, ni_out, xi))
  for mtd, tol in methods:
    if opt["verbose"]:
      print(579, f"Running method {mtd} with tol={tol}")
    if mtd == "Lcg": # Linear, cg
      L = L_from_A(A) #Laplacian
      v1, R1, status = L_sol(L, v_in, ni_in, ni_out, tol=tol)
      x1 = np.append(v1, v_in / R1) # convert v to x
      x1 = np.delete(x1, [ni_in, ni_out])
      def sol(): pass # Make empty sol object
      sol.success = (status == 0)
      sol.message = ""
      sol.nfev = -1
      sol.x = x1
      sol.fun = NL_res(A, w, v_in, ni_in, ni_out, x1)
    elif mtd in NL_methods:
      opt["tol"] = tol
      sol = NL_sol(A, w, v_in, ni_in, ni_out, xi=xi, method=mtd, opt=opt)
    else:
      print("Error: Unknown method")
    if opt["verbose"]:
      print(582, sol.message, sol.nfev)#, sol.x[-8:])
    # See if sol is an improvement and if it's good enough alredy
    if not sol.success:
      print("Warning: the solver did not converge")
      # I may still want to use the result if it's better than before
    #if sol.success:
    rx1 = np.linalg.norm(sol.fun)
    if opt["verbose"]:
      print(587, f"Previous r={rxi:.8f}. New r={rx1:.8f}")
    if rx1 < ftol: #sol.x is already good enough
      return sol
    if rx1 < rxi: #sol.x is better than the previous xi
      rxi = rx1
      xi = sol.x
      # Note: this doesn't affect the final result...
      xi[xi<0] = 0 # Keep xi within bounds
      xi[xi>v_in] = v_in
  return sol

def NL_Axb(A, b, w=1, xi=None, method="n-k", opt={}):
  """OLD
  Solve the equation rowsum( A.*sinh(wX) ) / w = b
  When A is the Adjacency Matrix, this returns the voltage at each node
    if I = k*sinh(wV) represents the I-V curve at each junction
  Parameters
  ----------
    A (NxN scipy matrix) : the Adjacency Matrix of the RN
    b (Nx1 np array) : the current vector
    w (float) : the coefficient on the argument for sinh(wx)
      This can be related to the 3rd derivative of the i-v curve at zero.
    xi (Nx1 np array) : initial guess for the solution. If None, xi=0
      Maybe providing this could speed it up a lot. Idk.
    method ("n-k", "hybr") : the solution method 
      (n-k and hybr are suggested, but there's also trf)
    opt (dict) : solver options, to be passed on to the solver.
      May include a "verbose" option.
      "xtol" is important for hybr
  
  Returns
  -------
    x (np array) : the solution vector
  """

  if xi is None:
    xi = np.zeros((A.shape[1], 1))
  if "verbose" not in opt:
    opt["verbose"] = debug
  if type(opt["verbose"]) != int:
    opt["verbose"] = 1 if opt["verbose"] else 0
  if opt["verbose"] > 2:
    db_print(f"Starting Nonlinear Solver: {method}")

  if method == "n-k":
    # Function used by N-K
    def residual(x):
      # Find the residual from the nonlinear verson of KCL
      vcols, vrows = np.meshgrid(x, x)
      sinharg = w * (vrows - vcols) # Argument to be 'sinh()'ed
      Asinh = A.multiply(np.sinh(sinharg)).toarray() # A .* sinh()
      sumI = Asinh.sum(1) / w # sum of currents leaving each node
      return sumI - np.hstack(b)
    #, iter=16000
    # DOES THIS WORK?
    # I think rdiff may need to be passed in a dictionary
    sol = spo.root(residual, xi, method="krylov", verbose=opt["verbose"], rdiff=.05)

  elif method in ("hybr", "trf"):
    def res_jac(x):
      # Returns the residual and the Jacobian for the given x
      # Find the residual from the nonlinear verson of KCL
      # See OneNote > Nonlinear system
      #print(218, np.max(x), np.min(x))
      vcols, vrows = np.meshgrid(x, x)
      sinharg = w * (vrows - vcols) # Argument to be 'sinh()'ed
      Asinh = A.multiply(np.sinh(sinharg)).toarray() # A .* sinh()
      sumI = Asinh.sum(1) / w # sum of currents leaving each node
      r = sumI - np.hstack(b)
      # Now find the Jacobian (See OneNote > Jacobian of NL)
      Acosh = A.multiply(np.cosh(sinharg)).toarray()
      J = np.diag(Acosh.sum(1)) - Acosh
      return r, J
    if method == "hybr":
      if "xtol" not in opt:
        opt["xtol"] = 1e-3
      # The only relevant option
      hopt = {"xtol" : opt["xtol"]}
      #nkw["maxfev"] = 1e6
      sol = spo.root(res_jac, xi, method="hybr", jac=True, options=hopt)
      #print(201, "nfev", sol.nfev)
    elif method == "trf":
      # "verbose" is a solver option for trf
      if "ftol" not in opt:
        opt["ftol"] = 1e-3
      if "xtol" not in opt:
        opt["xtol"] = 1e-3
      if "gtol" not in opt:
        opt["gtol"] = 1e-5
      last_x = None
      last_res = None
      last_jac = None
      # These functions only recalculate if needed
      # This mimics how the spo.root function handels jac=True
      def res(x):
        nonlocal last_res
        nonlocal last_jac
        nonlocal last_x
        if not np.all(x == last_x):
          last_res, last_jac = res_jac(x)
          last_x = x
        return last_res
      def jac(x):
        nonlocal last_res
        nonlocal last_jac
        nonlocal last_x
        if not np.all(x == last_x):
          last_res, last_jac = res_jac(x)
          last_x = x
        return last_jac
      xi = xi.flatten()
      sol = spo.least_squares(res, xi, jac=jac, method="trf", **opt)

  if opt["verbose"] > 1:
    db_print(f"Solver message: {sol.message}")
    #db_print("Done solving")
  return sol.x

def L_from_A(A):
  """Create a Laplacian from an Adjacency matrix
  """
  D = sps.diags(np.array(A.sum(1).flatten())[0])
  return D - A

def L_sol(Lpl, v_in, ni_in, ni_out, method="cg", tol=1e-5):
  """Solve the linear version of the system
  Parameters
  ----------
    Lpl (sps matrix) : Laplacian matrix for conductance
    v_in (float) : Voltage in
    ni_in (int) : index of the input node (V+)
    ni_out (int) : index of the output node (V-)
    method {"cg", "spsolve"} : Which linear solver to use
    atol (float) : Absolute tolerance for the norm of the residuals in cg
  Returns
  -------
    v (np array) : voltage at every node
    Req : Equivalent resistance (v_in / I)
    status : Success code. 0=success.
  """

  atol = tol
  # Set up the b column in Lpl*v=b
  N = Lpl.shape[0]
  b = np.zeros((N, 1))
  # In reality, this should be the current flowing in. However, since
  # we need the Req to know that, use 1A for now, then adjust later.
  b[ni_in] = 1
  b[ni_out] = -1
  # LINEAR: Apply 1A, then scale linearly to whatever voltage is desired
  if method == "cg":
    xi = np.zeros(N)
    xi[ni_in] = v_in
    #print(638, xi[-12:])
    #yell = lambda xk : print(635, xk[-12:])#(Lpl.dot(xk)-b)[-10:])
    # Use scipy's conjugate gradient method, since Lpl is symmetric
    v, status = spsl.cg(Lpl, b, x0=xi, atol=atol)#, callback=yell)
    # Set the ground pin to zero, adjusting the answers accordingly
    v -= v[ni_out]
    # Check the exit status
    if status != 0:
      db_print(f"Conjugate Gradient Error: code {status}")
      if status > 0:
        db_print("\tthe desired tolerance was not reached")
      else:
        db_print("\t\"illegal input or breakdown\"")
  elif method == "spsolve":
    # Ground the given pin, overwriting the last row for the grounding equation
    # That's fine, since all the information is repeated.
    Lpl[N-1] = sps.csr_matrix((1,N))
    Lpl[N-1, gdi] = 1
    b[N-1] = 0
    # Solve Lpl*v=b
    v = spsl.spsolve(Lpl, b)
    status = 0
  elif method == "splsqr":
    RES = spsl.lsqr(Lpl, b)
    v = RES[0]
    status = RES[1]
    # Set the ground pin to zero, adjusting the answers accordingly
    v -= v[ni_out]
    # Check the exit status
    if status != 1:
      db_print("Error: lsqr did not confidently find a good solution")
      db_print(f"Exit info: {str(RES[1:-1])}")
      db_print("spsl.lsqr really shouldn't be used for this problem.")
    else:
      # Match with cg status success code
      status = 0
  # Scale the voltages linearly
  Req = v[ni_in] - v[ni_out] #This is true since I was 1A.
  voltage_mult = v_in / Req
  v *= voltage_mult
  return v, Req, status


# TESTS
def LS_dist_test(DIM):
  """A test of the line segment distance function
  """
  #p0 = 1.5 - np.random.rand(DIM)
  p0 = np.array([2.5, 1])
  #v0 = 1 - 2*np.random.rand(DIM)
  v0 = np.array([.5, .5])
  #p1 = 1.5 - np.random.rand(DIM)
  p1 = np.array([1.5, 1])
  #v1 = 1 - 2*np.random.rand(DIM)
  v1 = np.array([-.5, .5])

  if DIM == 3:
    # Adjust to 3d
    p0 = np.pad(p0, (0,1))
    v0 = np.pad(v0, (0,1))
    p1 = np.pad(p1, (0,1))
    v1 = np.pad(v1, (0,1))
    # Plot
    xs = [p0[0], p1[0]]
    ys = [p0[1], p1[1]]
    zs = [p0[2], p1[2]]
    us = [v0[0], v1[0]]
    vs = [v0[1], v1[1]]
    ws = [v0[2], v1[2]]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.quiver(xs, ys, zs, us, vs, ws)#, normalize=True)
    ax.axes.set_xlim3d(0,2)
    ax.axes.set_ylim3d(0,2)
    ax.axes.set_zlim3d(0,2)
    # Mark p0 with a green dot
    ax.scatter3D(p0[0], p0[1], p0[2], color="green")
    fig.show()
  elif DIM == 2:
    # Plot
    xs = [p0[0], p1[0]]
    ys = [p0[1], p1[1]]
    us = [v0[0], v1[0]]
    vs = [v0[1], v1[1]]
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.quiver(xs, ys, us, vs, scale=1, angles='xy', scale_units='xy')
    #ax.set_xlim([0,2])
    ax.set_ylim([0,2])
    # Mark p0 with a green dot
    ax.scatter(p0[0], p0[1], color="green")
    fig.show()

  print(f"p0: {p0}, p1: {p1}")
  print(f"v0: {v0}, v1: {v1}")
  dmin, pmin = min_LS_dist(p0, p1, v0, v1)#, 1)
  print(f"Min dist = {dmin} @{pmin}")

  input("DONE.")

def LS_pt_dist_test(DIM):
  """A test of the line segment to point distance function
  """
  pt = np.array([1.5, 1])
  p = np.array([1, 1])
  v = np.array([1, 1])
  
  if DIM == 3:
    # Adjust to 3d
    pt = np.pad(pt, (0,1))
    p = np.pad(p, (0,1))
    v = np.pad(v, (0,1))
    # Plot
    xs = [p[0]]
    ys = [p[1]]
    zs = [p[2]]
    us = [v[0]]
    vs = [v[1]]
    ws = [v[2]]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.quiver(xs, ys, zs, us, vs, ws)#, normalize=True)
    ax.axes.set_xlim3d(0,2)
    ax.axes.set_ylim3d(0,2)
    ax.axes.set_zlim3d(0,2)
    # Mark p with a green dot
    ax.scatter3D(p[0], p[1], p[2], color="green")
    # Mark pt with a red dot
    ax.scatter3D(pt[0], pt[1], pt[2], color="red")
    fig.show()
  elif DIM == 2:
    # Plot
    xs = [p[0]]
    ys = [p[1]]
    us = [v[0]]
    vs = [v[1]]
    fig, ax = plt.subplots()
    ax.quiver(xs, ys, us, vs, scale=1, angles='xy', scale_units='xy')
    ax.set_xlim([0,2])
    ax.set_ylim([0,2])
    # Mark p with a green dot
    ax.scatter(p[0], p[1], color="green")
    # Mark pt with a red dot
    ax.scatter(pt[0], pt[1], color="red")
    fig.show()

  print(f"pt: {pt}")
  print(f"p: {p}")
  print(f"v: {v}")
  dmin, tmin = LS_pt_dist(pt, p, v)
  print(f"Min dist = {dmin} @{tmin}")

  input("DONE.")

#LS_dist_test(2)
#LS_pt_dist_test(2)
