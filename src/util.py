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
import optax

# List of strings (in lowercase) that are considered equivalent to True
true_strs = ("true", "t", "yes", "y", "on", "1")
# List of nonlinear solver methods
NL_methods = ("n-k", "hybr", "trf", "mlt", "custom", "optax-adam")
# Debugging boolean - to be set from cp.getboolean("exec", "debug")
debug = False
timing = False
# Options relevant to the sim log file
log_fname = "log_TEMP.txt"
use_stdout = False

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

def ainsrt2(x, i0, v0, i1, v1):
  """Insert a series of values at the specified indices into the given array.
  insertions : [(i0, v0), (i1, v1)]
  Like ainsrt, but specifically for inserting 2 values and compatible with jit.
  """
  N = x.size+2
  x01 = jnp.concatenate([ x[0:i0], v0, x[i0:i1-1], v1, x[i1-1:] ], axis=None)[0:N]
  x10 = jnp.concatenate([ x[0:i1], v1, x[i1:i0-1], v0, x[i0-1:] ], axis=None)[0:N]
  # Essentially it's an if statement here
  return jnp.where(i0 < i1, x01, x10)

def sim_log(*strings):
  """Record s in the sim log file
  """
  with open(log_fname, "a") as log_file:
    for s in strings:
      log_file.write(str(s) + " ")
    log_file.write("\n")

def db_print(*strings):
  """Custom wrapper / modification of the standard print()
  If "debug" setting is True:
    Prints to the sim log file
    If use_stdout is True, also prints to the screen with print()
  """
  if debug:
    sim_log(*strings)
    for s in strings:
      if use_stdout:
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
      sim_log(msg, "time:", t)

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
      sim_log("Please format N_in and N_out as binary")
      return None
    except KeyError:
      sim_log("Please include desired inputs and outputs")
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
  #db_print(f"p1: {p1}, v1: {v1}, CPt1: {CPt1}")
  #db_print(f"p1 + v1*CPt1 = {p1 + v1*CPt1}, CPdist: {CPdist}")
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
  
  #db_print(f"55: {CR_pts}")
  
  # Find the min, of the 9 critical points
  dmin = 1e6
  pmin = (0,0)
  for p in CR_pts:
    if p[2] < dmin:
      if p[0] >= 0 and p[0] <= 1 and p[1] >= 0 and p[1] <= 1:
        pmin = (p[0], p[1])
        dmin = p[2]
  
  return (dmin, pmin)

def parabola_opt(fx0, fx1, fx2):
  """Find the root of a parabola running through the points
  (0, fx0), (1,fx1), (2,fx2)
  """
  xc = (3*fx0 - 4*fx1 + fx2) / (2*fx0 - 4*fx1 + 2*fx2)
  #db_print(316, fx0, fx1, fx2, xc)
  if jnp.isnan(xc):
    return 1
  eps = 1e-6
  return max(min(xc, 2-eps), eps)

def line_min(f, x0, fx0, dx, mlt=1, fx2=None, nfev=0, fev_max=16):
  """Find the best step size multiplier to apply to dx
  """
  fx1 = f(x0 + dx*mlt)
  nfev += 1
  #db_print(323, fx0, fx1)
  if fx1 >= fx0:
    if nfev < fev_max:
      return line_min(f, x0, fx0, dx, mlt=.5*mlt, fx2=fx1, nfev=nfev)
    else:
      return .5*mlt, nfev
  if fx2 is None:
    fx2 = f(x0 + dx*mlt*2)
    nfev += 1
  if fx2 >= fx1:
    # mlt is in (0,2)
    return (parabola_opt(fx0, fx1, fx2))*mlt, nfev
  # mlt > 1
  fn = [fx0, fx1, fx2]
  i = 2
  while fn[i] < fn[i-1]:
    i += 1
    #fn[i] =
    fn.append( f(x0 + dx*mlt*i) )
    nfev += 1
    #db_print(337, fn[i])
  # The optimum is between the last three values
  return (i-2 + parabola_opt(*fn[-3:]))*mlt, nfev

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

def sum_node_I(v, A, w, ni):
  """Like RN.sum_currents, return the current sinked by the given node
  The difference is that this operates with the adjacency matrix.
  """
  dv = v[ni] - v
  i_in = - NL_I(A[ni,:], w, dv)
  return jnp.sum(i_in)

# These calculate the residuals for a given x vector in the NL system
def NL_res_j(x, A, w, v_in, ni_in, ni_out):
  """NL_res, but compatible with jit.
  Also take note of the changed order of arguments
  """
  N = x.size + 2
  v = ainsrt2(x, ni_in, v_in, ni_out, 0)[0:-1]
  vcols, vrows = jnp.meshgrid(v, v)
  sinharg = w * (vrows - vcols) # Argument to be 'sinh()'ed
  Asinh = jnp.multiply(A, jnp.sinh(sinharg))
  sumI = Asinh.sum(1) / w # sum of currents leaving each node
  res = sumI.at[ni_in].add(- x[-1])
  res = res.at[ni_out].add(x[-1])
  return res#[1:] # Throw one equation out
def NL_res_old(A, w, v_in, ni_in, ni_out, x):
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
      If "Lcg", then find an xi by solving the linear system with cg
      If "L_{nonlin_method}_{N} then solve sequentially for better xi-s
        by gradually increasing the nonlinearity N times.
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

  times = tic()

  if "verbose" not in opt:
    opt["verbose"] = debug
  if type(opt["verbose"]) != int:
    opt["verbose"] = 1 if opt["verbose"] else 0
  if opt["verbose"] > 2:
    db_print(f"Starting Nonlinear Solver: {method}")
  N = A.shape[1]-1 # len(x) = N_nodes + 1 - 2
  if xi is None:
    xi = jnp.zeros(N).at[ni_in].set(v_in)
  elif isinstance(xi, str):
    assert xi[0] == "L", "483: Unknown xi option"
    # Get xi from the linear system, solving with cg
    L = L_from_A(A)
    vL, RL, status = L_sol(L, v_in, ni_in, ni_out, tol=1e-7)
    xL = np.append(vL, v_in / RL) # convert v to x
    xL = np.delete(xL, [ni_in, ni_out])
    toc(times, "Lcg")
    if xi == "Lcg":
      xi = jnp.array(xL)
    else:
      # Presolving with smaller w. Format: "L_method_N"
      xi_parts = xi.split("_") 
      xi = jnp.array(xL)
      N = int(xi_parts[2])
      ximethod = xi_parts[1]
      assert ximethod in NL_methods
      # Make N simpler versions of the system to solve
      for i in range(N):
        wi = w * ( (i+1)/(N+1) )**2
        soli = NL_sol(A, wi, v_in, ni_in, ni_out, xi=xi, method=ximethod, 
          opt={"ftol":5e-5})
        #db_print(soli.__dict__)
        xi = soli.x
      db_print(503, "Done with presolving")
      toc(times, f"xi presolving, method={ximethod}")
  else:
    xi = jnp.array(xi)
  if hasattr(A, "toarray"):
    # Convert to jax array
    A = jnp.array(A.toarray())
  ubound = v_in * np.ones(N)
  ubound[-1] = np.inf # No bound on current
  bounds = (np.zeros(N)-1e-6, ubound+1e-6)

  # Create jit compiled versions of the function
  jres = jax.jit(NL_res_j, static_argnums=[2,3,4,5])
  res = lambda x: jres(x, A, w, v_in, ni_in, ni_out)
  jjac = jax.jit(jax.jacfwd(jres), static_argnums=[2,3,4,5])
  jac = lambda x: jjac(x, A, w, v_in, ni_in, ni_out)

  #toc(times, "jit")
  rxi = res(xi)
  db_print(f"||res(xi)||: {jnp.linalg.norm(rxi)}")
  toc(times, "res(xi)")
  if jnp.linalg.norm(rxi) < 1e-8: #TMP: replace with tol
    # If xi is already within tolerance, we're done
    def sol(): pass
    sol.x = xi
    sol.fun = rxi
    return sol

  if method == "mlt":
    # Chain multiple solvers
    #ltol = 5e-3
    xtol = opt["xtol"] if "xtol" in opt else 1e-5
    ftol = opt["ftol"] if "ftol" in opt else 1e-3
    # List of solvers and tolerances
    methods = [
      #("hybr", xtol*10),
      ("hybr", xtol*100),
      ("hybr", xtol),
      ("trf", xtol/16),
      ("trf", xtol/256),
      ("trf", xtol/4096)]
      #("custom", stol)]
      #("Lcg", 1e-7),
      #("hybr", ltol),
      #("trf", 10*stol),
      #("hybr", stol)]#,
      #("trf", stol),
      #("n-k", stol)]
    sol = NL_mlt(A, w, v_in, ni_in, ni_out, xi, methods, ftol=ftol, opt=opt)
    return sol

  elif method == "optax-adam":
    # Scalar conversion
    loss = lambda x,A,w,v_in,ni_in,ni_out: jnp.sum(jnp.square(NL_res_j(x, A,
      w, v_in, ni_in, ni_out)))
    jloss = jax.jit(loss, static_argnums=[2,3,4,5])
    jgrad = jax.jit(jax.grad(loss, 0), static_argnums=[2,3,4,5])
    jlx = lambda x: jloss(x,A,w,v_in,ni_in,ni_out) #"jlx" = Jitted Loss (x)
    jgx = lambda x: jgrad(x,A,w,v_in,ni_in,ni_out)
    # Parameters
    options = {
      "lrn_rate": 1e-3,
      "maxit": 4000
    }
    sol = NL_adam(jlx, jgx, xi, options)
    toc(times, f"Optax Adam solver (w={w})")

  elif method == "custom":
    options = {
      "maxit": 400,
      "xtol": opt["xtol"] if "xtol" in opt else 1e-10,
      "rtol": opt["ftol"] if "ftol" in opt else 1e-4,
      "leash": 8,
      "momentum": 0.1
    }
    sol = NL_custom_N(res, jac, xi, options)
    toc(times, f"Custom Newton solver (w={w})")

  elif method == "n-k":
    #residual = lambda x: NL_res(A, w, v_in, ni_in, ni_out, x)
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
    sol = spo.root(res, xi, method="krylov", options=nkopt)
  elif method == "hybr":
    # Make it square for hybr
    res = lambda x: jres(x, A, w, v_in, ni_in, ni_out)[1:]
    jac = lambda x: jjac(x, A, w, v_in, ni_in, ni_out)[1:]
    #res_jac = lambda x: NL_resjac(A, w, v_in, ni_in, ni_out, x)
    xtol = opt["xtol"] if "xtol" in opt else 1e-3
    if "tol" in opt:
      xtol = opt["tol"]
    hopt = {"xtol" : xtol,
      "maxfev": 1000000,
      "factor": 1}
    # Scale the variable for current so it's more significant
    hopt["diag"] = np.ones(N)
    hopt["diag"][-1] = N
    #db_print(f"583: xi={xi}")
    #sol = spo.root(res_jac, xi, method="hybr", jac=True, options=hopt)
    sol = spo.root(res, xi, method="hybr", jac=jac, options=hopt)
    toc(times, f"hybr (w={w})")
    #db_print(f"||res(sol.x)||: {jnp.linalg.norm(res(sol.x))}")
    #toc(times, "res(sol.x)")
  elif method == "trf":
    #res_jac = lambda x: NL_resjac(A, w, v_in, ni_in, ni_out, x)
    # The least_squares method doesn't take an options argument.
    #   Instead, expand options like this: **trf_opt
    #   "verbose" is an option in least_squares {0,1,2}
    trf_opt = {
      "bounds": bounds,
      "x_scale": "jac" # IDK about this...
    }
    # Scale the variable for current so it's more significant
    #opt["x_scale"] = np.ones(N)
    #opt["x_scale"] = N/10
    trf_opt["verbose"] = opt["verbose"] if "verbose" in opt else 0
    if "tol" in opt:
      trf_opt["xtol"] = opt["tol"]
    if "xtol" not in trf_opt:
      trf_opt["xtol"] = 1e-6
    # This ftol refers to SSR cost, not ||res||
    # Actually, it's relative (dF / F), not absolute cost
    trf_opt["ftol"] = opt["ftol"] if "ftol" in opt else 1e-3
    trf_opt["ftol"] = .5 * trf_opt["ftol"]**2 * 10 # *10 arbitrary
    trf_opt["gtol"] = opt["gtol"] if "gtol" in opt else 1e-10
    # Make sure xi is feasible
    db_print(617, len(xi[xi<0]))
    xi = xi.at[xi<0].set(0)
    db_print(619, len(xi[xi>v_in]))
    Ii = xi[-1]
    xi = xi.at[xi>v_in].set(v_in)
    xi = xi.at[-1].set(Ii) # x[-1] has no upper bound
    xi = xi.flatten()
    sol = spo.least_squares(res, xi, jac=jac, method="trf", **trf_opt)

  rxf = res(sol.x)
  db_print(f"||res(xf)||: {jnp.linalg.norm(rxf)}")
  toc(times, "NL_sol", total=True)
  return sol

def NL_adam(jlx, jgx, xi, options):
  """Adam solver from optax
  """
  # Set up optax
  params = {'x': xi}
  opt = optax.adam(options["lrn_rate"])
  state = opt.init(params)

  # Optimize
  it = 0
  while it < options["maxit"]:
    if it%10 == 0:
      rx = jlx(params['x'])
      gx = jgx(params['x'])
      #db_print(f"Iteration {it}: x[-5:]={params['x'][-5:]}, loss={rx}")
      db_print(f"Iteration {it};\tloss={rx:.6f};\tmax(grad)={jnp.max(gx):.6f}")
    grads = {'x': jgx(params['x'])}
    updates, state = opt.update(grads, state)
    params = optax.apply_updates(params, updates)
    it += 1

  def sol(): pass
  sol.x = params['x']
  sol.loss = jlx(sol.x)
  sol.grad = jgx(sol.x)
  sol.it = it
  sol.message = "MESSAGE STUB" # TMP
  sol.success = True # TMP
  return sol

def NL_custom_N(res, jac, xi, options):
  """Custom solver, using a modified version of Newton's method
  """
  # Deal with generic tol
  if "tol" in options:
    options["xtol"] = options["tol"]
  err = 1
  rtol = options["rtol"] if "rtol" in options else 1e-4
  nstep = 1
  nstep_2avg = 1
  xtol = options["xtol"] if "xtol" in options else 1e-9
  # How far it can go astray without resetting back to last min
  leash = options["leash"] if "leash" in options else 16
  nfev = 0
  it = 0
  maxit = options["maxit"] if "maxit" in options else 500
  fnorm = lambda x: jnp.linalg.norm(res(x))
  # This makes the newton steps more like acceleration than velocity
  momentum = options["momentum"] if "momentum" in options else 0.2
  laststep = jnp.zeros(xi.shape)
  x1 = xi
  r1 = res(x1)
  err = jnp.linalg.norm(r1)
  itsince = 0
  xm_used = False # Has this xm already been reset back to?
  xm = x1 # x_min - best x so far
  rm = r1
  errm = jnp.linalg.norm(rm)
  while(it < maxit and err > rtol and nstep_2avg > xtol):
    J1 = jac(x1)
    step, *_ = jnp.linalg.lstsq(J1, -r1)
    # Find the best step along that line
    step_mlt, lm_fev = line_min(fnorm, x1, err, step) # Simple 1D optimization
    db_print(536, step_mlt, lm_fev)
    nfev += lm_fev
    step *= step_mlt
    step += laststep*momentum # Apply momentum
    laststep = step
    x1 = x1 + step
    r1 = res(x1)
    nfev += 1
    err = jnp.linalg.norm(r1)
    if err < errm:
      xm = x1
      rm = r1
      errm = err
      itsince = 0
      xm_used = False
    else:
      itsince += 1
      if itsince > leash:
        if xm_used:
          db_print(f"No improvement in the last {leash} iterations since"
            " the last reset")
          # We're in a loop. No improvement has occurred since last reset
          break
        # Reset back to xm
        x1 = xm
        r1 = rm
        err = errm
        laststep = jnp.zeros(xi.shape)
        itsince = 0
        xm_used = True
        db_print("Reset back to x_min")

    it += 1
    last_nstep = nstep
    nstep = jnp.linalg.norm(step)
    # Use a running average to allow occasional tiny steps
    nstep_2avg = (nstep + last_nstep) / 2
    #dJ = jnp.linalg.det(J1)
    nJs = jnp.linalg.norm( jnp.dot(J1, step) )
    db_print(f"At it#{it}/{maxit}, err={err}, ||step||={nstep},"# |J|={dJ},"
      f" ||J*step||={nJs}")
  def sol(): pass
  sol.x = xm
  sol.fun = rm
  sol.err = errm
  sol.it = it
  sol.message = "MESSAGE STUB" # TMP
  sol.nfev = nfev
  sol.success = True # TMP
  return sol

def NL_adpt(A, w, v_in, ni_in, ni_out, xi, ftol, opt, rxi=None):
  """Adaptive method that keeps trying more careful solvers until
  it reaches ftol. Also considers KCL error.
    The following two values must be < ftol
      - ||res||
      - ptp(i_out, i_in, x[-1]) -- This represents KCL error as well as how
        accurate the current variable x[-1] is
  NOT FINISHED, NOT USED. I'll probably stick with NL_mlt
  """

  if rxi is None:
    # Initial r
    rxi = np.linalg.norm(NL_res_j(xi, A, w, v_in, ni_in, ni_out))
  method = "hybr" # Fastest, but less reliable sometimes
  tol = 10*ftol
  opt["tol"] = tol
  while True:
    if opt["verbose"]:
      db_print(783, f"Running method {method} with tol={tol}")
    sol = NL_sol(A, w, v_in, ni_in, ni_out, xi=xi, method=method, opt=opt)
    if opt["verbose"]:
      db_print(582, sol.message, sol.nfev)#, sol.x[-8:])
    # See if sol is an improvement and if it's good enough alredy
    if not sol.success:
      db_print("Warning: the solver did not converge")
      # I may still want to use the result if it's better than before
    #if sol.success:
    rx1 = np.linalg.norm(sol.fun)
    if opt["verbose"]:
      db_print(f"ADPT step: Previous r={rxi:.8f}. New r={rx1:.8f}")
    if rx1 < ftol: #sol.x is already good enough
      return sol
    if rx1 < rxi: #sol.x is better than the previous xi
      rxi = rx1
      xi = sol.x
  # NOT FINISHED
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

  # Initial r
  rxi = np.linalg.norm(NL_res_j(xi, A, w, v_in, ni_in, ni_out))
  for mtd, tol in methods:
    if opt["verbose"]:
      db_print(579, f"Running method {mtd} with tol={tol}")
    #if mtd == "Lcg": # Linear, cg
    #  v1, R1, status = L_sol(L, v_in, ni_in, ni_out, tol=tol)
    #  x1 = np.append(v1, v_in / R1) # convert v to x
    #  x1 = np.delete(x1, [ni_in, ni_out])
    #  def sol(): pass # Make empty sol object
    #  sol.success = (status == 0)
    #  sol.message = ""
    #  sol.nfev = -1
    #  sol.x = x1
    #  sol.fun = NL_res_j(x1, A, w, v_in, ni_in, ni_out)
    if mtd in NL_methods:
      opt["tol"] = tol
      #opt["ftol"] = ftol
      sol = NL_sol(A, w, v_in, ni_in, ni_out, xi=xi, method=mtd, opt=opt)
    else:
      db_print("Error: Unknown method")
    if opt["verbose"]:
      db_print(582, sol.message, sol.nfev)#, sol.x[-8:])
    # See if sol is an improvement and if it's good enough alredy
    if not sol.success:
      # I may still want to use the result if it's better than before
      # However, when you start from a bad starting place, it's worse.
      db_print("Warning: the solver did not converge")
    if sol.success:
      rx1 = jnp.linalg.norm(sol.fun)
      v = ainsrt2(sol.x, ni_in, v_in, ni_out, 0)[0:-1]
      i_in = - sum_node_I(v, A, w, ni_in)
      i_out = sum_node_I(v, A, w, ni_out)
      KCL_err = jnp.ptp(jnp.array( (i_in, i_out, sol.x[-1]) ))
      db_print(854, i_in, i_out, sol.x[-1], rx1)
      err = max(KCL_err, rx1)
      if opt["verbose"]:
        db_print(f"MLT step: Previous err={rxi:.8f}. New err={err:.8f}")
      if err < ftol: #sol.x is already good enough
        return sol
      if err < rxi: #sol.x is better than the previous xi
        rxi = err
        xi = sol.x
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
      #db_print(218, np.max(x), np.min(x))
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
      #db_print(201, "nfev", sol.nfev)
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
    #db_print(638, xi[-12:])
    #yell = lambda xk : db_print(635, xk[-12:])#(Lpl.dot(xk)-b)[-10:])
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

  db_print(f"p0: {p0}, p1: {p1}")
  db_print(f"v0: {v0}, v1: {v1}")
  dmin, pmin = min_LS_dist(p0, p1, v0, v1)#, 1)
  db_print(f"Min dist = {dmin} @{pmin}")

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

  db_print(f"pt: {pt}")
  db_print(f"p: {p}")
  db_print(f"v: {v}")
  dmin, tmin = LS_pt_dist(pt, p, v)
  db_print(f"Min dist = {dmin} @{tmin}")

  input("DONE.")

#LS_dist_test(2)
#LS_pt_dist_test(2)
