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

"""

import time
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy.sparse as sps
import torch as tc

# List of strings (in lowercase) that are considered equivalent to True
true_strs = ("true", "t", "yes", "y", "on", "1")
# Debugging boolean - to be set from cp.getboolean("exec", "debug")
debug = False
timing = False
# Options relevant to the sim log file
log_fname = "log_TEMP.txt"
use_stdout = False
log_indent = 0

# Set up torch device
gpu = 7
device = tc.device(f"cuda:{gpu}" if tc.cuda.is_available() else "cpu")
#device = tc.device(f"cuda" if tc.cuda.is_available() else "cpu")
tc.cuda.set_device(device)

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

def timestamp():
  """Generate a timestamp
  """
  return time.strftime("%Y%m%dT%H%M%S")

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

def ainsrt2_jax(x, i0, v0, i1, v1):
  """Insert a series of values at the specified indices into the given array.
  insertions : [(i0, v0), (i1, v1)]
  Like ainsrt, but specifically for inserting 2 values and compatible with jit.
  """
  N = x.size+2
  # TODO: NOT OPTIMIZED
  x01 = jnp.concatenate([ x[0:i0], v0, x[i0:i1-1], v1, x[i1-1:] ], axis=None)[0:N]
  x10 = jnp.concatenate([ x[0:i1], v1, x[i1:i0-1], v0, x[i0-1:] ], axis=None)[0:N]
  # Essentially it's an if statement here
  return jnp.where(i0 < i1, x01, x10)

def ainsrt2(x, i0, v0, i1, v1):
  """Insert a series of values at the specified indices into the given array.
  insertions : [(i0, v0), (i1, v1)]
  Like ainsrt, but specifically for inserting 2 values and compatible with jit.
  """
  if not tc.is_tensor(x):
    x = tc.tensor(x, device=device)
  N = x.size(0)+2
  v0 = tc.tensor([v0], dtype=tc.float64, device=device)
  v1 = tc.tensor([v1], dtype=tc.float64, device=device)
  # TODO: NOT OPTIMIZED
  x01 = tc.cat([ x[0:i0], v0, x[i0:i1-1], v1, x[i1-1:] ])[0:N]
  x10 = tc.cat([ x[0:i1], v1, x[i1:i0-1], v0, x[i0-1:] ])[0:N]
  # Essentially it's an if statement here
  condition = tc.tensor(i0 < i1, device=device)
  return tc.where(condition, x01, x10)

def sps_to_tct(M):
  """sps matrix to torch.tensor
  """
  coo = sps.coo_matrix(M)
  values = coo.data
  indices = np.vstack((coo.row, coo.col))
  i = tc.tensor(indices, dtype=tc.int32, device=device)
  v = tc.tensor(values, dtype=tc.float64, device=device)
  shape = coo.shape
  M = tc.sparse_coo_tensor(i, v, tc.Size(shape), dtype=tc.float64, device=device)
  return M.coalesce()

def sim_log(*strings):
  """Record s in the sim log file
  """
  with open(log_fname, "a") as log_file:
    log_file.write("\t"*log_indent)
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
  if tc.isnan(xc):
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
