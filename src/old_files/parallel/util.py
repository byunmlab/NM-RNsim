# Script that contains utility functions

import numpy as np
import matplotlib.pyplot as plt

# List of strings (in lowercase) that are considered equivalent to True
true_strs = ("true", "t", "yes", "y", "on", "1")

def str2arr(s):
  """Converts a comma-separated list of floats to a np.array.
    e.g. : s="1, 2.5, 3.14, 4"
  """
  return np.array(s.split(","), dtype=float)

def str2bool(s):
  """Returns bool(True) if the string s represents True
  """
  return str(s).lower() in true_strs

def db_print(cp, s):
  """Wrapper for standard print()
  Prints to screen if the "debug" setting is True
  """
  if cp.getboolean("exec", "debug"):
    print(s)

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