"""This script contains the functions related to solving the system

  I : Calculate nonlinear current
  P : Calculate nonlinear power
  NL_Axb : Solve the system sum( A.*sinh(wX) ) = b using a nonlinear solver

"""

import util
from util import db_print
from util import tic
from util import toc
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.optimize as spo
import numpy as np
import jax
import jax.numpy as jnp
import optax

# List of supported i-v curve functions
iv_funs = ("L", "sinh", "relu")
# List of supported nonlinear solver methods
NL_methods = ("n-k", "hybr", "trf", "mlt", "adpt", "custom", "optax-adam")
# Min tolerance. A little bigger than epsilon
tol_min = 1e-15

class RNOptRes(spo.OptimizeResult):
  """The object returned by RNsol
  This object must have:
    status
    v
    I_in
    Req
    nfev
  It may also have:
    x
    message
    Any other attribute of spo.OptimizeResult
  If initialized with x, v_in, ni_in, & ni_out, this is sufficient
  """
  def __init__(self, x=None, v=None, status=-999, message=None, nfev=-1,
    I_in=None, Req=None, v_in=None, ni_in=None, ni_out=None):
    """Constructor
    """
    super().__init__()
    self.nfev = nfev
    self.status = status
    if x is not None:
      self.x = x
      if (v_in is not None and ni_in is not None and ni_out is not None):
        set_xvIR(self, x, v_in, ni_in, ni_out)
    if v is not None:
      self.v = v
    if I_in is not None:
      self.I_in = I_in
    if Req is not None:
      self.Req = Req
    if message is not None:
      self.message = message

def set_xvIR(RNOR, x, v_in, ni_in, ni_out):
  """ Helper method for RNOptRes objects
  This isn't inside the RNOptRes class because of the dict attributes thing.
  """
  RNOR.x = x
  RNOR.v = util.ainsrt2(x, ni_in, v_in, ni_out, 0)[0:-1]
  RNOR.v = np.array(RNOR.v, dtype=np.float)
  I_in = x[-1]
  if np.abs(I_in) < 1e-30:
    Req = 1e30
  else:
    Req = v_in / I_in
  RNOR.I_in = I_in
  RNOR.Req = Req

def I(vp, vn, params):
  """Simple calculation of current between two nodes
  Parameters
  ----------
    vp: voltage at anode
    vn: voltage at cathode
    params (dict): Parameters for the current calculation.
      ivfun (str) : Which function? (L, sinh or relu)
      if ivfun==L:
        params["G"]: conductivity (iv slope)
      if ivfun==sinh:
        params["G"]: conductivity (iv slope @v=0)
        params["w"]: the coefficient on the argument for sinh(wx)
      if ivfun==relu:
        params["G"]: conductivity (iv slope @v>0)
  Returns
  -------
    i: Current. Positive current flows from anode to cathode by PSC
  """
  ivfun = params["ivfun"]
  if ivfun == "L":
    return params["G"] * (vp-vn)
  elif ivfun == "sinh":
    return params["G"]/params["w"] * np.sinh(params["w"]*(vp-vn))
  elif ivfun == "relu":
    i = params["G"]*(vp - vn)
    is_on = (i>0)
    # If this is a vector with all the nodes, then exempt the pins
    if hasattr(is_on, "__len__") and len(is_on) > max(params["pinids"]):
      #is_on = is_on.at[params["pinids"]].set(1)
      is_on = jax.ops.index_update(is_on, jnp.array(params["pinids"]), 1)
    elif "not_relu" in params and params["not_relu"]:
      # Exception for individual nodes to be non-relu
      is_on = 1
    return i * is_on
  else:
    return "ERROR: unknown ivfun"
def P(vp, vn, params):
  """Simple calculation of power (p=vi)
  Parameters
  ----------
    vp: voltage at anode
    vn: voltage at cathode
    params (dict): Parameters for the current calculation.
      ivfun (str) : Which function? (L, sinh or relu)
      if ivfun==L:
        params["G"]: conductivity (iv slope)
      if ivfun==sinh:
        params["G"]: conductivity (iv slope @v=0)
        params["w"]: the coefficient on the argument for sinh(wx)
      if ivfun==relu:
        params["G"]: conductivity (iv slope @v>0)
  Returns
  -------
    p: Power.
  """
  i = I(vp, vn, params)
  return (vp-vn)*i

def sum_node_I(v, A, ni, w, pinids, ivfun="sinh"):
  """Like RN.sum_currents, return the current sinked by the given node
  The difference is that this operates with the adjacency matrix.
  Parameters
  ----------
    v: vector of voltages
    A: adjacency matrix for conductance
    ni: index of node
    w: coefficient for sinh
    ivfun (str) : Which function? (sinh or relu)
  """
  vn = v[ni]*np.ones(v.shape)
  params = {
    "G": A[ni,:],
    "w": w,
    "pinids": pinids,
    "ivfun": ivfun
  }
  i_in = I(v, vn, params)
  return jnp.sum(i_in)

def NL_res_j(x, A, w, pinids, v_in, ni_in, ni_out, fun="sinh"):
  """NL_res, but compatible with jit.
  Also take note of the changed order of arguments
  """
  N = x.size + 2
  v = util.ainsrt2(x, ni_in, v_in, ni_out, 0)[0:-1]
  vcols, vrows = jnp.meshgrid(v, v)
  if fun == "sinh":
    sinharg = w * (vrows - vcols) # Argument to be 'sinh()'ed
    Asinh = jnp.multiply(A, jnp.sinh(sinharg))
    sumI = Asinh.sum(1) / w # sum of currents leaving each node
  elif fun == "relu":
    V = (vrows - vcols)
    pin_indices = jnp.array([ni_in, ni_out, *pinids]) # Only for the immediate input and output...
    vsrc = (V > 0)
    # It's always okay for current to flow in or out of the pins
    vsrc = vsrc.at[:,pin_indices].set(1) 
    vsrc = vsrc.at[pin_indices,:].set(1)
    vsnk = (V < 0)
    vsnk = vsnk.at[:,pin_indices].set(1)
    vsnk = vsnk.at[pin_indices,:].set(1)
    # This method says that current should flow from low index pins
    triui = jnp.triu_indices(N, 1)
    trili = jnp.tril_indices(N, -1)
    #G = A.at[triui].multiply(vsrc[triui])
    G = jax.ops.index_mul(A, triui, vsrc[triui])
    #G = G.at[trili].multiply(vsnk[trili])
    G = jax.ops.index_mul(G, trili, vsnk[trili])
    sumI = jnp.multiply(G, V).sum(1)
  res = sumI.at[ni_in].add(- x[-1])
  res = res.at[ni_out].add(x[-1])
  return res#[1:] # Throw one equation out (the first one)

def RNsol(params, options):
  """Solve the given RN system
  Parameters
  ----------
    params (dict): Parameters that define the problem
      A (NxN scipy matrix): the Adjacency Matrix of the RN
      w (float, for SINH): the coefficient on the argument for sinh(wx)
        This can be related to the 3rd derivative of the i-v curve at zero.
      pinids (tuple of int, for RELU): the indexes of the pins
      v_in (float): the applied voltage
      ni_in (int, [0,N-1]): the index of the input node
      ni_out (int, [0,N-1]): the index of the output node
      ivfun ("sinh" or "relu"): Which function to use
    options (dict): Options about how to solve it
      method (element of NL_methods): Which solver to use
      verbose (int, [0,3]): Verbosity
      xtol: tolerance in x
      ftol: tolerance in f
      xi (str or N+1 np array) : initial guess for the solution. If None, xi=0
        If "Lcg", then find an xi by solving the linear system with cg
        If "L_{nonlin_method}_{N} then solve sequentially for better xi-s
          by gradually increasing the nonlinearity N times. (for SINH)
  Returns
  -------
    result (object)
      result.v (N np array) : voltage vector.
      result.I_in (float) : current
      result.Req (float) : Equivalent Resistance
      result.code (int) : Success code.
        0: Success
        1: Invalid call to RNsol
      result.message (str) : code message
  """

  # Starting message
  if options["verbose"]:
    db_print(f"Solving RN...")
    util.log_indent += 1

  # Result object
  result = RNOptRes()

  v_in = params["v_in"]
  ni_in = params["ni_in"]
  ni_out = params["ni_out"]
  ivfun = params["ivfun"] if "ivfun" in params else "sinh"
  if ivfun not in iv_funs:
    result.status = 1
    result.message = "Unknown ivfun"
    return result

  if ivfun == "L":
    # Linear system
    if options["method"] in NL_methods:
      result.status = 1
      result.message = "Nonlinear method specified for linear ivfun"
      return result
    L = None
    if "L" not in params:
      if "A" not in params:
        result.status = 1
        result.message = "Missing parameter or option: 'A' 'L'"
        return result
      params["L"] = L_from_A(params["A"])
    v, Req, status = L_sol(params, options)
    result.v = v
    result.I_in = v_in / Req
    result.Req = Req
    result.status = status
  else:
    # Nonlinear system
    if options["method"] not in NL_methods:
      result.status = 1
      result.message = "options['method'] is not a valid nonlinear method"
      return result
    sol = NL_sol(params, options)
    set_xvIR(result, sol.x, v_in, ni_in, ni_out)
    #v = util.ainsrt(sol.x, [(ni_in, v_in), (ni_out, 0)])[0:-1]
    #v = np.array(v, dtype=np.float) #Convert back to np (float64)
    #I_in = sol.x[-1]
    #I_in = np.float64(I_in) #Convert back to np
    #if np.abs(I_in) < 1e-30:
    #  # Avoid divide by zero
    #  Req = 1e30 #TMP
    #else:
    #  Req = v_in / I_in
    result.status = sol.status if hasattr(sol, "status") else 0

  util.sim_log("166 RNsol: v[in], v[out], I_in, Req: ", 
    result.v[ni_in], result.v[ni_out], result.I_in, result.Req)
  # Return to normal indent
  if options["verbose"]:
    util.log_indent -= 1
  return result

def NL_sol(params, options):
  """Solve the given nonlinear problem.
  Parameters
  ----------
    params (dict): Parameters that define the problem
      A (NxN scipy matrix): the Adjacency Matrix of the RN
      w (float, for SINH): the coefficient on the argument for sinh(wx)
        This can be related to the 3rd derivative of the i-v curve at zero.
      pinids (tuple of int, for RELU): the indexes of the pins
      v_in (float): the applied voltage
      ni_in (int, [0,N-1]): the index of the input node
      ni_out (int, [0,N-1]): the index of the output node
      ivfun ("sinh" or "relu"): Which function to use
    options (dict): Options about how to solve it
      method (element of NL_methods): Which solver to use
      verbose (int, [0,3]): Verbosity
      xtol: tolerance in x
      ftol: tolerance in f
      xi (str or N+1 np array) : initial guess for the solution. If None, xi=0
        If "Lcg", then find an xi by solving the linear system with cg
        If "L_{nonlin_method}_{N} then solve sequentially for better xi-s
          by gradually increasing the nonlinearity N times. (for SINH)
  Returns
  -------
    sol (OptimizeResult) : the solver result
      Note: sol.x (N-1 np array) : the solution vector. 
      Note: x[-1] = I_in and x is missing v_in & v_out
  """

  times = tic()

  # Rename the contents of params for convenience
  A = params["A"]
  w = params["w"]
  pinids = params["pinids"]
  v_in = params["v_in"]
  ni_in = params["ni_in"]
  ni_out = params["ni_out"]
  ivfun = params["ivfun"] if "ivfun" in params else "sinh"

  # If necessary, convert A to a jnp array
  if hasattr(params["A"], "toarray"):
    A = jnp.array(A.toarray())

  # Now reload the edited params into fprms
  fprms = {} # Filtered params
  fprms["A"] = A
  fprms["w"] = w
  fprms["pinids"] = pinids
  fprms["v_in"] = v_in
  fprms["ni_in"] = ni_in
  fprms["ni_out"] = ni_out
  fprms["ivfun"] = ivfun

  # Build some things based on params
  N = A.shape[1]-1 # len(x) = N_nodes + 1 - 2
  ubound = v_in * np.ones(N)
  ubound[-1] = np.inf # No bound on current
  bounds = (np.zeros(N)-1e-6, ubound+1e-6)

  # Rename some of the contents of options for convenience
  xi = options["xi"]
  method = options["method"]
  verbose = options["verbose"] if "verbose" in options else util.debug

  # If verbose is not an int, then make it a 0/1 bool
  if type(verbose) != int:
    verbose = int(bool(verbose))

  # Come up with a good xi if needed
  if xi is None:
    xi = jnp.zeros(N).at[ni_in].set(v_in)
  elif isinstance(xi, str):
    assert xi[0] == "L", "483: Unknown xi option"
    # Get xi from the linear system, solving with cg
    L = L_from_A(params["A"]) # Use the sps version. Makes this a little fragile
    params_L = {
      # Use the sps version. Makes this a little fragile
      "L": L_from_A(params["A"]),
      "v_in": v_in,
      "ni_in": ni_in,
      "ni_out": ni_out
    }
    options_L = {
      "ftol": 1e-7
    }
    vL, RL, status = L_sol(params_L, options_L)
    xL = np.append(vL, v_in / RL) # convert v to x
    xL = np.delete(xL, [ni_in, ni_out])
    toc(times, "Lcg")
    if xi == "Lcg":
      xi = jnp.array(xL)
    else:
      # Presolving with smaller w. Format: "L_method_N"
      xi_parts = xi.split("_") 
      xi = jnp.array(xL)
      N_pre = int(xi_parts[2])
      ximethod = xi_parts[1]
      assert ximethod in NL_methods
      # Make N_pre simpler versions of the system to solve
      for i in range(N_pre):
        wi = w * ( (i+1)/(N_pre+1) )**2
        params_i = fprms.copy()
        params_i["w"] = wi
        options_i = {
          "xi": xi,
          "method": ximethod,
          "xtol": options["xtol"] if "xtol" in options else 1e-7,
          "ftol": options["ftol"] if "ftol" in options else 1e-6
        }
        soli = NL_sol(params_i, options_i)
        #soli = NL_sol(A, params, v_in, ni_in, ni_out, xi=xi, method=ximethod, 
        #  options={"ftol":5e-5})
        #db_print(soli.__dict__)
        xi = soli.x
      db_print(503, "Done with presolving")
      toc(times, f"xi presolving, method={ximethod}")
  else:
    xi = jnp.array(xi)

  fopt = {} # Filtered options
  fopt["xi"] = xi
  fopt["method"] = method
  fopt["verbose"] = verbose
  if "xtol" in options:
    fopt["xtol"] = options["xtol"]
  if "ftol" in options:
    fopt["ftol"] = options["ftol"]

  if verbose > 2:
    db_print(f"Starting Nonlinear Solver: {method}")

  # Create jit compiled versions of the function
  jres = jax.jit(NL_res_j, static_argnums=[2,3,4,5,6,7])
  res = lambda x: jres(x, A, w, pinids, v_in, ni_in, ni_out, ivfun)
  jjac = jax.jit(jax.jacfwd(jres), static_argnums=[2,3,4,5,6,7])
  jac = lambda x: jjac(x, A, w, pinids, v_in, ni_in, ni_out, ivfun)

  #toc(times, "jit")
  rxi = res(xi)
  db_print(f"||res(xi)||: {jnp.linalg.norm(rxi)}")
  toc(times)#, "res(xi)")
  if method != "adpt" and jnp.linalg.norm(rxi) < fopt["ftol"]: #1e-9:
    # For adpt, ftol has a different meaning. TODO: use a different name?
    # If xi is already within tolerance, we're done
    sol = RNOptRes(x=xi, status=0, nfev=1)
    sol.fun = rxi
    sol.message = "The linear model was sufficient in this case"
    return sol

  if method == "adpt":
    sol = NL_adpt(fprms, fopt)

  elif method == "mlt":
    # Chain multiple solvers
    #ltol = 5e-3
    xtol = options["xtol"] if "xtol" in options else 1e-5
    ftol = options["ftol"] if "ftol" in options else 1e-3
    # List of solvers and tolerances
    methods = [
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
    sol = NL_mlt(fprms, fopt, methods, ftol)
    return sol

  elif method == "optax-adam":
    # TODO: broken
    # Scalar conversion
    loss = lambda x,A,w,pinids,v_in,ni_in,ni_out: jnp.sum(jnp.square(NL_res_j(
      x, A, w, pinids, v_in, ni_in, ni_out, ivfun)))
    jloss = jax.jit(loss, static_argnums=[2,3,4,5,6])
    jgrad = jax.jit(jax.grad(loss, 0), static_argnums=[2,3,4,5,6])
    jlx = lambda x: jloss(x,A,w,pinids,v_in,ni_in,ni_out) #"jlx" = Jitted Loss (x)
    jgx = lambda x: jgrad(x,A,w,pinids,v_in,ni_in,ni_out)
    # Parameters
    adam_opt = {
      "lrn_rate": 1e-3,
      "maxit": 4000
    }
    sol = NL_adam(jlx, jgx, xi, adam_opt)
    toc(times, f"Optax Adam solver (w={w})")

  elif method == "custom":
    # Broken
    cust_opt = {
      "maxit": 400,
      "xtol": options["xtol"] if "xtol" in options else 1e-10,
      "rtol": options["ftol"] if "ftol" in options else 1e-4,
      "leash": 8,
      "momentum": 0.1
    }
    sol = NL_custom_N(res, jac, xi, cust_opt)
    toc(times, f"Custom Newton solver (w={w})")

  elif method == "n-k":
    #residual = lambda x: NL_res(A, w, v_in, ni_in, ni_out, x)
    xtol = options["xtol"] if "xtol" in options else 1e-3
    ftol = options["ftol"] if "ftol" in options else 1e-2
    if "tol" in options: # Generic tol
      xtol = options["tol"]
    nkopt = {
      "disp": True if options["verbose"] > 0 else False,
      "xtol": xtol,
      "ftol": ftol,
      "jac_options": {"rdiff": .05}
    }
    sol = spo.root(res, xi, method="krylov", options=nkopt)
  elif method == "hybr":
    # Make it square for hybr
    res = lambda x: jres(x, A, w, pinids, v_in, ni_in, ni_out, ivfun)[1:]
    jac = lambda x: jjac(x, A, w, pinids, v_in, ni_in, ni_out, ivfun)[1:]
    xtol = options["xtol"] if "xtol" in options else 1e-3
    if "tol" in options:
      xtol = options["tol"]
    hopt = {
      "xtol" : xtol,
      "maxfev": 1000000,
      "factor": 1
    }
    # Scale the variable for current so it's more significant
    hopt["diag"] = np.ones(N)
    hopt["diag"][-1] = N
    #db_print(f"583: xi={xi}")
    sol = spo.root(res, xi, method="hybr", jac=jac, options=hopt)
    toc(times, f"hybr (w={w})")
    #db_print(f"||res(sol.x)||: {jnp.linalg.norm(res(sol.x))}")
    #toc(times, "res(sol.x)")
  elif method == "trf":
    # The least_squares method doesn't take an options argument.
    #   Instead, expand options like this: **trf_opt
    #   "verbose" is an option in least_squares {0,1,2}
    trf_opt = {
      "bounds": bounds,
      "x_scale": "jac", # IDK about this...
    }
    # Scale the variable for current so it's more significant
    #opt["x_scale"] = np.ones(N)
    #opt["x_scale"] = N/10
    trf_opt["verbose"] = options["verbose"] if "verbose" in options else 0
    if "xtol" in options:
      trf_opt["xtol"] = options["xtol"]
    if "tol" in options:
      trf_opt["xtol"] = options["tol"]
    if "xtol" not in trf_opt:
      trf_opt["xtol"] = 1e-6
    # This ftol refers to SSR cost, not ||res||
    # Actually, it's relative (dF / F), not absolute cost
    trf_opt["ftol"] = options["ftol"] if "ftol" in options else 1e-3
    trf_opt["ftol"] = .5 * trf_opt["ftol"]**2 * 10 # *10 arbitrary
    trf_opt["gtol"] = options["gtol"] if "gtol" in options else 1e-10
    # Make sure xi is feasible
    #db_print(617, len(xi[xi<0]))
    xi = xi.at[xi<0].set(0)
    #db_print(619, len(xi[xi>v_in]))
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

  sol = RNOptRes()
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
    step_mlt, lm_fev = util.line_min(fnorm, x1, err, step) # Simple 1D optimization
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
  sol = RNOptRes()
  sol.x = xm
  sol.fun = rm
  sol.err = errm
  sol.it = it
  sol.message = "MESSAGE STUB" # TMP
  sol.nfev = nfev
  sol.success = True # TMP
  return sol

def NL_adpt(params, options):
  """Adaptive method that keeps trying more careful solvers
  This is the procedure which ends whenever err is small enough
    1. Start with hybr and xtol=1e-2
    2. Decrease xtol by 10x until it's no longer improving
    3. Switch to trf
    4. Decrease xtol by 10x until reaching 1e-12
  Aggregate error: err = max(||res||, ptp(i_out, i_in, x[-1]))
    This represents KCL error at each node as well as for the whole RN
    It also checks that x[-1] should be I
  Termination condition: err < x[-1] * ftol
    This way, if I=0.001A, the acceptable error is smaller than if I=10A
  """

  A = params["A"]
  w = params["w"]
  pinids = params["pinids"]
  v_in = params["v_in"]
  ni_in = params["ni_in"]
  ni_out = params["ni_out"]
  ivfun = params["ivfun"]
  xi = options["xi"]
  ftol = options["ftol"] if "ftol" in options else 1e-2
  opt_i = options.copy()
  opt_i["ftol"] = tol_min # We're using xtol

  # Set up initial sol
  sol = RNOptRes(xi, v_in=v_in, ni_in=ni_in, ni_out=ni_out)

  i = 0
  I = xi[-1]
  tol = 1e-2
  mtd = "hybr" # Start with hybr, then go to trf if hybr doesn't improve
  # Stopping error scales with current (down to tol_min)
  err_stop = max(I*ftol, tol_min)

  # Initial r
  err = np.linalg.norm(NL_res_j(xi, A, w, pinids, v_in, ni_in, ni_out, ivfun))
  while err > err_stop:
    if options["verbose"]:
      db_print(f"Running adaptive solver step {i}: method {mtd}, tol={tol}")
    opt_i["xtol"] = tol
    opt_i["method"] = mtd
    sol = NL_sol(params, opt_i)
    if options["verbose"]:
      db_print(f"NL_sol (nfev={sol.nfev}) message: {sol.message}")
    res = jnp.linalg.norm(sol.fun)
    set_xvIR(sol, sol.x, v_in, ni_in, ni_out)
    I = sol.I_in
    v = sol.v
    i_in = - sum_node_I(v, A, ni_in, w, pinids, ivfun=ivfun)
    i_out = sum_node_I(v, A, ni_out, w, pinids, ivfun=ivfun)
    ierr = jnp.ptp(jnp.array((i_in, i_out, I)))
    newerr = max(res, ierr)
    err_stop = max(I*ftol, tol_min)
    db_print(f"Error={newerr}, stopping err={err_stop}")
    # Set things up for the next round
    if newerr < err:
      # This x is better than the previous x
      err = newerr
      opt_i["xi"] = sol.x
    else:
      db_print("This adpt step did not reduce the error")
      if mtd == "hybr":
        mtd = "trf" # Switch to trf
        i += 1
        continue
      elif tol <= tol_min:
        db_print("Quitting Adaptive solver unsuccessfully")
        break
    i += 1
    if mtd == "hybr" and not sol.success:
      mtd = "trf"
    elif tol > tol_min: # No need to reduce tolerance if we're switching methods
      tol /= 10
    else:
      db_print("Quitting Adaptive solver unsuccessfully")
      break
  # TODO: Deal with when it found a local minimum.
  return sol

def NL_mlt(params, options, methods, final_ftol):
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
    final_ftol (float) : Stopping tolerance for the residual r
  Returns
  -------
    sol (OptimizeResult) : the solver result
  """

  A = params["A"]
  w = params["w"]
  pinids = params["pinids"]
  v_in = params["v_in"]
  ni_in = params["ni_in"]
  ni_out = params["ni_out"]
  ivfun = params["ivfun"]
  xi = options["xi"]
  opt_i = options.copy()

  # Initial r
  rxi = np.linalg.norm(NL_res_j(xi, A, w, pinids, v_in, ni_in, ni_out, ivfun))
  for mtd, tol in methods:
    if options["verbose"]:
      db_print(579, f"Running method {mtd} with tol={tol}")
    if mtd in NL_methods and mtd != "mlt":
      opt_i["xtol"] = tol
      opt_i["method"] = mtd
      sol = NL_sol(params, opt_i)
    else:
      db_print("Error: Unknown method")
    if options["verbose"]:
      db_print(582, sol.message, sol.nfev)#, sol.x[-8:])
    # See if sol is an improvement and if it's good enough alredy
    if not sol.success:
      # I may still want to use the result if it's better than before
      # However, when you start from a bad starting place, it's worse.
      db_print("Warning: the solver did not converge")
    if sol.success:
      rx1 = jnp.linalg.norm(sol.fun)
      rx1_max = jnp.max(sol.fun)
      v = util.ainsrt2(sol.x, ni_in, v_in, ni_out, 0)[0:-1]
      i_in = - sum_node_I(v, A, ni_in, w, pinids, ivfun=ivfun)
      i_out = sum_node_I(v, A, ni_out, w, pinids, ivfun=ivfun)
      KCL_err = jnp.ptp(jnp.array( (i_in, i_out, sol.x[-1]) ))
      db_print(854, i_in, i_out, sol.x[-1], rx1, rx1_max)
      err = max(KCL_err, rx1)
      if options["verbose"]:
        db_print(f"MLT step: Previous err={rxi:.8f}. New err={err:.8f}")
      if err < final_ftol: #sol.x is already good enough
        return sol
      if err < rxi: #sol.x is better than the previous xi
        rxi = err
        opt_i["xi"] = sol.x
  return sol

def L_from_A(A):
  """Create a Laplacian from an Adjacency matrix
  """
  if sps.issparse(A):
    D = sps.diags(np.array(A.sum(1)).squeeze())
  else:
    D = jnp.diag(A.sum(1))
  return D - A

def L_sol(params, options):
  """Solve the linear version of the system
  Parameters
  ----------
    params["L"] (sps matrix) : Laplacian matrix for conductance
    params["v_in"] (float) : Voltage in
    params["ni_in"] (int) : index of the input node (V+)
    params["ni_out"] (int) : index of the output node (V-)
    options["method"] {"cg", "spsolve"} : Which linear solver to use
    options["ftol"] (float) : Absolute tolerance for the norm of the residuals in cg
  Returns
  -------
    v (np array) : voltage at every node
    Req : Equivalent resistance (v_in / I)
    status : Success code. 0=success.
  """

  Lpl = params["L"]
  v_in = params["v_in"]
  ni_in = params["ni_in"]
  ni_out = params["ni_out"]
  method = options["method"] if "method" in options else "cg"
  atol = options["ftol"] if "ftol" in options else 1e-5
  
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
      #db_print("spsl.lsqr really shouldn't be used for this problem.")
    else:
      # Match with cg status success code
      status = 0
  # Scale the voltages linearly
  Req = v[ni_in] - v[ni_out] #This is true since I was 1A.
  voltage_mult = v_in / Req
  v *= voltage_mult

  return v, Req, status

