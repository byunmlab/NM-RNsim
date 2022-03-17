
import pandas as pd
import numpy as np

#IO_Ns = pd.read_csv("93_trsts/IO_XOR.csv", skipinitialspace=True).to_numpy()
IOfile = "IO_XOR.csv"

def str2np(s):
  """Take a string representation of a 2d np array and return a np array
  """
  s = s.strip()
  assert s[0] == "["
  s = s[1:-1] # Cut off major brackets
  arrs = []
  lines = s.split("\n")
  for line in lines:
    if "[" not in line:
      continue
    line = line.split("[")[1] # There may be text before the '['
    line_arr = line.strip("[] \n").split(" ")
    while "" in line_arr:
      # Remove any empty entries
      line_arr.remove("")
    for i, n in enumerate(line_arr):
      # Remove any whitespace still hiding next to the numbers
      line_arr[i] = n.strip()
    npa = np.array(line_arr, dtype=np.float64)
    arrs.append(npa)
  return np.array(arrs)

def Is2RMSR(out_currents, IO_Ns=None, mode="max", msg=None):
  if IO_Ns is None:
    IO_Ns = default_IO()
  N_out_pins = len(IO_Ns[0][1].split("b")[1]) #Assumes 0bXXX format
  desired_out_currents = np.zeros(out_currents.shape)
  for i in range(len(IO_Ns)):
    Ni, No = IO_Ns[i]
    for si in range(len(No)-1, 1, -1):
      # Loop backwards, since the LSB is output 0
      if No[si] == "1":
        sioi = N_out_pins-1 - (si-2) # convert string index to out pin #
        desired_out_currents[i,sioi] = 1
    if mode == "rel":
      N_HI_outs = np.sum(desired_out_currents[i,:])
      currents = out_currents[i,:]
      if N_HI_outs != 0:
        desired_out_currents[i,:] *= np.sum(currents) / N_HI_outs
  max_current = np.max(out_currents)
  if mode == "max":
    desired_out_currents *= max_current

  res = desired_out_currents - out_currents
  divisor = np.repeat(np.vstack(np.sum(out_currents, axis=1)), 2, axis=1)
  # Don't divide by zero. Let 0 current contribute 0 error. Warn?
  divisor[divisor < 1e-5] = 1
  # percent normalized residuals
  pnR = 100 * res / divisor
  # Root Mean Square Residuals (where those are percent normalized residuals)
  RMSR = np.sqrt( np.mean(np.square(pnR)) )
  
  if msg is not None:
    if False:
      print(f"The output currents {msg} are: ")
      for i in range(len(IO_Ns)):
        Ni, _ = IO_Ns[i]
        print(f"\tfor input {Ni}: {out_currents[i,:]}")
    # print(currents) # This is the lazy version
    print(f"The %RMSR {msg} is: ", RMSR)

  return RMSR

def default_IO():
  #IOfile = "93_trsts/IO_XOR.csv"
  return pd.read_csv(IOfile, skipinitialspace=True).to_numpy()

def str_RMSR(s):
  currents = str2np(s)
  pRMSRM = Is2RMSR(currents, msg="(max)", mode="max")
  pRMSRR = Is2RMSR(currents, msg="(rel)", mode="rel")
  return pRMSRM, pRMSRR

def test():
  print("TEST")
  s = """
  [[-5.78558302e-15  2.25965224e-14]
   [-7.79570375e-09 -1.89312860e-08]
   0bXX: [ 1.55198105e-02  6.70244525e-03]
   [ 1.55200724e-02  6.70219165e-03]]
  """
  str_RMSR(s)

#test()
