import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Parse the arguments
desc = "Plot the voltages in the given files"
parser = argparse.ArgumentParser(description=desc)
lfh = "Provide the filename of a csv with voltages in a linear RN"
parser.add_argument("linear_filename", help=lfh)
nlfh = "Provide the filename of a csv with voltages in a nonlinear RN"
parser.add_argument("nonlinear_filename", help=nlfh)
rh = "Whether to plot the residual of the difference between the linear and\
  nonlinear voltage"
parser.add_argument("-r", "--plot_residuals", action="store_true", help=rh)
args = parser.parse_args()

fn_vL = args.linear_filename
#"../../test_results/73_train/RN_73L0_trained_v.csv"
fn_vNL = args.nonlinear_filename
#"../../test_results/73_train/RN_73NL0_trained_v.csv"
plot_res = args.plot_residuals

extracted_title = fn_vL.split("/")[-1].split(".")[0] + "-vs-"
extracted_title += fn_vNL.split("/")[-1].split(".")[0]
fig_title = extracted_title
extensionL = fn_vL.split(".")[-1]
extensionNL = fn_vNL.split(".")[-1]

extension_error = "Please provide a csv of RN voltages"
if extensionL != "csv" or extensionNL != "csv":
  print(extension_error)
  quit()

#filetypes = ("csv", "gz", "json", "pickle")
#extension_error = "Please provide an RN file or a csv of RN voltages"
#if (extensionL not in filetypes) or (extensionNL not in filetypes):
#  print(extension_error)
#  quit()

voltage_key = "VOLTAGE"
id_key = "NODE ID"

df_vL = pd.read_csv(fn_vL, skipinitialspace=True)
vL = df_vL[voltage_key].to_numpy()
iL = df_vL[id_key].to_numpy()

df_vNL = pd.read_csv(fn_vNL, skipinitialspace=True)
vNL = df_vNL[voltage_key].to_numpy()
iNL = df_vNL[id_key].to_numpy()

vLf = []
vNLf = []
for n_L, id_L in enumerate(iL):
  if id_L in iNL:
    # This node is present in both
    vLf.append(vL[n_L])
    vNLf.append(vNL[np.where(iNL == id_L)[0][0]])

#print(35, iLf[-4:], iNLf[-4:])
#print(38, vLf[-4:], vNLf[-4:])

#vlim = 100
vlim = 1

#fig, ax = plt.subplots()
fig = plt.figure(figsize=(14,8))
ax = fig.add_subplot()
ax.scatter(vLf, vNLf, s=4)
if vlim == 100:
  ax.plot([0,100],[0,100], "--")
  ax.set_ylim(-5,105)
elif vlim == 1:
  ax.plot([0,1],[0,1], "--")
  ax.set_ylim(-.05,1.05)
ax.set_xlabel("Voltage in linear system")
ax.set_ylabel("Voltage in nonlinear system")
fig.suptitle(fig_title)
fig_name = f"plt_vs_{fig_title}.png"
fig.savefig(fig_name)

print("Figure saved to file:", fig_name)

if plot_res:
  # Residuals plot
  fig = plt.figure(figsize=(14,8))
  ax = fig.add_subplot()
  ax.plot([0,100],[0,0], "--")
  ax.scatter(vLf, np.array(vNLf) - np.array(vLf), s=4)
  #ax.set_ylim(-.05,.05)
  ax.set_ylim(-1,1)
  ax.set_xlabel("Voltage in linear system")
  ax.set_ylabel("Nonlinear voltage - linear voltage")
  fig.suptitle(fig_title)
  fig_name = f"plt_vs_res_{fig_title}.png"
  fig.savefig(fig_name)

  print("Figure saved to file:", fig_name)


