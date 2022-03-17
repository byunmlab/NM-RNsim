import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Parse the arguments
desc = "Plot the currents in the given files"
parser = argparse.ArgumentParser(description=desc)
lfh = "Provide the filename of a csv with currents in a linear RN"
parser.add_argument("linear_filename", help=lfh)
nlfh = "Provide the filename of a csv with currents in a nonlinear RN"
parser.add_argument("nonlinear_filename", help=nlfh)
rh = "Whether to plot the residual of the difference between the linear and\
  nonlinear current"
parser.add_argument("-r", "--plot_residuals", action="store_true", help=rh)
args = parser.parse_args()

fn_iL = args.linear_filename
#"../../test_results/73_train/RN_73L0_trained_i.csv"
fn_iNL = args.nonlinear_filename
#"../../test_results/73_train/RN_73NL0_trained_i.csv"
plot_res = args.plot_residuals

extracted_title = fn_iL.split("/")[-1].split(".")[0] + "-vs-"
extracted_title += fn_iNL.split("/")[-1].split(".")[0]
fig_title = extracted_title
extensionL = fn_iL.split(".")[-1]
extensionNL = fn_iNL.split(".")[-1]

extension_error = "Please provide a csv of RN currents"
if extensionL != "csv" or extensionNL != "csv":
  print(extension_error)
  quit()

#filetypes = ("csv", "gz", "json", "pickle")
#extension_error = "Please provide an RN file or a csv of RN currents"
#if (extensionL not in filetypes) or (extensionNL not in filetypes):
#  print(extension_error)
#  quit()

current_key = "THROUGH CURRENT"
id_key = "NODE ID"

df_iL = pd.read_csv(fn_iL, skipinitialspace=True)
idL = df_iL[id_key].to_numpy()
iL = df_iL[current_key].to_numpy()

df_iNL = pd.read_csv(fn_iNL, skipinitialspace=True)
idNL = df_iNL[id_key].to_numpy()
iNL = df_iNL[current_key].to_numpy()

#f for final.
iLf = []
iNLf = []
for n_L, id_L in enumerate(idL):
  if id_L in idNL:
    # This node is present in both
    iLf.append(iL[n_L])
    iNLf.append(iNL[np.where(idNL == id_L)[0][0]])

#print(35, idL[-4:], idNL[-4:])
#print(38, iLf[-4:], iNLf[-4:])

#ilim = 1
ilim = max(np.max(iNLf), np.max(iLf))
#print(66, ilim)

#fig, ax = plt.subplots()
fig = plt.figure(figsize=(14,8))
ax = fig.add_subplot()
#ax.set_xscale("log")
#ax.set_yscale("log")
ax.scatter(iLf, iNLf, s=4)

# Add a y-x line
if ilim == 100:
  ax.plot([0,100],[0,100], "--")
  ax.set_ylim(-5,105)
elif ilim == 1:
  ax.plot([0,1],[0,1], "--")
  ax.set_ylim(-.05,1.05)
else:
  ax.plot([0,ilim],[0,ilim], "--")

ax.set_xlabel("Current in linear system")
ax.set_ylabel("Current in nonlinear system")
fig.suptitle(fig_title)
fig_name = f"plt_is_{fig_title}.png"
fig.savefig(fig_name)

print("Figure saved to file:", fig_name)

if plot_res:
  # Residuals plot
  fig = plt.figure(figsize=(14,8))
  ax = fig.add_subplot()
  ax.plot([0,100],[0,0], "--")
  ax.scatter(iLf, np.array(iNLf) - np.array(iLf), s=4)
  #ax.set_ylim(-.05,.05)
  ax.set_ylim(-1,1)
  ax.set_xlabel("Current in linear system")
  ax.set_ylabel("Nonlinear current - linear current")
  fig.suptitle(fig_title)
  fig_name = f"plt_is_res_{fig_title}.png"
  fig.savefig(fig_name)

  print("Figure saved to file:", fig_name)


