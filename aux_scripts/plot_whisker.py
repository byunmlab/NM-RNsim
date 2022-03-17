import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Make a whisker plot")
parser.add_argument("filename", nargs="+", help="provide an RMSR (csv) file")
parser.add_argument("--old", action="store_true",
  help="Is this an older file format? If so, use the key %%SSR, not %%RMSR")
args = parser.parse_args()

y_key = "%SSR" if args.old else "%RMSR"

def whisker(ifname):
  if ifname[-4:] != ".csv":
    print("Please provide a .csv file")
    return
  fname_core = ifname[0:-4]
  #ifname = "RMSRs_19_all.csv"
  #ifname = "data9596_19.csv"
  #ofname = "RMSRs_19_whisker.png"
  #ofname = "9596whisker_19.png"
  ofname = fname_core + "_whisker.png"
  #title = "%RMSR (REL) Sim 9596, ks=1.9"
  title = f"%RMSR, {fname_core}"

  cutoff = 0 # Temp

  df = pd.read_csv(ifname)
  Epochs = df["Epoch"].unique()
  data = [None]*len(Epochs)

  for i, E in enumerate(Epochs):
    filtered = df[y_key][df["Epoch"] == E]
    filtered = filtered[filtered > cutoff]
    data[i] = filtered.to_numpy()

  f, a = plt.subplots()
  a.boxplot(data, positions=Epochs)
  a.set_title(title)
  a.set_xlabel("Epoch")
  a.set_ylabel("%RMSR (REL)")

  f.savefig(ofname)
  print("Figure saved to:", ofname)

for filename in args.filename:
  whisker(filename)
