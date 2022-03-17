import argparse
import numpy as np
import pandas as pd

desc = """Concatenate the data in multiple .csv files.
  This is good for when many similar sims have been run that return the same
  kinds of data, and they are to be plotted together."""
parser = argparse.ArgumentParser(description=desc)
fh = "Provide a list of csv files to concatenate"
parser.add_argument("filenames", nargs="+", help=fh)
Ah = """Whether to average all values that report about the same epoch.
  This requires that the csv files have the columns [Epoch, %%RMSR]"""
  # Btw, the % needs to be escaped for some reason in the above line
parser.add_argument("-A", "--avg_epochs", action="store_true", help=Ah)
parser.add_argument("-o", "--out_file", default="OUT.csv",
  help="Name of output file. Default=OUT.csv")
parser.add_argument("-c", "--cutoff", type=float,
  help="Will ignore all values lower than cutoff. default=None")
parser.add_argument("--old", action="store_true",
  help="Is this an older file format? If so, use the key %%SSR, not %%RMSR")
args = parser.parse_args()

outfile = args.out_file
#i0 = lambda f: 1+f.rindex("/") if ("/" in f) else 0
#trimmed_fnames = [f[i0(f):f.rindex(".csv")] for f in args.filenames]
#outfile = "_-_".join(trimmed_fnames) + ".csv"
#outfile = "OUT.csv"
cutoff = args.cutoff
y_key = "%SSR" if args.old else "%RMSR"

rd_csv = lambda f: pd.read_csv(f, skipinitialspace=True)
df = pd.concat(map(rd_csv, args.filenames))
if cutoff is not None:
  df = df[df[y_key] > cutoff]
else:
  df = df[df["Code"] < 3]

#print(df)
if args.avg_epochs:
  # Average all values that refer to the same Epoch
  #cutoff = 39 # Ignore small values...
  Epochs = df["Epoch"].unique()
  df2 = pd.DataFrame(np.zeros( (Epochs.shape[0], 2) ))
  df2.columns = ["Epoch", y_key]
  df2["Epoch"] = Epochs
  for E in Epochs:
    filtered = df[y_key][df["Epoch"] == E]
    #filtered = filtered[filtered > cutoff]
    df2.loc[E, y_key] = np.mean(filtered.to_numpy())
  df = df2

df.to_csv(outfile, index=None)
print(f"Combined data saved to {outfile}")

