import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set up argparse
desc = """Scatter plot RMSR against Epoch."""
parser = argparse.ArgumentParser(description=desc)
fh = "Provide a csv file to plot"
parser.add_argument("filename", help=fh)
parser.add_argument("-o", "--out_file", default="OUT.png",
  help="Name of output file. Default=OUT.png")
parser.add_argument("--old", action="store_true",
  help="Is this an older file format? If so, use the key %%SSR, not %%RMSR")
args = parser.parse_args()

#ifname = "../../test_results/95_trsts/95_RMSRs_AVG.csv"
#ofname = "../../test_results/95_trsts/95_RMSRs_AVG.png"
#ifname = "95_trsts/95_RMSRs_AVG.csv"
#ofname = "95_trsts/95_RMSRs_AVG.png"
#ifname = "RMSRs_101ABr2.csv"
#ofname = "RMSRs_101ABr2.png"
ifname = args.filename
ofname = args.out_file

y_key = "%SSR" if args.old else "%RMSR"

df = pd.read_csv(ifname)

f, a = plt.subplots()
a.scatter(df["Epoch"], df[y_key])

f.savefig(ofname)
print("Figure saved to ", ofname)
