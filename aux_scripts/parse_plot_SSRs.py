import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = "parsed_all.csv"
tag = "19"
ofile = f"data_{tag}.csv"
header = True
cutoff = 30

if header:
  df = pd.read_csv(filename)
else:
  df = pd.read_csv(filename, header=None)
S_sub = df["%RMSR"][ df["FNAME"].str.contains(tag)&( df["%RMSR"]>cutoff ) ]
E_sub = df["Epoch"][ df["FNAME"].str.contains(tag)&( df["%RMSR"]>cutoff ) ]
df2 = pd.concat([E_sub, S_sub], axis=1)
#print(df2)
df2.to_csv(ofile, index=False)
print("Data saved to:", ofile)

quit()

f, a = plt.subplots()
a.scatter(E_sub, S_sub)
fname = f"S{tag}_All"
f.savefig(fname)

# Average
E15_u = E15.unique()
S15_avgs = np.zeros(len(E15_u))
for i, E in enumerate(E15_u):
  S15_avgs[i] = np.mean( S15[E15 == E].to_numpy() )
f, a = plt.subplots()
a.scatter(E15_u, S15_avgs)
fname = f"S{tag}_Avg"
f.savefig(fname)


