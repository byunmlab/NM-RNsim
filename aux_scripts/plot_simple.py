import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#ifname = "../../test_results/95_trsts/95_RMSRs_AVG.csv"
#ofname = "../../test_results/95_trsts/95_RMSRs_AVG.png"
#ifname = "95_trsts/95_RMSRs_AVG.csv"
#ofname = "95_trsts/95_RMSRs_AVG.png"
ifname = "RMSRs_101ABr2.csv"
ofname = "RMSRs_101ABr2.png"

old = False
y_key = "%SSR" if old else "%RMSR"

df = pd.read_csv(ifname)

f, a = plt.subplots()
a.scatter(df["Epoch"], df[y_key])

f.savefig(ofname)
print("Figure saved to ", ofname)
