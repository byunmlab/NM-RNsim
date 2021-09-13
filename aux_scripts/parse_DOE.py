"""Take the csv files resulting from the DOE and parse them into one csv

Desired Format
--------------

ID    ks    flm0  flmr  ftpr  bpmr  pbfr  bf    RMIN  RDROP
-----------------------------------------------------------

"""

import pandas as pd
import numpy as np

parsed_file = "parsed_data.csv"
options_list_file = "options_list.csv"
options_df = pd.read_csv(options_list_file)

N = options_df.shape[0]
data_df = options_df.copy()
data_df["RMIN"] = - np.ones(N)
data_df["RDROP"] = - np.ones(N)

print(data_df)

for i in range(N):
  ID = data_df.at[i, "SIM_ID"]
  RMSRs_file = f"RMSRs_{ID}.csv"
  RMSRs_df = pd.read_csv(RMSRs_file)
  RMSR_min = np.min(RMSRs_df["%RMSR"][ RMSRs_df["Code"] < 3 ])
  RMSR0 = RMSRs_df["%RMSR"][ RMSRs_df["Code"] == -1 ]
  data_df.at[i, "RMIN"] = RMSR_min
  data_df.at[i, "RDROP"] = RMSR0 - RMSR_min

print(data_df)

data_df.to_csv(parsed_file)

