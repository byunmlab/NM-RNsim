import pandas as pd
import numpy as np
import ResistorNetwork as rn

# This file draws the results from multiple power tests 
# by reading them from a power_log file.

filename = "power_log_6.csv"
# in and out pins (0-indexed)
p0 = 1
p1 = 7

# Load the file
df = pd.read_csv(filename, skipinitialspace=True)

#print(df)
pos = df["PM POS"]
#print(pos)
A = np.zeros((len(pos), 2))
# Messy conversion to np array
for i in range(len(pos)):
  a1 = pos[i].strip(" []").split(" ")
  a1 = " ".join(a1).split()
  A[i,:] = np.array(a1)

#print(A)

# Make a plot
sH = 2.743 #Horizontal spacing (in this case vertical)
sV = 2.845
pR = sV/4 #pin R (taken to be 1/2 of sV/2)
shape = [0, 4*pR+sV, 0, 4*pR+4*sH]
# Input and Output pins
pins = [("pin1", 2*pR, 2*pR), ("pin2", 2*pR, 2*pR+sH), ("pin3", 2*pR, 2*pR+2*sH), 
  ("pin4", 2*pR, 2*pR+3*sH), ("pin5", 2*pR, 2*pR+4*sH), ("pin6", 2*pR+sV, 2*pR+.5*sH),
  ("pin7", 2*pR+sV, 2*pR+1.5*sH), ("pin8", 2*pR+sV, 2*pR+2.5*sH), ("pin9", 2*pR+sV, 2*pR+3.5*sH)]
# Make a dummy network with no nodes, just to use its draw function
RN = rn.ResistorNetwork(0, pins=pins, limits=shape, pin_r=pR)

fig0, ax0 = RN.draw()
ax0.plot(pins[p0][1], pins[p0][2], "gx", markersize=15)
ax0.plot(pins[p1][1], pins[p1][2], "gx", markersize=15)
for a in A:
  ax0.plot(a[0], a[1], "rx", markersize=10)
fig0.show()
input("Press Enter to continue")


