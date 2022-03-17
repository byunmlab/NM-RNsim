import networkx as nx
import ResistorNetwork as rn
import time
import math

# This script runs a power test.
# It is designed to be used in parallel.

NUM_NODES = 1600
SES_NUM = 7 #This should be set the same in all 3 files
filename = f"power_log_{SES_NUM}.csv"
file = open(filename, "a")

# Copied from RunPowerTest
# (http://www.interfacebus.com/D-sub-pin-positions-9-pin.jpg)
sH = 2.743 #Horizontal spacing (in this case vertical)
sV = 2.845
pR = sV/4 #pin R (taken to be 1/2 of sV/2)
# Approx. size of the DB9 in mm 
shape = [0, 4*pR+sV, 0, 4*pR+4*sH]
# Input and Output pins
pins = [("pin1", 2*pR, 2*pR), ("pin2", 2*pR, 2*pR+sH), ("pin3", 2*pR, 2*pR+2*sH), 
  ("pin4", 2*pR, 2*pR+3*sH), ("pin5", 2*pR, 2*pR+4*sH), ("pin6", 2*pR+sV, 2*pR+.5*sH),
  ("pin7", 2*pR+sV, 2*pR+1.5*sH), ("pin8", 2*pR+sV, 2*pR+2.5*sH), ("pin9", 2*pR+sV, 2*pR+3.5*sH)]

# Node radius for connections
nR = 0.55

t0 = time.time()

# Random seed
rs = int(math.modf(t0)[0]*100000)
RN = rn.ResistorNetwork(NUM_NODES, pins=pins, limits=shape, pin_r=pR, node_r=nR, rand_s=rs)
print("RN Built")
# Apply a voltage and find the highest power edge
p_max, edge_pm = RN.apply_v(25, "pin2", "pin8")
POS = RN.G.nodes[edge_pm[0]]['pos']
t1 = time.time()
# Save to file
file.write(str(NUM_NODES)+", ")
file.write(str(nR)+", ")
file.write(str(rs)+", ")
file.write(str(p_max)+", ")
pos_str = "[" + str(POS[0]) + " " + str(POS[1]) + "], "
file.write(pos_str)
file.write("DB9\n")
file.close()
print(f"Done. Sim calculation time: {t1-t0}")

#print("Done")

