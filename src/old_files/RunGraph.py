import networkx as nx
import matplotlib.pyplot as plt
import time
import ResistorNetwork as rn
NUM_NODES = 100
EPOCHS = 10


pins = [("in0", 0, 0.1), ("in1", 0, 0.9), ("out0", 1, 0.2), ("out1", 1, 0.8)]
RN = rn.ResistorNetwork(NUM_NODES, pins=pins)
#fig, ax = RN.draw()
#plt.show()
RN.setVoltages(["in1"],["out0", "out1"],5)
currents = RN.getOutputCurrents(["out0", "out1"])
req1= RN.R_pp("in1", "out0")
print("Resistance from in1 to out0 is ", req1)
print("The currents at output pins with in1 set high are ", 
  currents[0], " and ", currents[1])

#pull the pins high or low
bin = RN.pullOutputs(currents, [.2, .2])
print("The ouput pins read as ", bin)
desired = [0, 0]
print("The desired output is ", desired)
out = RN.compareOutputs(bin, desired)
print("The bad pins are ", out)
print("The initial size is ", RN.size())
RN.backPropagate(["in1"], ["out0"], 10, .1)
print("Total power as 10V / Req is ", 10**2 / req1)
print("The final size is ", RN.size())
print(RN.get())
print(bin)
print(out)
print("end")
