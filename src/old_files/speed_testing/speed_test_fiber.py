import time
import numpy as np
from ResistorNetwork import ResistorNetwork as RN
import matplotlib.pyplot as plt


N = 200

# Input and Output pins
pins_3d = [("in0", -1.5, -1.5, -1.5), ("in1", -1.5, 1.5, -1.5), ("in2", -1.5, 0, 1.5),
  ("out0", 1.5, -1.5, 1.5), ("out1", 1.5, 1.5, 1.5), ("out2", 1.5, 0, -1.5)]
pins = pins_3d

# Dimensions of network
limits_3d = [-2,2, -2,2, -2,2]
limits_2d = [-2,2, -2,2]
limits = limits_3d

t0 = time.time()

# Make the Resistor Network
rn = RN(N, node_r=.25, rand_s=17, pins=pins, fibers=True, limits=limits)

t1 = time.time()

fig, ax = rn.draw(width_attrib="cnd")
fig.show()

t2 = time.time()


print(f"N = {N}")
print(f"Building Network: {t1 - t0}")
print(f"Drawing: {t2 - t1}")


input("Pause")
