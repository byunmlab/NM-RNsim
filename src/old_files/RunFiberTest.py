""" Script for testing the implementation of fibers.
"""

from ResistorNetwork import ResistorNetwork as RN
import matplotlib.pyplot as plt


N = 200

# Input and Output pins
pins_3d = [("in0", -1.5, -1.5, -1.5), ("in1", -1.5, 1.5, -1.5), ("in2", -1.5, 0, 1.5),
  ("out0", 1.5, -1.5, 1.5), ("out1", 1.5, 1.5, 1.5), ("out2", 1.5, 0, -1.5)]
#pins_2d = [("in0", -1.5, -1.5), ("in1", -1.5, 1.5),
#  ("out0", 1.5, -1.5), ("out1", 1.5, 1.5)]
#pins_center = [("pin", 0, 0)]
pins = pins_3d

# Dimensions of network
limits_3d = [-2,2, -2,2, -2,2]
limits_2d = [-2,2, -2,2]
limits = limits_3d

# Make the Resistor Network
#rn = RN(N, node_r=.25, rand_s=11, pins=pins, pin_r=1, fibers=True, fl_mean=1, limits=limits)
rn = RN(N, node_r=.25, rand_s=16, pins=pins, fibers=True, limits=limits)

#fig, (ax0, ax1, ax2) = plt.subplots(1,3)
#fig, (ax0, ax1) = plt.subplots(1,2)

fig, ax = rn.draw(width_attrib="cnd")
#rn.draw(ax=ax1, width_attrib="cnd", edge_color="g")
#rn.draw_f(ax=ax1)
fig.show()

input("Pause")
