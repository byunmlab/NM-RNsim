# Neuromorphic Resistor Network Simulator
This program simulates resistor networks. It was created to inform the design
of a neuromorphic device created by the neuromorphic computing team in the BYU
[Perception, Control, Cognition Lab](https://pcc.cs.byu.edu/).

## Requirements
See requirements.txt for python packages needed.

The program was designed to be run with GPU acceleration. This is essential 
for solving larger networks in any reasonable amount of time.

## Physical Motivation
This simulator was created to model neuromorphic neural networks in physical
chips composed of nickel nanostrands suspended in resin.

Originally, our theory was that sending high current through the chip across 
undesired connections would burn out several fibers along the most important
pathways along that connection and thus increase the resistance, so that is
what was modeled here. In reality, it seems that the underlying processes are
much more complex, some of them increasing conductivity while others decrease
conductivity.

Future versions of this simulator could attempt to model some of these other
mechanisms to better represent the physical system.

