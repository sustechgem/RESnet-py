# RESnet-py

3D resistor network solution for d.c. forward simulation problems. Python implementation.

This is a python implementation of the Matlab code to solve for the electric field of subsurface under the excitation of d.c. sources.

Matlab code repository link：

- https://github.com/sustechgem/RESnet-m

Run the following scripts:

- RUNME.py: A self-testing program to ensure a corrent installation and to demonstrate the workflow of software

- Example_Halfspace.py: Verification of the numerical accuracy by comparing with the analytic solution of a half-space

- Example_Casing.py: Simulation of the surface electric field with the presence of steel well casing

- Example_Infrastructure.py: Effect of complex metallic infrastructure on the surface dc resistivity data

#### Note：

This code solves large sparse matrices by calling the MKL PyPardiso interface PyPardiso.py, and a testing code can be found in PyPardisoExample.py. 
PyPardiso.py is made available by Wang Ke.
