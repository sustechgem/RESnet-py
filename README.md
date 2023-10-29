# RESnet-py
Dikun Yang (yangdikun@gmail.com)

**3D resistor network solution for d.c. forward simulation problems. Python implementation.**

This is a python implementation of the Matlab code to solve for the electric field of subsurface under the excitation of d.c. sources.

Matlab code repository link：

- https://github.com/sustechgem/RESnet-m

Run the following scripts:

- RUNME.py: A self-testing program to ensure a corrent installation and to demonstrate the workflow of software

- Example_Halfspace.py: Verification of the numerical accuracy by comparing with the analytic solution of a half-space

- Example_Casing.py: Simulation of the surface electric field with the presence of steel well casing

- Example_Infrastructure.py: Effect of complex metallic infrastructure on the surface dc resistivity data

#### Note：

This code only requires Numpy and Scipy for scientific computing and Matplotlib for data visualization. There are no specific requirements for the package version.

The current code implementation solves large sparse matrices by calling the MKL PARDISO interface PyPardiso.py; a testing script PyPardisoExample.py is provided along with the interface function. 
The PARDISO solver comes with the package "mkl" as part of the standard installation of Numpy in Anaconda. Sometimes a "Segmentation 
fault" error occurs when calling the mkl library. The problem can be fixed by creating a new environment and freshly installing the recommended package versions specified in environment.yml. If the codes do not run on your computer, it is very likely the solver does not work properly. You have the option of replacing it with your own solver or making sure the DLL file name (e.g. mkl_rt.1) is correctly specified in PyPardiso.py. The current PyPardiso.py has included a few variants of the DLL file that have been found in different Numpy installations. 
