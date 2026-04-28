CAR T Tumor Simulation

Description

This program simulates tumor growth and CAR T therapy in a three dimensional tissue volume.

The model includes functional CAR T cells exhausted CAR T cells antigen positive tumor cells antigen low tumor cells chemokine immunosuppression and hypoxia.

The program uses a reaction diffusion chemotaxis model. Reactions are solved by a second order Runge Kutta method. Diffusion is solved implicitly with Jacobi iterations.

Requirements

A C ++ 17 compiler is required. OpenMP is optional and can be used for parallel execution.

Build

Compile the source file with any modern C plus plus compiler. Enable OpenMP only if your compiler supports it.

Run

Start the program from the command line. Parameters can be passed as key value arguments.

Main parameters

nx ny nz set the grid size.
Lx Ly Lz set the domain size.
dt sets the time step.
T sets the final simulation time.
output_every sets how often results are saved.
out_dir sets the output folder.
threads sets the number of OpenMP threads.

Output

The program saves CSV files to the output folder. Each file contains time grid indices coordinates and values of all model variables.

Variables

C functional CAR T cells.
E exhausted CAR T cells.
TA antigen positive tumor cells.
TB antigen low tumor cells.
S chemoattractant.
A immunosuppressive factor.
H hypoxia level.

Purpose

The program is intended for numerical study of tumor dynamics during CAR T therapy including cell infiltration exhaustion antigen escape immunosuppression and hypoxia.
