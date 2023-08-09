# Ising Model for Ferromagnets

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

The code shows the theory and implementation of the Ising Model for Ferromagnets. The general steps are:

1. Create a random initial grid, with each value in the grid depicting either $+1$ spin or a $-1$ spin.
1. Get the Energy and the Average spin of the system for the initial grids.
1. Create a function for calculating the equilibrium state after the initial states evolve for some large number of time steps. This function is the **Metropolis Algorithm**
1. Calculate how these equilibrium state energies and average spins change with a change in the temperature of the bath.
1. Calculate the changes in Heat Capacity as the bath temperature increases.
