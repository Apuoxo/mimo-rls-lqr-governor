[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

# MIMO-RLS-LQR Adaptive Governor for Multi-Cluster DVFS

Implementation of an adaptive DVFS governor based on **MIMO Recursive Least Squares (MIMO-RLS)** online system identification and **Linear Quadratic Regulator (LQR)** optimal control.

**Author**: Stanislav Usychenko

## Features

- 6-dimensional state vector
- Real-time MIMO-RLS system identification with forgetting factor
- Periodic LQR gain recomputation
- Closed-loop stability check (spectral radius)
- Simulation of heterogeneous multi-cluster processor dynamics

## Quick Start

```bash
python simulation.py