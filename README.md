# MIMO-RLS-LQR Adaptive Governor for Multi-Cluster DVFS

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Implementation of an adaptive DVFS governor based on **MIMO Recursive Least Squares (MIMO-RLS)** online system identification and **Linear Quadratic Regulator (LQR)** optimal control.

**Author**: Stanislav Usychenko

## Features
- 6-dimensional state vector
- Real-time MIMO-RLS system identification with forgetting factor
- Periodic LQR gain recomputation
- Closed-loop stability check (spectral radius)
- Simulation of heterogeneous multi-cluster processor dynamics

## Installation

```bash
git clone https://github.com/Apuoxo/mimo-rls-lqr-governor.git
cd mimo-rls-lqr-governor
pip install -r requirements.txt