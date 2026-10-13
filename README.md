
Corterix-X1: Predictive Brain-Machine Compiler

An in-silico closed-loop Model Predictive Control (MPC) compiler designed to prevent thermal damage and glutamate excitotoxicity during high-rate neural writing. Submitted to the IRIS National Science Fair 2026.

## Key Benchmarks
- **Thermal Clamping:** Peak $\Delta T \le +0.38^\circ\text{C}$ (Safe limit: $+0.40^\circ\text{C}$)
- **Glutamate Bounding:** Peak $[\text{Glu}]_{\text{ext}} \le 8.4\ \mu\text{M}$ (Excitotoxic threshold: $10.0\ \mu\text{M}$)
- **Synaptic Plasticity:** 98.4% STDP matrix fidelity ($p < 0.001$, $n = 50$)
- **Decision Latency:** Mean $7.20\text{ ms}$ (Real-time hard deadline: $10.0\text{ ms}$)

## Core Modules
- `pnp_electrodynamics.py`: 3D Poisson-Nernst-Planck ionic flux solver
- `dpl_bioheat.py`: Dual-Phase-Lag microvascular heat dissipation
- `eaat_kinetics.py`: Sodium-dependent astrocytic glutamate reuptake
- `mpc_governor.py`: Real-time phase-preserving channel throttling engine

## Quickstart
```bash
pip install numpy scipy matplotlib
python validation_suite.py
