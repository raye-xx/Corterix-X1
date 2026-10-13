
Corterix-X1: Predictive Brain-Machine Compiler

An in-silico closed-loop Model Predictive Control (MPC) compiler designed to prevent thermal damage and glutamate excitotoxicity during high-rate neural writing. Submitted to the IRIS National Science Fair 2026.

## Key Benchmarks
- **Thermal Clamping:** Peak $\Delta T \le +0.38^\circ\text{C}$ (Safe limit: $+0.40^\circ\text{C}$)
- **Glutamate Bounding:** Peak $[\text{Glu}]_{\text{ext}} \le 8.4\ \mu\text{M}$ (Excitotoxic threshold: $10.0\ \mu\text{M}$)
- **Synaptic Plasticity:** 98.4% STDP matrix fidelity ($p < 0.001$, $n = 50$)
- **Decision Latency:** Mean $7.20\text{ ms}$ (Real-time hard deadline: $10.0\text{ ms}$)

## Core Modules
- 'sens_ana.py'- Thermal sensitivity parametric sweep across stimulation frequency (10–100 Hz) and microvascular perfusion rates (50%–130% physiological baseline)
- 'mont_carlo.py'- Monte Carlo execution latency profile of the Corterix-X1 ROM MPC governor (N = 10,000 cycles).
- 'main.py'- 3.	Multi-physics closed-loop validation of the Corterix-X1 Brain-Machine Compiler across 2.0-second stimulation sequences


## Quickstart
```bash
pip install numpy scipy matplotlib
python validation_suite.py
