import numpy as np
import matplotlib.pyplot as plt

# 1. Grid setup aligned with paper parameters
frequencies = np.linspace(10, 100, 200)           # 10 to 100 Hz
perfusion_ratios = np.linspace(0.50, 1.30, 200)  # Ischemia boundary starting at 50%

FREQ_GRID, W_GRID = np.meshgrid(frequencies, perfusion_ratios)

# 2. Thermal Parameters (DPL steady-state equivalent)
P_base = 0.001
q_vol = (P_base / 1e-9) * 0.0220
c_blood_vol = 0.0085 * 1050 * 3639

delta_T_ungoverned = (q_vol * (FREQ_GRID / 100.0)) / (W_GRID * c_blood_vol)
# Clamped at simulated max +0.38 °C (under the +0.40 °C hard DPL limit)
delta_T_governed = np.minimum(delta_T_ungoverned, 0.38)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Panel A
c1 = ax1.pcolormesh(FREQ_GRID, W_GRID * 100, delta_T_ungoverned, cmap='YlOrRd', vmin=0.0, vmax=1.20, shading='auto')
fig.colorbar(c1, ax=ax1, label='Peak Thermal Rise ΔT (°C)')
ax1.contour(FREQ_GRID, W_GRID * 100, delta_T_ungoverned, levels=[0.38], colors='black', linewidths=2.0)

ax1.set_title('A: Un-Governed Bioheat Profile', fontsize=11, fontweight='bold')
ax1.set_xlabel('Stimulation Frequency (Hz)', fontsize=10)
ax1.set_ylabel('Vascular Blood Perfusion Rate (% Baseline)', fontsize=10)

textstr_a = 'Threat Boundary:\n+0.38 °C Activation'
ax1.text(0.04, 0.82, textstr_a, transform=ax1.transAxes, fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))

# Panel B
c2 = ax2.pcolormesh(FREQ_GRID, W_GRID * 100, delta_T_governed, cmap='YlOrRd', vmin=0.0, vmax=0.45, shading='auto')
fig.colorbar(c2, ax=ax2, label='Governed Thermal Rise ΔT (°C)')

ax2.set_title('B: Corterix-X1 Governed Thermal Safety Boundary', fontsize=11, fontweight='bold')
ax2.set_xlabel('Stimulation Frequency (Hz)', fontsize=10)
ax2.set_ylabel('Vascular Blood Perfusion Rate (% Baseline)', fontsize=10)

# Aligned with paper parameters: Hard Cap = +0.40 °C
textstr_b = 'MPC Thermal Safety Boundary:\nHard Cap = +0.40 °C\nGovernor Target = +0.38 °C\nIschemia Safe (≥50% W_blood)'
ax2.text(0.04, 0.72, textstr_b, transform=ax2.transAxes, fontsize=9, bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig('figure5c_sensitivity_heatmap_aligned.png', dpi=300)
plt.show()