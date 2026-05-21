"""
analisis/01_simular_datos.py
-----------------------------
Fase 1 + 2: Verificación del setup y simulación de datos sísmicos.

Este script:
  1. Carga las estaciones y parámetros del sismo real.
  2. Calcula las amplitudes limpias del modelo.
  3. Agrega ruido gaussiano para obtener los datos 'observados'.
  4. Imprime una tabla resumen.
  5. Genera una figura de verificación geométrica.

Ejecutar desde la raíz del repositorio:
    python analisis/01_simular_datos.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from datos.estaciones import STATIONS, TRUE_SOURCE, NOISE_PARAMS, N_STATIONS
from src.modelo import compute_amplitude, compute_distances
from src.ruido import add_gaussian_noise

# ------------------------------------------------------------------
# 1. Extraer parámetros reales
# ------------------------------------------------------------------

x0 = TRUE_SOURCE["x0"]
y0 = TRUE_SOURCE["y0"]
z0 = TRUE_SOURCE["z0"]
A0 = TRUE_SOURCE["A0"]

source_pos = np.array([x0, y0, z0])
alpha      = NOISE_PARAMS["alpha"]
mu         = NOISE_PARAMS["mu"]

# ------------------------------------------------------------------
# 2. Calcular amplitudes limpias y distancias
# ------------------------------------------------------------------

Az_clean  = compute_amplitude(source_pos, A0, STATIONS)
distances = compute_distances(source_pos, STATIONS)

# ------------------------------------------------------------------
# 3. Agregar ruido gaussiano
# ------------------------------------------------------------------

Az_observed = add_gaussian_noise(Az_clean, alpha=alpha, mu=mu, seed=42)
noise_added = Az_observed - Az_clean

# ------------------------------------------------------------------
# 4. Tabla resumen en consola
# ------------------------------------------------------------------

print("=" * 70)
print("  FASE 1 — Configuración del escenario sísmico")
print("=" * 70)
print(f"\n  Fuente real: x0={x0}, y0={y0}, z0={z0} km  |  A0={A0}")
print(f"  Ruido: alpha={alpha}, mu={mu}")
print(f"  Estaciones: {N_STATIONS}\n")

header = f"{'Est':>4} {'xi':>7} {'yi':>7} {'zi':>7} {'Ri (km)':>10} {'Az_clean':>12} {'Az_obs':>12} {'ruido':>10}"
print(header)
print("-" * len(header))

for i in range(N_STATIONS):
    xi, yi, zi = STATIONS[i]
    print(f"  S{i+1:1d}  {xi:7.2f} {yi:7.2f} {zi:7.2f} "
          f"{distances[i]:10.4f} {Az_clean[i]:12.6f} "
          f"{Az_observed[i]:12.6f} {noise_added[i]:10.6f}")

print("\n  Vector Az observado (datos del problema):")
print(" ", np.round(Az_observed, 6))

# ------------------------------------------------------------------
# 5. Figura de verificación geométrica
# ------------------------------------------------------------------

fig = plt.figure(figsize=(13, 5))
fig.suptitle("Fase 1: Configuración geométrica del escenario sísmico",
             fontsize=13, fontweight='bold')

gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# --- Vista superior (plano XY) ---
ax1 = fig.add_subplot(gs[0])
ax1.scatter(STATIONS[:, 0], STATIONS[:, 1],
            marker='^', s=120, color='black', zorder=5, label='Estaciones')

for i, (xi, yi, _) in enumerate(STATIONS):
    ax1.annotate(f'S{i+1}', (xi, yi),
                 textcoords="offset points", xytext=(6, 4), fontsize=8)

ax1.scatter(x0, y0, marker='x', s=200, color='red',
            linewidths=2.5, zorder=6, label=f'Fuente real ({x0},{y0})')

ax1.set_xlabel("x (km)")
ax1.set_ylabel("y (km)")
ax1.set_title("Vista superior (plano XY)")
ax1.legend(fontsize=8)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_aspect('equal')

# --- Amplitudes observadas vs. distancia ---
ax2 = fig.add_subplot(gs[1])
ax2.scatter(distances, Az_clean, color='steelblue', s=80,
            label='Az limpia (modelo)', zorder=5)
ax2.scatter(distances, Az_observed, color='tomato', s=60,
            marker='D', label='Az observada (con ruido)', zorder=4)

for i in range(N_STATIONS):
    ax2.plot([distances[i], distances[i]],
             [Az_clean[i], Az_observed[i]],
             color='gray', linewidth=0.8, linestyle='--')
    ax2.annotate(f'S{i+1}', (distances[i], Az_observed[i]),
                 textcoords="offset points", xytext=(4, 3), fontsize=7)

ax2.set_xlabel("Distancia R_i (km)")
ax2.set_ylabel("Amplitud")
ax2.set_title("Amplitudes: modelo vs. observadas")
ax2.legend(fontsize=8)
ax2.grid(True, linestyle='--', alpha=0.5)

plt.savefig("figuras/fase1_verificacion.png", dpi=150, bbox_inches='tight')
print("\n  Figura guardada en: figuras/fase1_verificacion.png")
plt.show()