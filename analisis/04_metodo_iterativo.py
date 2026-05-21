"""
analisis/04_metodo_iterativo.py
--------------------------------
Fase 5: Método iterativo de mínimos cuadrados (Jacobiano).

Este script:
  1. Define distintos puntos de partida m0 para probar robustez.
  2. Corre el método iterativo desde cada punto inicial.
  3. Compara la solución iterativa con la solución por fuerza bruta.
  4. Grafica la convergencia: E(k), ||Δm||(k), trayectoria en XY.
  5. Genera tabla resumen de resultados.

Ejecutar desde la raíz del repositorio:
    python analisis/04_metodo_iterativo.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from datos.estaciones   import STATIONS, TRUE_SOURCE, NOISE_PARAMS
from src.modelo         import compute_amplitude
from src.ruido          import add_gaussian_noise
from src.jacobiano      import iterative_least_squares
from src.funcion_error  import error_single

os.makedirs("figuras", exist_ok=True)

# ------------------------------------------------------------------
# 0. Reproducir datos observados
# ------------------------------------------------------------------

source_pos  = np.array([TRUE_SOURCE["x0"], TRUE_SOURCE["y0"], TRUE_SOURCE["z0"]])
A0_real     = TRUE_SOURCE["A0"]
Az_clean    = compute_amplitude(source_pos, A0_real, STATIONS)
Az_observed = add_gaussian_noise(Az_clean, NOISE_PARAMS["alpha"],
                                  NOISE_PARAMS["mu"], seed=42)

x0_r, y0_r, z0_r = TRUE_SOURCE["x0"], TRUE_SOURCE["y0"], TRUE_SOURCE["z0"]

# Resultados de fuerza bruta cargados desde scripts 02 y 07
import json

m_brute_fijo      = None
m_brute_perfilada = None

ruta_fijo = "datos/resultado_fuerza_bruta.json"
if os.path.exists(ruta_fijo):
    with open(ruta_fijo, "r") as f:
        rb = json.load(f)
    m_brute_fijo = np.array([rb["x_opt"], rb["y_opt"], rb["z_opt"], rb["A0_fijo"]])
    print(f"  Fuerza bruta (A0 fijo) cargada: {m_brute_fijo}")

ruta_perfilada = "datos/resultado_fuerza_bruta_perfilada.json"
if os.path.exists(ruta_perfilada):
    with open(ruta_perfilada, "r") as f:
        rp = json.load(f)
    m_brute_perfilada = np.array([rp["x_opt"], rp["y_opt"],
                                    rp["z_opt"], rp["A0_opt"]])
    print(f"  Fuerza bruta (perfilada) cargada: {m_brute_perfilada}")

if m_brute_fijo is None and m_brute_perfilada is None:
    print("  [!] No se encontró ningún resultado de fuerza bruta.")
    print("      Ejecuta primero los scripts 02 y/o 07.")
    sys.exit(1)

# ------------------------------------------------------------------
# 1. Puntos de partida
# ------------------------------------------------------------------

initial_models = {
    "Cerca (0, 0, -1)"        : np.array([ 0.0,  0.0, -1.0,  1.5]),
    "Lejos NE (4, 4, -0.5)"   : np.array([ 4.0,  4.0, -0.5,  1.0]),
    "Lejos SO (-5, -5, -2)"   : np.array([-5.0, -5.0, -2.0,  3.0]),
    "Muy cerca (-0.5, 0, -1)" : np.array([-0.5,  0.0, -1.0,  2.0]),
}

# ------------------------------------------------------------------
# 2. Correr método iterativo desde cada punto inicial
# ------------------------------------------------------------------

print("=" * 65)
print("  FASE 5 — Método iterativo de mínimos cuadrados (Jacobiano)")
print("=" * 65)
print(f"\n  Fuente real: ({x0_r}, {y0_r}, {z0_r}, A0={A0_real})\n")

results = {}

for label, m0 in initial_models.items():
    res = iterative_least_squares(
    m0          = m0,
    stations    = STATIONS,
    Az_observed = Az_observed,
    max_iter    = 200,
    tol         = 1e-8,
    damping     = 1e-4,   # Levenberg-Marquardt suave: evita divergencia
)   

    results[label] = res
    mf = res['m_final']
    err_pos = np.sqrt((mf[0]-x0_r)**2 + (mf[1]-y0_r)**2 + (mf[2]-z0_r)**2)

    status = "✓ Convergió" if res['converged'] else "✗ No convergió"
    print(f"  [{status}] {label}")
    print(f"    Inicio : ({m0[0]:.1f}, {m0[1]:.1f}, {m0[2]:.1f}, A0={m0[3]:.1f})")
    print(f"    Final  : ({mf[0]:.4f}, {mf[1]:.4f}, {mf[2]:.4f}, A0={mf[3]:.4f})")
    print(f"    Iters  : {res['n_iter']}  |  E final: {res['history_E'][-1]:.4e}"
          f"  |  Error pos: {err_pos:.4f} km\n")

# ------------------------------------------------------------------
# 3. Figura 1 — Convergencia de E(k) para todos los inicios
# ------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Convergencia del método iterativo de mínimos cuadrados",
             fontsize=13, fontweight='bold')

colors = ['steelblue', 'darkorange', 'tomato', 'mediumseagreen']

for (label, res), color in zip(results.items(), colors):
    iters = np.arange(len(res['history_E']))
    axes[0].semilogy(iters, res['history_E'], color=color,
                     linewidth=2, label=label)
    axes[1].semilogy(iters[1:], res['history_dm'][1:], color=color,
                     linewidth=2, label=label)

axes[0].set_xlabel("Iteración $k$", fontsize=11)
axes[0].set_ylabel("$E(\\mathbf{m}_k)$  (escala log)", fontsize=11)
axes[0].set_title("Error de ajuste $E$ vs. iteración")
axes[0].legend(fontsize=8)
axes[0].grid(True, linestyle='--', alpha=0.5)

axes[1].set_xlabel("Iteración $k$", fontsize=11)
axes[1].set_ylabel("$\\|\\Delta\\mathbf{m}\\|$  (escala log)", fontsize=11)
axes[1].set_title("Norma de la corrección $\\|\\Delta m\\|$ vs. iteración")
axes[1].legend(fontsize=8)
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("figuras/convergencia.png", dpi=150)
print("  Figura guardada: figuras/convergencia.png")
plt.close()

# ------------------------------------------------------------------
# 4. Figura 2 — Trayectoria iterativa en el plano XY
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 7))
ax.set_title("Trayectoria iterativa en el plano $XY$", fontsize=12)

for (label, res), color in zip(results.items(), colors):
    hist = res['history_m']
    ax.plot(hist[:, 0], hist[:, 1], '-o', color=color,
            markersize=3, linewidth=1.5, label=label, alpha=0.8)
    # Punto inicial
    ax.scatter(hist[0, 0], hist[0, 1], marker='s', s=80,
               color=color, zorder=5, edgecolors='black', linewidth=0.8)
    # Punto final
    ax.scatter(hist[-1, 0], hist[-1, 1], marker='D', s=80,
               color=color, zorder=5, edgecolors='black', linewidth=0.8)

# Fuente real
ax.scatter(x0_r, y0_r, marker='*', s=300, color='gold',
           edgecolors='black', linewidth=1, zorder=10,
           label=f'Fuente real ({x0_r}, {y0_r})')

# Estaciones
ax.scatter(STATIONS[:, 0], STATIONS[:, 1], marker='^', s=80,
           color='black', zorder=6, label='Estaciones')
for i, (xi, yi, _) in enumerate(STATIONS):
    ax.annotate(f'S{i+1}', (xi, yi), xytext=(5, 4),
                textcoords='offset points', fontsize=7)

ax.set_xlabel("$x$ (km)", fontsize=11)
ax.set_ylabel("$y$ (km)", fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
plt.tight_layout()
plt.savefig("figuras/trayectoria_iterativa.png", dpi=150)
print("  Figura guardada: figuras/trayectoria_iterativa.png")
plt.close()

# ------------------------------------------------------------------
# 5. Figura 3 — Evolución de cada parámetro (mejor inicio)
# ------------------------------------------------------------------

# Forzar el inicio "Cerca" para la figura de evolución
# (coincide con la descripción del documento LaTeX)
best_label = "Cerca (0, 0, -1)"
best_res   = results[best_label]
hist_m     = best_res['history_m']
param_names = ['$x_0$ (km)', '$y_0$ (km)', '$z_0$ (km)', '$A_0$']
param_real  = [x0_r, y0_r, z0_r, A0_real]

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
fig.suptitle(f"Evolución de parámetros — inicio: {best_label}",
             fontsize=12, fontweight='bold')

for idx, (ax, name, real) in enumerate(zip(axes.ravel(), param_names, param_real)):
    iters = np.arange(len(hist_m))
    ax.plot(iters, hist_m[:, idx], color='steelblue', linewidth=2)
    ax.axhline(real, color='tomato', linestyle='--',
               linewidth=1.5, label=f'Real = {real}')
    ax.set_xlabel("Iteración $k$")
    ax.set_ylabel(name)
    ax.set_title(f"Convergencia de {name}")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("figuras/evolucion_parametros.png", dpi=150)
print("  Figura guardada: figuras/evolucion_parametros.png")
plt.close()

# ------------------------------------------------------------------
# 6. Tabla comparativa final
# ------------------------------------------------------------------

print("\n" + "=" * 65)
print("  COMPARACIÓN FINAL")
print("=" * 65)
print(f"\n  {'Método':<30} {'x*':>8} {'y*':>8} {'z*':>8} {'A0*':>8} {'Err(km)':>10}")
print("  " + "-" * 63)

# Real
print(f"  {'Fuente real':<30} {x0_r:>8.3f} {y0_r:>8.3f} {z0_r:>8.3f} "
      f"{A0_real:>8.3f} {'—':>10}")

# Fuerza bruta con A0 fijo
if m_brute_fijo is not None:
    eb = np.sqrt((m_brute_fijo[0]-x0_r)**2
                 + (m_brute_fijo[1]-y0_r)**2
                 + (m_brute_fijo[2]-z0_r)**2)
    print(f"  {'Fuerza bruta (A0=2 fijo)':<30} "
          f"{m_brute_fijo[0]:>8.3f} {m_brute_fijo[1]:>8.3f} "
          f"{m_brute_fijo[2]:>8.3f} {m_brute_fijo[3]:>8.3f} {eb:>10.4f}")

# Fuerza bruta perfilada (4D)
if m_brute_perfilada is not None:
    eb = np.sqrt((m_brute_perfilada[0]-x0_r)**2
                 + (m_brute_perfilada[1]-y0_r)**2
                 + (m_brute_perfilada[2]-z0_r)**2)
    print(f"  {'Fuerza bruta (perfilada 4D)':<30} "
          f"{m_brute_perfilada[0]:>8.3f} {m_brute_perfilada[1]:>8.3f} "
          f"{m_brute_perfilada[2]:>8.3f} {m_brute_perfilada[3]:>8.3f} {eb:>10.4f}")
# Iterativo
for label, res in results.items():
    mf  = res['m_final']
    ep  = np.sqrt((mf[0]-x0_r)**2 + (mf[1]-y0_r)**2 + (mf[2]-z0_r)**2)
    tag = label[:28]
    print(f"  {tag:<30} {mf[0]:>8.4f} {mf[1]:>8.4f} {mf[2]:>8.4f} "
          f"{mf[3]:>8.4f} {ep:>10.4f}")

print()