"""
analisis/06_analisis_hessiano.py
---------------------------------
Fase 6: Análisis del Hessiano, covarianza y barras de error.

Este script:
  1. Carga la solución estimada del método iterativo (ejecuta script 04).
  2. Calcula el jacobiano G en la solución y el Hessiano H = 2 G^T G.
  3. Estima la matriz de covarianza C = σ² (G^T G)^{-1}.
  4. Reporta las barras de error en cada parámetro.
  5. Analiza autovalores/autovectores de G^T G para identificar
     las direcciones mejor y peor resueltas en el espacio de parámetros.
  6. Genera figuras:
     - Mapa de calor del Hessiano y de la matriz de correlación.
     - Diagrama de barras de los autovalores.

Ejecutar desde la raíz del repositorio:
    python analisis/06_analisis_hessiano.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from datos.estaciones   import STATIONS, TRUE_SOURCE, NOISE_PARAMS
from src.modelo         import compute_amplitude
from src.ruido          import add_gaussian_noise
from src.jacobiano      import iterative_least_squares, compute_covariance_analysis

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

# ------------------------------------------------------------------
# 1. Obtener la solución estimada (corremos el iterativo desde un buen inicio)
# ------------------------------------------------------------------

print("=" * 65)
print("  FASE 6 — Análisis del Hessiano, covarianza y barras de error")
print("=" * 65)

m0 = np.array([0.0, 0.0, -1.0, 1.5])
res = iterative_least_squares(
    m0=m0,
    stations=STATIONS,
    Az_observed=Az_observed,
    max_iter=200,
    tol=1e-8,
    damping=1e-4,
)
m_hat = res['m_final']

print(f"\n  Solución estimada: m_hat = {np.round(m_hat, 4)}")
print(f"  Iteraciones: {res['n_iter']}, "
      f"E_final = {res['history_E'][-1]:.4e}\n")

# ------------------------------------------------------------------
# 2. Análisis de covarianza
# ------------------------------------------------------------------

analysis = compute_covariance_analysis(m_hat, STATIONS, Az_observed)

G              = analysis['G']
GtG            = analysis['GtG']
C              = analysis['covariance']
sigmas         = analysis['sigmas']
eigvals        = analysis['eigenvalues']
eigvecs        = analysis['eigenvectors']
cond           = analysis['condition_num']
sigma_residual = analysis['sigma_residual']
E_min          = analysis['E_min']
dof            = analysis['dof']

param_names = ['x0', 'y0', 'z0', 'A0']
param_real  = [x0_r, y0_r, z0_r, A0_real]

# ------------------------------------------------------------------
# 3. Reporte en consola
# ------------------------------------------------------------------

print("-" * 65)
print("  Estadística del residual")
print("-" * 65)
print(f"  M (estaciones)        : {len(STATIONS)}")
print(f"  p (parámetros)        : 4")
print(f"  Grados de libertad    : {dof}")
print(f"  E* (residual mínimo)  : {E_min:.4e}")
print(f"  σ residual estimada   : {sigma_residual:.4e}")

print("\n" + "-" * 65)
print("  Barras de error (1σ) por parámetro")
print("-" * 65)
print(f"  {'Param':>6} {'Real':>10} {'Estimado':>12} {'Sigma':>14}"
      f" {'Sigma rel.':>12}")
for i, name in enumerate(param_names):
    sigma_rel = sigmas[i] / abs(m_hat[i]) * 100 if m_hat[i] != 0 else np.nan
    print(f"  {name:>6} {param_real[i]:>10.4f} {m_hat[i]:>12.4f} "
          f"{sigmas[i]:>14.4e} {sigma_rel:>11.2f}%")

# Comprobar si los valores reales caen dentro de las barras de 2σ
print("\n  ¿Valor real dentro del intervalo de 2σ?")
for i, name in enumerate(param_names):
    lo = m_hat[i] - 2*sigmas[i]
    hi = m_hat[i] + 2*sigmas[i]
    dentro = lo <= param_real[i] <= hi
    marca = "✓" if dentro else "✗"
    print(f"    {marca} {name}: [{lo:.4f}, {hi:.4f}]  "
          f"real = {param_real[i]}")

print("\n" + "-" * 65)
print("  Autoanálisis de G^T G")
print("-" * 65)
print(f"  Número de condición κ = λ_max / λ_min = {cond:.4e}")
print(f"\n  Autovalores (de mayor = mejor resuelto a menor = peor resuelto):")
for i, lam in enumerate(eigvals):
    print(f"    λ_{i+1} = {lam:.4e}")

print(f"\n  Autovectores (direcciones en el espacio [x0, y0, z0, A0]):")
print(f"  {'':<8} {'v_1':>14} {'v_2':>14} {'v_3':>14} {'v_4':>14}")
print(f"  {'':<8} {'(mejor)':>14} {'':>14} {'':>14} {'(peor)':>14}")
for i, name in enumerate(param_names):
    print(f"  {name:<8} " + " ".join(f"{eigvecs[i,j]:>14.4f}" for j in range(4)))

# Identificar la coordenada dominante del peor autovector
peor = eigvecs[:, -1]
idx_dom = np.argmax(np.abs(peor))
print(f"\n  → La dirección peor resuelta está dominada por '{param_names[idx_dom]}' "
      f"(|componente| = {abs(peor[idx_dom]):.3f}).")
print(f"    Esto confirma cuantitativamente la baja sensibilidad en "
      f"{param_names[idx_dom]}.")

# ------------------------------------------------------------------
# 4. Figura 1 — Matriz de correlación
# ------------------------------------------------------------------

# Matriz de correlación a partir de la covarianza
D = np.sqrt(np.diag(C))
corr = C / np.outer(D, D)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle("Análisis del Hessiano en la solución estimada",
             fontsize=13, fontweight='bold')

# Hessiano (G^T G) en escala logarítmica
H = 2 * GtG
im0 = axes[0].imshow(np.log10(np.abs(H) + 1e-30), cmap='viridis')
axes[0].set_xticks(range(4))
axes[0].set_yticks(range(4))
axes[0].set_xticklabels(param_names)
axes[0].set_yticklabels(param_names)
axes[0].set_title("$\\log_{10}|H_{ij}|$  con  $H = 2\\, G^T G$")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

# Anotar valores
for i in range(4):
    for j in range(4):
        axes[0].text(j, i, f"{H[i,j]:.2e}",
                     ha='center', va='center',
                     color='white' if H[i,j] < H.max()*0.5 else 'black',
                     fontsize=8)

# Matriz de correlación
im1 = axes[1].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
axes[1].set_xticks(range(4))
axes[1].set_yticks(range(4))
axes[1].set_xticklabels(param_names)
axes[1].set_yticklabels(param_names)
axes[1].set_title("Matriz de correlación entre parámetros")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

for i in range(4):
    for j in range(4):
        axes[1].text(j, i, f"{corr[i,j]:.2f}",
                     ha='center', va='center',
                     color='black', fontsize=9)

plt.tight_layout()
plt.savefig("figuras/hessiano_correlacion.png", dpi=150)
print("\n  Figura guardada: figuras/hessiano_correlacion.png")
plt.close()

# ------------------------------------------------------------------
# 5. Figura 2 — Autovalores de G^T G
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))
indices = np.arange(1, 5)
bars = ax.bar(indices, eigvals, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'],
              edgecolor='black', linewidth=0.8)
ax.set_yscale('log')
ax.set_xticks(indices)
ax.set_xticklabels([f'$\\lambda_{i}$' for i in indices])
ax.set_ylabel('Autovalor (escala log)', fontsize=11)
ax.set_title(f"Autovalores de $G^T G$  —  $\\kappa$ = {cond:.2e}",
             fontsize=12)

# Anotar valores
for bar, lam in zip(bars, eigvals):
    ax.text(bar.get_x() + bar.get_width()/2, lam,
            f'{lam:.2e}', ha='center', va='bottom', fontsize=9)

ax.grid(True, linestyle='--', alpha=0.4, which='both', axis='y')
plt.tight_layout()
plt.savefig("figuras/autovalores_GtG.png", dpi=150)
print("  Figura guardada: figuras/autovalores_GtG.png")
plt.close()

# ------------------------------------------------------------------
# 6. Figura 3 — Solución con barras de error
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))
xpos = np.arange(4)
ax.errorbar(xpos, m_hat, yerr=2*sigmas, fmt='o',
            color='steelblue', ecolor='steelblue',
            elinewidth=2, capsize=8, markersize=10,
            label='Estimación ± 2σ')
ax.scatter(xpos, param_real, marker='x', s=150, color='tomato',
           linewidths=2.5, zorder=5, label='Valor real')

ax.set_xticks(xpos)
ax.set_xticklabels(param_names, fontsize=12)
ax.set_ylabel('Valor', fontsize=11)
ax.set_title('Parámetros estimados con barras de error (2σ)',
             fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("figuras/parametros_con_error.png", dpi=150)
print("  Figura guardada: figuras/parametros_con_error.png")
plt.close()

print("\n  ¡Fase 6 completada!\n")