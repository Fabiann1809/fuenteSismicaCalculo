"""
analysis/02_error_exploration.py
---------------------------------
Fase 3 + 4: Exploración de la función de error E(x, y, z).

Este script:
  1. Evalúa E en una grilla 3D (100 x 100 x 100 puntos).
  2. Genera mapas de calor para cortes z = k seleccionados.
  3. Marca en cada mapa el mínimo local y la posición real.
  4. Construye la curva E_min(z) y encuentra la profundidad óptima.
  5. Grafica la trayectoria del mínimo (x*(z), y*(z)).
  6. Guarda todas las figuras en figures/.

Ejecutar desde la raíz del repositorio:
    python analysis/02_error_exploration.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from datos.estaciones   import STATIONS, TRUE_SOURCE, NOISE_PARAMS
from src.modelo         import compute_amplitude
from src.ruido          import add_gaussian_noise
from src.funcion_error  import error_grid_slice, error_grid_3d

os.makedirs("figuras/heatmaps", exist_ok=True)

# ------------------------------------------------------------------
# 0. Reproducir datos observados (misma semilla que Fase 1)
# ------------------------------------------------------------------

source_pos  = np.array([TRUE_SOURCE["x0"], TRUE_SOURCE["y0"], TRUE_SOURCE["z0"]])
A0          = TRUE_SOURCE["A0"]
Az_clean    = compute_amplitude(source_pos, A0, STATIONS)
Az_observed = add_gaussian_noise(Az_clean, NOISE_PARAMS["alpha"],
                                  NOISE_PARAMS["mu"], seed=42)

x0_real, y0_real, z0_real = TRUE_SOURCE["x0"], TRUE_SOURCE["y0"], TRUE_SOURCE["z0"]

# ------------------------------------------------------------------
# 1. Definir grilla 3D  (100 x 100 x 100)
# ------------------------------------------------------------------

Nx, Ny, Nz = 100, 100, 100

x_grid = np.linspace(-8, 8, Nx)
y_grid = np.linspace(-8, 8, Ny)
z_grid = np.linspace(-3, 0, Nz)

print("=" * 60)
print("  FASE 3+4 — Exploración de la función de error E(x,y,z)")
print("=" * 60)
print(f"\n  Grilla: {Nx} x {Ny} x {Nz} = {Nx*Ny*Nz:,} puntos")
print(f"  x ∈ [{x_grid[0]}, {x_grid[-1]}] km")
print(f"  y ∈ [{y_grid[0]}, {y_grid[-1]}] km")
print(f"  z ∈ [{z_grid[0]:.2f}, {z_grid[-1]:.2f}] km\n")

# ------------------------------------------------------------------
# 2. Evaluar E en toda la grilla 3D
# ------------------------------------------------------------------

print("  Evaluando E(x,y,z) en la grilla 3D...")
E_cube = np.zeros((Nz, Ny, Nx))

for iz in tqdm(range(Nz), desc="  Procesando cortes z", ncols=60):
    E_cube[iz] = error_grid_slice(
        x_grid, y_grid, z_grid[iz], A0, STATIONS, Az_observed
    )

print(f"\n  E mínimo global: {E_cube.min():.6e}")
print(f"  E máximo global: {E_cube.max():.6e}")

# Guardar cubo para reutilización en script 03
np.save("figuras/E_cubo.npy", E_cube)
print("  Cubo guardado en: figuras/E_cubo.npy")

# ------------------------------------------------------------------
# 3. Curva E_min(z) y profundidad óptima
# ------------------------------------------------------------------

# Mínimo de E en el plano (x,y) para cada z
E_min_z   = E_cube.min(axis=(1, 2))          # shape (Nz,)

# Índice del z óptimo
iz_opt    = np.argmin(E_min_z)
z_opt     = z_grid[iz_opt]
E_opt     = E_min_z[iz_opt]

# Posición (x*, y*) en el plano óptimo
slice_opt = E_cube[iz_opt]
iy_opt, ix_opt = np.unravel_index(np.argmin(slice_opt), slice_opt.shape)
x_opt     = x_grid[ix_opt]
y_opt     = y_grid[iy_opt]


print(f"\n  Mínimo global encontrado:")
print(f"    x* = {x_opt:.3f} km  (real: {x0_real})")
print(f"    y* = {y_opt:.3f} km  (real: {y0_real})")
print(f"    z* = {z_opt:.3f} km  (real: {z0_real})")
print(f"    E* = {E_opt:.6e}")

# ------------------------------------------------------------------
# Guardar resultado de fuerza bruta para uso en script 04
# ------------------------------------------------------------------

os.makedirs("datos", exist_ok=True)
resultado_fuerza_bruta = {
    "x_opt": float(x_opt),
    "y_opt": float(y_opt),
    "z_opt": float(z_opt),
    "A0_fijo": float(A0),
    "E_opt": float(E_opt),
}

import json
with open("datos/resultado_fuerza_bruta.json", "w") as f:
    json.dump(resultado_fuerza_bruta, f, indent=2)
print(f"\n  Resultado de fuerza bruta guardado: datos/resultado_fuerza_bruta.json")
# ------------------------------------------------------------------
# 4. Figura 1 — Curva E_min(z)
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(z_grid, E_min_z, color='steelblue', linewidth=2)
ax.axvline(z_opt, color='tomato', linestyle='--',
           linewidth=1.5, label=f'$z^*$ = {z_opt:.3f} km (estimado)')
ax.axvline(z0_real, color='green', linestyle=':',
           linewidth=1.5, label=f'$z_0$ = {z0_real} km (real)')
ax.set_xlabel("Profundidad $z$ (km)", fontsize=12)
ax.set_ylabel("$E_{\\min}(z)$", fontsize=12)
ax.set_title("Función de error mínimo en función de la profundidad", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("figuras/emin_curve.png", dpi=150)
print("\n  Figura guardada: figuras/emin_curve.png")
plt.close()

# ------------------------------------------------------------------
# 5. Figura 2 — Trayectoria (x*(z), y*(z))
# ------------------------------------------------------------------

x_star = np.zeros(Nz)
y_star = np.zeros(Nz)

for iz in range(Nz):
    iy_, ix_ = np.unravel_index(np.argmin(E_cube[iz]), (Ny, Nx))
    x_star[iz] = x_grid[ix_]
    y_star[iz] = y_grid[iy_]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Trayectoria del mínimo $(x^*(z),\\, y^*(z))$",
             fontsize=13, fontweight='bold')

# x*(z)
axes[0].plot(z_grid, x_star, color='steelblue', linewidth=1.5)
axes[0].axhline(x0_real, color='tomato', linestyle='--',
                label=f'$x_0$ real = {x0_real}')
axes[0].axvline(z_opt, color='gray', linestyle=':',
                linewidth=1, label=f'$z^*$ = {z_opt:.3f}')
axes[0].set_xlabel("$z$ (km)")
axes[0].set_ylabel("$x^*(z)$ (km)")
axes[0].set_title("Componente $x$")
axes[0].legend(fontsize=9)
axes[0].grid(True, linestyle='--', alpha=0.5)

# y*(z)
axes[1].plot(z_grid, y_star, color='darkorange', linewidth=1.5)
axes[1].axhline(y0_real, color='tomato', linestyle='--',
                label=f'$y_0$ real = {y0_real}')
axes[1].axvline(z_opt, color='gray', linestyle=':',
                linewidth=1, label=f'$z^*$ = {z_opt:.3f}')
axes[1].set_xlabel("$z$ (km)")
axes[1].set_ylabel("$y^*(z)$ (km)")
axes[1].set_title("Componente $y$")
axes[1].legend(fontsize=9)
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("figuras/trajectory_xy.png", dpi=150)
print("  Figura guardada: figuras/trajectory_xy.png")
plt.close()

# ------------------------------------------------------------------
# 6. Figura 3 — Mapas de calor para z seleccionados
# ------------------------------------------------------------------

# Elegir 6 cortes representativos: incluir z_opt y z_real
z_candidates = [z_grid[0], z_grid[20], z_grid[40],
                z_opt, z0_real, z_grid[-1]]
# Eliminar duplicados manteniendo orden
seen = set()
z_selected = []
for z in z_candidates:
    key = round(z, 3)
    if key not in seen:
        seen.add(key)
        z_selected.append(z)

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("Mapas de calor de $E(x, y, z=k)$ para distintos cortes",
             fontsize=13, fontweight='bold')

for ax, z_k in zip(axes.ravel(), z_selected):
    iz_k   = np.argmin(np.abs(z_grid - z_k))
    E_slice = E_cube[iz_k]

    im = ax.imshow(
        np.log10(E_slice + 1e-30),
        origin='lower',
        extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
        aspect='auto',
        cmap='hot_r'
    )
    plt.colorbar(im, ax=ax, label='$\\log_{10}(E)$', fraction=0.046)

    # Mínimo local en este corte
    iy_k, ix_k = np.unravel_index(np.argmin(E_slice), E_slice.shape)
    ax.scatter(x_grid[ix_k], y_grid[iy_k],
               marker='*', s=150, color='cyan', zorder=5,
               label=f'mín ({x_grid[ix_k]:.2f}, {y_grid[iy_k]:.2f})')

    # Posición real de la fuente
    ax.scatter(x0_real, y0_real,
               marker='x', s=100, color='lime', linewidths=2,
               zorder=6, label=f'real ({x0_real}, {y0_real})')

    ax.set_title(f"$z$ = {z_grid[iz_k]:.3f} km", fontsize=10)
    ax.set_xlabel("$x$ (km)", fontsize=9)
    ax.set_ylabel("$y$ (km)", fontsize=9)
    ax.legend(fontsize=7, loc='upper right')

plt.tight_layout()
plt.savefig("figuras/heatmaps/heatmaps_selected.png", dpi=150)
print("  Figura guardada: figuras/heatmaps/heatmaps_selected.png")
plt.close()

# ------------------------------------------------------------------
# 7. Figura 4 — Mapa de calor detallado del corte óptimo
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))
E_opt_slice = E_cube[iz_opt]

im = ax.imshow(
    np.log10(E_opt_slice + 1e-30),
    origin='lower',
    extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
    aspect='auto',
    cmap='hot_r'
)
plt.colorbar(im, ax=ax, label='$\\log_{10}(E)$')

# Contornos
ax.contour(x_grid, y_grid,
           np.log10(E_opt_slice + 1e-30),
           levels=10, colors='white', linewidths=0.5, alpha=0.4)

ax.scatter(x_opt, y_opt, marker='*', s=250, color='cyan',
           zorder=5, label=f'Mínimo estimado ({x_opt:.2f}, {y_opt:.2f})')
ax.scatter(x0_real, y0_real, marker='x', s=200, color='lime',
           linewidths=2.5, zorder=6,
           label=f'Fuente real ({x0_real}, {y0_real})')
ax.scatter(STATIONS[:, 0], STATIONS[:, 1],
           marker='^', s=80, color='deepskyblue', zorder=4,
           label='Estaciones')

ax.set_title(f"Corte óptimo: $z^*$ = {z_opt:.3f} km", fontsize=12)
ax.set_xlabel("$x$ (km)", fontsize=11)
ax.set_ylabel("$y$ (km)", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figuras/heatmaps/heatmap_optimal.png", dpi=150)
print("  Figura guardada: figuras/heatmaps/heatmap_optimal.png")
plt.close()

print("\n  ¡Fase 3+4 completada!")
print(f"  Estimación: ({x_opt:.3f}, {y_opt:.3f}, {z_opt:.3f})")
print(f"  Real:       ({x0_real}, {y0_real}, {z0_real})")
err_pos = np.sqrt((x_opt-x0_real)**2 + (y_opt-y0_real)**2 + (z_opt-z0_real)**2)
print(f"  Error posicional: {err_pos:.4f} km")