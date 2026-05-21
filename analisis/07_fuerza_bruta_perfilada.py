"""
analisis/07_fuerza_bruta_perfilada.py
--------------------------------------
Fase 7: Fuerza bruta con función de error perfilada.

Este script:
  1. Calcula la función de error perfilada E_tilde(x,y,z) sobre la grilla 3D,
     donde A0 ya está estimado analíticamente en cada celda.
  2. Encuentra el mínimo global: (x*, y*, z*, A0*).
  3. Compara con el resultado del método iterativo (Fase 5).
  4. Genera un nuevo JSON con la solución 4D de fuerza bruta.

Esto resuelve la inconsistencia de la versión anterior donde la fuerza bruta
fijaba A0=2.0 mientras que el iterativo lo estimaba.

Ejecutar desde la raíz del repositorio:
    python analisis/07_fuerza_bruta_perfilada.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from datos.estaciones    import STATIONS, TRUE_SOURCE, NOISE_PARAMS
from src.modelo          import compute_amplitude
from src.ruido           import add_gaussian_noise
from src.funcion_error   import error_profiled_slice

os.makedirs("figuras", exist_ok=True)
os.makedirs("datos", exist_ok=True)

# ------------------------------------------------------------------
# 0. Reproducir datos observados
# ------------------------------------------------------------------

pos_real    = np.array([TRUE_SOURCE["x0"], TRUE_SOURCE["y0"], TRUE_SOURCE["z0"]])
A0_real     = TRUE_SOURCE["A0"]
Az_clean    = compute_amplitude(pos_real, A0_real, STATIONS)
Az_observed = add_gaussian_noise(Az_clean, NOISE_PARAMS["alpha"],
                                  NOISE_PARAMS["mu"], seed=42)

x0_r, y0_r, z0_r = TRUE_SOURCE["x0"], TRUE_SOURCE["y0"], TRUE_SOURCE["z0"]

# ------------------------------------------------------------------
# 1. Grilla 3D (misma resolución que el script 02)
# ------------------------------------------------------------------

Nx, Ny, Nz = 100, 100, 100
x_grid = np.linspace(-8, 8, Nx)
y_grid = np.linspace(-8, 8, Ny)
z_grid = np.linspace(-3, 0, Nz)

print("=" * 65)
print("  FASE 7 — Fuerza bruta con función de error perfilada")
print("=" * 65)
print(f"\n  Grilla: {Nx}x{Ny}x{Nz} = {Nx*Ny*Nz:,} puntos")
print(f"  A0 se estima analíticamente en cada celda (no se fija a priori).\n")

# ------------------------------------------------------------------
# 2. Evaluar E perfilada en toda la grilla
# ------------------------------------------------------------------

E_cube_profiled  = np.zeros((Nz, Ny, Nx))
A0_cube_profiled = np.zeros((Nz, Ny, Nx))

for iz in tqdm(range(Nz), desc="  Cortes z", ncols=60):
    E_slice, A0_slice = error_profiled_slice(
        x_grid, y_grid, z_grid[iz], STATIONS, Az_observed
    )
    E_cube_profiled[iz]  = E_slice
    A0_cube_profiled[iz] = A0_slice

np.save("figuras/E_cubo_perfilado.npy",  E_cube_profiled)
np.save("figuras/A0_cubo_perfilado.npy", A0_cube_profiled)
print("\n  Cubos guardados: figuras/E_cubo_perfilado.npy, figuras/A0_cubo_perfilado.npy")

# ------------------------------------------------------------------
# 3. Mínimo global 4D
# ------------------------------------------------------------------

iz_opt, iy_opt, ix_opt = np.unravel_index(
    np.argmin(E_cube_profiled), E_cube_profiled.shape
)

x_opt  = x_grid[ix_opt]
y_opt  = y_grid[iy_opt]
z_opt  = z_grid[iz_opt]
A0_opt = A0_cube_profiled[iz_opt, iy_opt, ix_opt]
E_opt  = E_cube_profiled[iz_opt, iy_opt, ix_opt]

print("\n" + "-" * 55)
print("  Resultado: fuerza bruta perfilada (4D)")
print("-" * 55)
print(f"  x* = {x_opt:.4f} km   (real: {x0_r})")
print(f"  y* = {y_opt:.4f} km   (real: {y0_r})")
print(f"  z* = {z_opt:.4f} km   (real: {z0_r})")
print(f"  A0* = {A0_opt:.4f}    (real: {A0_real})")
print(f"  E* = {E_opt:.4e}")

err_pos = np.sqrt((x_opt-x0_r)**2 + (y_opt-y0_r)**2 + (z_opt-z0_r)**2)
print(f"  Error posicional: {err_pos:.4f} km")

# ------------------------------------------------------------------
# 4. Guardar resultado para uso en script 04
# ------------------------------------------------------------------

resultado = {
    "metodo"       : "fuerza_bruta_perfilada",
    "x_opt"        : float(x_opt),
    "y_opt"        : float(y_opt),
    "z_opt"        : float(z_opt),
    "A0_opt"       : float(A0_opt),
    "E_opt"        : float(E_opt),
    "error_pos_km" : float(err_pos),
    "grilla"       : [Nx, Ny, Nz],
}

with open("datos/resultado_fuerza_bruta_perfilada.json", "w") as f:
    json.dump(resultado, f, indent=2)
print(f"\n  Resultado guardado: datos/resultado_fuerza_bruta_perfilada.json")

# ------------------------------------------------------------------
# 5. Figura — Comparación E original (A0 fijo) vs. E perfilada
# ------------------------------------------------------------------

# Cargar el cubo original (A0=2.0 fijo) si existe
cubo_original_path = "figuras/E_cubo.npy"
if os.path.exists(cubo_original_path):
    E_cube_original = np.load(cubo_original_path)

    E_min_z_orig  = E_cube_original.min(axis=(1, 2))
    E_min_z_prof  = E_cube_profiled.min(axis=(1, 2))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(z_grid, E_min_z_orig, color='steelblue', linewidth=2,
                label='E con $A_0 = 2.0$ (fijo)')
    ax.semilogy(z_grid, E_min_z_prof, color='tomato', linewidth=2,
                label='E perfilada ($\\hat{A}_0$ analítico)')
    ax.axvline(z0_r, color='seagreen', linestyle=':',
               linewidth=2, label=f'$z_0$ real = {z0_r} km')
    ax.set_xlabel("Profundidad $z$ (km)", fontsize=11)
    ax.set_ylabel("$E_{\\min}(z)$ (escala log)", fontsize=11)
    ax.set_title("Comparación: error con $A_0$ fijo vs. perfilado",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4, which='both')
    plt.tight_layout()
    plt.savefig("figuras/comparacion_perfilada.png", dpi=150)
    print(f"  Figura guardada: figuras/comparacion_perfilada.png")
    plt.close()

# ------------------------------------------------------------------
# 6. Figura — Mapa de A0 estimado en el plano óptimo
# ------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(f"Plano óptimo $z^* = {z_opt:.3f}$ km",
             fontsize=13, fontweight='bold')

# E perfilada
im0 = axes[0].imshow(np.log10(E_cube_profiled[iz_opt] + 1e-30),
                      origin='lower',
                      extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                      aspect='auto', cmap='hot_r')
axes[0].scatter(x_opt, y_opt, marker='*', s=200, color='cyan',
                zorder=5, label='Mínimo')
axes[0].scatter(x0_r, y0_r, marker='x', s=150, color='lime',
                linewidths=2.5, zorder=6, label='Real')
axes[0].set_xlabel("$x$ (km)")
axes[0].set_ylabel("$y$ (km)")
axes[0].set_title("$\\log_{10}(\\tilde{E})$ perfilada")
axes[0].legend(fontsize=9)
plt.colorbar(im0, ax=axes[0], fraction=0.046)

# A0 estimado
im1 = axes[1].imshow(A0_cube_profiled[iz_opt],
                      origin='lower',
                      extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
                      aspect='auto', cmap='viridis',
                      vmin=0, vmax=4)
axes[1].scatter(x_opt, y_opt, marker='*', s=200, color='red',
                zorder=5, label=f'$\\hat{{A}}_0$ = {A0_opt:.3f}')
axes[1].scatter(x0_r, y0_r, marker='x', s=150, color='white',
                linewidths=2.5, zorder=6, label=f'Real = {A0_real}')
axes[1].set_xlabel("$x$ (km)")
axes[1].set_ylabel("$y$ (km)")
axes[1].set_title("$\\hat{A}_0(x, y)$ óptimo analítico")
axes[1].legend(fontsize=9)
plt.colorbar(im1, ax=axes[1], fraction=0.046)

plt.tight_layout()
plt.savefig("figuras/mapa_A0_perfilado.png", dpi=150)
print(f"  Figura guardada: figuras/mapa_A0_perfilado.png")
plt.close()

print("\n  ¡Fase 7 completada!")