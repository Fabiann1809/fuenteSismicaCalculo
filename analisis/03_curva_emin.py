"""
analisis/03_curva_emin.py
--------------------------
Fase 4 (parte 2): Construcción y análisis de la función reducida

    E_min(z) = min_{x,y} E(x, y, z)

Este script:
  1. Carga el cubo E(x,y,z) si ya fue calculado, o lo recalcula.
  2. Construye la curva E_min(z).
  3. Determina z* (profundidad óptima) y la posición (x*, y*) asociada.
  4. Analiza la estabilidad: ancho del pozo mínimo (FWHM).
  5. Genera todas las figuras relacionadas con E_min(z).

Puede ejecutarse de forma independiente al script 02.

Ejecutar desde la raíz del repositorio:
    python analisis/03_curva_emin.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from datos.estaciones       import STATIONS as ESTACIONES
from datos.estaciones       import TRUE_SOURCE as FUENTE_REAL
from datos.estaciones       import NOISE_PARAMS as PARAMS_RUIDO
from src.modelo             import compute_amplitude as calcular_amplitud
from src.ruido              import add_gaussian_noise as agregar_ruido_gaussiano
from src.funcion_error      import error_grid_slice as error_corte_2d

os.makedirs("figuras", exist_ok=True)

# ------------------------------------------------------------------
# 0. Reproducir datos observados (misma semilla)
# ------------------------------------------------------------------

pos_real    = np.array([FUENTE_REAL["x0"], FUENTE_REAL["y0"], FUENTE_REAL["z0"]])
A0          = FUENTE_REAL["A0"]
Az_limpia   = calcular_amplitud(pos_real, A0, ESTACIONES)
Az_obs      = agregar_ruido_gaussiano(Az_limpia, PARAMS_RUIDO["alpha"],
                                       PARAMS_RUIDO["mu"], seed=42)

x0_r = FUENTE_REAL["x0"]
y0_r = FUENTE_REAL["y0"]
z0_r = FUENTE_REAL["z0"]

# ------------------------------------------------------------------
# 1. Definir grilla (misma que script 02)
# ------------------------------------------------------------------

Nx, Ny, Nz = 100, 100, 100
x_grid = np.linspace(-8, 8, Nx)
y_grid = np.linspace(-8, 8, Ny)
z_grid = np.linspace(-3, 0, Nz)

# ------------------------------------------------------------------
# 2. Intentar cargar cubo precalculado, si no recalcular
# ------------------------------------------------------------------

cubo_path = "figuras/E_cubo.npy"

if os.path.exists(cubo_path):
    print("  Cargando cubo E(x,y,z) precalculado...")
    E_cubo = np.load(cubo_path)
else:
    print("  Cubo no encontrado. Recalculando (ejecuta 02 primero para más rapidez)...")
    from tqdm import tqdm
    E_cubo = np.zeros((Nz, Ny, Nx))
    for iz in tqdm(range(Nz), desc="  Calculando cortes", ncols=60):
        E_cubo[iz] = error_corte_2d(x_grid, y_grid, z_grid[iz],
                                     A0, ESTACIONES, Az_obs)
    np.save(cubo_path, E_cubo)
    print(f"  Cubo guardado en {cubo_path}")

# ------------------------------------------------------------------
# 3. Construir E_min(z) y encontrar z*
# ------------------------------------------------------------------

E_min_z = E_cubo.min(axis=(1, 2))          # shape (Nz,)

iz_opt  = np.argmin(E_min_z)
z_opt   = z_grid[iz_opt]
E_opt   = E_min_z[iz_opt]

# Posición (x*, y*) en el corte óptimo
corte_opt      = E_cubo[iz_opt]
iy_opt, ix_opt = np.unravel_index(np.argmin(corte_opt), corte_opt.shape)
x_opt          = x_grid[ix_opt]
y_opt          = y_grid[iy_opt]

print("\n" + "=" * 55)
print("  ANÁLISIS DE E_min(z)")
print("=" * 55)
print(f"\n  z*  estimado : {z_opt:.4f} km   (real: {z0_r})")
print(f"  x*  estimado : {x_opt:.4f} km   (real: {x0_r})")
print(f"  y*  estimado : {y_opt:.4f} km   (real: {y0_r})")
print(f"  E*           : {E_opt:.4e}")

# ------------------------------------------------------------------
# 4. Análisis de estabilidad: FWHM del pozo mínimo
# ------------------------------------------------------------------

# Half-maximum: valor a mitad entre el mínimo y el máximo de E_min_z
E_half = E_opt + 0.5 * (E_min_z.max() - E_opt)
indices_bajo = np.where(E_min_z <= E_half)[0]

if len(indices_bajo) > 0:
    z_fwhm_lo = z_grid[indices_bajo[0]]
    z_fwhm_hi = z_grid[indices_bajo[-1]]
    fwhm      = abs(z_fwhm_hi - z_fwhm_lo)
else:
    z_fwhm_lo = z_fwhm_hi = z_opt
    fwhm = 0.0

print(f"\n  Ancho del pozo mínimo (FWHM):")
print(f"    z ∈ [{z_fwhm_lo:.3f}, {z_fwhm_hi:.3f}] km")
print(f"    FWHM = {fwhm:.4f} km")
print(f"  → {'Mínimo bien definido' if fwhm < 0.5 else 'Mínimo amplio — menor sensibilidad en z'}")

# ------------------------------------------------------------------
# 5. Figura 1 — Curva E_min(z) con análisis de estabilidad
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(z_grid, E_min_z, color='steelblue', linewidth=2.5,
        label='$E_{\\min}(z)$')

# Zona FWHM sombreada
ax.axvspan(z_fwhm_lo, z_fwhm_hi, alpha=0.12, color='steelblue',
           label=f'FWHM = {fwhm:.3f} km')
ax.axhline(E_half, color='steelblue', linestyle=':', linewidth=1,
           alpha=0.6)

# Marcadores
ax.axvline(z_opt, color='tomato', linestyle='--', linewidth=2,
           label=f'$z^*$ estimado = {z_opt:.3f} km')
ax.axvline(z0_r, color='seagreen', linestyle=':', linewidth=2,
           label=f'$z_0$ real = {z0_r} km')

# Anotación del mínimo
ax.annotate(f'  $E^*$ = {E_opt:.2e}',
            xy=(z_opt, E_opt),
            xytext=(z_opt + 0.3, E_opt + 0.05*(E_min_z.max() - E_opt)),
            fontsize=9, color='tomato',
            arrowprops=dict(arrowstyle='->', color='tomato', lw=1.2))

ax.set_xlabel("Profundidad $z$ (km)", fontsize=12)
ax.set_ylabel("$E_{\\min}(z)$", fontsize=12)
ax.set_title("Función de error mínimo en función de la profundidad", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("figuras/curva_emin.png", dpi=150)
print("\n  Figura guardada: figuras/curva_emin.png")
plt.close()

# ------------------------------------------------------------------
# 6. Figura 2 — Trayectoria (x*(z), y*(z))
# ------------------------------------------------------------------

x_star = np.zeros(Nz)
y_star = np.zeros(Nz)

for iz in range(Nz):
    iy_, ix_ = np.unravel_index(np.argmin(E_cubo[iz]), (Ny, Nx))
    x_star[iz] = x_grid[ix_]
    y_star[iz] = y_grid[iy_]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Trayectoria del mínimo $(x^*(z),\\, y^*(z))$",
             fontsize=13, fontweight='bold')

for ax, vals, real, label, color in zip(
    axes,
    [x_star, y_star],
    [x0_r, y0_r],
    ['$x^*(z)$ (km)', '$y^*(z)$ (km)'],
    ['steelblue', 'darkorange']
):
    ax.plot(z_grid, vals, color=color, linewidth=2)
    ax.axhline(real, color='tomato', linestyle='--',
               linewidth=1.5, label=f'Valor real = {real}')
    ax.axvline(z_opt, color='gray', linestyle=':',
               linewidth=1.2, label=f'$z^*$ = {z_opt:.3f}')
    ax.set_xlabel("$z$ (km)", fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(f"Componente {label.split('(')[0].strip()}")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("figuras/trayectoria_xy.png", dpi=150)
print("  Figura guardada: figuras/trayectoria_xy.png")
plt.close()

# ------------------------------------------------------------------
# 7. Figura 3 — E_min(z) en escala logarítmica
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogy(z_grid, E_min_z, color='steelblue', linewidth=2.5,
            label='$E_{\\min}(z)$')
ax.axvline(z_opt, color='tomato', linestyle='--', linewidth=2,
           label=f'$z^*$ = {z_opt:.3f} km')
ax.axvline(z0_r, color='seagreen', linestyle=':', linewidth=2,
           label=f'$z_0$ real = {z0_r} km')
ax.set_xlabel("Profundidad $z$ (km)", fontsize=12)
ax.set_ylabel("$E_{\\min}(z)$  (escala log)", fontsize=12)
ax.set_title("$E_{\\min}(z)$ en escala logarítmica", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.5, which='both')
plt.tight_layout()
plt.savefig("figuras/curva_emin_log.png", dpi=150)
print("  Figura guardada: figuras/curva_emin_log.png")
plt.close()

print(f"\n  ¡Script 03 completado!")