"""
analisis/05_generar_video.py
-----------------------------
Genera una animación (video .mp4 o .gif) de los mapas de calor
de E(x, y, z=k) barriendo todos los cortes z.

El video muestra cómo evoluciona la función de error al variar
la profundidad, permitiendo visualizar en qué z el mínimo es
más claro y localizado.

Requisitos:
    - El cubo E_cubo.npy debe existir (ejecutar 02 primero)
    - pip install imageio[ffmpeg]  para .mp4
    - pip install imageio          para .gif

Ejecutar desde la raíz del repositorio:
    python analisis/05_generar_video.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

from datos.estaciones  import STATIONS, TRUE_SOURCE, NOISE_PARAMS
from src.modelo        import compute_amplitude
from src.ruido         import add_gaussian_noise

os.makedirs("figuras",        exist_ok=True)
os.makedirs("video",          exist_ok=True)
os.makedirs("figuras/frames", exist_ok=True)

# ------------------------------------------------------------------
# 0. Reproducir datos y configuración
# ------------------------------------------------------------------

pos_real  = np.array([TRUE_SOURCE["x0"], TRUE_SOURCE["y0"], TRUE_SOURCE["z0"]])
A0        = TRUE_SOURCE["A0"]
Az_limpia = compute_amplitude(pos_real, A0, STATIONS)
Az_obs    = add_gaussian_noise(Az_limpia, NOISE_PARAMS["alpha"],
                                NOISE_PARAMS["mu"], seed=42)

x0_r = TRUE_SOURCE["x0"]
y0_r = TRUE_SOURCE["y0"]
z0_r = TRUE_SOURCE["z0"]

Nx, Ny, Nz = 100, 100, 100
x_grid = np.linspace(-8, 8, Nx)
y_grid = np.linspace(-8, 8, Ny)
z_grid = np.linspace(-3, 0, Nz)

# ------------------------------------------------------------------
# 1. Cargar cubo precalculado
# ------------------------------------------------------------------

cubo_path = "figuras/E_cubo.npy"

if not os.path.exists(cubo_path):
    print("  [!] No se encontró E_cubo.npy")
    print("      Ejecuta primero: python analisis/02_exploracion_error.py")
    sys.exit(1)

print("  Cargando cubo E(x,y,z)...")
E_cubo = np.load(cubo_path)

# ------------------------------------------------------------------
# 2. Preprocesar: escala logarítmica y normalización global
# ------------------------------------------------------------------

E_log     = np.log10(E_cubo + 1e-30)
vmin_glob = E_log.min()
vmax_glob = E_log.max()

# Curva E_min(z) para panel lateral
E_min_z = E_cubo.min(axis=(1, 2))
iz_opt  = np.argmin(E_min_z)
z_opt   = z_grid[iz_opt]

# ------------------------------------------------------------------
# 3. Generar frames
# ------------------------------------------------------------------

print(f"\n  Generando {Nz} frames...")

for iz in tqdm(range(Nz), desc="  Renderizando frames", ncols=65):
    fig = plt.figure(figsize=(13, 5.5))
    fig.patch.set_facecolor('#0f0f1a')

    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.2], wspace=0.08)

    # --- Panel izquierdo: mapa de calor ---
    ax_map = fig.add_subplot(gs[0])
    ax_map.set_facecolor('#0f0f1a')

    im = ax_map.imshow(
        E_log[iz],
        origin='lower',
        extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]],
        aspect='auto',
        cmap='inferno',
        vmin=vmin_glob,
        vmax=vmax_glob,
        interpolation='bilinear'
    )

    # Contornos suaves
    ax_map.contour(
        x_grid, y_grid, E_log[iz],
        levels=8, colors='white', linewidths=0.4, alpha=0.25
    )

    # Mínimo en este corte
    iy_k, ix_k = np.unravel_index(np.argmin(E_cubo[iz]), (Ny, Nx))
    ax_map.scatter(x_grid[ix_k], y_grid[iy_k],
                   marker='*', s=180, color='cyan', zorder=6,
                   label=f'Mín: ({x_grid[ix_k]:.2f}, {y_grid[iy_k]:.2f})')

    # Fuente real
    ax_map.scatter(x0_r, y0_r, marker='x', s=140, color='lime',
                   linewidths=2.2, zorder=7, label=f'Real: ({x0_r}, {y0_r})')

    # Estaciones
    ax_map.scatter(STATIONS[:, 0], STATIONS[:, 1],
                   marker='^', s=60, color='deepskyblue',
                   zorder=5, alpha=0.8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax_map, fraction=0.035, pad=0.02)
    cbar.set_label('$\\log_{10}(E)$', color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    # Formato
    ax_map.set_xlabel("$x$ (km)", color='white', fontsize=11)
    ax_map.set_ylabel("$y$ (km)", color='white', fontsize=11)
    ax_map.tick_params(colors='white')
    for spine in ax_map.spines.values():
        spine.set_edgecolor('white')
    leg = ax_map.legend(fontsize=8, loc='upper right',
                        facecolor='#1a1a2e', labelcolor='white',
                        edgecolor='gray')

    # Título con z actual
    ax_map.set_title(
        f"$E(x,\\, y,\\, z = {z_grid[iz]:.3f}$ km$)$",
        color='white', fontsize=13, pad=8
    )

    # --- Panel derecho: curva E_min(z) con cursor ---
    ax_curve = fig.add_subplot(gs[1])
    ax_curve.set_facecolor('#0f0f1a')

    ax_curve.semilogy(E_min_z, z_grid, color='steelblue',
                      linewidth=1.8, alpha=0.9)

    # Cursor en z actual
    ax_curve.axhline(z_grid[iz], color='cyan', linewidth=1.5,
                     linestyle='--', alpha=0.8)
    ax_curve.scatter(E_min_z[iz], z_grid[iz],
                     color='cyan', s=80, zorder=6)

    # z real y z óptimo
    ax_curve.axhline(z0_r, color='lime', linewidth=1.2,
                     linestyle=':', alpha=0.7, label=f'$z_0$ real')
    ax_curve.axhline(z_opt, color='tomato', linewidth=1.2,
                     linestyle=':', alpha=0.7, label=f'$z^*$')

    ax_curve.set_xlabel("$E_{\\min}$", color='white', fontsize=10)
    ax_curve.set_ylabel("$z$ (km)", color='white', fontsize=10)
    ax_curve.set_title("$E_{\\min}(z)$", color='white', fontsize=11)
    ax_curve.tick_params(colors='white')
    ax_curve.yaxis.set_label_position('right')
    ax_curve.yaxis.tick_right()
    for spine in ax_curve.spines.values():
        spine.set_edgecolor('#444')
    ax_curve.legend(fontsize=7, facecolor='#1a1a2e',
                    labelcolor='white', edgecolor='gray')
    ax_curve.grid(True, linestyle='--', alpha=0.2, color='gray')

    # Progreso en esquina
    pct = (iz + 1) / Nz * 100
    fig.text(0.01, 0.01, f"Frame {iz+1}/{Nz}  ({pct:.0f}%)",
             color='gray', fontsize=8)

    plt.tight_layout(pad=0.5)
    fig.savefig(f"figuras/frames/frame_{iz:04d}.png",
                dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)

print(f"  {Nz} frames guardados en figuras/frames/")

# ------------------------------------------------------------------
# 4. Ensamblar video
# ------------------------------------------------------------------

try:
    import imageio
    video_path = "video/mapa_calor_sismica.mp4"
    frames_paths = [f"figuras/frames/frame_{iz:04d}.png" for iz in range(Nz)]

    print(f"\n  Ensamblando video .mp4...")
    writer = imageio.get_writer(video_path, fps=15, codec='libx264',
                                 quality=8, macro_block_size=None)
    for fp in tqdm(frames_paths, desc="  Escribiendo frames", ncols=65):
        writer.append_data(imageio.imread(fp))
    writer.close()
    print(f"  Video guardado: {video_path}")

except Exception as e:
    print(f"\n  [!] No se pudo generar .mp4: {e}")
    print("      Generando .gif como alternativa...")
    try:
        import imageio
        gif_path  = "video/mapa_calor_sismica.gif"
        # Para gif razonable usar solo 1 de cada 2 frames
        frames_gif = [imageio.imread(f"figuras/frames/frame_{iz:04d}.png")
                      for iz in range(0, Nz, 2)]
        imageio.mimsave(gif_path, frames_gif, fps=10, loop=0)
        print(f"  GIF guardado: {gif_path}")
    except Exception as e2:
        print(f"  [!] Error generando GIF: {e2}")
        print("      Los frames PNG están disponibles en figuras/frames/")

print("\n  ¡Script 05 completado!")