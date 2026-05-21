"""
src/utils.py
-------------
Funciones utilitarias compartidas entre todos los scripts de análisis.

Centraliza la lógica repetida para:
  - Reproducir los datos observados (siempre con la misma semilla)
  - Imprimir tablas de resultados en consola
  - Calcular métricas de error posicional
  - Formatear resultados comparativos
"""

import numpy as np


def get_observed_data(stations, true_source, noise_params, seed=42):
    """
    Genera los datos observados reproducibles.

    Encapsula el bloque que se repite en todos los scripts:
        source_pos  = [x0, y0, z0]
        Az_clean    = modelo sin ruido
        Az_observed = Az_clean + ruido gaussiano

    Parámetros
    ----------
    stations     : np.ndarray (M, 3) — coordenadas de estaciones
    true_source  : dict con claves x0, y0, z0, A0
    noise_params : dict con claves alpha, mu
    seed         : int — semilla aleatoria (default 42)

    Retorna
    -------
    dict con claves:
        'source_pos'  : np.ndarray [x0, y0, z0]
        'A0'          : float
        'Az_clean'    : np.ndarray (M,) — amplitudes sin ruido
        'Az_observed' : np.ndarray (M,) — amplitudes con ruido
        'distances'   : np.ndarray (M,) — distancias R_i
    """
    from src.modelo import compute_amplitude, compute_distances
    from src.ruido import add_gaussian_noise

    source_pos = np.array([true_source["x0"],
                            true_source["y0"],
                            true_source["z0"]])
    A0         = true_source["A0"]

    Az_clean    = compute_amplitude(source_pos, A0, stations)
    distances   = compute_distances(source_pos, stations)
    Az_observed = add_gaussian_noise(
        Az_clean,
        alpha = noise_params["alpha"],
        mu    = noise_params["mu"],
        seed  = seed
    )

    return {
        "source_pos"  : source_pos,
        "A0"          : A0,
        "Az_clean"    : Az_clean,
        "Az_observed" : Az_observed,
        "distances"   : distances,
    }


def positional_error(estimated, true_source):
    """
    Calcula el error posicional euclidiano en km.

    Parámetros
    ----------
    estimated   : array-like [x, y, z] o dict con x0, y0, z0
    true_source : dict con claves x0, y0, z0

    Retorna
    -------
    float — distancia entre posición estimada y real (km)
    """
    if isinstance(estimated, dict):
        est = np.array([estimated["x0"], estimated["y0"], estimated["z0"]])
    else:
        est = np.asarray(estimated[:3])

    real = np.array([true_source["x0"], true_source["y0"], true_source["z0"]])
    return float(np.linalg.norm(est - real))


def print_data_table(stations, distances, Az_clean, Az_observed):
    """
    Imprime la tabla de datos simulados en consola.

    Parámetros
    ----------
    stations    : np.ndarray (M, 3)
    distances   : np.ndarray (M,)
    Az_clean    : np.ndarray (M,)
    Az_observed : np.ndarray (M,)
    """
    noise = Az_observed - Az_clean
    header = (f"{'Est':>4} {'xi':>7} {'yi':>7} {'zi':>7} "
              f"{'Ri (km)':>10} {'Az_clean':>12} {'Az_obs':>12} {'ruido':>10}")
    print(header)
    print("-" * len(header))
    for i, (xi, yi, zi) in enumerate(stations):
        print(f"  S{i+1:1d}  {xi:7.2f} {yi:7.2f} {zi:7.2f} "
              f"{distances[i]:10.4f} {Az_clean[i]:12.6f} "
              f"{Az_observed[i]:12.6f} {noise[i]:10.6f}")


def print_comparison_table(results_dict, true_source, brute_force=None):
    """
    Imprime tabla comparativa de métodos de estimación.

    Parámetros
    ----------
    results_dict  : dict {nombre: array [x, y, z, A0]}
                    Resultados de distintos métodos o inicios.
    true_source   : dict con x0, y0, z0, A0
    brute_force   : array [x, y, z, A0] opcional — resultado fuerza bruta
    """
    x0_r = true_source["x0"]
    y0_r = true_source["y0"]
    z0_r = true_source["z0"]
    A0_r = true_source["A0"]

    print(f"\n  {'Método':<32} {'x*':>8} {'y*':>8} {'z*':>8} "
          f"{'A0*':>8} {'Err(km)':>10}")
    print("  " + "-" * 68)

    # Fila real
    print(f"  {'Fuente real':<32} {x0_r:>8.3f} {y0_r:>8.3f} "
          f"{z0_r:>8.3f} {A0_r:>8.3f} {'—':>10}")

    # Fuerza bruta (opcional)
    if brute_force is not None:
        ep = positional_error(brute_force, true_source)
        print(f"  {'Fuerza bruta (100³)':<32} {brute_force[0]:>8.3f} "
              f"{brute_force[1]:>8.3f} {brute_force[2]:>8.3f} "
              f"{brute_force[3]:>8.3f} {ep:>10.4f}")

    # Métodos iterativos
    for nombre, m in results_dict.items():
        ep  = positional_error(m, true_source)
        tag = nombre[:30]
        print(f"  {tag:<32} {m[0]:>8.4f} {m[1]:>8.4f} {m[2]:>8.4f} "
              f"{m[3]:>8.4f} {ep:>10.4f}")


def load_or_compute_cube(x_grid, y_grid, z_grid, A0,
                         stations, Az_observed,
                         path="figuras/E_cubo.npy"):
    """
    Carga el cubo E(x,y,z) desde disco si existe, si no lo calcula.

    Parámetros
    ----------
    x_grid, y_grid, z_grid : arrays 1D
    A0          : float
    stations    : np.ndarray (M, 3)
    Az_observed : np.ndarray (M,)
    path        : str — ruta del archivo .npy

    Retorna
    -------
    np.ndarray shape (Nz, Ny, Nx)
    """
    import os
    from src.funcion_error import error_grid_slice

    if os.path.exists(path):
        print(f"  Cargando cubo desde {path}...")
        return np.load(path)

    print(f"  Cubo no encontrado. Calculando...")
    from tqdm import tqdm
    Nz, Ny, Nx = len(z_grid), len(y_grid), len(x_grid)
    E_cubo = np.zeros((Nz, Ny, Nx))
    for iz in tqdm(range(Nz), desc="  Cortes z", ncols=60):
        E_cubo[iz] = error_grid_slice(
            x_grid, y_grid, z_grid[iz], A0, stations, Az_observed
        )
    np.save(path, E_cubo)
    print(f"  Cubo guardado en {path}")
    return E_cubo