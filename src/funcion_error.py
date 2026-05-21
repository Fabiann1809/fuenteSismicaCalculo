"""
src/error_function.py
---------------------
Construcción de la función de error E(x, y, z).

La función de error cuantifica la discrepancia entre las amplitudes
observadas (datos) y las predichas por el modelo para una posición
candidata de la fuente:

    E(x, y, z) = sum_i ( Az_i_obs - Az_i_pred(x, y, z, A0) )^2

Para evaluar E en una grilla 3D de forma eficiente se utiliza
vectorización con NumPy (sin loops explícitos sobre la grilla).

Referencia: Ecuación (7) del enunciado del proyecto.
"""

import numpy as np


def error_single(candidate: np.ndarray,
                 A0: float,
                 stations: np.ndarray,
                 Az_observed: np.ndarray) -> float:
    """
    Evalúa E en un único punto candidato (x, y, z).

    Parámetros
    ----------
    candidate    : array [x, y, z] — posición candidata de la fuente
    A0           : float           — amplitud en el origen
    stations     : array (M, 3)    — coordenadas de las estaciones
    Az_observed  : array (M,)      — amplitudes observadas

    Retorna
    -------
    float
        Valor escalar de E en ese punto.
    """
    diff    = stations - candidate          # (M, 3)
    R       = np.sqrt(np.sum(diff**2, axis=1))  # (M,)
    Az_pred = A0 * np.exp(-R) / R          # (M,)
    return float(np.sum((Az_observed - Az_pred)**2))


def error_grid_slice(x_grid: np.ndarray,
                     y_grid: np.ndarray,
                     z_val: float,
                     A0: float,
                     stations: np.ndarray,
                     Az_observed: np.ndarray) -> np.ndarray:
    """
    Evalúa E(x, y, z=cte) en un corte 2D completo de forma vectorizada.

    Parámetros
    ----------
    x_grid      : array (Nx,)  — valores de x en la grilla
    y_grid      : array (Ny,)  — valores de y en la grilla
    z_val       : float        — profundidad fija del corte
    A0          : float        — amplitud en el origen
    stations    : array (M, 3) — coordenadas de las estaciones
    Az_observed : array (M,)   — amplitudes observadas

    Retorna
    -------
    np.ndarray shape (Ny, Nx)
        Mapa 2D de la función de error para ese corte z = z_val.
        (Ny filas = eje y, Nx columnas = eje x — convenio imshow)
    """
    # Construir grilla 2D: shape (Ny, Nx)
    XX, YY = np.meshgrid(x_grid, y_grid)   # (Ny, Nx)
    ZZ     = np.full_like(XX, z_val)       # (Ny, Nx)

    # Aplanar para vectorizar: (Ny*Nx, 3)
    candidates = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

    # Diferencias con cada estación: (Ny*Nx, M, 3)
    # stations shape (M, 3) → broadcast con (Ny*Nx, 1, 3)
    diff = candidates[:, np.newaxis, :] - stations[np.newaxis, :, :]

    # Distancias: (Ny*Nx, M)
    R = np.sqrt(np.sum(diff**2, axis=2))

    # Amplitudes predichas: (Ny*Nx, M)
    Az_pred = A0 * np.exp(-R) / R

    # Error cuadrático por estación, sumado: (Ny*Nx,)
    E_flat = np.sum((Az_observed[np.newaxis, :] - Az_pred)**2, axis=1)

    # Reshape al mapa 2D
    return E_flat.reshape(XX.shape)


def error_grid_3d(x_grid: np.ndarray,
                  y_grid: np.ndarray,
                  z_grid: np.ndarray,
                  A0: float,
                  stations: np.ndarray,
                  Az_observed: np.ndarray) -> np.ndarray:
    """
    Evalúa E(x, y, z) en toda la grilla 3D.

    Parámetros
    ----------
    x_grid, y_grid, z_grid : arrays 1D — ejes de la grilla
    A0           : float
    stations     : array (M, 3)
    Az_observed  : array (M,)

    Retorna
    -------
    np.ndarray shape (Nz, Ny, Nx)
        Cubo de errores. Índices: [iz, iy, ix]
    """
    Nx, Ny, Nz = len(x_grid), len(y_grid), len(z_grid)
    E_cube = np.zeros((Nz, Ny, Nx))

    for iz, z_val in enumerate(z_grid):
        E_cube[iz] = error_grid_slice(
            x_grid, y_grid, z_val, A0, stations, Az_observed
        )

    return E_cube
def estimate_A0_optimal(x_grid: np.ndarray,
                         y_grid: np.ndarray,
                         z_val: float,
                         stations: np.ndarray,
                         Az_observed: np.ndarray) -> np.ndarray:
    """
    Estimación analítica de A0 que minimiza E para cada (x, y) en un corte z=cte.

    Derivando ∂E/∂A0 = 0 a (x,y,z) fijo:

        A0_hat(x,y,z) = sum_i (Az_obs_i * u_i) / sum_i (u_i^2)
        donde u_i = exp(-R_i) / R_i

    Parámetros
    ----------
    x_grid, y_grid : arrays 1D
    z_val          : float
    stations       : array (M, 3)
    Az_observed    : array (M,)

    Retorna
    -------
    np.ndarray shape (Ny, Nx)
        Mapa 2D de A0_hat para ese corte.
    """
    XX, YY = np.meshgrid(x_grid, y_grid)
    ZZ     = np.full_like(XX, z_val)

    candidates = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    diff = candidates[:, np.newaxis, :] - stations[np.newaxis, :, :]
    R    = np.sqrt(np.sum(diff**2, axis=2))      # (N, M)
    u    = np.exp(-R) / R                         # (N, M)

    numer = np.sum(Az_observed[np.newaxis, :] * u, axis=1)   # (N,)
    denom = np.sum(u**2, axis=1)                              # (N,)

    A0_hat = numer / denom
    return A0_hat.reshape(XX.shape)


def error_profiled_slice(x_grid: np.ndarray,
                          y_grid: np.ndarray,
                          z_val: float,
                          stations: np.ndarray,
                          Az_observed: np.ndarray) -> tuple:
    """
    Evalúa la función de error perfilada (con A0 optimizado en cada celda).

        E_perfilada(x,y,z) = min_A0 E(x,y,z,A0)
                           = sum_i (Az_obs_i - A0_hat * u_i)^2

    Esta función no requiere fijar A0 y permite que la búsqueda por grilla
    estime simultáneamente la posición y la amplitud, igual que el iterativo.

    Parámetros
    ----------
    x_grid, y_grid : arrays 1D
    z_val          : float
    stations       : array (M, 3)
    Az_observed    : array (M,)

    Retorna
    -------
    (E_map, A0_map) : tupla de np.ndarray, ambos shape (Ny, Nx)
        E_map  : función de error perfilada
        A0_map : A0 óptimo en cada (x,y)
    """
    XX, YY = np.meshgrid(x_grid, y_grid)
    ZZ     = np.full_like(XX, z_val)

    candidates = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    diff = candidates[:, np.newaxis, :] - stations[np.newaxis, :, :]
    R    = np.sqrt(np.sum(diff**2, axis=2))       # (N, M)
    u    = np.exp(-R) / R                          # (N, M)

    # A0 óptimo por celda
    numer  = np.sum(Az_observed[np.newaxis, :] * u, axis=1)
    denom  = np.sum(u**2, axis=1)
    A0_hat = numer / denom                         # (N,)

    # Amplitudes predichas con A0 óptimo
    Az_pred = A0_hat[:, np.newaxis] * u            # (N, M)

    # Error perfilado
    E_flat = np.sum((Az_observed[np.newaxis, :] - Az_pred)**2, axis=1)

    Ny, Nx = XX.shape
    return E_flat.reshape(Ny, Nx), A0_hat.reshape(Ny, Nx)