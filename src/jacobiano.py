"""
src/jacobian.py
---------------
Cálculo del jacobiano G y método iterativo de mínimos cuadrados
para la localización de la fuente sísmica.

El jacobiano G es una matriz (M x 4) donde cada fila corresponde
a una estación y cada columna a la derivada parcial de Az_i
respecto a cada parámetro del modelo m = [x0, y0, z0, A0]:

    G[i, 0] = dAz_i / dx0
    G[i, 1] = dAz_i / dy0
    G[i, 2] = dAz_i / dz0
    G[i, 3] = dAz_i / dA0

Derivadas analíticas:

    Sea f(R) = A0 * exp(-R) / R
    dAz_i/dx0 = A0 * exp(-R_i) * (xi - x0) / R_i^2 * (1/R_i + 1)
    (análogo para y0 y z0)
    dAz_i/dA0 = exp(-R_i) / R_i

El esquema iterativo:
    Δm = (G^T G)^{-1} G^T ΔAz
    m_{k+1} = m_k + Δm

Referencias: Ecuaciones (8)-(11) del enunciado del proyecto.
"""

import numpy as np


def compute_jacobian(m: np.ndarray, stations: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz jacobiana G en el punto m.

    Parámetros
    ----------
    m        : array [x0, y0, z0, A0] — modelo actual
    stations : array (M, 3)           — coordenadas de estaciones

    Retorna
    -------
    np.ndarray shape (M, 4)
        Matriz jacobiana G.
    """
    x0, y0, z0, A0 = m
    M = len(stations)
    G = np.zeros((M, 4))

    for i, (xi, yi, zi) in enumerate(stations):
        dx = xi - x0
        dy = yi - y0
        dz = zi - z0
        R  = np.sqrt(dx**2 + dy**2 + dz**2)

        exp_R  = np.exp(-R)
        factor = A0 * exp_R * (1.0/R + 1.0) / (R**2)

        G[i, 0] = factor * dx          # dAz/dx0
        G[i, 1] = factor * dy          # dAz/dy0
        G[i, 2] = factor * dz          # dAz/dz0
        G[i, 3] = exp_R / R            # dAz/dA0

    return G


def iterative_least_squares(m0: np.ndarray,
                             stations: np.ndarray,
                             Az_observed: np.ndarray,
                             max_iter: int = 100,
                             tol: float = 1e-8,
                             damping: float = 0.0) -> dict:
    """
    Método iterativo de mínimos cuadrados para estimar m.

    En cada iteración:
        1. Calcula Az predichas con el modelo actual m_k
        2. Calcula el residual ΔAz = Az_obs - Az_pred
        3. Calcula el jacobiano G en m_k
        4. Resuelve Δm = (G^T G + λI)^{-1} G^T ΔAz
        5. Actualiza m_{k+1} = m_k + Δm

    El parámetro `damping` (λ) es una regularización de Tikhonov
    que estabiliza la inversión cuando G^T G es casi singular.

    Parámetros
    ----------
    m0          : array [x0, y0, z0, A0] — modelo inicial
    stations    : array (M, 3)
    Az_observed : array (M,)
    max_iter    : int   — máximo de iteraciones
    tol         : float — criterio de convergencia (norma de Δm)
    damping     : float — parámetro de regularización λ (default 0)

    Retorna
    -------
    dict con claves:
        'm_final'    : array [x0, y0, z0, A0] — solución estimada
        'history_m'  : array (iter, 4)         — evolución de m
        'history_E'  : array (iter,)            — evolución del error
        'history_dm' : array (iter,)            — norma de Δm
        'converged'  : bool
        'n_iter'     : int
    """
    from src.modelo import compute_amplitude
    from src.funcion_error import error_single

    m_k = m0.copy().astype(float)
    M   = len(stations)

    history_m  = [m_k.copy()]
    history_E  = [error_single(m_k[:3], m_k[3], stations, Az_observed)]
    history_dm = [np.inf]

    converged = False

    for k in range(max_iter):
        # 1. Amplitudes predichas con m_k
        Az_pred = compute_amplitude(m_k[:3], m_k[3], stations)

        # 2. Residual
        delta_Az = Az_observed - Az_pred      # (M,)

        # 3. Jacobiano
        G = compute_jacobian(m_k, stations)   # (M, 4)

        # 4. Mínimos cuadrados: Δm = (G^T G + λI)^{-1} G^T ΔAz
        GtG = G.T @ G                         # (4, 4)
        if damping > 0:
            GtG += damping * np.eye(4)
        Gt_dAz = G.T @ delta_Az              # (4,)

        try:
            delta_m = np.linalg.solve(GtG, Gt_dAz)
        except np.linalg.LinAlgError:
            print(f"  [!] Sistema singular en iteración {k+1}, deteniendo.")
            break

        # 5. Actualizar modelo
        m_k = m_k + delta_m

        # Registrar
        norm_dm = np.linalg.norm(delta_m)
        E_k     = error_single(m_k[:3], m_k[3], stations, Az_observed)

        history_m.append(m_k.copy())
        history_E.append(E_k)
        history_dm.append(norm_dm)

        # Criterio de convergencia
        if norm_dm < tol:
            converged = True
            break

    return {
        'm_final'   : m_k,
        'history_m' : np.array(history_m),
        'history_E' : np.array(history_E),
        'history_dm': np.array(history_dm),
        'converged' : converged,
        'n_iter'    : len(history_E) - 1,
    }