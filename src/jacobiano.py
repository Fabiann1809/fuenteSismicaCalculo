"""
src/jacobiano.py
----------------
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

El esquema iterativo implementado es Levenberg-Marquardt con
backtracking: rechaza pasos que aumenten el error o produzcan
A0 no físico (negativo), ajustando λ dinámicamente.

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
                             damping: float = 1e-4) -> dict:
    """
    Método iterativo de mínimos cuadrados (Levenberg-Marquardt con
    backtracking) para estimar m = [x0, y0, z0, A0].

    En cada iteración:
        1. Calcula Az predichas con el modelo actual m_k.
        2. Calcula el residual ΔAz = Az_obs - Az_pred y el error E_k.
        3. Calcula el jacobiano G en m_k.
        4. Intenta el paso Δm = (G^T G + λI)^{-1} G^T ΔAz.
        5. Si el paso reduce E, lo acepta y baja λ.
           Si no, sube λ y reintenta (acercándose a descenso de gradiente).
        6. Rechaza pasos que hagan A0 ≤ 0 (no físico).

    Parámetros
    ----------
    m0          : array [x0, y0, z0, A0] — modelo inicial
    stations    : array (M, 3)
    Az_observed : array (M,)
    max_iter    : int   — máximo de iteraciones
    tol         : float — criterio de convergencia (norma de Δm)
    damping     : float — λ inicial de Levenberg-Marquardt

    Retorna
    -------
    dict con claves:
        'm_final'    : array [x0, y0, z0, A0] — solución estimada
        'history_m'  : array (iter, 4)
        'history_E'  : array (iter,)
        'history_dm' : array (iter,)
        'converged'  : bool
        'n_iter'     : int
    """
    from src.modelo import compute_amplitude
    from src.funcion_error import error_single

    m_k = m0.copy().astype(float)

    history_m  = [m_k.copy()]
    history_E  = [error_single(m_k[:3], m_k[3], stations, Az_observed)]
    history_dm = [np.inf]

    converged = False
    lam = damping if damping > 0 else 1e-10

    for k in range(max_iter):
        # 1. Amplitudes predichas con m_k
        Az_pred = compute_amplitude(m_k[:3], m_k[3], stations)

        # 2. Residual y error actual
        delta_Az = Az_observed - Az_pred
        E_actual = float(np.sum(delta_Az**2))

        # 3. Jacobiano
        G = compute_jacobian(m_k, stations)

        # 4. Sistema normal
        GtG    = G.T @ G
        Gt_dAz = G.T @ delta_Az

        # 5. Búsqueda con damping adaptativo (backtracking)
        step_accepted = False
        delta_m = np.zeros(4)

        for intento in range(20):
            try:
                delta_m = np.linalg.solve(GtG + lam * np.eye(4), Gt_dAz)
            except np.linalg.LinAlgError:
                lam *= 10
                continue

            m_trial = m_k + delta_m

            # Filtro físico: rechazar A0 no positivo
            if m_trial[3] <= 1e-6:
                lam *= 10
                continue

            # Evaluar nuevo error
            Az_trial = compute_amplitude(m_trial[:3], m_trial[3], stations)
            E_trial  = float(np.sum((Az_observed - Az_trial)**2))

            if E_trial < E_actual:
                # Paso aceptado: bajar λ para próxima iteración
                m_k = m_trial
                lam = max(lam / 10, 1e-12)
                step_accepted = True
                break
            else:
                # Paso rechazado: subir λ y reintentar
                lam *= 10

        if not step_accepted:
            # Si no se pudo mejorar pero ya estábamos en el mínimo (error pequeño
            # y norma de los últimos pasos chica), considerar convergido al piso de ruido.
            if len(history_dm) >= 3 and max(history_dm[-3:]) < 1e-4:
                converged = True
                print(f"  [✓] Piso de ruido alcanzado en iteración {k+1} "
                      f"(no se puede reducir más el error).")
            else:
                print(f"  [!] No se pudo dar un paso útil en iteración {k+1}, deteniendo.")
            break

        # Registrar
        norm_dm = float(np.linalg.norm(delta_m))
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
def compute_covariance_analysis(m_hat: np.ndarray,
                                  stations: np.ndarray,
                                  Az_observed: np.ndarray) -> dict:
    """
    Análisis de covarianza, autovalores y autovectores en la solución m_hat.

    Calcula:
        - G = jacobiano en m_hat
        - H = 2 G^T G (Hessiano aproximado de E en el mínimo)
        - C = sigma² · (G^T G)^{-1}  (matriz de covarianza)
        - sigmas = sqrt(diag(C))      (barras de error en cada parámetro)
        - autovalores y autovectores de G^T G (direcciones bien/mal resueltas)
        - número de condición κ = λ_max / λ_min

    Parámetros
    ----------
    m_hat       : array [x0, y0, z0, A0] — solución estimada
    stations    : array (M, 3)
    Az_observed : array (M,)

    Retorna
    -------
    dict con claves:
        'G'              : jacobiano (M, 4)
        'GtG'            : G^T G  (4, 4)
        'covariance'     : matriz de covarianza C (4, 4)
        'sigmas'         : array (4,) — incertidumbres en (x0, y0, z0, A0)
        'eigenvalues'    : array (4,) — autovalores de G^T G (descendente)
        'eigenvectors'   : array (4, 4) — autovectores como columnas
        'condition_num'  : float — número de condición de G^T G
        'sigma_residual' : float — sigma del residual estimada
        'E_min'          : float — error en el mínimo
        'dof'            : int   — grados de libertad (M - 4)
    """
    from src.modelo import compute_amplitude

    M = len(stations)
    p = 4  # número de parámetros
    dof = M - p

    if dof <= 0:
        raise ValueError(f"Grados de libertad no positivos: M={M}, p={p}")

    # Jacobiano en la solución
    G = compute_jacobian(m_hat, stations)
    GtG = G.T @ G

    # Error en el mínimo y sigma residual estimada
    Az_pred = compute_amplitude(m_hat[:3], m_hat[3], stations)
    residuals = Az_observed - Az_pred
    E_min = float(np.sum(residuals**2))
    sigma2 = E_min / dof
    sigma_residual = float(np.sqrt(sigma2))

    # Matriz de covarianza
    try:
        GtG_inv = np.linalg.inv(GtG)
        covariance = sigma2 * GtG_inv
        sigmas = np.sqrt(np.diag(covariance))
    except np.linalg.LinAlgError:
        covariance = np.full((p, p), np.nan)
        sigmas = np.full(p, np.nan)

    # Autovalores y autovectores de G^T G (ordenados de mayor a menor)
    eigvals, eigvecs = np.linalg.eigh(GtG)  # eigh: simétrica → ordenados ascendente
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    condition_num = float(eigvals[0] / eigvals[-1]) if eigvals[-1] > 0 else np.inf

    return {
        'G'              : G,
        'GtG'            : GtG,
        'covariance'     : covariance,
        'sigmas'         : sigmas,
        'eigenvalues'    : eigvals,
        'eigenvectors'   : eigvecs,
        'condition_num'  : condition_num,
        'sigma_residual' : sigma_residual,
        'E_min'          : E_min,
        'dof'            : dof,
    }