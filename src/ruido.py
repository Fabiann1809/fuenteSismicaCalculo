"""
src/noise.py
------------
Generación de ruido gaussiano para los datos sísmicos simulados.

El ruido sigue una distribución normal con:
    mu    = 0
    sigma = alpha * A0 * exp(-R_i) / R_i

Es decir, el nivel de ruido es proporcional a la amplitud local
de la señal en cada estación. Esto modela un ruido realista donde
las estaciones más lejanas (señal más débil) tienen menos ruido
absoluto pero igual incertidumbre relativa.

Referencia: Ecuación (3) del enunciado del proyecto.
"""

import numpy as np


def compute_sigma(amplitudes_clean: np.ndarray, alpha: float) -> np.ndarray:
    """
    Calcula la desviación estándar del ruido para cada estación.

        sigma_i = alpha * |Az_i_clean|

    Parámetros
    ----------
    amplitudes_clean : np.ndarray shape (M,)
        Amplitudes predichas sin ruido.
    alpha : float
        Nivel de ruido (0.05 = 5%).

    Retorna
    -------
    np.ndarray shape (M,)
        Vector de desviaciones estándar por estación.
    """
    return alpha * np.abs(amplitudes_clean)


def add_gaussian_noise(amplitudes_clean: np.ndarray,
                       alpha: float,
                       mu: float = 0.0,
                       seed: int = 42) -> np.ndarray:
    """
    Agrega ruido gaussiano a las amplitudes limpias.

        Az_i = Az_i_clean + epsilon_i
        epsilon_i ~ N(mu, sigma_i^2)

    Parámetros
    ----------
    amplitudes_clean : np.ndarray shape (M,)
        Amplitudes sin ruido del modelo.
    alpha : float
        Nivel de ruido relativo (0.05 para ruido moderado).
    mu : float
        Media del ruido (default = 0.0).
    seed : int
        Semilla aleatoria para reproducibilidad.

    Retorna
    -------
    np.ndarray shape (M,)
        Amplitudes observadas con ruido.
    """
    rng = np.random.default_rng(seed)
    sigma = compute_sigma(amplitudes_clean, alpha)
    noise = rng.normal(loc=mu, scale=sigma)
    return amplitudes_clean + noise