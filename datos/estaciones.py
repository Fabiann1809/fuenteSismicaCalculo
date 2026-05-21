"""
data/stations.py
----------------
Coordenadas de las estaciones sísmicas y parámetros reales
del sismo simulado.

Unidades: kilómetros (km)
"""

import numpy as np

# ------------------------------------------------------------------
# Estaciones sísmicas (xi, yi, zi)
# Distribuidas espacialmente para cubrir bien el área de estudio.
# Se definen 8 estaciones inspiradas en la distribución de la
# Figura 1 del enunciado.
# ------------------------------------------------------------------

STATIONS = np.array([
    [-5.0,  4.0,  0.0],   # S1 - noroeste, superficie
    [ 0.0,  5.0,  0.0],   # S2 - norte, superficie
    [ 5.0,  3.0,  0.0],   # S3 - noreste, superficie
    [-4.0,  0.0, -0.5],   # S4 - oeste, ligera profundidad
    [ 4.0,  0.0, -0.5],   # S5 - este, ligera profundidad
    [-3.0, -4.0, -0.3],   # S6 - suroeste
    [ 2.0, -5.0, -0.2],   # S7 - sur
    [ 6.0, -3.0,  0.0],   # S8 - sureste, superficie
])

N_STATIONS = len(STATIONS)  # 8 estaciones

# ------------------------------------------------------------------
# Parámetros reales del sismo (la "verdad" que queremos recuperar)
# ------------------------------------------------------------------

TRUE_SOURCE = {
    "x0": -1.0,   # km
    "y0":  0.5,   # km
    "z0": -1.2,   # km  (profundidad, valor negativo = bajo tierra)
    "A0":  2.0,   # amplitud en el origen (unidades arbitrarias)
}

# ------------------------------------------------------------------
# Parámetros del ruido gaussiano
# ------------------------------------------------------------------

NOISE_PARAMS = {
    "alpha": 0.05,   # nivel de ruido moderado (5%)
    "mu":    0.0,    # media del ruido
}