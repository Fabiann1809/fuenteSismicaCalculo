"""
src/model.py
------------
Implementación del modelo de atenuación de amplitudes sísmicas.

Modelo utilizado (ondas de cuerpo):

    Az_i = A0 * exp(-R_i) / R_i

donde:
    R_i = sqrt((xi - x0)^2 + (yi - y0)^2 + (zi - z0)^2)

Referencias: Ecuaciones (1) y (2) del enunciado del proyecto.
"""

import numpy as np


def compute_distance(station: np.ndarray, source: np.ndarray) -> float:
    """
    Calcula la distancia euclidiana entre una estación y la fuente.

    Parámetros
    ----------
    station : array [xi, yi, zi]
        Posición de la estación sísmica.
    source : array [x0, y0, z0]
        Posición de la fuente sísmica.

    Retorna
    -------
    float
        Distancia R_i en km.
    """
    diff = station - source
    return np.sqrt(np.sum(diff**2))


def compute_amplitude(source_pos: np.ndarray, A0: float,
                       stations: np.ndarray) -> np.ndarray:
    """
    Calcula las amplitudes predichas por el modelo para todas
    las estaciones (sin ruido).

        Az_i = A0 * exp(-R_i) / R_i

    Parámetros
    ----------
    source_pos : array [x0, y0, z0]
        Posición de la fuente sísmica.
    A0 : float
        Amplitud en el origen.
    stations : array shape (M, 3)
        Coordenadas de las M estaciones.

    Retorna
    -------
    np.ndarray shape (M,)
        Vector de amplitudes predichas Az'.
    """
    amplitudes = np.zeros(len(stations))

    for i, station in enumerate(stations):
        R_i = compute_distance(station, source_pos)
        amplitudes[i] = A0 * np.exp(-R_i) / R_i

    return amplitudes


def compute_distances(source_pos: np.ndarray,
                      stations: np.ndarray) -> np.ndarray:
    """
    Calcula el vector de distancias R_i para todas las estaciones.

    Parámetros
    ----------
    source_pos : array [x0, y0, z0]
    stations   : array shape (M, 3)

    Retorna
    -------
    np.ndarray shape (M,)
        Vector de distancias en km.
    """
    distances = np.zeros(len(stations))
    for i, station in enumerate(stations):
        distances[i] = compute_distance(station, source_pos)
    return distances