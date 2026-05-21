# Localización de Fuente Sísmica por Mínimos Cuadrados

Proyecto para estimar la posición (x, y, z) de una fuente sísmica a partir de
amplitudes registradas en estaciones superficiales, usando un modelo de
atenuación geométrica y el método iterativo de Gauss-Newton.

## Estructura del proyecto

```
fuenteSismicaCalculo/
├── datos/                  # Coordenadas de estaciones y datos simulados
├── src/                    # Módulos del modelo físico y métodos numéricos
├── analisis/               # Scripts de cada fase del proyecto
├── figuras/                # Gráficas generadas
│   └── mapas_calor/
├── video/                  # Video animado de los mapas de calor
├── docs/                   # Documento LaTeX
└── requirements.txt
```

## Instalación de dependencias

```bash
pip install -r requirements.txt
```

## Orden de ejecución

Ejecutar desde la raíz del proyecto:

```bash
python analisis/01_simular_datos.py
python analisis/02_exploracion_error.py
python analisis/03_curva_emin.py
python analisis/04_metodo_iterativo.py
python analisis/05_generar_video.py   # requiere FFmpeg
```

## Modelo físico

La amplitud de la onda sísmica a distancia $r$ de la fuente se modela como:

$$A(r) = \frac{A_0}{r^n}$$

La función de error cuadrático es:

$$E(x,y,z) = \sum_{i=1}^{N} \left[ A_i^{\text{obs}} - A(r_i) \right]^2$$

## Dependencias

- `numpy` — álgebra lineal y arreglos numéricos
- `matplotlib` — visualización y animaciones
- `scipy` — optimización (opcional)
