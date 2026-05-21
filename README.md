# Localización de una Fuente Sísmica — Problema Inverso

Proyecto académico para estimar la posición $(x_0, y_0, z_0)$ y la amplitud
$A_0$ de una fuente sísmica a partir de amplitudes registradas en estaciones
superficiales, mediante la minimización de una función de error multivariable.

Desarrollado para la asignatura **Cálculo Multivariable** del programa de
Ingeniería de Software, Universidad Cooperativa de Colombia (Pasto, Nariño).

---

## Modelo físico

La amplitud sísmica observada en la estación $i$ se modela como atenuación
geométrica más absorción anelástica con ruido gaussiano:

$$
A_{z_i} = A_0 \, \frac{e^{-R_i}}{R_i} + \epsilon_i,
\qquad
R_i = \sqrt{(x_i - x_0)^2 + (y_i - y_0)^2 + (z_i - z_0)^2}
$$

donde $\epsilon_i \sim \mathcal{N}(0, \sigma_i^2)$ con
$\sigma_i = \alpha \cdot A_0 \, e^{-R_i}/R_i$.

La función de error de mínimos cuadrados es:

$$
E(\mathbf{m}) = \sum_{i=1}^{M} \left(A_{z_i} - A_0 \, \frac{e^{-R_i}}{R_i}\right)^2,
\qquad \mathbf{m} = (x_0, y_0, z_0, A_0)
$$

El problema consiste en encontrar $\mathbf{m}^* = \arg\min E(\mathbf{m})$.

---

## Métodos implementados

1. **Fuerza bruta sobre grilla 3D** ($100^3$ puntos) con $A_0 = 2.0$ fijo.
2. **Fuerza bruta perfilada** que estima $\widehat{A}_0(x,y,z)$ analíticamente
   en cada celda usando la fórmula cerrada
   $\widehat{A}_0 = \sum_i A_{z_i} u_i \big/ \sum_i u_i^2$
   con $u_i = e^{-R_i}/R_i$.
3. **Método iterativo de Levenberg–Marquardt con backtracking** y filtro
   físico ($A_0 > 0$).
4. **Análisis de incertidumbre** mediante la matriz de covarianza
   $\widehat{\mathbf{C}} = \hat{\sigma}^2 (\mathbf{G}^T\mathbf{G})^{-1}$ y
   descomposición espectral del Hessiano.

---

## Estructura del repositorio

```
fuenteSismicaCalculo/
├── datos/                  # Coordenadas de estaciones, parámetros del sismo
│   ├── estaciones.py
│   ├── resultado_fuerza_bruta.json
│   └── resultado_fuerza_bruta_perfilada.json
├── src/                    # Módulos del modelo físico y métodos numéricos
│   ├── modelo.py           # Modelo de atenuación
│   ├── ruido.py            # Generación de ruido gaussiano
│   ├── funcion_error.py    # Función de error y versión perfilada
│   ├── jacobiano.py        # Jacobiano analítico y Levenberg-Marquardt
│   └── utilidades.py       # Funciones auxiliares
├── analisis/               # Scripts numerados por fase
│   ├── 01_simular_datos.py
│   ├── 02_exploracion_error.py
│   ├── 03_curva_emin.py
│   ├── 04_metodo_iterativo.py
│   ├── 05_generar_video.py
│   ├── 06_analisis_hessiano.py
│   └── 07_fuerza_bruta_perfilada.py
├── figuras/                # Gráficas generadas
│   └── heatmaps/
├── video/                  # Animación de mapas de calor
├── docs/                   # Documento LaTeX
│   ├── main.tex
│   ├── main.pdf
│   └── referencias.bib
├── requirements.txt
└── README.md
```

---

## Instalación

Se recomienda Python 3.10 o superior. Crea un entorno virtual y luego:

```bash
pip install -r requirements.txt
```

Dependencias principales:

- `numpy` — álgebra lineal y arreglos numéricos
- `scipy` — utilidades de optimización (opcional)
- `matplotlib` — visualización y animaciones
- `tqdm` — barras de progreso
- `imageio` — generación de video/GIF (opcional)

---

## Orden de ejecución

Ejecutar desde la **raíz del repositorio** (no desde `analisis/`):

```bash
python analisis/01_simular_datos.py
python analisis/02_exploracion_error.py
python analisis/03_curva_emin.py
python analisis/04_metodo_iterativo.py
python analisis/05_generar_video.py        # opcional: requiere ffmpeg
python analisis/06_analisis_hessiano.py
python analisis/07_fuerza_bruta_perfilada.py
```

Los scripts 02 y 07 generan datos cacheados (`E_cubo.npy`, `E_cubo_perfilado.npy`)
que son reutilizados por los demás. Los scripts 03 a 07 dependen de la
salida del script 02.

---

## Resultados principales

Con la configuración por defecto (8 estaciones, ruido moderado $\alpha = 0.05$,
semilla `seed=42`), los tres métodos producen las siguientes estimaciones para
la fuente real $\mathbf{m}_{\text{real}} = (-1.0,\, 0.5,\, -1.2,\, 2.0)$:

| Método                          | $x^*$  | $y^*$ | $z^*$  | $A_0^*$ | Error (km) |
|---------------------------------|--------|-------|--------|---------|------------|
| Fuerza bruta ($A_0 = 2$ fijo)   | -1.051 | 0.566 | -1.212 | 2.000   | 0.084      |
| Fuerza bruta perfilada (4D)     | -1.051 | 0.566 | -1.151 | 1.962   | 0.096      |
| Levenberg–Marquardt             | -1.073 | 0.518 | -1.203 | 1.917   | 0.075      |

El análisis del Hessiano arroja $\kappa(\mathbf{G}^T\mathbf{G}) = 1.36 \times 10^{5}$
e incertidumbres $\sigma_{z_0} = 0.60$ km vs. $\sigma_{x_0} \approx 0.035$ km,
evidenciando cuantitativamente el clásico **trade-off profundidad–amplitud**
de la sismología con redes superficiales.

---

## Compilación del documento

Desde `docs/`:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

O bien, en un solo comando con `latexmk`:

```bash
latexmk -pdf main.tex
```

---

## Integrantes

- Leider Fabián Chipu Erazo
- Miguel Francisco Ruales
- Juan José Rueda
- Andrés Maya
- Camilo Castro

**Asignatura:** Cálculo Multivariable
**Docente:** M.Sc. Alejandro Molina
**Universidad Cooperativa de Colombia** — Pasto, Nariño

---

