"""Compara los umbrales de Isolation Forest del Laboratorio 6."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from desafios.comun import guardar_tabla
from originales.base import SEED, datos_segmentacion


def main():
    datos_base = datos_segmentacion()
    generador = np.random.default_rng(SEED)
    datos_anadidos = pd.DataFrame(
        generador.uniform(-12, 12, size=(12, 2)), columns=datos_base.columns
    )
    datos = pd.concat([datos_base, datos_anadidos], ignore_index=True)
    es_anadida = datos.index >= len(datos_base)

    print(f"Observaciones: {len(datos)} | Base: {len(datos_base)} "
          f"| Añadidas: {len(datos_anadidos)} | Semilla: {SEED}")
    print(f"Columnas: {', '.join(datos.columns)}")
    resultados = []
    for contaminacion in (0.02, 0.04, 0.08):
        bosque = IsolationForest(contamination=contaminacion, random_state=SEED)
        modelo = make_pipeline(StandardScaler(), bosque).fit(datos)
        alerta = modelo.predict(datos) == -1
        resultados.append({
            "contamination": contaminacion,
            "alertas": alerta.sum(),
            "sin_alerta": (~alerta).sum(),
            "anadidas_con_alerta": (alerta & es_anadida).sum(),
            "base_con_alerta": (alerta & ~es_anadida).sum(),
            "anadidas_sin_alerta": (~alerta & es_anadida).sum(),
        })
        print(f"Índices con alerta para contamination={contaminacion:.2f}: "
              f"{datos.index[alerta].tolist()}")

    print("\nComparación de umbrales")
    guardar_tabla("tabla_contamination", resultados, carpeta=Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()
