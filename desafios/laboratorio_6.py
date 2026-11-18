import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from desafios.comun import guardar_tabla
from originales.base import SEED, datos_segmentacion

base = datos_segmentacion()
rng = np.random.default_rng(SEED)
inyectadas = pd.DataFrame(rng.uniform(-12, 12, size=(12, 2)), columns=base.columns)
Z = pd.concat([base, inyectadas], ignore_index=True)
es_inyectada = np.arange(len(Z)) >= len(base)

print(f"Observaciones: {len(Z)} | Inyectadas: {es_inyectada.sum()} | Semilla: {SEED}")
filas = []
for contaminacion, arboles in [(0.04, 100), (0.02, 100), (0.08, 100), (0.04, 250)]:
    bosque = IsolationForest(contamination=contaminacion, n_estimators=arboles, random_state=SEED)
    modelo = make_pipeline(StandardScaler(), bosque)
    alerta = modelo.fit(Z).predict(Z) == -1
    detectadas = (alerta & es_inyectada).sum()
    filas.append({
        "contamination": contaminacion, "n_estimators": arboles,
        "alertas": alerta.sum(), "inyectadas_detectadas": detectadas,
        "alertas_en_datos_base": (alerta & ~es_inyectada).sum(),
        "recuperacion_inyectadas": detectadas / len(inyectadas),
    })

guardar_tabla("L6", filas)
print("La procedencia de los puntos se usa solo para interpretar las alertas.")
