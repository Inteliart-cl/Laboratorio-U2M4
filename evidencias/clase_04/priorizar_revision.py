from math import floor

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from desafios.comun import ROOT, cv_clasificacion, guardar_tabla, particion_clasificacion
from originales.base import SEED

PRESUPUESTO = 0.40


def evaluar_revision(modelo, X, y):
    puntajes = modelo.predict_proba(X)[:, 1]
    cupos = floor(PRESUPUESTO * len(y))
    orden = np.argsort(-puntajes, kind="stable")
    detectados = int((y.iloc[orden[:cupos]] == 1).sum())
    casos_riesgo = int((y == 1).sum())
    return {
        "casos": len(y),
        "revisiones": cupos,
        "riesgo_total": casos_riesgo,
        "riesgo_detectado": detectados,
        "riesgo_omitido": casos_riesgo - detectados,
        "recall_riesgo": detectados / casos_riesgo,
        "precision_revision": detectados / cupos,
    }


def main():
    Xtr, Xte, ytr, yte = particion_clasificacion()
    candidatos = {
        "Regresión logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Bosque aleatorio": RandomForestClassifier(n_estimators=100, random_state=SEED),
    }
    pliegues = list(cv_clasificacion().split(Xtr, ytr))
    filas = []
    totales = {}
    for nombre, candidato in candidatos.items():
        total = 0
        for numero, (ajuste, validacion) in enumerate(pliegues, start=1):
            modelo = clone(candidato).fit(Xtr.iloc[ajuste], ytr.iloc[ajuste])
            resultado = evaluar_revision(modelo, Xtr.iloc[validacion], ytr.iloc[validacion])
            filas.append({"etapa": "Validación", "pliegue": numero, "modelo": nombre, **resultado})
            total += resultado["riesgo_detectado"]
        totales[nombre] = total
        print(f"{nombre}: {total} casos de riesgo detectados en validación")

    elegido = max(totales, key=totales.get)
    print(f"Modelo elegido con el mismo presupuesto de revisión: {elegido}")
    modelo = candidatos[elegido].fit(Xtr, ytr)
    resultado = evaluar_revision(modelo, Xte, yte)
    filas.append({"etapa": "Prueba del laboratorio", "pliegue": "—", "modelo": elegido, **resultado})
    guardar_tabla("priorizacion_riesgo", filas, carpeta=ROOT / "evidencias" / "clase_04")


if __name__ == "__main__":
    main()
