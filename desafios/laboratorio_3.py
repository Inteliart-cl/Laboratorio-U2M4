import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from desafios.comun import cv_clasificacion, guardar_tabla, particion_clasificacion
from originales.base import SEED, datos_segmentacion

Xtr, Xte, ytr, yte = particion_clasificacion()
candidatos = [
    ("Árbol base", DecisionTreeClassifier(max_depth=4, random_state=SEED)),
    ("Árbol más profundo", DecisionTreeClassifier(max_depth=8, random_state=SEED)),
    ("Árbol con hojas mayores", DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, random_state=SEED)),
    ("Bosque de referencia", RandomForestClassifier(n_estimators=100, random_state=SEED)),
]
filas = []
importancias = pd.DataFrame(index=Xtr.columns)
for nombre, modelo in candidatos:
    scores = cross_val_score(modelo, Xtr, ytr, cv=cv_clasificacion(), scoring="accuracy")
    parametros = modelo.get_params()
    filas.append({
        "modelo": nombre,
        "max_depth": str(parametros["max_depth"]),
        "min_samples_leaf": parametros["min_samples_leaf"],
        "n_estimators": parametros.get("n_estimators", "—"),
        "accuracy_cv": scores.mean(), "desv_cv": scores.std(),
    })
    modelo.fit(Xtr, ytr)
    importancias[nombre] = modelo.feature_importances_

guardar_tabla("L3", filas)
mejor = max(range(len(filas)), key=lambda i: filas[i]["accuracy_cv"])
nombre, modelo = candidatos[mejor]
print(f"Prueba final de {nombre}: accuracy={modelo.score(Xte, yte):.4f}")
print("Importancias por reducción de impureza (ajuste en entrenamiento):")
print(importancias.to_string(float_format=lambda valor: f"{valor:.3f}"))

Z = StandardScaler().fit_transform(datos_segmentacion())
pca = PCA(n_components=2)
coords = pca.fit_transform(Z)
pc1, pc2 = pca.explained_variance_ratio_
print(f"PCA: {coords.shape} | PC1={pc1:.2%} | PC2={pc2:.2%} | Acumulada={pc1 + pc2:.2%}")
