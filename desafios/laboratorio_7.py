from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from desafios.comun import cv_clasificacion, guardar_tabla, particion_clasificacion

Xtr, Xte, ytr, yte = particion_clasificacion()
pipeline = Pipeline([("escala", StandardScaler()), ("modelo", KNeighborsClassifier())])
vecinos_base = [3, 5, 7, 9]
grillas = [
    ("Base", vecinos_base, ["uniform"]),
    ("Más valores de k", [1, 3, 5, 7, 9, 11, 13, 15], ["uniform"]),
    ("Ponderación por distancia", vecinos_base, ["uniform", "distance"]),
]
filas = []
busquedas = []
for nombre, vecinos, pesos in grillas:
    parametros = {"modelo__n_neighbors": vecinos, "modelo__weights": pesos}
    busqueda = GridSearchCV(pipeline, parametros, cv=cv_clasificacion(), scoring="accuracy")
    busqueda.fit(Xtr, ytr)
    filas.append({
        "grilla": nombre, "n_neighbors_probados": str(vecinos), "weights_probados": ", ".join(pesos),
        "mejor_k": busqueda.best_params_["modelo__n_neighbors"],
        "mejor_weights": busqueda.best_params_["modelo__weights"],
        "accuracy_cv": busqueda.best_score_,
    })
    busquedas.append(busqueda)

guardar_tabla("L7", filas)
# El orden favorece la grilla base cuando la mejor puntuación es igual.
mejor = max(range(len(filas)), key=lambda i: filas[i]["accuracy_cv"])
print(f"Selección: {filas[mejor]['grilla']} | {busquedas[mejor].best_params_}")
print(f"Prueba final: accuracy={busquedas[mejor].score(Xte, yte):.4f}")
