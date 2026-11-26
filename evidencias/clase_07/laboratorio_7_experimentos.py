"""Reproduce las tres búsquedas y evalúa el modelo elegido por CV."""

from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from desafios.comun import cv_clasificacion, guardar_tabla, particion_clasificacion


def main():
    Xtr, Xte, ytr, yte = particion_clasificacion()
    pipeline = Pipeline([
        ("escala", StandardScaler()),
        ("modelo", KNeighborsClassifier()),
    ])
    vecinos_base = [3, 5, 7, 9]
    rejillas = [
        ("Inicial", {"modelo__n_neighbors": vecinos_base}),
        ("A", {"modelo__n_neighbors": [1, 3, 5, 7, 9, 11, 15]}),
        ("B", {
            "modelo__n_neighbors": vecinos_base,
            "modelo__weights": ["uniform", "distance"],
            "modelo__p": [1, 2],
        }),
    ]
    cv = list(cv_clasificacion().split(Xtr, ytr))
    busquedas = {
        nombre: GridSearchCV(pipeline, parametros, cv=cv, scoring="accuracy").fit(Xtr, ytr)
        for nombre, parametros in rejillas
    }

    # El conjunto de prueba se consulta después de seleccionar por validación.
    nombre_elegido = max(busquedas, key=lambda nombre: busquedas[nombre].best_score_)
    print(f"Configuración elegida por CV: {nombre_elegido}")
    print(f"Entrenamiento: {len(Xtr)} | Test: {len(Xte)} | Rondas de CV: {len(cv)}")
    filas = []
    for nombre, busqueda in busquedas.items():
        parametros = busqueda.best_estimator_.named_steps["modelo"].get_params()
        desviacion = busqueda.cv_results_["std_test_score"][busqueda.best_index_]
        filas.append({
            "configuracion": nombre,
            "combinaciones": len(busqueda.cv_results_["params"]),
            "n_neighbors": parametros["n_neighbors"],
            "p": parametros["p"],
            "weights": parametros["weights"],
            "accuracy_cv": busqueda.best_score_,
            "desv_cv": desviacion,
            "accuracy_test": busqueda.score(Xte, yte),
        })

    carpeta = Path(__file__).resolve().parent
    guardar_tabla("02_tabla_resultados", filas, carpeta=carpeta)
    print("Test de Inicial y A: registro didáctico.")

    elegido = busquedas[nombre_elegido]
    pred = elegido.predict(Xte)
    print(f"Mejores parámetros: {elegido.best_params_}")
    print("Matriz de confusión (filas reales, columnas predichas, orden 0 y 1):")
    print(confusion_matrix(yte, pred, labels=[0, 1]))
    print("Reporte de clasificación:")
    print(classification_report(
        yte, pred, labels=[0, 1], target_names=["Sin riesgo", "Con riesgo"],
        digits=3, zero_division=0,
    ))


if __name__ == "__main__":
    main()
