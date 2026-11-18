from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, precision_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from desafios.comun import cv_clasificacion, guardar_tabla, particion_clasificacion
from originales.base import SEED

Xtr, Xte, ytr, yte = particion_clasificacion()
candidatos = [
    ("Logística", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
    ("Bosque base", RandomForestClassifier(n_estimators=100, random_state=SEED)),
    ("SVM", make_pipeline(StandardScaler(), SVC())),
    ("Bosque de 250 árboles", RandomForestClassifier(n_estimators=250, random_state=SEED)),
    ("Bosque con profundidad 6", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED)),
]
metricas = {
    "accuracy": "accuracy",
    "recall_riesgo": "recall",
    "precision_riesgo": make_scorer(precision_score, zero_division=0),
}
filas = []
for nombre, modelo in candidatos:
    scores = cross_validate(modelo, Xtr, ytr, cv=cv_clasificacion(), scoring=metricas)
    parametros = modelo.get_params()
    filas.append({
        "modelo": nombre, "n_estimators": parametros.get("n_estimators", "—"),
        "max_depth": str(parametros["max_depth"]) if "max_depth" in parametros else "—",
        "accuracy_cv": scores["test_accuracy"].mean(),
        "recall_riesgo_cv": scores["test_recall_riesgo"].mean(),
        "precision_riesgo_cv": scores["test_precision_riesgo"].mean(),
    })

guardar_tabla("L8", filas)
mejor = max(range(len(filas)), key=lambda i: filas[i]["accuracy_cv"])
nombre, modelo = candidatos[mejor]
prediccion = modelo.fit(Xtr, ytr).predict(Xte)
print(f"Modelo seleccionado por accuracy CV: {nombre}")
print("Prueba final | clase 1 = riesgo")
print(classification_report(yte, prediccion, digits=4, zero_division=0))
print("Matriz de confusión (orden 0, 1):")
print(confusion_matrix(yte, prediccion, labels=[0, 1]))
