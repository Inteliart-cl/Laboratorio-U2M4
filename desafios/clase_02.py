from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from desafios.comun import ROOT, guardar_tabla, particion_clasificacion
from originales.base import SEED, datos_segmentacion


def main():
    carpeta = ROOT / "evidencias" / "clase_02"
    Xtr, Xte, ytr, yte = particion_clasificacion()
    modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    prediccion = modelo.fit(Xtr, ytr).predict(Xte)
    reporte = classification_report(yte, prediccion, output_dict=True, zero_division=0)
    print(f"Clasificación de riesgo | Entrenamiento: {len(Xtr)} | Prueba: {len(Xte)} | Semilla: {SEED}")
    guardar_tabla("clasificacion", [
        {"categoria": nombre, **reporte[clave]}
        for clave, nombre in [("0", "Clase 0"), ("1", "Clase 1"), ("macro avg", "Macro avg"), ("weighted avg", "Weighted avg")]
    ], carpeta=carpeta)
    print(f"Accuracy: {reporte['accuracy']:.4f} | Matriz de confusión (orden 0, 1):")
    print(confusion_matrix(yte, prediccion, labels=[0, 1]))

    Z = datos_segmentacion()
    configuraciones = [("Original", 0.28, 5), ("A", 0.20, 5), ("B", 0.45, 5), ("C", 0.28, 8)]
    resultados = []
    for experimento, eps, min_samples in configuraciones:
        dbscan = make_pipeline(StandardScaler(), DBSCAN(eps=eps, min_samples=min_samples))
        etiquetas = dbscan.fit_predict(Z)
        resultados.append({
            "experimento": experimento, "eps": eps, "min_samples": min_samples,
            "grupos": len(set(etiquetas) - {-1}), "ruido": (etiquetas == -1).sum(),
        })
    print(f"\nDBSCAN | {len(Z)} observaciones | Variables: {', '.join(Z.columns)}")
    guardar_tabla("dbscan", resultados, carpeta=carpeta)


if __name__ == "__main__":
    main()
