from pathlib import Path

from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from desafios.comun import guardar_tabla, particion_clasificacion
from originales.base import SEED, datos_segmentacion


def main():
    carpeta = Path(__file__).resolve().parent
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

    revision = yte.to_frame("riesgo")
    revision["prediccion"] = prediccion
    revision["puntaje"] = modelo.predict_proba(Xte)[:, 1]
    negativos = revision.loc[revision["prediccion"] == 0]
    cercanos = negativos.nlargest(10, "puntaje").rename_axis("registro").reset_index()
    print("\nSimulación | Diez casos clasificados como sin riesgo más próximos al umbral")
    guardar_tabla("revision_umbral", cercanos, carpeta=carpeta)
    print(f"Casos de riesgo recuperados: {cercanos['riesgo'].sum()} de {len(cercanos)} revisados")

    Z = datos_segmentacion()
    configuraciones = [("Radio menor", 0.18, 5), ("Original", 0.28, 5), ("Radio mayor", 0.45, 5)]
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
