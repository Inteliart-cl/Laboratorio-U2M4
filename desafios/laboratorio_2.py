from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from desafios.comun import guardar_tabla, particion_clasificacion
from originales.base import datos_segmentacion

Xtr, Xte, ytr, yte = particion_clasificacion()
clasificador = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
prediccion = clasificador.fit(Xtr, ytr).predict(Xte)
print("Regresión logística | Prueba estratificada")
print(classification_report(yte, prediccion, digits=4, zero_division=0))
print("Matriz de confusión (orden 0, 1):")
print(confusion_matrix(yte, prediccion, labels=[0, 1]))

Z = StandardScaler().fit_transform(datos_segmentacion())
filas = []
for eps, minimo in [(0.28, 5), (0.15, 5), (0.45, 5), (0.28, 10)]:
    etiquetas = DBSCAN(eps=eps, min_samples=minimo).fit_predict(Z)
    ruido = (etiquetas == -1).sum()
    filas.append({
        "eps": eps, "min_samples": minimo, "grupos": len(set(etiquetas) - {-1}),
        "ruido": ruido, "ruido_pct": 100 * ruido / len(Z),
    })

print("DBSCAN | Segmentación de 360 observaciones")
guardar_tabla("L2", filas)
