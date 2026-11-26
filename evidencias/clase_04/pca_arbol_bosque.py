from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from desafios.comun import guardar_tabla, particion_clasificacion
from originales.base import SEED, datos_segmentacion


def main():
    carpeta = Path(__file__).resolve().parent
    Z = StandardScaler().fit_transform(datos_segmentacion())
    filas_pca = []
    for componentes in (2, 1):
        pca = PCA(n_components=componentes)
        coordenadas = pca.fit_transform(Z)
        filas_pca.append({
            "componentes": componentes,
            "forma": str(coordenadas.shape),
            "varianza_retenida_pct": 100 * pca.explained_variance_ratio_.sum(),
        })
    print("PCA | Frecuencia y valor estandarizados")
    guardar_tabla("pca", filas_pca, carpeta=carpeta)

    Xtr, Xte, ytr, yte = particion_clasificacion()
    arboles = [
        ("Árbol", profundidad, DecisionTreeClassifier(max_depth=profundidad, random_state=SEED))
        for profundidad in (2, 4, 8)
    ]
    bosque = ("Bosque", 100, RandomForestClassifier(n_estimators=100, random_state=SEED))
    filas_modelos = []
    for nombre, valor, modelo in arboles + [bosque]:
        modelo.fit(Xtr, ytr)
        filas_modelos.append({
            "modelo": nombre,
            "valor": valor,
            "accuracy": modelo.score(Xte, yte),
            **dict(zip(Xtr.columns, modelo.feature_importances_)),
        })
    print(f"\nClasificación | Entrenamiento: {len(Xtr)} | Prueba: {len(Xte)} | Semilla: {SEED}")
    print("Valor = max_depth para Árbol y n_estimators para Bosque")
    guardar_tabla("modelos", filas_modelos, carpeta=carpeta)


if __name__ == "__main__":
    main()
