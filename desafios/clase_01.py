import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from desafios.comun import ROOT, guardar_tabla
from originales.base import SEED, datos_regresion


def evaluar_regresion(X, y, proporcion):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=proporcion, random_state=SEED)
    prediccion = LinearRegression().fit(Xtr, ytr).predict(Xte)
    return {
        "prueba_pct": int(proporcion * 100),
        "filas_entrenamiento": len(Xtr),
        "filas_prueba": len(Xte),
        "mae": mean_absolute_error(yte, prediccion),
        "r2": r2_score(yte, prediccion),
    }, ytr, yte


def main():
    X, y = datos_regresion()
    carpeta = ROOT / "evidencias" / "clase_01"
    print(f"Datos: {X.shape[0]} filas, {X.shape[1]} indicadores | Objetivo: {y.name} | Semilla: {SEED}")
    print(f"Entradas: {', '.join(X.columns)}")
    print(f"Nulos X/y: {X.isna().sum().sum()}/{y.isna().sum()} | Filas duplicadas en X: {X.duplicated().sum()}")
    print(f"indicador_1: media={X['indicador_1'].mean():.4f} | desviación estándar={X['indicador_1'].std():.4f}")

    original, ytr, yte = evaluar_regresion(X, y, 0.20)
    exploratorio, _, _ = evaluar_regresion(X, y, 0.30)
    print("\nComparación de particiones")
    guardar_tabla("particiones", [
        {"escenario": "A · original", **original},
        {"escenario": "B · exploratorio", **exploratorio},
    ], carpeta=carpeta)

    promedio = ytr.mean()
    mae_referencia = mean_absolute_error(yte, np.full(len(yte), promedio))
    print(f"\nComparación sobre la prueba original | Media de y de entrenamiento: {promedio:.4f}")
    guardar_tabla("referencia_media", [
        {"modelo": "Regresión lineal", "mae": original["mae"]},
        {"modelo": "Media de entrenamiento", "mae": mae_referencia},
    ], carpeta=carpeta)


if __name__ == "__main__":
    main()
