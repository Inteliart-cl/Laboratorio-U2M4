from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from originales.base import SEED, datos_clasificacion

ROOT = Path(__file__).resolve().parents[1]


def particion_clasificacion():
    X, y = datos_clasificacion()
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def cv_clasificacion():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def guardar_tabla(nombre, filas, *, carpeta=ROOT / "evidencias" / "resultados"):
    tabla = pd.DataFrame(filas)
    carpeta.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(carpeta / f"{nombre}.csv", index=False, sep=";", decimal=",", encoding="utf-8-sig")
    print(tabla.to_string(index=False, float_format=lambda valor: f"{valor:.4f}"))
