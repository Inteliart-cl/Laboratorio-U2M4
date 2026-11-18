from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier

from desafios.comun import cv_clasificacion, guardar_tabla, particion_clasificacion
from originales.base import SEED

Xtr, Xte, ytr, yte = particion_clasificacion()
configuraciones = [(profundidad, 1) for profundidad in [2, 4, 8, None]]
configuraciones += [(4, 8), (8, 8), (None, 8)]
filas = []
modelos = []
for profundidad, minimo in configuraciones:
    modelo = DecisionTreeClassifier(max_depth=profundidad, min_samples_leaf=minimo, random_state=SEED)
    scores = cross_validate(modelo, Xtr, ytr, cv=cv_clasificacion(), scoring="accuracy", return_train_score=True)
    train = scores["train_score"].mean()
    validacion = scores["test_score"].mean()
    filas.append({
        "max_depth": str(profundidad), "min_samples_leaf": minimo,
        "accuracy_train_cv": train, "accuracy_validacion_cv": validacion,
        "brecha": train - validacion,
    })
    modelos.append(modelo)

guardar_tabla("L4", filas)
# En caso de empate, se elige la menor profundidad y luego la hoja mínima más grande.
mejor = max(
    range(len(filas)),
    key=lambda i: (filas[i]["accuracy_validacion_cv"], -(configuraciones[i][0] or float("inf")), configuraciones[i][1]),
)
profundidad, minimo = configuraciones[mejor]
test = modelos[mejor].fit(Xtr, ytr).score(Xte, yte)
print(f"Selección por validación: max_depth={profundidad}, min_samples_leaf={minimo}")
print(f"Prueba final: accuracy={test:.4f}")
