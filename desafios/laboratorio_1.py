from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

from desafios.comun import guardar_tabla
from originales.base import SEED, datos_regresion

X, y = datos_regresion()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
configuraciones = [
    ("Base", True, False),
    ("Sin intercepto", False, False),
    ("Coeficientes no negativos", True, True),
]

print(f"Observaciones: {len(X)} | Variables: {X.shape[1]} | Entrenamiento: {len(Xtr)} | Prueba: {len(Xte)}")
print(f"Nulos en X/y: {X.isna().sum().sum()}/{y.isna().sum()} | Filas duplicadas en X: {X.duplicated().sum()}")
filas = []
modelos = []
for nombre, intercepto, positivos in configuraciones:
    modelo = LinearRegression(fit_intercept=intercepto, positive=positivos)
    scores = cross_validate(modelo, Xtr, ytr, cv=cv, scoring=("neg_mean_absolute_error", "r2"))
    filas.append({
        "configuracion": nombre, "fit_intercept": intercepto, "positive": positivos,
        "mae_cv": -scores["test_neg_mean_absolute_error"].mean(), "r2_cv": scores["test_r2"].mean(),
    })
    modelos.append(modelo)

guardar_tabla("L1", filas)
mejor = min(range(len(filas)), key=lambda i: filas[i]["mae_cv"])
prediccion = modelos[mejor].fit(Xtr, ytr).predict(Xte)
print(f"Selección por MAE CV: {filas[mejor]['configuracion']}")
print(f"Prueba final: MAE={mean_absolute_error(yte, prediccion):.4f} | R²={r2_score(yte, prediccion):.4f}")
