import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from desafios.comun import ROOT, cv_clasificacion, guardar_tabla, particion_clasificacion
from originales.base import SEED, datos_segmentacion

Xtr, Xte, ytr, yte = particion_clasificacion()
candidatos = [
    ("SVM base", make_pipeline(StandardScaler(), SVC(C=1, gamma="scale")), 1, "scale"),
    ("SVM C mayor", make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")), 10, "scale"),
    ("SVM gamma menor", make_pipeline(StandardScaler(), SVC(C=1, gamma=0.01)), 1, "0.01"),
    ("Naive Bayes", GaussianNB(), "—", "—"),
    ("K-NN (k=5)", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)), "—", "—"),
]
filas = []
for nombre, modelo, c, gamma in candidatos:
    cv = cross_val_score(modelo, Xtr, ytr, cv=cv_clasificacion(), scoring="accuracy")
    filas.append({"modelo": nombre, "C": c, "gamma": gamma, "accuracy_cv": cv.mean(), "desv_cv": cv.std()})

guardar_tabla("L5_clasificacion", filas)
mejor = max(range(len(filas)), key=lambda i: filas[i]["accuracy_cv"])
nombre, modelo, _, _ = candidatos[mejor]
print(f"Prueba final de {nombre}: accuracy={modelo.fit(Xtr, ytr).score(Xte, yte):.4f}")

Z = StandardScaler().fit_transform(datos_segmentacion())
kmeans = KMeans(n_clusters=3, n_init=10, random_state=SEED)
etiquetas_kmeans = kmeans.fit_predict(Z)
etiquetas_jerarquico = AgglomerativeClustering(n_clusters=3).fit_predict(Z)
guardar_tabla("L5_agrupamiento", [
    {"modelo": nombre, "grupos": 3, "silhouette": silhouette_score(Z, etiquetas)}
    for nombre, etiquetas in [("K-Means", etiquetas_kmeans), ("Jerárquico", etiquetas_jerarquico)]
])

coords = TSNE(n_components=2, perplexity=30, random_state=SEED).fit_transform(Z)
print(f"t-SNE: {coords.shape} | gráfico coloreado con etiquetas de K-Means")
carpeta = ROOT / "evidencias" / "graficos"
carpeta.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
for grupo, color in enumerate(["#235789", "#bc4b25", "#338260"]):
    puntos = coords[etiquetas_kmeans == grupo]
    ax.scatter(puntos[:, 0], puntos[:, 1], s=19, alpha=0.76, color=color, label=f"Grupo {grupo + 1}")
ax.set(title="Proyección t-SNE de la segmentación sintética", xlabel="Componente t-SNE 1", ylabel="Componente t-SNE 2")
ax.legend(title="K-Means", loc="best")
fig.savefig(carpeta / "L5_tsne.png", dpi=160)
plt.close(fig)
