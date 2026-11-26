from originales.base import *
X,y=datos_clasificacion()
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=SEED)
print("Modelo       | Parámetro      | Accuracy de prueba")
for c in (1.0, 0.5, 2.0):
    modelo = Pipeline([
        ("escala", StandardScaler()),
        ("modelo", SVC(C=c)),
    ])
    modelo.fit(Xtr, ytr)
    print(f"SVM          | C={c:<12} | {modelo.score(Xte, yte):.3f}")

modelo = GaussianNB().fit(Xtr, ytr)
print(f"Naive Bayes  | Base           | {modelo.score(Xte, yte):.3f}")

for vecinos in (5, 3, 9):
    modelo = Pipeline([
        ("escala", StandardScaler()),
        ("modelo", KNeighborsClassifier(n_neighbors=vecinos)),
    ])
    modelo.fit(Xtr, ytr)
    print(f"K-NN         | Vecinos={vecinos:<6} | {modelo.score(Xte, yte):.3f}")

Z = StandardScaler().fit_transform(datos_segmentacion())
print("\nGrupos | Silhouette K-Means | Silhouette jerárquico")
for grupos in (2, 3, 4):
    kmeans = KMeans(n_clusters=grupos, n_init=10, random_state=SEED)
    jerarquico = AgglomerativeClustering(n_clusters=grupos)
    sil_kmeans = silhouette_score(Z, kmeans.fit_predict(Z))
    sil_jerarquico = silhouette_score(Z, jerarquico.fit_predict(Z))
    print(f"{grupos:>6} | {sil_kmeans:>18.3f} | {sil_jerarquico:>21.3f}")

import matplotlib.pyplot as plt

proyeccion = TSNE(
    n_components=2, random_state=SEED, perplexity=30
).fit_transform(Z)
print("t-SNE:", proyeccion.shape)

plt.figure(figsize=(7, 5))
plt.scatter(proyeccion[:, 0], proyeccion[:, 1], s=14, alpha=0.7)
plt.title("Proyección t-SNE de los datos de segmentación")
plt.xlabel("Dimensión t-SNE 1")
plt.ylabel("Dimensión t-SNE 2")
plt.tight_layout()
plt.savefig("evidencias/clase_05/tsne.png", dpi=140)
plt.show()
