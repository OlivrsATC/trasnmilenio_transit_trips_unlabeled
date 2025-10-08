import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# Cargar datos
df = pd.read_csv("transmilenio_transit_trips_unlabeled.csv")

# Codificar variables categóricas
label_cols = ['origin', 'dest', 'origin_line', 'dest_line']
encoder = LabelEncoder()
for col in label_cols:
    df[col] = encoder.fit_transform(df[col])

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# -------------------
# Método Elbow y Silhouette
# -------------------
inertia = []
sil_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# Graficar Elbow y Silhouette
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(K_range, inertia, 'bo-', markersize=8)
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inercia')
plt.title('Método Elbow')
plt.subplot(1,2,2)
plt.plot(K_range, sil_scores, 'ro-', markersize=8)
plt.xlabel('Número de clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score por k')
plt.tight_layout()
plt.show()

# Seleccionar k óptimo
optimal_k = K_range[sil_scores.index(max(sil_scores))]
print(f"Número óptimo de clusters: {optimal_k}")

# Entrenar K-Means final
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_final.fit(X_scaled)
df['cluster'] = kmeans_final.labels_

# Visualización 2D con PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_reduced[:,0], X_reduced[:,1], c=df['cluster'], cmap='viridis', s=30)
plt.title(f"Agrupación de rutas TransMilenio con K-Means (k={optimal_k})")
plt.xlabel("Componente principal 1")
plt.ylabel("Componente principal 2")
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.show()

# KNN no supervisado
knn = NearestNeighbors(n_neighbors=3)
knn.fit(X_scaled)
distances, indices = knn.kneighbors(X_scaled)
print("Distancias a los 3 vecinos más cercanos (primeras 5 filas):")
print(distances[:5])
