# Proyecto IA: Clustering de rutas de TransMilenio

## Descripción

Este proyecto aplica **métodos de aprendizaje no supervisado** para analizar rutas del sistema de transporte masivo TransMilenio.  
Se utilizan algoritmos de **K-Means** y **KNN no supervisado** para agrupar viajes similares y encontrar patrones en los datos, considerando características como distancia, hora, número de transbordos y líneas de origen/destino.

El objetivo es mostrar cómo la inteligencia artificial puede ayudar a la **optimización de rutas y análisis de transporte urbano**.

---

## Contenido del repositorio

| Archivo | Descripción |
|---------|-------------|
| `transmilenio_transit_trips_unlabeled.csv` | Dataset con 2201 registros de viajes de TransMilenio. |
| `clustering_transmilenio.ipynb` | Notebook con el código en Python que realiza: preprocesamiento, K-Means, KNN, visualización PCA, método Elbow y cálculo de Silhouette Score. |
| `README.md` | Este archivo de descripción del proyecto. |

---

## Requisitos

Para ejecutar el proyecto necesitas tener instaladas las siguientes librerías de Python:


```bash
pip install pandas matplotlib scikit-learn


