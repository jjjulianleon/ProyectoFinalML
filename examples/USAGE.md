# 📖 Ejemplos de Uso del Proyecto

## 🚀 Ejecución Rápida

### Opción 1: Google Colab (Recomendado)

Abre el notebook directamente en Colab:
- Click en el badge "Open in Colab" en README.md
- O abre: https://colab.research.google.com/github/jjjulianleon/ProyectoFinalML/blob/main/notebooks/ProyectoFinal_ML.ipynb
- El notebook se encarga de todo automáticamente

### Opción 2: Script Python

```bash
# 1. Clonar repositorio
git clone https://github.com/jjjulianleon/ProyectoFinalML.git
cd ProyectoFinalML

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar API key (si se usa OpenAI)
cp .env.example .env
# Editar .env y agregar tu API key

# 4. Ejecutar pipeline completo
python examples/run_full_pipeline.py
```

---

## 💻 Ejemplos de Código

### Ejemplo 1: Cargar Datos de Ejemplo

```python
from src.etl.generate_sample_data import generate_sample_cooperativas_data

# Generar 50 cooperativas de ejemplo
df = generate_sample_cooperativas_data(n_samples=50)

print(df.head())
print(df.info())
print(df['rating'].value_counts())
```

### Ejemplo 2: Análisis de Clustering

```python
import numpy as np
from src.models.clustering import ClusteringAnalyzer

# Preparar datos (sin rating)
numeric_cols = df.select_dtypes(include=[np.number]).columns
X = df[numeric_cols]

# Crear analizador
analyzer = ClusteringAnalyzer(X, random_state=42)
X_scaled = analyzer.preprocess_data()

# Encontrar k óptimo
k_results = analyzer.find_optimal_k(k_range=range(2, 11))
print(k_results)

# Aplicar K-Means
kmeans_labels, metrics = analyzer.kmeans_clustering(n_clusters=3)
print(f"Silhouette Score: {metrics['silhouette']:.3f}")
```

### Ejemplo 3: Comparar Múltiples Algoritmos

```python
# K-Means
km_labels, km_metrics = analyzer.kmeans_clustering(n_clusters=3)

# Agglomerative Clustering
agg_labels, agg_metrics = analyzer.agglomerative_clustering(n_clusters=3)

# DBSCAN
dbscan_labels, dbscan_metrics = analyzer.dbscan_clustering(eps=0.5, min_samples=5)

# Resumen
summary = analyzer.get_summary()
print(summary)
```

### Ejemplo 4: Semi-Supervised Learning

```python
from src.models.semi_supervised import SemiSupervisedLearner

# Preparar datos con ratings
df_with_rating = df[numeric_cols + ['rating']].copy()

# Crear learner
semi = SemiSupervisedLearner(df_with_rating, target_column='rating')
X_semi, y_semi = semi.preprocess_data()

# Baseline supervisado
baseline = semi.supervised_baseline()
print(f"Accuracy baseline: {baseline['accuracy']:.3f}")

# Label Propagation con 30% datos etiquetados
lp_results = semi.label_propagation(labeled_ratio=0.3)
print(f"Accuracy (30% labeled): {lp_results['accuracy']:.3f}")

# Self-Training con 30% datos etiquetados
st_results = semi.self_training(labeled_ratio=0.3)
print(f"Accuracy (30% labeled): {st_results['accuracy']:.3f}")
```

### Ejemplo 5: Comparar Ratios de Labels

```python
# Evaluar rendimiento con diferentes ratios
ratios = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
results = semi.compare_ratios(ratios=ratios)

print(results)
results.to_csv('semi_supervised_results.csv', index=False)
```

### Ejemplo 6: Descarga Automática de PDFs

```python
from src.etl.pdf_downloader import PDFDownloader

# Crear descargador
downloader = PDFDownloader(output_dir="data/raw")

# URLs para descargar
urls = [
    "https://www.asis.fin.ec/...",
    "https://www.seps.gob.ec/..."
]

# Descargar
results = downloader.download_pdfs_batch(urls)

print(f"✓ Descargados: {len(results['successful'])}")
print(f"✗ Fallidos: {len(results['failed'])}")
```

### Ejemplo 7: Extraer Datos de PDFs con OpenAI

```python
import os
from dotenv import load_dotenv
from src.etl.data_extractor import DataExtractor, read_pdf_text

# Cargar API key
load_dotenv()

# Crear extractor
extractor = DataExtractor(
    api_key=os.getenv('OPENAI_API_KEY'),
    model='gpt-3.5-turbo'
)

# Leer PDF
pdf_path = "data/raw/documento.pdf"
text = read_pdf_text(pdf_path)

# Extraer datos estructurados
data = extractor.extract_from_text(text, cooperativa_name="Ejemplo")
print(data)
```

### Ejemplo 8: Visualización con t-SNE

```python
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Preparar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_cols])

# Aplicar t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Visualizar
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                     c=pd.Categorical(df['rating']).codes,
                     cmap='viridis', s=100, alpha=0.6)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('Cooperativas - Visualización t-SNE')
plt.colorbar(scatter, label='Rating')
plt.show()
```

### Ejemplo 9: Matriz de Confusión

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Comparar clustering vs ratings
ratings_encoded = pd.Categorical(df['rating']).codes
cm = confusion_matrix(ratings_encoded, kmeans_labels)

# Visualizar
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Rating Real')
plt.xlabel('Cluster Predicho')
plt.title('K-Means vs Ratings Reales')
plt.show()
```

### Ejemplo 10: Análisis Detallado de Clusters

```python
# Agregar cluster a dataframe
df['cluster'] = kmeans_labels

# Estadísticas por cluster
for cluster in np.unique(kmeans_labels):
    print(f"\n=== CLUSTER {cluster} ===")
    cluster_data = df[df['cluster'] == cluster]

    print(f"Tamaño: {len(cluster_data)}")
    print(f"Distribución de ratings:")
    print(cluster_data['rating'].value_counts())
    print(f"\nPromedio de indicadores:")
    print(cluster_data[numeric_cols].mean())
```

---

## 📊 Análisis Estadístico

### Correlaciones Altas

```python
# Encontrar variables altamente correlacionadas
corr_matrix = df[numeric_cols].corr()

# Triangular superior
import numpy as np
upper = np.triu(corr_matrix, k=1)

# Encontrar correlaciones > 0.8
for i, col1 in enumerate(numeric_cols):
    for j, col2 in enumerate(numeric_cols):
        if i < j and abs(upper[i, j]) > 0.8:
            print(f"{col1} <-> {col2}: {upper[i, j]:.3f}")
```

### Outliers

```python
from scipy import stats

# Método Z-score
z_scores = np.abs(stats.zscore(df[numeric_cols]))
outliers = (z_scores > 3).any(axis=1)

print(f"Outliers detectados: {outliers.sum()}")
print(df[outliers])
```

### Validación Cruzada

```python
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Evaluar estabilidad del clustering
scores = []
for i in range(10):
    km = KMeans(n_clusters=3, random_state=i)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    scores.append(score)

print(f"Mean Silhouette Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

---

## 🎨 Visualizaciones Personalizadas

### Gráfico de Clusters Customizado

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Scatter por rating
ax = axes[0, 0]
for rating in df['rating'].unique():
    mask = df['rating'] == rating
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
              label=f'Rating {rating}', s=100, alpha=0.6)
ax.set_title('Cooperativas por Rating')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Scatter por cluster
ax = axes[0, 1]
scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_labels,
                    cmap='viridis', s=100, alpha=0.6)
ax.set_title('K-Means Clusters')
plt.colorbar(scatter, ax=ax)
ax.grid(alpha=0.3)

# Plot 3: Silhouette vs k
ax = axes[1, 0]
ax.plot(k_results['k'], k_results['silhouette'], 'o-', linewidth=2)
ax.set_xlabel('k')
ax.set_ylabel('Silhouette Score')
ax.set_title('Elbow Method')
ax.grid(alpha=0.3)

# Plot 4: Comparison de algoritmos
ax = axes[1, 1]
metrics_names = ['K-Means', 'Agglomerative', 'DBSCAN']
silhouette_scores = [
    kmeans_metrics['silhouette'],
    agg_metrics['silhouette'],
    dbscan_metrics['silhouette']
]
ax.bar(metrics_names, silhouette_scores)
ax.set_ylabel('Silhouette Score')
ax.set_title('Comparación de Algoritmos')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

---

## 🔧 Debugging y Solución de Problemas

### Verificar Datos

```python
# Chequeo básico
print(f"Shape: {df.shape}")
print(f"Dtypes:\n{df.dtypes}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")

# Estadísticas
print(df.describe())

# Valores únicos por clase
print(df['rating'].value_counts())
```

### Diagnóstico de Clustering

```python
# Verificar que clustering funcionó
print(f"Clusters únicos: {np.unique(kmeans_labels)}")
print(f"Tamaño de clusters: {np.bincount(kmeans_labels)}")

# Comprobar convergencia
km = KMeans(n_clusters=3, verbose=1)
km.fit(X_scaled)
print(f"Inertia: {km.inertia_:.2f}")
print(f"Iteraciones: {km.n_iter_}")
```

### Performance

```python
import time

# Medir tiempo de ejecución
start = time.time()
labels = analyzer.kmeans_clustering(3)[0]
elapsed = time.time() - start
print(f"Tiempo: {elapsed:.3f}s")

# Memoria
import sys
print(f"Tamaño del dataset: {sys.getsizeof(X_scaled) / 1024 / 1024:.2f} MB")
```

---

## 📚 Recursos Adicionales

- [scikit-learn Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [t-SNE Papers](https://lvdmaaten.github.io/tsne/)

---

**Última actualización:** Noviembre 2025
