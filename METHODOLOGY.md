# 🔬 Metodología del Proyecto

## 📋 Resumen Ejecutivo

Este proyecto implementa técnicas de **Machine Learning no supervisado y semi-supervisado** para analizar y agrupar cooperativas de ahorro y crédito del Segmento 1 en Ecuador según sus indicadores financieros.

---

## 🎯 Objetivos

### Objetivos Principales

1. **Identificar grupos naturales** de cooperativas con comportamientos financieros similares
2. **Validar coherencia** entre clusters automáticos y ratings de riesgo externos
3. **Comparar enfoques** supervisados vs semi-supervisados
4. **Evaluar impacto** de cantidad de datos etiquetados en rendimiento

### Hipótesis

- Los indicadores financieros contienen patrones que permiten agrupar cooperativas naturalmente
- Estos grupos mostrarán cierta correlación con ratings de riesgo asignados externamente
- El aprendizaje semi-supervisado puede aproximar rendimiento supervisado con menos datos etiquetados

---

## 📊 FASE 1: ADQUISICIÓN Y PREPARACIÓN DE DATOS

### 1.1 Obtención de Datos

**Método:** Web Scraping + Extracción con LLM

```
PDFs de Reportes Financieros
            ↓
    [PDF Downloader]
            ↓
Archivos PDF en data/raw/
            ↓
    [PDF Text Reader]
            ↓
Texto extraído
            ↓
    [OpenAI API]
            ↓
JSON Estructurado
            ↓
    [Consolidación]
            ↓
CSV consolidado → data/processed/cooperativas_data.csv
```

**Fuentes:**
- SEPS (Superintendencia de Economía Popular y Solidaria)
- ASIS (Reportes de indicadores financieros)
- Reportes directos de cooperativas

**Variables extraídas:** 12+ indicadores financieros

### 1.2 Limpieza de Datos

```python
# Validaciones realizadas:
1. Verificar valores faltantes
2. Detectar outliers estadísticos
3. Validar rangos de valores (0-1 para ratios)
4. Remover duplicados
5. Verificar tipos de datos
```

**Resultado esperado:**
- 50-200 cooperativas
- 12 variables numéricas + 2 categóricas
- 0% valores faltantes (después de limpieza)

---

## 📈 FASE 2: ANÁLISIS EXPLORATORIO (EDA)

### 2.1 Estadística Descriptiva

```python
# Para cada variable:
- Media, mediana, desviación estándar
- Rango (mín, máx)
- Percentiles (Q1, Q3)
- Distribución por rating
```

### 2.2 Análisis de Correlaciones

```python
# Crear matriz de correlación de Pearson
# Objetivo: Identificar:
- Variables altamente correlacionadas (|r| > 0.8)
- Posibles redundancias
- Relaciones lineales con el rating
```

### 2.3 Reducción Dimensional (t-SNE)

```python
# Aplicar t-SNE para visualización 2D
# Parámetros:
- n_components = 2
- perplexity = 30
- n_iter = 1000
- random_state = 42

# Objetivo: Visualizar separación natural de cooperativas
# Color: Rating real
# Resultado: Gráfico interactivo
```

### 2.4 Visualizaciones Generadas

1. **Distribuciones por rating** - Histogramas de cada variable
2. **Matriz de correlación** - Heatmap de correlaciones
3. **t-SNE plot** - Espacio 2D coloreado por rating
4. **Box plots** - Distribución por grupo

---

## 🤖 FASE 3: CLUSTERING NO SUPERVISADO

### 3.1 Normalización de Datos

```python
# Aplicar StandardScaler
# Razón: Diferentes escalas entre variables
X_scaled = StandardScaler().fit_transform(X)
# Resultado: Media=0, Desv.Est.=1 para cada variable
```

### 3.2 Algoritmo 1: K-Means (Baseline)

**Característica:** Partition-based clustering

```
1. Inicializar k centroides aleatoriamente
2. Asignar cada punto al centroide más cercano
3. Recalcular centroides como media de los clusters
4. Repetir hasta convergencia
```

**Ventajas:**
- Rápido y escalable
- Fácil de interpretar
- Determinístico (con seed)

**Desventajas:**
- Requiere especificar k de antemano
- Sensible a inicialización
- Asume clusters esféricos

**Hiperparámetros:**
- `n_clusters`: Determinado por Elbow Method / Silhouette
- `n_init`: 10 (número de veces a correr)
- `random_state`: 42 (reproducibilidad)

### 3.3 Algoritmo 2: Agglomerative Clustering

**Característica:** Hierarchical (bottom-up)

```
1. Inicio: cada punto es su propio cluster
2. Fusión iterativa de clusters más similares
3. Hasta obtener k clusters
```

**Ventajas:**
- Produce dendrogramas (histórico de fusiones)
- No requiere inicialización aleatoria
- Más estable que K-Means

**Desventajas:**
- Mayor complejidad computacional O(n²)
- No es aplicable a datasets muy grandes

**Hiperparámetros:**
- `n_clusters`: Mismo que K-Means
- `linkage`: 'ward' (minimiza varianza intra-cluster)

### 3.4 Algoritmo 3: DBSCAN

**Característica:** Density-based clustering

```
1. Para cada punto no visitado:
   - Encontrar vecinos dentro de eps
   - Si >= min_samples: crear cluster
   - Expandir cluster recursivamente
2. Puntos aislados marcados como ruido (-1)
```

**Ventajas:**
- Detecta clusters de forma arbitraria
- Identifica outliers (ruido)
- No requiere especificar k

**Desventajas:**
- Sensible a hiperparámetros (eps, min_samples)
- Rendimiento variable en clusters de densidades diferentes

**Hiperparámetros:**
- `eps`: Radio máximo (determinar mediante k-distance graph)
- `min_samples`: Mínimo puntos para formar cluster (default: 2*dim)

### 3.5 Determinación del Número Óptimo de Clusters

**Método 1: Elbow Method**
```python
for k in range(2, 11):
    inertia = KMeans(k).fit(X).inertia_
# Gráficar inertia vs k
# Codo indica k óptimo
```

**Método 2: Silhouette Score**
```python
# s(i) = (b(i) - a(i)) / max(a(i), b(i))
# a(i): distancia promedio a otros puntos del mismo cluster
# b(i): distancia promedio a puntos del cluster más cercano
# Rango: -1 a 1 (mayor es mejor)

for k in range(2, 11):
    score = silhouette_score(X, KMeans(k).fit_predict(X))
# k con score máximo es óptimo
```

**Método 3: Davies-Bouldin Index**
```python
# DB = 1/k * Σ max(Si + Sj / dij)
# Si: dispersión promedio dentro del cluster i
# dij: distancia entre centroides i y j
# Rango: 0 a ∞ (menor es mejor)
```

---

## 📊 FASE 4: EVALUACIÓN DE CLUSTERING

### 4.1 Métricas Intrínsecas (sin labels)

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| **Silhouette Score** | promedio de s(i) para todos los puntos | -1 a 1, mayor es mejor |
| **Davies-Bouldin Index** | promedio de ratios de similitud | 0 a ∞, menor es mejor |
| **Calinski-Harabasz Index** | ratio de dispersión entre/intra cluster | mayor es mejor |

### 4.2 Métricas Extrínsecas (con labels reales)

| Métrica | Definición |
|---------|-----------|
| **Adjusted Rand Index (ARI)** | Acuerdo entre dos clusterings (ajustado por azar) |
| **Normalized Mutual Information** | Información compartida entre clustering y labels |
| **Purity** | Proporción de puntos en cluster puro |

```python
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(labels_true, labels_pred)
# -1 ≤ ARI ≤ 1
# 1: acuerdo perfecto
# 0: acuerdo aleatorio
# <0: peor que acuerdo aleatorio
```

### 4.3 Matriz de Confusión

```python
# Comparar cluster predicho vs rating real
cm = confusion_matrix(ratings_encoded, cluster_labels)
# Visualizar como heatmap
```

---

## 🧠 FASE 5: SEMI-SUPERVISED LEARNING

### 5.1 Enfoque 1: Baseline Supervisado

```python
# Entrenar con 100% de datos etiquetados
LogisticRegression().fit(X, y)
# Baseline de referencia para comparar
```

**Métricas calculadas:**
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)

### 5.2 Enfoque 2: Label Propagation

**Algoritmo:**

```
1. Crear grafo de similitud entre puntos
2. Inicializar labels conocidos
3. Propagar labels iterativamente:
   label(i) = argmax Σ_j(kernel(i,j) * label(j))
4. Repetir hasta convergencia
```

**Ventajas:**
- Simple y eficaz
- Funciona bien con pocos labels

**Parámetros:**
- `kernel`: 'rbf' (Radial Basis Function)
- `gamma`: 20 (ancho del kernel)

### 5.3 Enfoque 3: Self-Training

**Algoritmo:**

```
1. Entrenar modelo con datos etiquetados
2. Predecir en datos no etiquetados
3. Agregar predicciones confiables al conjunto etiquetado
4. Reentrenar
5. Repetir hasta convergencia
```

**Ventajas:**
- Iterativo y mejora gradual
- Usa modelo base flexible

**Parámetros:**
- `base_estimator`: DecisionTreeClassifier
- `threshold`: 0.75 (confianza mínima)
- `max_iter`: 10 (iteraciones máximas)

### 5.4 Comparación Variando Labels

```python
# Para ratios [10%, 20%, 30%, 50%, 70%]:
for ratio in ratios:
    # Seleccionar aleatoriamente ratio% de datos como etiquetados
    # Entrenar Label Propagation
    # Entrenar Self-Training
    # Calcular métricas
    # Comparar contra baseline
```

**Objetivo:** Encontrar punto de inflexión donde semi-supervised se acerca a supervisado.

---

## 📊 FASE 6: ANÁLISIS E INTERPRETACIÓN

### 6.1 Patrones Observados en Clustering

Analizar para cada cluster:
- Distribución de ratings
- Características financieras distintivas
- Interpretación económica

### 6.2 Discrepancias entre Clusters y Ratings

Investigar:
- Cooperativas asignadas a cluster diferente de su rating
- Posibles razones (indicadores más recientes, cambios operacionales)
- Implicaciones para política de crédito

### 6.3 Performance del Semi-Supervised Learning

Conclusiones sobre:
- Cuántos datos etiquetados son necesarios
- Cuál método es más apropiado para esta tarea
- Viabilidad de implementación en producción

---

## 📁 Estructura de Salidas

```
figures/
├── 01_distribucion_por_rating.png
├── 02_matriz_correlacion.png
├── 03_tsne_visualization.png
├── 04_elbow_analysis.png
├── 05_clustering_results_tsne.png
├── 06_confusion_matrices.png
└── 07_semi_supervised_comparison.png

data/processed/
├── cooperativas_data.csv
├── cooperativas_clustered.csv
├── clustering_metrics.csv
└── semi_supervised_results.csv
```

---

## 🔍 Validaciones Implementadas

1. **Verificación de datos:**
   - Valores en rangos válidos
   - Ausencia de NaN después de limpieza
   - Duplicados removidos

2. **Validación de clustering:**
   - k ≥ 2 clusters
   - Todos los clusters contienen ≥ 1 punto
   - Labels en rango [0, k-1]

3. **Validación de semi-supervised:**
   - Mínimo 1 label por clase
   - Ratio de labels en [0, 1]
   - Métricas calculadas para todas las clases

---

## 📚 Referencias Teóricas

### Clustering
1. Lloyd, S. (1982). Least squares quantization in PCM
2. Rousseeuw, P. J. (1987). Silhouettes
3. Davies & Bouldin (1979). A cluster separation measure
4. Ester et al. (1996). A density-based algorithm for discovering clusters

### Semi-Supervised Learning
1. Zhou et al. (2004). Learning with Local and Global Consistency
2. Chapelle et al. (2006). Semi-Supervised Learning

### Métricas
1. Hubert & Arabie (1985). Comparing partitions
2. Vinh et al. (2009). Information theoretic measures for clusterings

---

## 🚀 Implementación en Producción

Para llevar este análisis a producción:

1. **Pipeline automático:**
   - Descargas mensuales de reportes
   - Procesamiento automático
   - Actualización de clusters

2. **Monitoreo:**
   - Estabilidad temporal de clusters
   - Drift en indicadores financieros
   - Performance de predicciones

3. **API REST:**
   - Predecir cluster/rating para nueva cooperativa
   - Consultar métricas históricas
   - Exportar reportes

---

**Fecha de elaboración:** Noviembre 2025

**Última revisión:** Noviembre 2025
