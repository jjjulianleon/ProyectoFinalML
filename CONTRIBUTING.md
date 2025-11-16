# 🤝 Contribuyendo al Proyecto

Gracias por tu interés en mejorar este proyecto. Esta guía te ayudará a entender cómo contribuir.

---

## 📋 Proceso General

1. Fork el repositorio
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Hacer cambios y commits descriptivos
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

---

## 🎯 Áreas de Contribución

### 1. Nuevos Algoritmos de Clustering

**Ubicación:** `src/models/clustering.py`

**Pasos:**
```python
# Agregar nuevo método a ClusteringAnalyzer
def gaussian_mixture_clustering(self, n_components: int) -> Tuple[np.ndarray, Dict]:
    """Implementar Gaussian Mixture Models."""
    # Implementar GMM
    # Calcular métricas
    # Retornar labels y métricas
    return labels, metrics
```

**Consideraciones:**
- Mantener consistencia con interfaz existente
- Incluir docstrings
- Calcular métricas estándar (Silhouette, Davies-Bouldin)
- Agregar test

### 2. Métodos de Semi-Supervised Learning

**Ubicación:** `src/models/semi_supervised.py`

**Pasos:**
```python
# Agregar nuevo método a SemiSupervisedLearner
def manifold_propagation(self, labeled_ratio: float = 0.2) -> Dict:
    """Implementar Manifold-based semi-supervised learning."""
    # Implementar algoritmo
    # Calcular métricas
    # Retornar diccionario con resultados
    return metrics
```

**Consideraciones:**
- Mantener parámetro `labeled_ratio` consistente
- Comparar contra baseline supervisado
- Documentar hiperparámetros

### 3. Mejoras en Extracción de Datos

**Ubicación:** `src/etl/data_extractor.py`

**Ideas:**
- Soportar más formatos (Excel, JSON, XML)
- Mejorar extracción de datos con Vision API
- Agregar validación de datos
- Implementar caché de resultados

### 4. Visualizaciones

**Ubicación:** `notebooks/ProyectoFinal_ML.ipynb`

**Ideas:**
- Gráficos interactivos con Plotly
- Dendrogramas para Agglomerative Clustering
- Visualizaciones 3D
- Dashboards con Streamlit

### 5. Documentación

**Áreas:**
- Traducción a otros idiomas
- Tutoriales adicionales
- FAQ
- Guías de interpretación de resultados

---

## 💻 Configuración de Desarrollo

### 1. Crear Ambiente de Desarrollo

```bash
git clone https://github.com/jjjulianleon/ProyectoFinalML.git
cd ProyectoFinalML

python -m venv venv_dev
source venv_dev/bin/activate  # Windows: venv_dev\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Configurar Pre-commit Hooks (Opcional)

```bash
pip install pre-commit
pre-commit install
```

### 3. Ejecutar Tests

```bash
pytest tests/
```

---

## ✅ Checklist de Calidad

Antes de hacer un Pull Request, verifica:

- [ ] Código sigue la guía de estilo (PEP 8)
- [ ] Docstrings en español o inglés clara
- [ ] Tipos de datos anotados
- [ ] Tests incluidos
- [ ] README actualizado si aplica
- [ ] No hay código comentado sin razón
- [ ] No hay credenciales expuestas
- [ ] Compatible con Python 3.8+

---

## 🔍 Guía de Estilo

### Nombres de Variables

```python
# ✓ Bien
cooperativa_data = pd.DataFrame()
n_clusters = 3
silhouette_score = 0.75

# ✗ Mal
coop_data = pd.DataFrame()
k = 3
s_score = 0.75
```

### Docstrings

```python
def calculate_metrics(labels: np.ndarray, X: np.ndarray) -> Dict:
    """
    Calcula métricas de evaluación de clustering.

    Args:
        labels: Array de etiquetas de cluster (0, 1, ..., k-1)
        X: Array de características (n_samples, n_features)

    Returns:
        Diccionario con métricas:
        - silhouette: Silhouette Score
        - davies_bouldin: Davies-Bouldin Index
        - calinski_harabasz: Calinski-Harabasz Index

    Raises:
        ValueError: Si n_clusters < 2
        TypeError: Si X no es np.ndarray

    Example:
        >>> labels = np.array([0, 0, 1, 1])
        >>> X = np.random.randn(4, 2)
        >>> metrics = calculate_metrics(labels, X)
        >>> print(metrics['silhouette'])
        0.75
    """
    # Implementación
    pass
```

### Comentarios

```python
# ✓ Bien - Explica el por qué
# Usar StandardScaler porque las variables tienen escalas diferentes
scaler = StandardScaler()

# ✗ Mal - Repite el código
# Crear StandardScaler
scaler = StandardScaler()
```

---

## 🧪 Testing

### Estructura de Tests

```
tests/
├── __init__.py
├── test_clustering.py
├── test_semi_supervised.py
└── test_etl.py
```

### Ejemplo de Test

```python
import pytest
import numpy as np
from src.models.clustering import ClusteringAnalyzer

def test_kmeans_clustering():
    """Test que K-Means produce el número correcto de clusters."""
    # Arrange
    X = np.random.randn(100, 5)
    analyzer = ClusteringAnalyzer(X)
    analyzer.preprocess_data()

    # Act
    labels, metrics = analyzer.kmeans_clustering(n_clusters=3)

    # Assert
    assert len(np.unique(labels)) == 3
    assert metrics['n_clusters'] == 3
    assert -1 <= metrics['silhouette'] <= 1

def test_invalid_k():
    """Test que k < 2 raise ValueError."""
    X = np.random.randn(100, 5)
    analyzer = ClusteringAnalyzer(X)
    analyzer.preprocess_data()

    with pytest.raises(ValueError):
        analyzer.kmeans_clustering(n_clusters=1)
```

### Ejecutar Tests

```bash
# Todos los tests
pytest

# Tests específicos
pytest tests/test_clustering.py

# Con cobertura
pytest --cov=src tests/
```

---

## 📝 Mensajes de Commit

Sigue el formato Conventional Commits:

```
type(scope): subject

body

footer
```

### Tipos

- `feat`: Nueva feature
- `fix`: Corrección de bug
- `docs`: Cambios en documentación
- `style`: Cambios de formato (sin cambios de código)
- `refactor`: Refactorización sin cambios de comportamiento
- `perf`: Mejoras de performance
- `test`: Agregar o actualizar tests
- `ci`: Cambios en CI/CD

### Ejemplos

```
feat(clustering): Add spectral clustering algorithm

- Implement SpectralClustering wrapper
- Add to ClusteringAnalyzer
- Include evaluation metrics

Closes #15
```

```
fix(semi_supervised): Handle edge case with no labeled data

Previously crashed when labeled_ratio=0. Now uses baseline model.
```

---

## 🚀 Guía de Release

1. Actualizar versión en `src/__init__.py`
2. Actualizar CHANGELOG.md
3. Crear Git tag: `git tag -a v1.0.0 -m "Release v1.0.0"`
4. Push a GitHub: `git push origin v1.0.0`
5. Crear Release en GitHub con notas

---

## 📞 Comunicación

- **Issues**: Para reportar bugs o sugerir features
- **Discussions**: Para preguntas generales
- **Pull Requests**: Para enviar cambios

---

## 🎓 Recursos para Aprender

- [Git Workflow Guide](https://guides.github.com/)
- [Python Style Guide](https://pep8.org/)
- [Machine Learning Best Practices](https://scikit-learn.org/)
- [Software Testing](https://pytest.org/)

---

## 🙏 Reconocimientos

Tu contribución será reconocida en:
- README.md (Contributors section)
- Commit history
- Release notes

---

## ❓ Preguntas?

- Abre un issue en GitHub
- Revisa documentación existente
- Contacta a los mantenedores

¡Gracias por contribuir! 🎉
