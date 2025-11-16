# Proyecto Final ML: Clustering y Semi-Supervised Learning de Cooperativas del Segmento 1 en Ecuador

## 📚 Descripción del Proyecto

Este proyecto aplica técnicas de **Machine Learning no supervisado y semi-supervisado** para:
1. **Agrupar cooperativas de ahorro y crédito** del Segmento 1 en Ecuador según características financieras
2. **Validar coherencia de clusters** contra calificaciones de riesgo reales
3. **Comparar enfoques supervisados y semi-supervisados**

## 🎯 Objetivos

- Construir dataset consolidado de indicadores financieros
- Implementar 3+ algoritmos de clustering (K-Means, Agglomerative, DBSCAN)
- Evaluar con métricas: Silhouette Score, Davies-Bouldin Index
- Comparar con ratings reales de agencias externas
- Implementar semi-supervised learning con label ratio variable

## 📁 Estructura del Proyecto

```
├── data/
│   ├── raw/              # PDFs descargados originales
│   └── processed/        # Datos limpios y estructurados (.csv, .xlsx)
├── src/
│   ├── etl/             # Scripts de extracción y transformación
│   │   ├── pdf_downloader.py
│   │   └── data_extractor.py
│   └── models/          # Modelos de clustering y semi-supervised
│       ├── clustering.py
│       └── semi_supervised.py
├── notebooks/
│   └── ProyectoFinal_ML.ipynb    # Notebook principal ejecutable
├── figures/             # Gráficos y visualizaciones
├── requirements.txt     # Dependencias
├── .env.example        # Plantilla de variables de entorno
└── README.md           # Este archivo
```

## 🚀 Inicio Rápido

### 1. Clonar repositorio
```bash
git clone https://github.com/jjjulianleon/ProyectoFinalML.git
cd ProyectoFinalML
```

### 2. Configurar ambiente
```bash
# Crear archivo .env
cp .env.example .env

# Editar .env y agregar tu API Key de OpenAI
# OPENAI_API_KEY=sk-proj-...
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar en Google Colab (Recomendado)

**Click aquí para abrir directamente en Colab:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjjulianleon/ProyectoFinalML/blob/main/notebooks/ProyectoFinal_ML.ipynb)

El notebook está optimizado para ejecutarse completamente en Colab con instalación automática de dependencias.

O en Jupyter local:
```bash
jupyter notebook notebooks/ProyectoFinal_ML.ipynb
```

## 📋 Fases del Proyecto

### Fase 1: Obtención de Datos
- Descarga automática de PDFs desde lista de URLs
- Extracción de datos con OpenAI API (sin OCR)
- Limpieza y estructuración

### Fase 2: Análisis Exploratorio (EDA)
- Estadísticas descriptivas
- Detección de valores faltantes
- Análisis de correlaciones
- Reducción dimensional con t-SNE

### Fase 3: Clustering No Supervisado
- **K-Means** (baseline): Elección óptima de k
- **Agglomerative Clustering**: Análisis jerárquico
- **DBSCAN**: Clustering basado en densidad
- Evaluación y comparación con ratings reales

### Fase 4: Semi-Supervised Learning
- Baseline supervisado como referencia
- Label Propagation y Self-Training
- Análisis de ratio labeled/unlabeled
- Comparación de rendimiento

## 🔑 Variables de Entorno Requeridas

```
OPENAI_API_KEY     # Tu API key de OpenAI (necesario para Fase 1)
MODEL_NAME         # gpt-3.5-turbo (desarrollo) o gpt-4 (producción)
```

## 📊 Indicadores Financieros Analizados

- **Calidad de Activos**: Activos improductivos/Total, Activos productivos/Pasivo
- **Morosidad**: Tasa de morosidad total, Cobertura de cartera
- **Eficiencia**: Gastos operacionales/Activo, Gastos personal/Activo promedio
- **Rentabilidad**: ROA, ROE
- **Liquidez**: Cartera/Depósitos, Fondos disponibles/Depósitos corto plazo
- **Vulnerabilidad**: Cartera improductiva/Patrimonio

## 📈 Métricas de Evaluación

### Clustering
- **Silhouette Score**: Mide cohesión y separación
- **Davies-Bouldin Index**: Ratio de similitud intra vs inter-cluster

### Semi-Supervised
- **Accuracy, Precision, Recall, F1-Score**
- **Comparación con baseline supervisado**

## 🤖 Modelos Utilizados

| Fase | Modelo | Propósito |
|------|--------|----------|
| EDA | t-SNE | Visualización de dimensionalidad |
| Clustering | K-Means | Baseline no supervisado |
| Clustering | Agglomerative | Clustering jerárquico |
| Clustering | DBSCAN | Clustering basado en densidad |
| Semi-Supervised | Label Propagation | Propagación de etiquetas |
| Semi-Supervised | Self-Training | Auto-entrenamiento iterativo |

## 📌 Requisitos Especiales

- Python 3.8+
- API Key de OpenAI válida
- Acceso a internet (para descargar PDFs)
- Mínimo 4GB RAM (recomendado 8GB para análisis completo)

## 🔍 Fuentes de Datos

- [Superintendencia de Economía Popular y Solidaria (SEPS)](https://www.seps.gob.ec)
- Reportes financieros institucionales
- Ejemplo: [Indicadores Financieros ASIS](https://www.asis.fin.ec/wp-content/uploads/2020/08/2025-06-Indicadores-Financieros_ago_mkt_2025.pdf)

## 👨‍💻 Autor

Desarrollado como Proyecto Final del curso de Machine Learning

## 📄 Licencia

MIT License

## 📞 Soporte

Para problemas, crear un issue en el repositorio.

---

**Última actualización**: Noviembre 2025
