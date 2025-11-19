"""
Módulo de clustering para análisis de cooperativas.
Implementa K-Means, Agglomerative Clustering y DBSCAN.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import logging

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringAnalyzer:
    """Análisis completo de clustering para datos financieros."""

    def __init__(self, data: pd.DataFrame, random_state: int = 42):
        """
        Inicializa el analizador de clustering.

        Args:
            data: DataFrame con variables financieras
            random_state: Para reproducibilidad
        """
        self.data = data
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.data_scaled = None
        self.clusters = {}
        self.metrics = {}

    def preprocess_data(self) -> np.ndarray:
        """
        Preprocesa y escala los datos.

        Returns:
            Array escalado
        """
        logger.info("Preprocesando datos...")

        # Remover filas con NaN
        data_clean = self.data.dropna()

        # Escalar datos
        self.data_scaled = self.scaler.fit_transform(data_clean)

        logger.info(f"✓ Datos escalados: {self.data_scaled.shape}")
        return self.data_scaled

    def find_optimal_k(self, k_range: range = range(2, 11)) -> Dict:
        """
        Encuentra el número óptimo de clusters para K-Means.
        Valida k automáticamente para datasets pequeños.

        Args:
            k_range: Rango de k a evaluar

        Returns:
            DataFrame con métricas para cada k
        """
        logger.info("Buscando k óptimo...")

        n_samples = self.data_scaled.shape[0]

        # Validar que k sea válido para el tamaño del dataset
        # Regla: k debe ser < n/3 para clustering significativo
        max_k = max(2, n_samples // 3)

        # Filtrar k_range
        valid_k_range = [k for k in k_range if k <= max_k and k < n_samples]

        if len(valid_k_range) == 0:
            logger.error(f"❌ No hay valores válidos de k para {n_samples} muestras")
            logger.error(f"   Max k válido: {max_k}, muestras: {n_samples}")
            # Retornar k_range mínimo válido
            valid_k_range = list(range(2, min(max_k + 1, 4)))

        if len(valid_k_range) < len(k_range):
            logger.warning(f"⚠️  k_range ajustado para dataset pequeño")
            logger.warning(f"   Evaluando k ∈ {valid_k_range}")

        # Adaptive n_init para datasets pequeños
        if n_samples < 50:
            n_init = 20  # Más iteraciones para estabilidad
        else:
            n_init = 10  # Default

        results = {
            'k': [],
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }

        for k in valid_k_range:
            try:
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=self.random_state,
                    n_init=n_init,
                    init='k-means++'  # Mejor para datos pequeños
                )
                labels = kmeans.fit_predict(self.data_scaled)

                # Calcular métricas
                silhouette = silhouette_score(self.data_scaled, labels)
                db_index = davies_bouldin_score(self.data_scaled, labels)
                ch_index = calinski_harabasz_score(self.data_scaled, labels)

                results['k'].append(k)
                results['silhouette'].append(silhouette)
                results['davies_bouldin'].append(db_index)
                results['calinski_harabasz'].append(ch_index)

                logger.info(
                    f"  k={k}: Silhouette={silhouette:.3f}, DB={db_index:.3f}, CH={ch_index:.1f}"
                )
            except Exception as e:
                logger.warning(f"  ⚠️  k={k} falló: {e}")
                continue

        if not results['k']:
            logger.error("❌ No se pudieron calcular métricas para ningún k")
            raise ValueError("find_optimal_k: no valid k values")

        return pd.DataFrame(results)

    def kmeans_clustering(self, n_clusters: int) -> Tuple[np.ndarray, Dict]:
        """
        Aplica K-Means clustering.

        Args:
            n_clusters: Número de clusters

        Returns:
            Tupla (labels, métricas)
        """
        logger.info(f"\n▶ K-Means (k={n_clusters})")

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        labels = kmeans.fit_predict(self.data_scaled)

        metrics = self._calculate_metrics("KMeans", labels)

        self.clusters['kmeans'] = labels
        self.metrics['kmeans'] = metrics

        return labels, metrics

    def agglomerative_clustering(self, n_clusters: int) -> Tuple[np.ndarray, Dict]:
        """
        Aplica Agglomerative Clustering (jerárquico).

        Args:
            n_clusters: Número de clusters

        Returns:
            Tupla (labels, métricas)
        """
        logger.info(f"\n▶ Agglomerative Clustering (k={n_clusters})")

        agg = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        labels = agg.fit_predict(self.data_scaled)

        metrics = self._calculate_metrics("Agglomerative", labels)

        self.clusters['agglomerative'] = labels
        self.metrics['agglomerative'] = metrics

        return labels, metrics

    def dbscan_clustering(self, eps: float = None, min_samples: int = None) -> Tuple[np.ndarray, Dict]:
        """
        Aplica DBSCAN clustering con parámetros adaptativos para datasets pequeños.

        Args:
            eps: Radio máximo de los vecinos (None = estimar automáticamente)
            min_samples: Número mínimo de muestras (None = estimar automáticamente)

        Returns:
            Tupla (labels, métricas)
        """
        n_samples = self.data_scaled.shape[0]

        # Parámetros adaptativos para datasets pequeños
        if min_samples is None:
            if n_samples < 20:
                min_samples = max(2, n_samples // 10)  # Muy flexible para datos pequeños
                logger.info(f"⚠️  Dataset pequeño: min_samples={min_samples}")
            elif n_samples < 50:
                min_samples = 3  # Moderadamente flexible
            else:
                min_samples = 5  # Default

        if eps is None:
            # Estimar eps del k-distance graph
            from sklearn.neighbors import NearestNeighbors
            try:
                # Usar k=min_samples-1 para estimar eps
                k = min(min_samples, n_samples - 1)
                neighbors = NearestNeighbors(n_neighbors=k)
                neighbors.fit(self.data_scaled)
                distances = neighbors.kneighbors_graph().data
                eps = np.percentile(distances, 50)  # Mediana de distancias
                logger.info(f"  eps estimado: {eps:.3f}")
            except Exception as e:
                logger.warning(f"  No se pudo estimar eps: {e}, usando default=0.5")
                eps = 0.5

        logger.info(f"\n▶ DBSCAN (eps={eps:.3f}, min_samples={min_samples})")

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.data_scaled)

        # DBSCAN puede marcar puntos como ruido (-1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        # Validar resultados
        if n_clusters == 0:
            logger.warning(f"  ⚠️  DBSCAN encontró 0 clusters (todos como ruido)")
            logger.warning(f"      Considerando ajustar eps o min_samples")
        else:
            logger.info(f"  Clusters encontrados: {n_clusters}, Ruido: {n_noise}")

        metrics = self._calculate_metrics("DBSCAN", labels)

        self.clusters['dbscan'] = labels
        self.metrics['dbscan'] = metrics

        return labels, metrics

    def _calculate_metrics(self, algorithm_name: str, labels: np.ndarray) -> Dict:
        """
        Calcula métricas de evaluación del clustering.

        Args:
            algorithm_name: Nombre del algoritmo
            labels: Labels de clustering

        Returns:
            Diccionario con métricas
        """
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        metrics = {
            'algorithm': algorithm_name,
            'n_clusters': n_clusters,
            'silhouette': None,
            'davies_bouldin': None,
            'calinski_harabasz': None
        }

        # Validar que hay al menos 2 clusters
        if n_clusters >= 2 and len(set(labels)) > 1:
            try:
                metrics['silhouette'] = silhouette_score(self.data_scaled, labels)
                metrics['davies_bouldin'] = davies_bouldin_score(self.data_scaled, labels)
                metrics['calinski_harabasz'] = calinski_harabasz_score(self.data_scaled, labels)

                logger.info(
                    f"  Silhouette: {metrics['silhouette']:.3f}, "
                    f"DB: {metrics['davies_bouldin']:.3f}"
                )
            except Exception as e:
                logger.warning(f"  Error calculando métricas: {e}")

        return metrics

    def compare_with_ratings(self, ratings: np.ndarray) -> Dict:
        """
        Compara clusters con ratings reales usando Adjusted Rand Index.

        Args:
            ratings: Array con ratings reales

        Returns:
            Diccionario con comparaciones
        """
        logger.info("\n📊 Comparación con ratings reales:")

        comparisons = {}
        for algo_name, labels in self.clusters.items():
            # DBSCAN puede tener -1 (ruido), los mapearmos a un cluster especial
            labels_clean = np.where(labels == -1, -1, labels)

            ari = adjusted_rand_score(ratings, labels_clean)
            comparisons[algo_name] = {'ari': ari}

            logger.info(f"  {algo_name}: Adjusted Rand Index = {ari:.3f}")

        return comparisons

    def get_summary(self) -> pd.DataFrame:
        """
        Obtiene resumen de métricas de todos los algoritmos.

        Returns:
            DataFrame con resumen
        """
        summary_data = []

        for algo_name, metrics in self.metrics.items():
            summary_data.append(metrics)

        return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Ejemplo de uso
    print("Módulo de clustering configurado.")
