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

        Args:
            k_range: Rango de k a evaluar

        Returns:
            Diccionario con métricas para cada k
        """
        logger.info("Buscando k óptimo...")

        results = {
            'k': [],
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
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

    def dbscan_clustering(self, eps: float = 0.5, min_samples: int = 5) -> Tuple[np.ndarray, Dict]:
        """
        Aplica DBSCAN clustering.

        Args:
            eps: Radio máximo de los vecinos
            min_samples: Número mínimo de muestras para forma un cluster

        Returns:
            Tupla (labels, métricas)
        """
        logger.info(f"\n▶ DBSCAN (eps={eps}, min_samples={min_samples})")

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(self.data_scaled)

        # DBSCAN puede marcar puntos como ruido (-1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

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
