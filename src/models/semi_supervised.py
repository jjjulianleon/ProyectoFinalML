"""
Módulo de Semi-Supervised Learning para clasificación de cooperativas.
Compara Label Propagation, Self-Training con baseline supervisado.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

from sklearn.semi_supervised import LabelPropagation, SelfTrainingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemiSupervisedLearner:
    """Análisis de semi-supervised learning con ratio variable de labels."""

    def __init__(self, data: pd.DataFrame, target_column: str, random_state: int = 42):
        """
        Inicializa el learner semi-supervisado.

        Args:
            data: DataFrame con variables
            target_column: Nombre de la columna objetivo (ratings)
            random_state: Para reproducibilidad
        """
        self.data = data
        self.target_column = target_column
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.data_scaled = None
        self.y_encoded = None
        self.results = []

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesa datos.

        Returns:
            Tupla (X escalado, y codificado)
        """
        logger.info("Preprocesando datos...")

        # Remover NaN
        data_clean = self.data.dropna()

        # Variables independientes
        X = data_clean.drop(columns=[self.target_column])
        y = data_clean[self.target_column]

        # Escalar X
        X_scaled = self.scaler.fit_transform(X)

        # Codificar y
        y_encoded = self.label_encoder.fit_transform(y)

        self.data_scaled = X_scaled
        self.y_encoded = y_encoded
        self.X_features = X

        logger.info(f"✓ Datos procesados: {X_scaled.shape}")
        logger.info(f"  Clases: {self.label_encoder.classes_}")

        return X_scaled, y_encoded

    def supervised_baseline(self) -> Dict:
        """
        Entrena baseline completamente supervisado.

        Returns:
            Diccionario con resultados
        """
        logger.info("\n▶ Baseline Supervisado (100% labeled)")

        clf = LogisticRegression(max_iter=1000, random_state=self.random_state)
        clf.fit(self.data_scaled, self.y_encoded)

        y_pred = clf.predict(self.data_scaled)

        metrics = {
            'method': 'Supervised Baseline',
            'labeled_ratio': 1.0,
            'accuracy': accuracy_score(self.y_encoded, y_pred),
            'precision': precision_score(self.y_encoded, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(self.y_encoded, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(self.y_encoded, y_pred, average='weighted', zero_division=0)
        }

        logger.info(f"  Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}")

        self.baseline_metrics = metrics
        return metrics

    def label_propagation(self, labeled_ratio: float = 0.2) -> Dict:
        """
        Aplica Label Propagation semi-supervisado.

        Args:
            labeled_ratio: Proporción de datos etiquetados (0-1)

        Returns:
            Diccionario con resultados
        """
        logger.info(f"\n▶ Label Propagation (labeled={labeled_ratio:.0%})")

        y_semi = self._create_semi_supervised_labels(labeled_ratio)

        clf = LabelPropagation(kernel='rbf', gamma=20)
        clf.fit(self.data_scaled, y_semi)

        y_pred = clf.predict(self.data_scaled)

        metrics = {
            'method': 'Label Propagation',
            'labeled_ratio': labeled_ratio,
            'accuracy': accuracy_score(self.y_encoded, y_pred),
            'precision': precision_score(self.y_encoded, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(self.y_encoded, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(self.y_encoded, y_pred, average='weighted', zero_division=0)
        }

        logger.info(f"  Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}")
        self.results.append(metrics)

        return metrics

    def self_training(self, labeled_ratio: float = 0.2) -> Dict:
        """
        Aplica Self-Training semi-supervisado.

        Args:
            labeled_ratio: Proporción de datos etiquetados (0-1)

        Returns:
            Diccionario con resultados
        """
        logger.info(f"\n▶ Self-Training (labeled={labeled_ratio:.0%})")

        y_semi = self._create_semi_supervised_labels(labeled_ratio)

        # Base estimator
        base_clf = DecisionTreeClassifier(max_depth=10, random_state=self.random_state)

        # Self-training
        clf = SelfTrainingClassifier(
            base_estimator=base_clf,
            threshold=0.75,
            max_iter=10
        )
        clf.fit(self.data_scaled, y_semi)

        y_pred = clf.predict(self.data_scaled)

        metrics = {
            'method': 'Self-Training',
            'labeled_ratio': labeled_ratio,
            'accuracy': accuracy_score(self.y_encoded, y_pred),
            'precision': precision_score(self.y_encoded, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(self.y_encoded, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(self.y_encoded, y_pred, average='weighted', zero_division=0)
        }

        logger.info(f"  Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}")
        self.results.append(metrics)

        return metrics

    def compare_ratios(self, ratios: list = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]) -> pd.DataFrame:
        """
        Compara rendimiento variando el ratio de datos etiquetados.

        Args:
            ratios: Lista de ratios a evaluar

        Returns:
            DataFrame con resultados
        """
        logger.info(f"\n📊 Comparando ratios de labeled data: {ratios}")

        all_results = [self.baseline_metrics] if hasattr(self, 'baseline_metrics') else []

        for ratio in ratios:
            if ratio < 1.0:
                # Label Propagation
                lp_results = self.label_propagation(ratio)
                all_results.append(lp_results)

                # Self-Training
                st_results = self.self_training(ratio)
                all_results.append(st_results)

        return pd.DataFrame(all_results)

    def _create_semi_supervised_labels(self, labeled_ratio: float) -> np.ndarray:
        """
        Crea array de labels semi-supervisado.

        Args:
            labeled_ratio: Proporción de datos etiquetados

        Returns:
            Array con labels (-1 para no etiquetado)
        """
        n_samples = len(self.y_encoded)
        n_labeled = int(n_samples * labeled_ratio)

        # Índices aleatorios para labels
        labeled_indices = np.random.choice(
            n_samples,
            size=n_labeled,
            replace=False
        )

        # Crear array de labels (-1 = no etiquetado)
        y_semi = np.full(n_samples, -1, dtype=int)
        y_semi[labeled_indices] = self.y_encoded[labeled_indices]

        return y_semi

    def get_results_summary(self) -> pd.DataFrame:
        """
        Obtiene resumen de todos los resultados.

        Returns:
            DataFrame con resumen
        """
        if not self.results:
            logger.warning("No hay resultados para mostrar")
            return pd.DataFrame()

        return pd.DataFrame(self.results)


if __name__ == "__main__":
    print("Módulo de semi-supervised learning configurado.")
