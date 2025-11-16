"""
Ejemplo completo: Ejecutar el pipeline de clustering y semi-supervised learning.

Este script demuestra cómo usar todos los módulos del proyecto.
"""

import sys
import os
from pathlib import Path

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from etl.generate_sample_data import generate_sample_cooperativas_data
from models.clustering import ClusteringAnalyzer
from models.semi_supervised import SemiSupervisedLearner


def main():
    """Ejecutar pipeline completo."""

    print("=" * 70)
    print("PROYECTO FINAL ML - PIPELINE COMPLETO")
    print("=" * 70)

    # ========================================================================
    # PASO 1: CARGAR DATOS
    # ========================================================================
    print("\n[1/5] CARGANDO DATOS...")
    print("-" * 70)

    df = generate_sample_cooperativas_data(n_samples=50)
    print(f"✓ Dataset cargado: {df.shape}")
    print(f"  Variables: {df.columns.tolist()}")
    print(f"  Distribución de ratings:")
    print(df['rating'].value_counts().sort_index())

    # Guardar datos
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    df.to_csv('data/processed/cooperativas_data.csv', index=False)
    print(f"✓ Datos guardados en: data/processed/cooperativas_data.csv")

    # ========================================================================
    # PASO 2: CLUSTERING
    # ========================================================================
    print("\n[2/5] EJECUTANDO CLUSTERING...")
    print("-" * 70)

    # Seleccionar variables numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Inicializar analizador
    analyzer = ClusteringAnalyzer(df[numeric_cols], random_state=42)
    X_scaled = analyzer.preprocess_data()

    # Encontrar k óptimo
    print("\n▶ Evaluando número óptimo de clusters...")
    k_results = analyzer.find_optimal_k(k_range=range(2, 9))

    # Seleccionar k óptimo
    optimal_k = k_results.loc[k_results['silhouette'].idxmax(), 'k'].astype(int)
    print(f"\n✓ k óptimo: {optimal_k}")

    # Aplicar algoritmos
    print("\n▶ Aplicando K-Means...")
    kmeans_labels, kmeans_metrics = analyzer.kmeans_clustering(n_clusters=optimal_k)

    print("\n▶ Aplicando Agglomerative Clustering...")
    agg_labels, agg_metrics = analyzer.agglomerative_clustering(n_clusters=optimal_k)

    print("\n▶ Aplicando DBSCAN...")
    dbscan_labels, dbscan_metrics = analyzer.dbscan_clustering(eps=0.8, min_samples=4)

    # Resumen
    print("\n" + "=" * 70)
    clustering_summary = analyzer.get_summary()
    print("RESUMEN DE CLUSTERING:")
    print(clustering_summary.to_string(index=False))
    print("=" * 70)

    # Guardar resultados
    clustering_summary.to_csv('data/processed/clustering_metrics.csv', index=False)
    print("\n✓ Métricas guardadas en: data/processed/clustering_metrics.csv")

    # ========================================================================
    # PASO 3: COMPARACIÓN CON RATINGS REALES
    # ========================================================================
    print("\n[3/5] COMPARANDO CON RATINGS REALES...")
    print("-" * 70)

    le = LabelEncoder()
    ratings_encoded = le.fit_transform(df['rating'])

    comparison = analyzer.compare_with_ratings(ratings_encoded)

    # ========================================================================
    # PASO 4: SEMI-SUPERVISED LEARNING
    # ========================================================================
    print("\n[4/5] EJECUTANDO SEMI-SUPERVISED LEARNING...")
    print("-" * 70)

    # Incluir rating en el análisis
    df_with_rating = df[numeric_cols + ['rating']].copy()

    semi_learner = SemiSupervisedLearner(
        df_with_rating,
        target_column='rating',
        random_state=42
    )
    X_semi, y_semi = semi_learner.preprocess_data()

    # Baseline supervisado
    print("\n▶ Entrenando baseline supervisado (100% labeled)...")
    baseline = semi_learner.supervised_baseline()
    print(f"  Accuracy: {baseline['accuracy']:.3f}")

    # Comparar con ratios variables
    print("\n▶ Evaluando semi-supervised learning con diferentes ratios...")
    ratios = [0.1, 0.2, 0.3, 0.5, 0.7]
    results_df = semi_learner.compare_ratios(ratios=ratios)

    print("\n" + "=" * 70)
    print("RESUMEN DE SEMI-SUPERVISED LEARNING:")
    print(results_df.to_string(index=False))
    print("=" * 70)

    # Guardar resultados
    results_df.to_csv('data/processed/semi_supervised_results.csv', index=False)
    print("\n✓ Resultados guardados en: data/processed/semi_supervised_results.csv")

    # ========================================================================
    # PASO 5: RESULTADOS FINALES
    # ========================================================================
    print("\n[5/5] GENERANDO REPORTE FINAL...")
    print("-" * 70)

    # Agregar clusters al dataframe original
    df_final = df.copy()
    df_final['cluster_kmeans'] = kmeans_labels
    df_final['cluster_agglomerative'] = agg_labels
    df_final['cluster_dbscan'] = dbscan_labels

    # Guardar
    df_final.to_csv('data/processed/cooperativas_clustered.csv', index=False)
    print("✓ Datos clustered guardados")

    # Análisis final
    print("\n" + "=" * 70)
    print("ANÁLISIS: CLUSTERS K-MEANS vs RATINGS")
    print("=" * 70)

    print("\n📊 Distribución de ratings por cluster:")
    crosstab = pd.crosstab(df_final['rating'], df_final['cluster_kmeans'], margins=True)
    print(crosstab)

    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 70)

    print("\n📁 Archivos generados:")
    print("  • data/processed/cooperativas_data.csv")
    print("  • data/processed/cooperativas_clustered.csv")
    print("  • data/processed/clustering_metrics.csv")
    print("  • data/processed/semi_supervised_results.csv")

    print("\n💡 Resultados principales:")
    print(f"  • Número óptimo de clusters: {optimal_k}")
    print(f"  • Silhouette Score (K-Means): {kmeans_metrics['silhouette']:.3f}")
    print(f"  • Davies-Bouldin Index (K-Means): {kmeans_metrics['davies_bouldin']:.3f}")
    print(f"  • Accuracy baseline supervisado: {baseline['accuracy']:.3f}")

    print("\n📊 Próximos pasos:")
    print("  1. Visualizar gráficos en el notebook de Colab")
    print("  2. Analizar clusters en detalle")
    print("  3. Comparar con expertos en finanzas")
    print("  4. Iterar sobre ajustes de hiperparámetros")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
