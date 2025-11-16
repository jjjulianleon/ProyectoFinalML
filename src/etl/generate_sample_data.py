"""
Genera datos de ejemplo de cooperativas para pruebas del proyecto.
Simula indicadores financieros realistas.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_sample_cooperativas_data(n_samples: int = 50) -> pd.DataFrame:
    """
    Genera dataset de ejemplo con cooperativas.

    Args:
        n_samples: Número de cooperativas a generar

    Returns:
        DataFrame con datos simulados
    """
    np.random.seed(42)

    # Definir ratings y sus características
    ratings = ['A', 'B', 'C', 'D']
    rating_samples = {
        'A': n_samples // 4,
        'B': n_samples // 4,
        'C': n_samples // 4,
        'D': n_samples // 4
    }

    data = []

    for rating, count in rating_samples.items():
        for i in range(count):
            # Generar datos financieros correlacionados con el rating
            if rating == 'A':
                # Cooperativas sanas
                activos_improductivos = np.random.normal(0.03, 0.01)
                morosidad = np.random.normal(0.02, 0.01)
                roa = np.random.normal(0.025, 0.005)
                roe = np.random.normal(0.15, 0.03)
                gastos_operacion = np.random.normal(0.04, 0.01)
                cartera_depositos = np.random.normal(0.85, 0.05)

            elif rating == 'B':
                activos_improductivos = np.random.normal(0.07, 0.02)
                morosidad = np.random.normal(0.05, 0.02)
                roa = np.random.normal(0.015, 0.005)
                roe = np.random.normal(0.10, 0.03)
                gastos_operacion = np.random.normal(0.06, 0.02)
                cartera_depositos = np.random.normal(0.75, 0.08)

            elif rating == 'C':
                activos_improductivos = np.random.normal(0.12, 0.03)
                morosidad = np.random.normal(0.10, 0.03)
                roa = np.random.normal(0.005, 0.005)
                roe = np.random.normal(0.05, 0.03)
                gastos_operacion = np.random.normal(0.09, 0.02)
                cartera_depositos = np.random.normal(0.65, 0.10)

            else:  # D
                activos_improductivos = np.random.normal(0.18, 0.04)
                morosidad = np.random.normal(0.18, 0.04)
                roa = np.random.normal(-0.01, 0.01)
                roe = np.random.normal(-0.05, 0.03)
                gastos_operacion = np.random.normal(0.12, 0.03)
                cartera_depositos = np.random.normal(0.50, 0.10)

            # Asegurar valores dentro de rangos realistas
            activos_improductivos = np.clip(activos_improductivos, 0, 1)
            morosidad = np.clip(morosidad, 0, 1)
            gastos_operacion = np.clip(gastos_operacion, 0, 1)
            cartera_depositos = np.clip(cartera_depositos, 0, 1)

            data.append({
                'cooperativa': f'Cooperativa_{rating}_{i+1:03d}',
                'rating': rating,
                'activos_improductivos_total': activos_improductivos,
                'activos_productivos_pasivo': 1 - activos_improductivos + np.random.normal(0, 0.05),
                'morosidad_total': morosidad,
                'cobertura_cartera': 1 - morosidad + np.random.normal(0, 0.05),
                'gastos_operacion_activo': gastos_operacion,
                'gastos_personal_activo': gastos_operacion * np.random.uniform(0.3, 0.6),
                'roa': roa,
                'roe': roe,
                'cartera_depositos': cartera_depositos,
                'fondos_disponibles_depositos': np.random.uniform(0.1, 0.4),
                'cartera_improductiva_patrimonio': np.random.uniform(0, 0.5) if rating in ['C', 'D'] else np.random.uniform(0, 0.1),
            })

    df = pd.DataFrame(data)

    # Asegurar valores no negativos
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(lower=0)

    return df


def save_sample_data(output_path: str = "data/processed/cooperativas_sample.csv"):
    """
    Genera y guarda datos de ejemplo.

    Args:
        output_path: Ruta de salida
    """
    df = generate_sample_cooperativas_data(n_samples=50)

    # Crear directorio si no existe
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"✓ Datos de ejemplo guardados en: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"\nPrimeras filas:")
    print(df.head())
    print(f"\nDistribución de ratings:")
    print(df['rating'].value_counts().sort_index())

    return df


if __name__ == "__main__":
    save_sample_data()
