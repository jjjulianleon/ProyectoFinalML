"""
Pipeline ETL completo para extracción automática de datos de cooperativas.
DESCARGA automática de PDFs + EXTRACCIÓN con OpenAI API.

Uso:
    python src/etl/run_etl_pipeline.py
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
import logging

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from etl.pdf_downloader import PDFDownloader
from etl.data_extractor import DataExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_urls_from_file(filepath: str) -> List[Tuple[str, str, str]]:
    """
    Carga URLs desde archivo de texto.

    Formato esperado por línea:
    URL | Nombre Cooperativa | Rating

    Args:
        filepath: Ruta del archivo con URLs

    Returns:
        Lista de tuplas (url, nombre, rating)
    """
    urls_data = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Saltar comentarios y líneas vacías
            if not line or line.startswith('#'):
                continue

            # Parsear línea
            parts = [p.strip() for p in line.split('|')]

            if len(parts) >= 3:
                url, nombre, rating = parts[0], parts[1], parts[2]
                urls_data.append((url, nombre, rating))
            elif len(parts) == 1:
                # Solo URL, sin nombre ni rating
                urls_data.append((parts[0], "", ""))

    return urls_data


def run_etl_pipeline(
    urls_file: str = "data/cooperativas_urls.txt",
    output_csv: str = "data/processed/cooperativas_data.csv",
    download_dir: str = "data/raw"
):
    """
    Ejecuta el pipeline ETL completo.

    Fases:
    1. Descarga PDFs desde URLs
    2. Extrae texto de PDFs con pdfplumber
    3. Procesa texto con OpenAI API
    4. Guarda datos estructurados en CSV

    Args:
        urls_file: Archivo con URLs de PDFs
        output_csv: Archivo de salida CSV
        download_dir: Directorio para PDFs descargados
    """
    logger.info("="*70)
    logger.info("🚀 INICIANDO PIPELINE ETL - EXTRACCIÓN AUTOMÁTICA DE DATOS")
    logger.info("="*70)

    # FASE 1: Cargar URLs
    logger.info("\n📋 FASE 1: Cargando URLs...")
    urls_data = load_urls_from_file(urls_file)

    if not urls_data:
        logger.error("❌ No se encontraron URLs en el archivo")
        return

    logger.info(f"✓ URLs cargadas: {len(urls_data)}")
    for url, nombre, rating in urls_data:
        logger.info(f"  • {nombre or 'Sin nombre'} (Rating: {rating or 'N/A'})")

    # FASE 2: Descargar PDFs
    logger.info("\n📥 FASE 2: Descargando PDFs...")
    downloader = PDFDownloader(output_dir=download_dir)

    urls_only = [url for url, _, _ in urls_data]
    download_results = downloader.download_pdfs_batch(urls_only)

    if not download_results['successful']:
        logger.error("❌ CRÍTICO: No se descargaron PDFs exitosamente")
        logger.error(f"   PDFs intentados: {len(urls_data)}")
        logger.error(f"   PDFs descargados: 0")
        if download_results['failed']:
            logger.error(f"   PDFs que fallaron: {len(download_results['failed'])}")
            for failed in download_results['failed']:
                logger.error(f"      • {failed['url']}")
                logger.error(f"        Error: {failed.get('error', 'Unknown')}")
        raise ValueError("No se pudieron descargar PDFs - Extracción de datos reales falló")

    # Mostrar resumen de descargas
    logger.info(f"✓ PDFs descargados exitosamente: {len(download_results['successful'])}")
    if download_results['failed']:
        logger.warning(f"⚠️  PDFs que fallaron: {len(download_results['failed'])}")
        for failed in download_results['failed']:
            logger.warning(f"    • {failed['url']}")

    # FASE 3: Extraer datos con OpenAI API
    logger.info("\n🤖 FASE 3: Extrayendo datos con OpenAI API...")

    extractor = DataExtractor()

    # Preparar lista de PDFs para procesar
    # Mapear PDFs descargados con su metadata (nombre, rating)
    pdf_paths_to_process = []

    for result in download_results['successful']:
        pdf_url = result['url']
        pdf_path = result['path']

        # Buscar metadata correspondiente a esta URL
        metadata = next((item for item in urls_data if item[0] == pdf_url), None)

        if metadata:
            _, nombre, rating = metadata
            pdf_paths_to_process.append((pdf_path, nombre, rating))
        else:
            # Si no se encuentra metadata, usar valores por defecto
            pdf_paths_to_process.append((pdf_path, "", ""))

    # Procesar todos los PDFs
    if not pdf_paths_to_process:
        logger.error("❌ CRÍTICO: No hay PDFs para procesar")
        logger.error(f"   PDFs descargados: {len(download_results['successful'])}")
        logger.error(f"   PDFs listos: {len(pdf_paths_to_process)}")
        raise ValueError("No hay PDFs para procesar - Algo salió mal en mapeo de metadata")

    logger.info(f"  Procesando {len(pdf_paths_to_process)} PDFs con OpenAI...")
    df = extractor.process_batch(pdf_paths_to_process)

    if df is None or df.empty:
        logger.error("❌ CRÍTICO: No se extrajeron datos de los PDFs")
        logger.error(f"   PDFs procesados: {len(pdf_paths_to_process)}")
        logger.error(f"   Filas extraídas: 0")
        logger.error("   Posibles causas:")
        logger.error("      • PDFs contienen solo imágenes (requiere OCR)")
        logger.error("      • Error en extracción con OpenAI API")
        logger.error("      • PDFs están dañados o vacíos")
        logger.error("      • Límite de API alcanzado")
        raise ValueError("Extracción de datos falló - No se extrajo información de los PDFs")

    # FASE 4: Guardar datos
    logger.info("\n💾 FASE 4: Guardando datos...")
    df_saved = extractor.save_extracted_data(output_csv)

    # Resumen final
    logger.info("\n" + "="*70)
    logger.info("✅ PIPELINE ETL COMPLETADO EXITOSAMENTE")
    logger.info("="*70)
    logger.info(f"\n📊 RESUMEN:")
    logger.info(f"  • PDFs procesados: {len(df)}")
    logger.info(f"  • Variables extraídas: {len(df.columns)}")
    logger.info(f"  • Archivo guardado: {output_csv}")
    logger.info(f"\n📈 Distribución de Ratings:")
    if 'rating' in df.columns:
        print(df['rating'].value_counts().to_string())
    logger.info("\n" + "="*70)

    return df


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PIPELINE ETL - EXTRACCIÓN AUTOMÁTICA DE DATOS DE COOPERATIVAS")
    print("="*70)

    # Ejecutar pipeline
    df = run_etl_pipeline()

    if df is not None and not df.empty:
        print("\n✅ Datos extraídos exitosamente")
        print(f"\nPrimeras filas:")
        print(df.head().to_string())
    else:
        print("\n❌ Error en el pipeline ETL")
