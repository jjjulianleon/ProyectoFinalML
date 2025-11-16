"""
Módulo para descargar PDFs de indicadores financieros de cooperativas.
Descarga automática desde URLs y manejo de errores.
"""

import os
import requests
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFDownloader:
    """Descargador de PDFs con manejo robusto de errores."""

    def __init__(self, output_dir: str = "data/raw"):
        """
        Inicializa el descargador.

        Args:
            output_dir: Directorio donde guardar PDFs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_pdf(self, url: str, filename: str = None) -> Tuple[bool, str]:
        """
        Descarga un PDF desde una URL.

        Args:
            url: URL del PDF
            filename: Nombre del archivo (si es None, se extrae de la URL)

        Returns:
            Tupla (éxito, ruta del archivo o mensaje de error)
        """
        try:
            # Extraer nombre del archivo si no se proporciona
            if filename is None:
                filename = url.split('/')[-1]
                if not filename.endswith('.pdf'):
                    filename = f"{filename}.pdf"

            file_path = self.output_dir / filename

            # Saltar si el archivo ya existe
            if file_path.exists():
                logger.info(f"✓ Archivo ya existe: {filename}")
                return True, str(file_path)

            # Descargar con timeout
            logger.info(f"Descargando: {filename}")
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()

            # Guardar archivo
            total_size = int(response.headers.get('content-length', 0))
            with open(file_path, 'wb') as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    f.write(response.content)

            logger.info(f"✓ Descargado exitosamente: {filename}")
            return True, str(file_path)

        except requests.RequestException as e:
            logger.error(f"✗ Error descargando {url}: {str(e)}")
            return False, f"Error de conexión: {str(e)}"
        except Exception as e:
            logger.error(f"✗ Error inesperado con {filename}: {str(e)}")
            return False, f"Error inesperado: {str(e)}"

    def download_pdfs_batch(self, urls: List[str]) -> dict:
        """
        Descarga múltiples PDFs.

        Args:
            urls: Lista de URLs para descargar

        Returns:
            Diccionario con resultados de descarga
        """
        results = {
            'successful': [],
            'failed': [],
            'total': len(urls)
        }

        logger.info(f"\nDescargando {len(urls)} PDFs...")

        for url in tqdm(urls, desc="Descarga en progreso"):
            success, result = self.download_pdf(url)
            if success:
                results['successful'].append({'url': url, 'path': result})
            else:
                results['failed'].append({'url': url, 'error': result})

        # Resumen
        logger.info(f"\n{'='*50}")
        logger.info(f"Descarga completada:")
        logger.info(f"  ✓ Exitosos: {len(results['successful'])}/{results['total']}")
        logger.info(f"  ✗ Fallidos: {len(results['failed'])}/{results['total']}")
        logger.info(f"{'='*50}\n")

        return results


def load_urls_from_file(filepath: str) -> List[str]:
    """
    Carga URLs desde un archivo (una por línea).

    Args:
        filepath: Ruta del archivo de URLs

    Returns:
        Lista de URLs
    """
    urls = []
    with open(filepath, 'r') as f:
        for line in f:
            url = line.strip()
            if url and not url.startswith('#'):
                urls.append(url)
    return urls


if __name__ == "__main__":
    # Ejemplo de uso
    downloader = PDFDownloader()

    # URLs de ejemplo (estas serían las URLs reales de los PDFs)
    example_urls = [
        "https://www.asis.fin.ec/wp-content/uploads/2020/08/2025-06-Indicadores-Financieros_ago_mkt_2025.pdf",
    ]

    results = downloader.download_pdfs_batch(example_urls)
    print(f"\nResultados: {results}")
