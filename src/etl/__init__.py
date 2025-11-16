"""
Módulo ETL para procesamiento de datos.
"""

from .pdf_downloader import PDFDownloader, load_urls_from_file
from .data_extractor import DataExtractor

__all__ = ['PDFDownloader', 'DataExtractor', 'load_urls_from_file']
