"""Test rápido del sistema"""
import sys
import os
from pathlib import Path

# Config
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from dotenv import load_dotenv
load_dotenv(override=True)

# Test básico
print("="*60)
print("TEST RAPIDO DEL SISTEMA")
print("="*60)

# 1. API Key
api_key = os.getenv('OPENAI_API_KEY')
modelo = os.getenv('MODEL_NAME')
print(f"\n[1] Variables de entorno")
print(f"    API Key: {'OK' if api_key else 'FALTA'}")
print(f"    Modelo: {modelo}")

# 2. DataExtractor
print(f"\n[2] Inicializando DataExtractor...")
try:
    from etl.data_extractor import DataExtractor
    extractor = DataExtractor()
    print(f"    OK - Modelo: {extractor.model}")
except Exception as e:
    print(f"    ERROR: {e}")
    sys.exit(1)

# 3. PDFDownloader
print(f"\n[3] Inicializando PDFDownloader...")
try:
    from etl.pdf_downloader import PDFDownloader
    downloader = PDFDownloader()
    print(f"    OK - Directorio: {downloader.output_dir}")
except Exception as e:
    print(f"    ERROR: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("SISTEMA LISTO PARA USAR")
print("="*60)
print("\nPara ejecutar el pipeline completo:")
print("  python src/etl/run_etl_pipeline.py")
