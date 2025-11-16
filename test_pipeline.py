"""
Script de prueba del pipeline ETL
"""

import sys
import os
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("="*70)
print("TEST DEL PIPELINE ETL")
print("="*70)

# Test 1: Verificar variables de entorno
print("\n[1] Verificando variables de entorno...")
from dotenv import load_dotenv
load_dotenv(override=True)

api_key = os.getenv('OPENAI_API_KEY')
model = os.getenv('MODEL_NAME', 'gpt-3.5-turbo')

print(f"  API Key: {'CONFIGURADA' if api_key else 'NO CONFIGURADA'}")
print(f"  Modelo: {model}")

if not api_key:
    print("\nERROR: Configura OPENAI_API_KEY en .env")
    sys.exit(1)

# Test 2: Verificar imports
print("\n[2] Verificando imports...")
try:
    from etl.pdf_downloader import PDFDownloader
    from etl.data_extractor import DataExtractor
    print("  OK: Modulos importados correctamente")
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

# Test 3: Inicializar DataExtractor
print("\n[3] Inicializando DataExtractor...")
try:
    extractor = DataExtractor()
    print(f"  OK: Extractor inicializado")
    print(f"  Modelo: {extractor.model}")
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

# Test 4: Verificar archivo de URLs
print("\n[4] Verificando archivo de URLs...")
urls_file = "data/cooperativas_urls.txt"
if Path(urls_file).exists():
    with open(urls_file, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    print(f"  OK: Archivo encontrado con {len(lines)} URLs")
else:
    print(f"  ADVERTENCIA: Archivo {urls_file} no encontrado")

print("\n" + "="*70)
print("RESULTADO: Sistema configurado correctamente")
print("="*70)
print("\nPara ejecutar el pipeline completo:")
print("  python src/etl/run_etl_pipeline.py")
