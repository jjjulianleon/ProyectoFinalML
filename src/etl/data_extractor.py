"""
Módulo para extraer datos de PDFs usando OpenAI API.
Convierte documentos PDF en tablas estructuradas CSV usando LLM.
ACTUALIZADO: Ahora usa pdfplumber + OpenAI API para extracción automática 100%
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from tqdm import tqdm
from dotenv import load_dotenv
import re

try:
    from openai import OpenAI
    import pdfplumber
except ImportError as e:
    print(f"Error de importación: {e}")
    print("Instala: pip install openai pdfplumber")

# Cargar variables de entorno (override para forzar recarga)
load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataExtractor:
    """Extrae datos de PDFs usando pdfplumber + OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, model: str = None):
        """
        Inicializa el extractor con OpenAI API.

        Args:
            api_key: API key de OpenAI (si es None, usa .env)
            model: Modelo a usar (si es None, usa .env)
        """
        # Cargar API key desde .env
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "❌ API Key no encontrada. Configura OPENAI_API_KEY en .env"
            )

        # Cargar modelo desde .env
        self.model = model or os.getenv('MODEL_NAME', 'gpt-3.5-turbo')

        # Inicializar cliente OpenAI
        self.client = OpenAI(api_key=self.api_key)
        self.extracted_data = []

        logger.info(f"✓ DataExtractor inicializado")
        logger.info(f"  Modelo: {self.model}")
        logger.info(f"  API Key: {self.api_key[:20]}...")

    def read_pdf_text(self, pdf_path: str) -> str:
        """
        Lee texto de un PDF usando pdfplumber.

        Args:
            pdf_path: Ruta del archivo PDF

        Returns:
            Texto extraído del PDF
        """
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"  📄 Leyendo PDF: {len(pdf.pages)} páginas")
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"

            logger.info(f"  ✓ Texto extraído: {len(text)} caracteres")
            return text

        except Exception as e:
            logger.error(f"  ❌ Error leyendo PDF: {str(e)}")
            return ""

    def extract_from_pdf(self, pdf_path: str, cooperativa_name: str = "", rating: str = "") -> Optional[Dict]:
        """
        Extrae datos financieros de un PDF usando pdfplumber + OpenAI.

        Args:
            pdf_path: Ruta del archivo PDF
            cooperativa_name: Nombre de la cooperativa (opcional)
            rating: Rating de la cooperativa (opcional)

        Returns:
            Diccionario con datos extraídos
        """
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 Procesando: {Path(pdf_path).name}")
            logger.info(f"{'='*60}")

            # Paso 1: Extraer texto del PDF
            text = self.read_pdf_text(pdf_path)
            if not text:
                logger.error("  ❌ No se pudo extraer texto del PDF")
                return None

            # Paso 2: Enviar a OpenAI para extracción estructurada
            logger.info("  🤖 Enviando a OpenAI API para extracción...")
            data = self.extract_from_text(text, cooperativa_name, rating)

            if data:
                logger.info(f"  ✓ Datos extraídos exitosamente")
                self.extracted_data.append(data)
                return data
            else:
                logger.error(f"  ❌ No se pudieron extraer datos")
                return None

        except Exception as e:
            logger.error(f"  ❌ Error procesando PDF: {str(e)}")
            return None

    def extract_from_text(self, text: str, cooperativa_name: str = "", rating: str = "") -> Optional[Dict]:
        """
        Extrae datos estructurados de texto usando OpenAI API.

        Args:
            text: Texto con indicadores financieros
            cooperativa_name: Nombre de la cooperativa
            rating: Rating de la cooperativa

        Returns:
            Diccionario con datos extraídos
        """
        try:
            # Limitar texto para no exceder límite de tokens (8000 chars ~ 2000 tokens)
            text_truncated = text[:8000]

            # Prompt mejorado para extracción de indicadores financieros
            prompt = f"""
Eres un analista financiero experto. Tu tarea es extraer TODOS los indicadores financieros de un documento PDF de una cooperativa de ahorro y crédito en Ecuador.

**INFORMACIÓN CONOCIDA:**
- Cooperativa: {cooperativa_name if cooperativa_name else "EXTRAER DEL DOCUMENTO"}
- Rating: {rating if rating else "EXTRAER DEL DOCUMENTO"}

**TEXTO DEL DOCUMENTO:**
{text_truncated}

**INSTRUCCIONES:**
1. Extrae los siguientes indicadores financieros (busca nombres similares o equivalentes):
   - Estructura y calidad de activos
   - Morosidad total o cartera en riesgo
   - ROA (Return on Assets) o Rentabilidad sobre activos
   - ROE (Return on Equity) o Rentabilidad sobre patrimonio
   - Eficiencia operacional (gastos operativos/activos)
   - Liquidez
   - Cobertura de cartera problemática
   - Solvencia o suficiencia patrimonial

2. Devuelve un JSON con este formato EXACTO (valores numéricos en decimales, ej: 5% = 0.05):

{{
    "cooperativa": "nombre completo",
    "rating": "A/B/C/BB/BBB/etc",
    "activos_improductivos_total": 0.00,
    "activos_productivos_pasivo": 0.00,
    "morosidad_total": 0.00,
    "cobertura_cartera": 0.00,
    "gastos_operacion_activo": 0.00,
    "gastos_personal_activo": 0.00,
    "roa": 0.00,
    "roe": 0.00,
    "cartera_depositos": 0.00,
    "fondos_disponibles_depositos": 0.00,
    "cartera_improductiva_patrimonio": 0.00
}}

3. Si un indicador no está disponible, usa null
4. Convierte porcentajes a decimales (ej: 15.5% → 0.155)
5. Retorna SOLO el JSON, sin texto adicional

JSON:
"""

            # Llamar a OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un analista financiero experto especializado en cooperativas de ahorro y crédito en Ecuador. Extraes datos con precisión y devuelves JSON válido."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1500
            )

            # Obtener respuesta
            response_text = response.choices[0].message.content.strip()

            # Limpiar respuesta (a veces OpenAI agrega markdown)
            response_text = re.sub(r'^```json\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
            response_text = response_text.strip()

            # Parsear JSON
            data = json.loads(response_text)

            # Validar estructura
            required_fields = ['cooperativa', 'rating']
            for field in required_fields:
                if field not in data:
                    logger.warning(f"  ⚠️  Campo faltante: {field}")

            # Sobrescribir con datos conocidos si están disponibles
            if cooperativa_name:
                data['cooperativa'] = cooperativa_name
            if rating:
                data['rating'] = rating

            logger.info(f"  ✓ OpenAI API: Datos extraídos")
            logger.info(f"    Cooperativa: {data.get('cooperativa', 'N/A')}")
            logger.info(f"    Rating: {data.get('rating', 'N/A')}")

            return data

        except json.JSONDecodeError as e:
            logger.error(f"  ❌ Error parseando JSON: {str(e)}")
            logger.error(f"  Respuesta: {response_text[:200]}...")
            return None
        except Exception as e:
            logger.error(f"  ❌ Error extrayendo datos: {str(e)}")
            return None

    def process_batch(self, pdf_paths: List[Tuple[str, str, str]]) -> pd.DataFrame:
        """
        Procesa múltiples PDFs.

        Args:
            pdf_paths: Lista de tuplas (ruta_pdf, nombre_cooperativa, rating)

        Returns:
            DataFrame con todos los datos extraídos
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 INICIANDO EXTRACCIÓN AUTOMÁTICA DE {len(pdf_paths)} PDFs")
        logger.info(f"{'='*60}\n")

        self.extracted_data = []

        for pdf_path, coop_name, coop_rating in tqdm(pdf_paths, desc="Procesando PDFs"):
            self.extract_from_pdf(pdf_path, coop_name, coop_rating)

        # Convertir a DataFrame
        if self.extracted_data:
            df = pd.DataFrame(self.extracted_data)
            logger.info(f"\n{'='*60}")
            logger.info(f"✅ EXTRACCIÓN COMPLETADA")
            logger.info(f"  Total cooperativas: {len(df)}")
            logger.info(f"  Variables extraídas: {len(df.columns)}")
            logger.info(f"{'='*60}\n")
            return df
        else:
            logger.warning("⚠️  No se extrajeron datos")
            return pd.DataFrame()

    def save_extracted_data(self, output_path: str = "data/processed/cooperativas_data.csv") -> pd.DataFrame:
        """
        Guarda datos extraídos en CSV.

        Args:
            output_path: Ruta del archivo de salida

        Returns:
            DataFrame guardado
        """
        if not self.extracted_data:
            logger.warning("⚠️  No hay datos para guardar")
            return pd.DataFrame()

        df = pd.DataFrame(self.extracted_data)

        # Crear directorio si no existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Guardar CSV
        df.to_csv(output_path, index=False)

        logger.info(f"\n{'='*60}")
        logger.info(f"💾 DATOS GUARDADOS")
        logger.info(f"  Archivo: {output_path}")
        logger.info(f"  Cooperativas: {len(df)}")
        logger.info(f"  Variables: {len(df.columns)}")
        logger.info(f"{'='*60}\n")

        return df


if __name__ == "__main__":
    # Ejemplo de uso
    print("="*60)
    print("PRUEBA DE DATA EXTRACTOR")
    print("="*60)

    extractor = DataExtractor()

    # Probar con un PDF de ejemplo
    print("\n✓ Extractor configurado y listo")
    print(f"  Modelo: {extractor.model}")
    print(f"  API Key configurada: {'✓' if extractor.api_key else '✗'}")
