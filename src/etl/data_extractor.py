"""
Módulo para extraer datos de PDFs usando OpenAI API.
Convierte documentos PDF en tablas estructuradas CSV.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    print("Instala openai: pip install openai")

# Cargar variables de entorno
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataExtractor:
    """Extrae datos de PDFs usando OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Inicializa el extractor.

        Args:
            api_key: API key de OpenAI (si es None, usa variable de entorno)
            model: Modelo a usar (gpt-3.5-turbo para desarrollo, gpt-4 para producción)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API Key no encontrada. Configura OPENAI_API_KEY en .env"
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.extracted_data = []

    def extract_pdf_content(self, pdf_path: str) -> Optional[str]:
        """
        Extrae texto de un PDF usando OpenAI.

        Args:
            pdf_path: Ruta del archivo PDF

        Returns:
            Texto extraído o None si hay error
        """
        try:
            # Leer archivo PDF
            with open(pdf_path, 'rb') as pdf_file:
                pdf_content = pdf_file.read()

            # Crear mensaje para OpenAI
            logger.info(f"Procesando: {Path(pdf_path).name}")

            # Crear prompt para extraer datos estructurados
            extraction_prompt = """
            Extrae todos los indicadores financieros de este documento PDF.

            Estructura la información en formato JSON con los siguientes campos:
            {
                "cooperativa": "nombre de la cooperativa",
                "rating": "A, B, C, etc.",
                "fecha": "fecha del reporte",
                "indicadores": {
                    "activos_improductivos_total": número,
                    "activos_productivos_pasivo": número,
                    "morosidad_total": número,
                    "cobertura_cartera": número,
                    "gastos_operacion_activo": número,
                    "gastos_personal_activo": número,
                    "roa": número,
                    "roe": número,
                    "cartera_depositos": número,
                    "fondos_disponibles_depositos": número,
                    "cartera_improductiva_patrimonio": número
                }
            }

            Si algún dato no está disponible, usa null.
            Retorna SOLO el JSON válido, sin explicaciones adicionales.
            """

            # Llamar a OpenAI (para PDFs grandes, puede ser necesario procesar en chunks)
            # Nota: OpenAI recomienda usar vision API para PDFs, pero para este proyecto
            # usaremos una estrategia alternativa de lectura de PDF primero

            logger.warning(
                "⚠️  Para procesar PDFs completos, necesitas usar File API o Vision API de OpenAI"
            )
            logger.info("Usa pdfplumber para extraer texto primero, luego envía a OpenAI")

            return None

        except Exception as e:
            logger.error(f"Error procesando {pdf_path}: {str(e)}")
            return None

    def extract_from_text(self, text: str, cooperativa_name: str = "") -> Optional[Dict]:
        """
        Extrae datos estructurados de texto usando OpenAI.

        Args:
            text: Texto con indicadores financieros
            cooperativa_name: Nombre de la cooperativa

        Returns:
            Diccionario con datos extraídos
        """
        try:
            prompt = f"""
            Extrae los indicadores financieros del siguiente texto y devuelve un JSON válido.

            Nombre de cooperativa: {cooperativa_name}

            Texto:
            {text[:2000]}  # Limitar a 2000 caracteres para no exceder límite de tokens

            Formato esperado:
            {{
                "cooperativa": "nombre",
                "rating": "A/B/C/etc",
                "activos_improductivos_total": 0.0,
                "activos_productivos_pasivo": 0.0,
                "morosidad_total": 0.0,
                "cobertura_cartera": 0.0,
                "gastos_operacion_activo": 0.0,
                "gastos_personal_activo": 0.0,
                "roa": 0.0,
                "roe": 0.0,
                "cartera_depositos": 0.0,
                "fondos_disponibles_depositos": 0.0,
                "cartera_improductiva_patrimonio": 0.0
            }}

            Retorna SOLO JSON válido.
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Eres un analista financiero experto. Extrae datos y devuelve JSON válido."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1000
            )

            # Parsear respuesta JSON
            response_text = response.choices[0].message.content
            data = json.loads(response_text)

            return data

        except json.JSONDecodeError:
            logger.error(f"Error parseando JSON para {cooperativa_name}")
            return None
        except Exception as e:
            logger.error(f"Error extrayendo datos: {str(e)}")
            return None

    def save_extracted_data(self, output_path: str = "data/processed/cooperativas_data.csv"):
        """
        Guarda datos extraídos en CSV.

        Args:
            output_path: Ruta del archivo de salida
        """
        if not self.extracted_data:
            logger.warning("No hay datos para guardar")
            return

        df = pd.DataFrame(self.extracted_data)

        # Crear directorio si no existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"✓ Datos guardados en: {output_path}")
        logger.info(f"  Cooperativas: {len(df)}")
        logger.info(f"  Variables: {len(df.columns)}")

        return df


def read_pdf_text(pdf_path: str) -> str:
    """
    Lee texto de un PDF usando pdfplumber.

    Args:
        pdf_path: Ruta del PDF

    Returns:
        Texto extraído
    """
    try:
        import pdfplumber

        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

        return text

    except ImportError:
        logger.error("Instala pdfplumber: pip install pdfplumber")
        return ""
    except Exception as e:
        logger.error(f"Error leyendo PDF: {str(e)}")
        return ""


if __name__ == "__main__":
    # Ejemplo de uso
    extractor = DataExtractor(model="gpt-3.5-turbo")

    # Ejemplo: extraer datos de un archivo de texto
    sample_data = {
        "cooperativa": "Cooperativa de Ejemplo",
        "rating": "B",
        "activos_improductivos_total": 0.05,
        "roa": 0.02
    }

    print("Extractor configurado y listo.")
    print(f"Modelo a usar: {extractor.model}")
