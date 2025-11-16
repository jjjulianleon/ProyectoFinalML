# 📖 Guía Completa de Configuración

## 🔑 Configuración de API Key de OpenAI

### Paso 1: Obtener API Key

1. Ir a [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Crear una cuenta o iniciar sesión
3. Crear nueva API key
4. Copiar la clave (⚠️ Solo se muestra una vez)

### Paso 2: Configurar Variables de Entorno

#### Opción A: En Google Colab

El notebook automáticamente pide la API key cuando se ejecuta. Simplemente copia tu API key cuando se te solicite.

#### Opción B: En máquina local

1. Crear archivo `.env` en la raíz del proyecto:

```bash
cp .env.example .env
```

2. Editar `.env` y agregar tu API key:

```
OPENAI_API_KEY=sk-proj-...
MODEL_NAME=gpt-3.5-turbo
```

⚠️ **SEGURIDAD**: Nunca hacer commit del archivo `.env` con credenciales reales. Ya está en `.gitignore`.

---

## 📥 Obtención de Datos

### Opción 1: Usar Datos de Ejemplo (Recomendado para pruebas)

El notebook incluye generador automático de datos de ejemplo:

```python
from src.etl.generate_sample_data import generate_sample_cooperativas_data

df = generate_sample_cooperativas_data(n_samples=50)
```

✅ **Ventaja**: Funciona sin necesidad de PDFs reales
⏱️ **Tiempo**: Ejecución inmediata

### Opción 2: Descargar Datos Reales de PDFs

#### Paso 1: Agregar URLs de PDFs

Editar `data/raw/urls_pdfs.txt` y agregar URLs:

```
https://www.asis.fin.ec/wp-content/uploads/2020/08/2025-06-Indicadores-Financieros_ago_mkt_2025.pdf
https://www.seps.gob.ec/...
```

#### Paso 2: Ejecutar Descargador de PDFs

```python
from src.etl.pdf_downloader import PDFDownloader
from src.etl.data_extractor import DataExtractor, read_pdf_text

# Descargar PDFs
downloader = PDFDownloader(output_dir="data/raw")
urls = ["https://...", "https://..."]
results = downloader.download_pdfs_batch(urls)

# Extraer texto de PDFs
for pdf_path in results['successful']:
    text = read_pdf_text(pdf_path['path'])
    # Procesar texto...
```

#### Fuentes de Datos Reales

1. **SEPS (Superintendencia de Economía Popular y Solidaria)**
   - URL: https://www.seps.gob.ec
   - Datos: Reportes financieros de cooperativas
   - Frecuencia: Mensual/Trimestral

2. **ASIS (Asociación de Supervisores)**
   - URL: https://www.asis.fin.ec
   - Datos: Indicadores financieros consolidados
   - Ejemplo: https://www.asis.fin.ec/wp-content/uploads/2020/08/2025-06-Indicadores-Financieros_ago_mkt_2025.pdf

3. **Reportes Directos de Cooperativas**
   - Algunos bancos publican reportes financieros
   - Acceso a través de portales corporativos

---

## 🚀 Ejecución del Proyecto

### En Google Colab (Recomendado)

1. Click en badge "Open in Colab" en README.md
2. El notebook clona automáticamente el repositorio
3. Instala dependencias
4. Configura API key cuando se solicite
5. ¡Ejecuta las celdas!

### En Máquina Local

```bash
# 1. Clonar repositorio
git clone https://github.com/jjjulianleon/ProyectoFinalML.git
cd ProyectoFinalML

# 2. Crear ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar .env
cp .env.example .env
# Editar .env y agregar API key

# 5. Ejecutar Jupyter
jupyter notebook notebooks/ProyectoFinal_ML.ipynb
```

---

## 📊 Estructura de Datos Esperada

### Formato de Entrada (CSV)

```
cooperativa,rating,activos_improductivos_total,activos_productivos_pasivo,...
"Coop A",A,0.03,1.02,...
"Coop B",B,0.07,0.95,...
```

### Variables Requeridas

| Variable | Descripción | Rango |
|----------|-------------|-------|
| `cooperativa` | Nombre de la cooperativa | string |
| `rating` | Calificación de riesgo (A, B, C, D) | categorical |
| `activos_improductivos_total` | Ratio activos improductivos | 0-1 |
| `activos_productivos_pasivo` | Ratio activos productivos | 0-2 |
| `morosidad_total` | Tasa de morosidad | 0-1 |
| `cobertura_cartera` | Cobertura de cartera | 0-1 |
| `gastos_operacion_activo` | Ratio gastos operacionales | 0-1 |
| `gastos_personal_activo` | Ratio gastos de personal | 0-1 |
| `roa` | Return on Assets | -0.5-0.5 |
| `roe` | Return on Equity | -1-1 |
| `cartera_depositos` | Ratio cartera/depósitos | 0-2 |
| `fondos_disponibles_depositos` | Ratio fondos disponibles | 0-1 |
| `cartera_improductiva_patrimonio` | Ratio cartera improductiva | 0-1 |

---

## 🔧 Troubleshooting

### Error: "API key not found"

```python
# Verificar que .env contiene:
OPENAI_API_KEY=sk-proj-...

# O en Colab, ingresar la key cuando se solicite
```

### Error: "No module named 'openai'"

```bash
pip install -r requirements.txt
# O instalar manualmente:
pip install openai
```

### Error: "PDF no se puede procesar"

OpenAI API requiere que los PDFs se pasen como base64 o mediante Vision API. Alternativa:

```python
import pdfplumber

# Extraer texto del PDF primero
with pdfplumber.open("documento.pdf") as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text()

# Luego enviar texto a OpenAI
```

### Colab: "Repositorio ya existe"

```bash
# Limpiar y reintentar
rm -rf ProyectoFinalML
git clone https://github.com/jjjulianleon/ProyectoFinalML.git
```

---

## 📈 Mejoras Futuras

- [ ] Integración con Vision API de OpenAI para procesamiento directo de PDFs
- [ ] Base de datos PostgreSQL para almacenar datos procesados
- [ ] API REST para acceder a modelos entrenados
- [ ] Dashboard interactivo con Streamlit
- [ ] Validación temporal de estabilidad de clusters
- [ ] Análisis de sensibilidad de indicadores financieros

---

## 📞 Soporte

Para preguntas específicas:

1. Revisar sección de troubleshooting arriba
2. Consultar documentación en comentarios del código
3. Crear un issue en GitHub
4. Contactar a profesores del curso

---

**Última actualización**: Noviembre 2025
