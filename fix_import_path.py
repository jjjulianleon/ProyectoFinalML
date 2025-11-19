import json
import os

nb_path = 'notebooks/ProyectoFinal_ML.ipynb'

print(f"Reading {nb_path}...")
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell with imports (Cell 2 usually, based on view_file it's execution_count 2)
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if "sys.path.insert(0, os.path.join(os.getcwd(), 'src'))" in source_str or "from etl.generate_sample_data" in source_str:
            print("Found target cell.")
            # Replace the source with robust path handling
            new_source = [
                "import os\n",
                "import sys\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from dotenv import load_dotenv\n",
                "\n",
                "# Configuración de Rutas (Robusta para Local y Colab)\n",
                "current_dir = os.getcwd()\n",
                "print(f\"📂 Directorio actual: {current_dir}\")\n",
                "\n",
                "# Determinar la raíz del proyecto\n",
                "if current_dir.endswith('notebooks'):\n",
                "    # Si estamos en notebooks/, subir un nivel\n",
                "    project_root = os.path.abspath(os.path.join(current_dir, '..'))\n",
                "else:\n",
                "    # Asumir que estamos en la raíz o en Colab (donde se clona en root)\n",
                "    project_root = current_dir\n",
                "\n",
                "print(f\"📂 Raíz del proyecto: {project_root}\")\n",
                "\n",
                "# Agregar 'src' al path para poder importar módulos\n",
                "src_path = os.path.join(project_root, 'src')\n",
                "if src_path not in sys.path:\n",
                "    sys.path.insert(0, src_path)\n",
                "    print(f\"✓ Agregado al path: {src_path}\")\n",
                "\n",
                "# Cargar variables de entorno\n",
                "env_path = os.path.join(project_root, '.env')\n",
                "load_dotenv(env_path)\n",
                "\n",
                "# Imports de módulos locales\n",
                "try:\n",
                "    from etl.generate_sample_data import generate_sample_cooperativas_data\n",
                "    from models.clustering import ClusteringAnalyzer\n",
                "    from models.semi_supervised import SemiSupervisedLearner\n",
                "    print(\"✓ Módulos locales importados correctamente\")\n",
                "except ImportError as e:\n",
                "    print(f\"❌ Error importando módulos: {e}\")\n",
                "    print(f\"   sys.path: {sys.path}\")\n",
                "    # Listar contenido de src para debug\n",
                "    if os.path.exists(src_path):\n",
                "        print(f\"   Contenido de {src_path}: {os.listdir(src_path)}\")\n",
                "    else:\n",
                "        print(f\"   ❌ La carpeta {src_path} no existe\")\n",
                "\n",
                "# Configuración de visualización\n",
                "sns.set(style=\"whitegrid\")\n",
                "plt.rcParams['figure.figsize'] = (12, 8)\n",
                "RANDOM_STATE = 42\n"
            ]
            cell['source'] = new_source
            found = True
            break

if found:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook import path fixed successfully.")
else:
    print("Target cell not found.")
