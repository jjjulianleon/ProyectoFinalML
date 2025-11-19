filename = 'notebooks/ProyectoFinal_ML.ipynb'
with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "numeric_cols =" in line:
        print(f"Found at line {i+1}: {line.strip()}")
