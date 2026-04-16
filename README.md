# Manual de Uso

## Requerimientos

- Python 3.12 o superior
- pip
- conda

## Instalación

### Conda

1. Clonar el repositorio
2. Crear un entorno virtual con conda
3. Instalar las dependencias con pip
```bash
git clone https://github.com/xehelex/SAD.git
cd SAD
conda create -n clasificador python=3.12
conda activate clasificador

pip install -r requirements.txt
```

### PyCharm

1. Clonar el repositorio
2. Crear un entorno virtual con PyCharm
3. Instalar las dependencias con pip
```bash
git clone https://github.com/xehelex/SAD.git
cd SAD

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```


## Estructura del Proyecto

- Main.py &rarr; Entrenamiento del modelo
- ScriptEvaluar.py &rarr; Predicción con modelo entrenado
- clasificador.json &rarr; Configuración del modelo
- requirements.txt &rarr; Dependencias
- output/ &rarr; Resultados generados

## Ayuda

### Entrenamiento

```bash
python Main.py --help
=== Clasificador ===

usage: Main.py [-h] -f FILE -p PREDICTION [-c CPU] [-v] [--debug]

Practica de algoritmos de clasificación de datos.

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  Fichero csv (/Path_to_file)
  -p PREDICTION         Columna a predecir
  -c CPU                Número de CPUs a utilizar [-1 para usar todos]
  -v, --verbose         Muestra métricas por terminal
  --debug               Modo debug
```

### Evaluación / Predicción

```bash
python ScriptEvaluar.py --help
=== Predicción clasificador ===

usage: ScriptEvaluar.py [-h] -f FILE -p PREDICTION [--debug]

optional arguments:
  -h, --help            show this help message and exit
  -f FILE               Fichero csv (/Path_to_file)
  -p PREDICTION         Columna objetivo
  --debug               Modo debug
```

## Uso

### Entrenamiento básico

```bash
python Main.py -f dataset.csv -p columna_objetivo
```

### Entrenamiento avanzado

```bash
python Main.py -f dataset.csv -p columna_objetivo -c 4 -v --debug
```

### Predicción

```bash
python ScriptEvaluar.py -f dataset_test.csv -p columna_objetivo
```

## Configuración (JSON)

```json
{
  "preprocessing": {
    "unique_category_threshold": 10,      //Límite para considerar una columna categórica
    "missing_values": "impute",           //Estrategia para missing values (impute, delete, none)
    "impute_strategy": "most_frequent",   //Modo de imputación, para que aplique, la estrategia ha de ser "impute"(mean, median, most_frequent)
    "scaling": "minMax",                  //Estrategia de escalado (minMax, standard, normalizer, maxAbs)
    "text_process": "tf-idf",             //Estrategia de procesado de texto (tf-idf, bow)
    "sampling": "oversampling",           //Estrategia de balanceo (oversampling, udnersampling, auto)
    "drop_features": []                   //Columnas a eliminar
  },
  "knn_config": {                         //*NOTA: si los array de enteros empiezan por -1: se genera un rango [-1, inicio, fin, paso]
    "k": [-1, 0, 5, 1],                   //Número de vecinos (array de enteros)*
    "p": [1, 2],                          //Distancia (1: Manhattan, 2: Euclídea)
    "weights": ["uniform", "distance"],   //Peso (uniform, distance)
    "pares": false                        //Permitir valores pares en k
  },
  "decision_tree_config": {
    "criterion": ["gini", "entropy"],     //Criterio para las particiones (gini, entropy)
    "max_depth": [-1, 4, 8, 1],           //Profundidad máxima (array de enteros)*
    "min_samples_split": [2, 5, 10],      //Muestras mínimas para dividir (array de enteros)*
    "min_samples_leaf": [1, 2, 4],        //Muestras mínimas por hoja (array de enteros)*
    "max_features": ["sqrt", "log2"]      //Features máximas a considerar (sqrt, log2)
  },
  "rf_config": {
    "n_estimators": [50, 100, 150],       //Número de árboles (array de enteros)*
    "max_depth": [5, 10, null],           //Profundidad máxima (array de enteros)*
    "min_samples_split": [2, 5, 10],      //Muestras mínimas para dividir (array de enteros)*
    "min_samples_leaf": [1, 2, 4],        //Muestras mínimas por hoja (array de enteros)*
    "bootstrap": [true, false],           //Uso de muestreo bootstrap (true, false)
    "max_features": ["sqrt", "log2"]      //Features máximas a considerar (sqrt, log2)
  },
  "test_size": "0.2",                     //Porcentaje para dividir la población
  "algorithm": "knn",                     //Algoritmo clasificador (knn, decision_tree, random_forest)
  "estimator": "f1_micro"                 //Métrica de optimización (f1_micro, f1_macro, f1_weighted, accuracy, recall, precision etc.)
}
```

## Características del Proyecto

- Preprocesamiento automático
  - Missing values
  - Escalado
  - Texto (NLP con NLTK)
  - Codificación categórica
- Soporte para:
  - kNN
  - Decision Tree
  - Random Forest
- Búsqueda de hiperparámetros con GridSearchCV
- Visualización:
  - Matrzi de confusión
  - Informe de clasificación
  - Gráficas de rendimiento