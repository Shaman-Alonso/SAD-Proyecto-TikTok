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
git clone https://github.com/Shaman-Alonso/SAD-Proyecto-TikTok
cd SAD-Proyecto-TikTok
conda create -n clasificador python=3.12
conda activate clasificador

pip install -r requirements.txt
```

### PyCharm

1. Clonar el repositorio
2. Crear un entorno virtual con PyCharm
3. Instalar las dependencias con pip
```bash
git clone https://github.com/Shaman-Alonso/SAD-Proyecto-TikTok
cd SAD-Proyecto-TikTok

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```


## Estructura del Proyecto

- Main.py &rarr; Script principal. Orquesta la ejecución de los diferentes módulos según los parametros indicados
- Preprocesador.py &rarr; Modulo de limpieza y preparación de datos. Se encarga del filtrado, tratamiento de nulos, mapeo de etiquetas y de la partición segura del dataset
- clasificador.py &rarr; Modulo de ML tradicional. Contiene el entrenamiento, balanceo y evaluación de los algoritmos clasicos.
- requirements.txt &rarr; Dependencias
- output/ &rarr; Resultados generados
- generativa.py &rarr; Pipeline de IA Generativa basado en LangChain y Ollama. Ejecuta la clasificacion de sentimientos, el aumento de datos sinteticos con filtro de duplicados y la evaluación.
- Clustering.py &rarr; Módulo de aprendizaje no supervisado. Aplica algoritmos de agrupamiento y modelado de tópicos para descubrir que temas especificos hablan los usuarios de las reseñas positivas y negativas
- Plotter.py &rarr; Script de visualización de datos.
- clasificador.json &rarr; Configuración del modelo
- generativa.json &rarr; Configuración del LLM

## Ayuda

### Entrenamiento

```bash
python Main.py --help
=== Clasificador ===
usage: Main.py [-h] -f FILE -m MODE -p PREDICTION [-c CPU] [-v] [--debug] [--config CONFIG]

Practica de algoritmos de clasificación de datos.

options:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  Fichero csv (/Path_to_file)
  -m MODE, --mode MODE  Modo: Train, Test o Cluster
  -p PREDICTION, --prediction PREDICTION
                        Columna a predecir (Nombre de la columna)
  -c CPU, --cpu CPU     Número de CPUs a utilizar [-1 para usar todos]
  -v, --verbose         Muestra las metricas por la terminal
  --debug               Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]
  --config CONFIG       Archivo JSON de configuración
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

### Generativa (Clasificación / Aumento de datos / Evaluación)
```bash
python generativa.py --help
usage: generativa.py [-h] [--config CONFIG] [--sample SAMPLE] [--shots SHOTS] [--data DATA] [--mode MODE] [--test_data TEST_DATA]

Clasificacion Ollama

options:
  -h, --help            show this help message and exit
  --config CONFIG       Ruta al archivo JSON con los parametros del modelo
  --sample SAMPLE       Numero de filas a evaluar
  --shots SHOTS         Numero de ejemplos
  --data DATA           Ruta al archivo CSV
  --mode MODE           Modos: classify, data_augmentation o evaluar
  --test_data TEST_DATA
                        Ruta al archivo CSV de test
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
### Clasificación generativa
```bash
python generativa.py --config generativa.json --sample -1 --shots 0 --data dataset.csv --mode classify
```
La clasificacion generativa genera dos archivos en output/:
1. "reporte_generativo.csv" donde se guarda el nombre del modelo, parametros, prompts y resultados
2. "logs_[modelo]_[parametros]" donde se guardan todas las entradas, salidas, el valor real y la respuesta cruda
### Aumento de datos generativa
```bash
python generativa.py --config generativa.json --sample -1 --shots 0 --data dataset.csv --mode data_augmentation
```
El aumento de datos generativo genera un archivo augmented.csv con las nuevas instancias sinteticas siguiendo la misma estructura del .csv de entrada

### Evaluación generativa
```bash
python generativa.py --config generativa.json --sample -1 --shots 3 --data dataset.csv --mode data_augmentation --test_data datos_evaluar.csv
```
La evaluación generativa genera un archivo test_predictions_[modelo].csv con la estructura de las instancias evaluadas y la prediccion realizada.

## clasificador (JSON)

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
## generativa (JSON)

```json
{
  "model": {
    "model": "qwen2.5:7b", //nombre del modelo que queremos ejecutar
    "temperature": 0.6, //Temperatura
    "num_predict": 80, //Limite de tokens a generar
    "top_k": 40, //Top_k
    "top_p": 0.85, //Top_p
    "stop": ["\n\n","Review:","User:","New version:"]}, //Lista de secuencias que detienen la generación

  "settings": {
    "prompt_zeroshot": "Classify the following review about TIKTOK using only one of the following words [positive, neutral, negative], remember to use ONLY ONE WORD\n    Text:{texto_nuevo}\n    Classification:",
    "prompt_fewshot_prefix":"You are an expert sentiment analyzer. Classify the following app reviews using EXACTLY ONE WORD: positive, neutral, or negative. Do not use emojis, conversational text, or punctuation. Look at these examples:\n",
    "prompt_fewshot_suffix": "\nReview: {texto_nuevo}\nClassification:",
    "prompt_augmentation": "Rewrite the following {etiqueta} app review in a different way, keeping the same meaning. Respond ONLY with the new text.\nReview: {texto}\nNew version:",
    "prompt_augmentation_few_shot_prefix": "You are an expert copywriter. Rewrite the given app reviews keeping their original sentiment and meaning intact, but using different words. Respond ONLY with the new rewritten text. Here are some examples:\n\n",
    "aug_fewshot_list": [//Lista de ejemplos usados en los shots
      {
        "original": "The app crashes every time I open it, completely useless.",
        "paraphrase": "I can't even get past the loading screen before it closes. Terrible experience."
      },
      {
        "original": "It's okay, nothing special but it gets the job done.",
        "paraphrase": "An average application that covers the basics without standing out."
      },
      {
        "original": "Absolutely love it! Best social media app out there.",
        "paraphrase": "I am obsessed with this! It is easily the greatest platform available right now."
      }
    ],
    "prompt_augmentation_few_shot_suffix": "Now, rewrite this {etiqueta} review.\nReview: {texto}\nNew version:"
  }
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