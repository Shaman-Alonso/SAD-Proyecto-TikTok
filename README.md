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

- Main.py           &rarr; Script principal. Orquesta la ejecución de los diferentes módulos según los parametros indicados
- Preprocesador.py  &rarr; Modulo de limpieza y preparación de datos. Se encarga del filtrado, tratamiento de nulos, mapeo de etiquetas y de la partición segura del dataset
- Clasificador.py   &rarr; Modulo de ML tradicional. Contiene el entrenamiento, balanceo y evaluación de los algoritmos clasicos.
- generativa.py     &rarr; Pipeline de IA Generativa basado en LangChain y Ollama. Ejecuta la clasificacion de sentimientos, el aumento de datos sinteticos con filtro de duplicados y la evaluación.
- Clustering.py     &rarr; Módulo de aprendizaje no supervisado. Aplica algoritmos de agrupamiento y modelado de tópicos para descubrir que temas especificos hablan los usuarios de las reseñas positivas y negativas
- requirements.txt  &rarr; Dependencias
- output/           &rarr; Resultados generados
- clasificador.json &rarr; Configuración del modelo y preproceso
- generativa.json   &rarr; Configuración del LLM

## Ayuda

### Tradicional (Entrenamiento, Evaluar y Clustering)

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

### Clasificación tradicional

#### Entrenamiento
```bash
python Main.py -f dataset.csv -p columna_objetivo -m "Train" -v --debug
```

#### Predicción

```bash
python Main.py -f dataset.csv -p columna_objetivo -m "Test" -v --debug
```
### Clustering (LDA / NMF)
Nota: el algoritmo de clustering (LDA o NMF) y sus parámetros se configuran en `clasificador.json` dentro del apartado `"clustering"`.
```bash
python Main.py -f dataset.csv -p columna_objetivo -m "Cluster" -v --debug
```
### Clasificación generativa
```bash
python generativa.py --config generativa.json --sample -1 --shots 0 --data dataset.csv --mode classify
```
La clasificación generativa genera dos archivos en output/:
1. "reporte_generativo.csv" donde se guarda el nombre del modelo, parametros, prompts y resultados
2. "logs_[modelo]_[parametros]" donde se guardan todas las entradas, salidas, el valor real y la respuesta cruda
### Aumento de datos generativa
```bash
python generativa.py --config generativa.json --sample -1 --shots 0 --data dataset.csv --mode data_augmentation
```
El aumento de datos generativo genera un archivo augmented.csv con las nuevas instancias sintéticas siguiendo la misma estructura del .csv de entrada

### Evaluación generativa
```bash
python generativa.py --config generativa.json --sample -1 --shots 3 --data dataset.csv --mode evaluar --test_data datos_evaluar.csv
```
La evaluación generativa genera un archivo test_predictions_[modelo].csv con la estructura de las instancias evaluadas y la predicción realizada.

## clasificador (JSON)

```json
{
  "preprocessing": {
    "unique_category_threshold": 10,      // Límite de valores distintos para tratar una columna como categórica
    "missing_values": "delete",           // Estrategia para nulos (delete: elimina la fila, impute: rellena)
    "impute_strategy": "mean",            // Imputación (mean, median, most_frequent)
    "scaling": "minMax",                  // Escalado de variables numéricas (minMax: rango 0-1, standard: media 0 y std 1)
    "cat2num": "ordinal",                 // Conversión de categorías a números (ordinal, one-hot)
    "text_process": "tf-idf",             // Vectorización de texto (tf-idf, bow)
    "sampling": "",                       // Balanceo de clases (oversampling, undersampling, auto)
    "drop_features": ["reviewId", "location", "date", "gender"], // Columnas a excluir del entrenamiento
    "lemmatization": "lem"                // Reducción de palabras a su raíz léxica
  },
  
  "knn_config": {                         //*NOTA: si empieza por -1: se genera rango [inicio, fin, paso]
      "k": [-1, 1, 10, 1],                // Número de vecinos a evaluar
      "p": [1, 2],                        // Métrica de distancia (1: Manhattan, 2: Euclídea)
      "weights": ["uniform", "distance"], // Peso de los vecinos según cercanía
      "pares": false                      // Permitir o evitar valores de K pares
  },
  "decision_tree_config": {
      "criterion": ["gini", "entropy"],   // Función para medir la calidad de la división
      "max_depth": [-1, 4, 8, 1],         // Profundidad máxima del árbol*
      "min_samples_split": [2, 5, 10],    // Muestras mínimas para dividir un nodo interno
      "min_samples_leaf": [1, 2, 4],      // Muestras mínimas requeridas en una hoja
      "max_features": ["sqrt", "log2"]    // Número de variables a considerar para la mejor división
  },
  "rf_config": {
      "n_estimators": [50, 100, 150, 200, 500], // Número de árboles en el bosque
      "max_depth": [5, 10, 20],           // Profundidad máxima de los árboles
      "min_samples_split": [2, 5, 10],    // Muestras mínimas para dividir nodos
      "min_samples_leaf": [1, 2, 4],      // Muestras mínimas en hojas
      "bootstrap": [true, false],         // Selección de muestras (con o sin reemplazo)
      "max_features": ["sqrt", "log2"]    // Variables máximas por árbol
  },
  "nb_config": {
      "alpha": [0.001, 0.01, 0.1, 1.0],   // Parámetro de suavizado (Laplace/Lidstone)
      "fit_prior": [true, false]          // Aprender o no las probabilidades a priori de las clases
  },
  "lr_config": {
      "C": [0.01, 0.1, 0.4, 5, 10],       // Inversa de la fuerza de regularización (menor valor = mayor regularización)
      "l1_ratio": [0, 0.5, 1],            // Mezcla ElasticNet (0: L2, 1: L1)
      "solver": ["saga", "lbfgs"],        // Algoritmo de optimización
      "max_iter": [5000, 10000]           // Número máximo de iteraciones para converger
  },

  "test_size": 0.15,                      // Porcentaje de datos para test final (evaluación ciega)
  "dev_size": 0.15,                       // Porcentaje de datos para validación (ajuste de hiperparámetros)
  "algorithm": "logistic_regression",     // Algoritmo clasificador seleccionado
  "estimator": "f1_macro",                // Métrica de optimización principal

  "clustering": {
      "cluster": "lda",                   // Algoritmo de agrupamiento/temática (lda, nmf)
      "textClustering": "content",        // Campo sobre el que se aplica el clustering

      "lda": {
        "num_topics": [5],                // Número de temas/grupos a identificar
        "passes": [10, 20],               // Número de pasadas por el corpus durante el entrenamiento
        "iterations": [100]               // Máximo de iteraciones en cada pasada
      },
      "nmf": {
        "num_topics": [2, 3, 4, 5, 6],    // Número de temas a evaluar en NMF
        "passes": [5, 10, 20],            // Repeticiones del proceso
        "iterations": [50, 100, 200]      // Límite de iteraciones para la convergencia de matrices
      }
  }
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

- **Preprocesamiento Automático** (`Preprocesador.py`)
  - **Missing values**: Gestión de nulos mediante eliminación (`delete`) o imputación (`mean`, `median`, `most_frequent`).
  - **Escalado**: Normalización de datos con `MinMaxScaler`, `StandardScaler`, `MaxAbsScaler` o `Normalizer`.
  - **Texto (NLP)**: Limpieza avanzada con `NLTK` (eliminación de stop-words, puntuación) y técnicas de **Lemmatization** o **Stemming**.
  - **Codificación categórica**: Transformación automática de etiquetas mediante `OrdinalEncoder` o `OneHotEncoder`.
  - **Balanceo de datos**: Soporte para `SMOTE`, `Oversampling` y `Undersampling` a través de `imblearn`.

- **Modelos de Clasificación Soportados** (`Clasificador.py`)
  - **KNN** (K-Nearest Neighbors)
  - **Decision Tree**
  - **Random Forest**
  - **Naive Bayes** (Multinomial)
  - **Logistic Regression** (con soporte para solvers `SAGA` y `LBFGS`)

- **Aprendizaje No Supervisado** (`Clustering.py`)
  - **LDA** (Latent Dirichlet Allocation) para detección de temáticas.
  - **NMF** (Non-negative Matrix Factorization) como alternativa de clustering.
  - Análisis por sentimiento (positivo/negativo).

- **IA Generativa y Aumento** (`generativa.py`)
  - Clasificación **Zero-shot** y **Few-shot** utilizando modelos locales vía **Ollama** (ej. `qwen2.5`, `gemma2`).
  - **Data Augmentation**: Generación de muestras sintéticas mediante parafraseo para robustecer el entrenamiento.

- **Optimización y Evaluación**
  - Búsqueda de hiperparámetros mediante **GridSearchCV**.
  - **Visualización**:
    - Matriz de confusión.
    - Informe detallado de clasificación (`Precision`, `Recall`, `F1-macro/micro`...).
    - Gráficas de métricas de coherencia y rendimiento.