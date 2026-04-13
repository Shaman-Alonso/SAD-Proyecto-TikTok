# -*- coding: utf-8 -*-
"""
Script para la implementación del algoritmo de clasificación

SI HACEMOS LA DISTINCION DE DOS FICHEROS: TRAIN/DEV Y TEST, HAY QUE PREPROCESAR LOS DOS FICHEROS.

nombre: modelo-hyperparam1-hyperparam2...
ej1: knn-p1-wU-k5...        -> 0.85
ej2: knn-p1-wU-k500...      -> 0.848

"""

import random
import sys
import signal
import argparse
import pandas as pd
import numpy as np
import string
import pickle
import time
import json
import csv
import os
from colorama import Fore
# Sklearn
from sklearn.calibration import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
# Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# Imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm

# Funciones auxiliares

def signal_handler(sig, frame):
    """Manejador de Ctrl+C para cierre ordenado"""
    print(f'\n[!] Interrupción detectada (Señal {sig}). Iniciando protocolo de cierre...')
    print('\nFinalizando ejecución de forma ordenada. ¡Adiós!')
    sys.exit(0)

def parse_args():
    """
    Función para parsear los argumentos de entrada
    """
    parse = argparse.ArgumentParser(description="Practica de algoritmos de clasificación de datos.")
    parse.add_argument("-m", "--mode", help="Modo de ejecución (train o test)", required=True)
    parse.add_argument("-f", "--file", help="Fichero csv (/Path_to_file)", required=True)
    parse.add_argument("-a", "--algorithm", help="Algoritmo a ejecutar (kNN, decision_tree o random_forest)", required=True)
    parse.add_argument("-p", "--prediction", help="Columna a predecir (Nombre de la columna)", required=True)
    parse.add_argument("-e", "--estimator", help="Estimador a utilizar para elegir el mejor modelo https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter", required=False, default=None)
    parse.add_argument("-c", "--cpu", help="Número de CPUs a utilizar [-1 para usar todos]", required=False, default=-1, type=int)
    parse.add_argument("-v", "--verbose", help="Muestra las metricas por la terminal", required=False, default=False, action="store_true")
    parse.add_argument("--debug", help="Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]", required=False, default=False, action="store_true")
    # Parseamos los argumentos
    args = parse.parse_args()
    
    # Leemos los parametros del JSON
    with open('config.json') as json_file:
        config = json.load(json_file)
    
    # Juntamos todo en una variable
    for key, value in config.items():
        setattr(args, key, value)
    
    # Parseamos los argumentos
    return args
    
def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    try:
        data = pd.read_csv(file, encoding='utf-8')
        print(Fore.GREEN+"Datos cargados con éxito"+Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED+"Error al cargar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)

# Funciones para calcular métricas

# TODO Aqui poned lo que hayais hecho

def calculate_fscore(y_true, y_pred):
    """
    Calcula el F1-score micro y macro.

    Parámetros:
    - y_true: Etiquetas reales.
    - y_pred: Etiquetas predichas por el modelo.

    Retorna:
    - Tupla con (f1_micro, f1_macro)
    """
    # Calculamos el F1 micro (bueno si las clases están desbalanceadas y quieres ver el global)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

    # Calculamos el F1 macro (bueno para tratar todas las clases por igual, independientemente de su tamaño)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return f1_micro, f1_macro


def calculate_classification_report(y_true, y_pred):
    """
    Genera un informe de texto que muestra las principales métricas de clasificación.
    """
    # Devuelve un string formateado con precision, recall, f1-score y support por cada clase
    return classification_report(y_true, y_pred, zero_division=0)


def calculate_confusion_matrix(y_true, y_pred):
    """
    Calcula la matriz de confusión para evaluar la precisión de una clasificación.
    """
    # Devuelve una matriz (array 2D) con los verdaderos positivos, falsos positivos, etc.
    return confusion_matrix(y_true, y_pred)

# Funciones para preprocesar los datos

def select_features():
    """
    Separa las características del conjunto de datos en características numéricas, de texto y categóricas.

    Returns:
        numerical_feature (DataFrame): DataFrame que contiene las características numéricas.
        text_feature (DataFrame): DataFrame que contiene las características de texto.
        categorical_feature (DataFrame): DataFrame que contiene las características categóricas.
    """
    try:
        # Numerical features
        numerical_feature = data.select_dtypes(include=['int64', 'float64']) # Columnas numéricas
        if args.prediction in numerical_feature.columns:
            numerical_feature = numerical_feature.drop(columns=[args.prediction])
        # Categorical features
        categorical_feature = data.select_dtypes(include='object')
        categorical_feature = categorical_feature.loc[:, categorical_feature.nunique() <= args.preprocessing["unique_category_threshold"]]
        
        # Text features
        text_feature = data.select_dtypes(include='object').drop(columns=categorical_feature.columns)

        print(Fore.GREEN+"Datos separados con éxito"+Fore.RESET)
        
        if args.debug:
            print(Fore.MAGENTA+"> Columnas numéricas:\n"+Fore.RESET, numerical_feature.columns)
            print(Fore.MAGENTA+"> Columnas de texto:\n"+Fore.RESET, text_feature.columns)
            print(Fore.MAGENTA+"> Columnas categóricas:\n"+Fore.RESET, categorical_feature.columns)
        return numerical_feature, text_feature, categorical_feature
    except Exception as e:
        print(Fore.RED+"Error al separar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)

def process_missing_values(numerical_feature, categorical_feature):
    """
    Procesa los valores faltantes en los datos según la estrategia especificada en los argumentos. Procesa los valores faltantes en los datos según la estrategia especificada en los argumentos.

    Args:
        numerical_feature (DataFrame): El DataFrame que contiene las características numéricas.
        categorical_feature (DataFrame): El DataFrame que contiene las características categóricas.

    Returns:
        None

    Raises:
        None
    """

    """
    
    """
    #TODO aqui lo que hayais hecho
    global data
    try:
        # 1. Miramos si el JSON dice que hay que imputar o borrar
        mode = args.preprocessing["missing_values"]  # "impute" o "drop"

        if mode == "drop":
            data.dropna(inplace=True)
            print(Fore.GREEN + "Filas con nulos eliminadas" + Fore.RESET)

        elif mode == "impute":
            strategy = args.preprocessing["impute_strategy"]  # "mean", "median", etc.

            # Imputar numéricos según la estrategia del JSON
            for col in numerical_feature.columns:
                if strategy == "mean":
                    data[col] = data[col].fillna(data[col].mean())
                elif strategy == "median":
                    data[col] = data[col].fillna(data[col].median())

            # Imputar categóricos (siempre con la moda, es lo más lógico)
            for col in categorical_feature.columns:
                if not data[col].mode().empty:
                    data[col] = data[col].fillna(data[col].mode()[0])

            print(Fore.GREEN + f"Valores imputados usando estrategia: {strategy}" + Fore.RESET)

    except Exception as e:
        print(Fore.RED + "Error en process_missing_values" + Fore.RESET);
        print(e);
        sys.exit(1)

def reescaler(numerical_feature):
    """
    Rescala las características numéricas en el conjunto de datos utilizando diferentes métodos de escala.

    Args:
        numerical_feature (DataFrame): El dataframe que contiene las características numéricas.

    Returns:
        None

    Raises:
        Exception: Si hay un error al reescalar los datos.

    """
    #TODO aqui reescalar
    """
    Rescala las características numéricas en el conjunto de datos utilizando diferentes métodos de escala.
    """
    global data
    try:
        # 1. Comprobar si hay columnas numéricas para escalar
        if numerical_feature.columns.size > 0:
            strategy = args.preprocessing["scaling"]
            print(f"{Fore.CYAN}Aplicando reescalado: {strategy}{Fore.RESET}")

            # 2. Seleccionar el escalador según el JSON
            if strategy == "standard":
                scaler = StandardScaler()
            elif strategy == "min-max":
                scaler = MinMaxScaler()
            elif strategy == "max-abs":
                scaler = MaxAbsScaler()
            elif strategy == "normalizer":
                scaler = Normalizer()
            else:
                print(Fore.YELLOW + f"Estrategia de escalado '{strategy}' no reconocida. Saltando..." + Fore.RESET)
                return

            # 3. Ajustar y transformar las columnas numéricas
            # Solo actuamos sobre las columnas que nos han pasado como numéricas
            cols_to_scale = numerical_feature.columns
            data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

            print(Fore.GREEN + f"Datos numéricos reescalados con éxito usando {strategy}" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No hay características numéricas para reescalar" + Fore.RESET)

    except Exception as e:
        print(Fore.RED + "Error al reescalar los datos" + Fore.RESET)
        print(e)
        sys.exit(1)

def cat2num(categorical_feature):
    """
    Convierte las características categóricas en características numéricas utilizando la codificación de etiquetas.

    Parámetros:
    categorical_feature (DataFrame): El DataFrame que contiene las características categóricas a convertir.

    """
#TODO aqui lo que haga falta para pasar de categorial a numerico
    global data
    try:
        # Comprobamos si realmente hay columnas categóricas que transformar
        if categorical_feature.columns.size > 0:
            print(f"{Fore.CYAN}Convirtiendo variables categóricas a numéricas...{Fore.RESET}")

            # Inicializamos el codificador
            le = LabelEncoder()

            # Recorremos cada columna categórica y la transformamos
            for col in categorical_feature.columns:
                # Usamos astype(str) por precaución, por si algún número o nulo se coló como texto
                data[col] = le.fit_transform(data[col].astype(str))

            print(Fore.GREEN + "Variables categóricas convertidas con éxito" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No hay columnas categóricas para convertir" + Fore.RESET)

    except Exception as e:
        print(Fore.RED + "Error al convertir las variables categóricas" + Fore.RESET)
        print(e)
        sys.exit(1)

def simplify_text(text_feature):
    """
    Función que simplifica el texto de una columna dada en un DataFrame. lower,stemmer, tokenizer, stopwords del NLTK....
    
    Parámetros:
    - text_feature: DataFrame - El DataFrame que contiene la columna de texto a simplificar.
    
    Retorna:
    None
    """
    #TODO aqui lo que sea preciso en caso de tener texto
    global data
    try:
        if text_feature.columns.size > 0:
            # Inicializamos herramientas de NLTK
            stop_words = set(stopwords.words('english'))
            stemmer = PorterStemmer()

            for col in text_feature.columns:
                print(f"{Fore.CYAN}Tokenizando columna: {col}{Fore.RESET}")

                def tokenize_logic(text):
                    if pd.isna(text):
                        return []
                    # 1. Pasar a minúsculas y Tokenizar
                    tokens = word_tokenize(str(text).lower())

                    # 2. Filtrar Stop words y signos de puntuación
                    # También filtramos por .isalpha() para limpiar ruido
                    filtered_tokens = [
                        t for t in tokens
                        if t not in stop_words and t not in string.punctuation
                    ]

                    # 3. Stemming (Raíz de la palabra)
                    stemmed_tokens = [stemmer.stem(t) for t in filtered_tokens]

                    return stemmed_tokens

                # Creamos la nueva columna (cuerpoMensajeToken en la imagen)
                data[col] = data[col].apply(tokenize_logic)
                # Opcional: Si quieres que el resto del script (TF-IDF) use este texto limpio,
                # a veces es mejor guardarlo como string unido por espacios:
                # data[new_col_name] = data[new_col_name].apply(lambda x: ' '.join(x))

            print(Fore.GREEN + "Texto simplificado y tokenizado con éxito" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No hay columnas de texto para simplificar" + Fore.RESET)

    except Exception as e:
        print(Fore.RED + "Error en simplify_text" + Fore.RESET)
        print(e)
        sys.exit(1)

def process_text(text_feature):
    """
    Procesa las características de texto utilizando técnicas de vectorización como TF-IDF o BOW.

    Parámetros:
    text_feature (pandas.DataFrame): Un DataFrame que contiene las características de texto a procesar.

    """
    global data
    try:
        if text_feature.columns.size > 0:
            if args.preprocessing["text_process"] == "tf-idf":               
               tfidf_vectorizer = TfidfVectorizer()
               text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
               tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
               text_features_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
               data = pd.concat([data, text_features_df], axis=1)
               data.drop(text_feature.columns, axis=1, inplace=True)
               print(Fore.GREEN+"Texto tratado con éxito usando TF-IDF"+Fore.RESET)
            elif args.preprocessing["text_process"] == "bow":
                bow_vecotirizer = CountVectorizer()
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                bow_matrix = bow_vecotirizer.fit_transform(text_data)
                text_features_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vecotirizer.get_feature_names_out())
                data = pd.concat([data, text_features_df], axis=1)
                print(Fore.GREEN+"Texto tratado con éxito usando BOW"+Fore.RESET)
            else:
                print(Fore.YELLOW+"No se están tratando los textos"+Fore.RESET)
        else:
            print(Fore.YELLOW+"No se han encontrado columnas de texto a procesar"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al tratar el texto"+Fore.RESET)
        print(e)
        sys.exit(1)

def over_under_sampling():
    """
    Realiza oversampling o undersampling en los datos según la estrategia especificada en args.preprocessing["sampling"].
    
    Args:
        None
    
    Returns:
        None
    
    Raises:
        Exception: Si ocurre algún error al realizar el oversampling o undersampling.
    """
    #TODO aqui lo que haya que hacerse
    global data
    try:
        # 1. Miramos qué estrategia dice el JSON
        # Asumimos que en el JSON tienes: args.preprocessing["sampling"] = "over", "under" o "none"
        strategy = args.preprocessing.get("sampling", "none")

        if strategy == "none":
            print(Fore.YELLOW + "No se aplica muestreo (clases tal cual)" + Fore.RESET)
            return

        print(f"{Fore.CYAN}Aplicando técnica de muestreo: {strategy}{Fore.RESET}")

        # 2. Necesitamos separar X e y para que imblearn pueda trabajar
        target_col = args.prediction
        y = data[target_col]
        X = data.drop(columns=[target_col])

        # 3. Aplicar el algoritmo correspondiente
        if strategy == "under":
            sampler = RandomUnderSampler(random_state=42)
        elif strategy == "over":
            sampler = RandomOverSampler(random_state=42)
        else:
            print(Fore.RED + f"Estrategia '{strategy}' no reconocida" + Fore.RESET)
            return

        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # 4. Reconstruimos el DataFrame global 'data' con los nuevos datos equilibrados
        data = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=[target_col])], axis=1)

        print(Fore.GREEN + f"Muestreo completado. Nuevas dimensiones: {data.shape}" + Fore.RESET)

    except Exception as e:
        print(Fore.RED + "Error en over_under_sampling" + Fore.RESET)
        print(e)
        sys.exit(1)

def drop_features():
    """
    Elimina las columnas especificadas del conjunto de datos.

    Parámetros:
    features (list): Lista de nombres de columnas a eliminar.

    """
    global data
    try:
        data = data.drop(columns=args.preprocessing["drop_features"])
        print(Fore.GREEN+"Columnas eliminadas con éxito"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al eliminar columnas"+Fore.RESET)
        print(e)
        sys.exit(1)

def preprocesar_datos():
    """
    Función para preprocesar los datos
        1. Separamos los datos por tipos (Categoriales, numéricos y textos)
        2. Pasar los datos de categoriales a numéricos 
        3. Tratamos missing values (Eliminar y imputar)
        4. Reescalamos los datos datos (MinMax, Normalizer, MaxAbsScaler)
        TODO 5. Simplificamos el texto (Normalizar, eliminar stopwords, stemming y ordenar alfabéticamente)
        6. Tratamos el texto (TF-IDF, BOW)
        7. Realizamos Oversampling o Undersampling
        8. Borrar columnas no necesarias
    :param data: Datos a preprocesar
    :return: Datos preprocesados y divididos en train y test
    """
    # Separamos los datos por tipos
    numerical_feature, text_feature, categorical_feature = select_features()

    # Simplificamos el texto
    simplify_text(text_feature)

    # Pasar los datos a categoriales a numéricos
    cat2num(categorical_feature)

    # Tratamos missing values
    process_missing_values(numerical_feature, categorical_feature)

    # Reescalamos los datos numéricos
    reescaler(numerical_feature)
    
    # Tratamos el texto
    process_text(text_feature)
    
    # Realizamos Oversampling o Undersampling
    over_under_sampling()

    drop_features()

    return data

# Funciones para entrenar un modelo

def divide_data():
    """
    Función que divide los datos en conjuntos de entrenamiento y desarrollo.

    Parámetros:
    - data: DataFrame que contiene los datos.
    - args: Objeto que contiene los argumentos necesarios para la división de datos.

    Retorna:
    - x_train: DataFrame con las características de entrenamiento.
    - x_dev: DataFrame con las características de desarrollo.
    - y_train: Serie con las etiquetas de entrenamiento.
    - y_dev: Serie con las etiquetas de desarrollo.
    """
    # Sacamos la columna a predecir
    global data
    try:
        # 1. Extraemos el nombre de la columna objetivo de los argumentos del terminal (-p)
        target_col = args.prediction

        # Verificamos por seguridad que la columna existe en el dataframe
        if target_col not in data.columns:
            print(Fore.RED + f"Error: La columna objetivo '{target_col}' no existe en los datos." + Fore.RESET)
            sys.exit(1)

        # 2. Separamos la "X" (características) y la "Y" (etiqueta a predecir)
        y = data[target_col]
        x = data.drop(columns=[target_col])

        # 3. Dividimos en Train (80%) y Dev (20%)
        # random_state=42 asegura que si ejecutas el script 10 veces,
        # siempre haga la misma partición aleatoria (ideal para debugear)
        x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.2, random_state=42)
        return x_train, x_dev, y_train, y_dev

    except Exception as e:
        print(Fore.RED + "Error al dividir los datos" + Fore.RESET)
        print(e)
        sys.exit(1)
 
def save_model(gs):
    """
    Guarda el modelo y los resultados de la búsqueda de hiperparámetros en archivos.

    Parámetros:
    - gs: objeto GridSearchCV, el cual contiene el modelo y los resultados de la búsqueda de hiperparámetros.

    Excepciones:
    - Exception: Si ocurre algún error al guardar el modelo.

    """
    try:
        with open('output/modelo.pkl', 'wb') as file:
            pickle.dump(gs, file)
            print(Fore.CYAN+"Modelo guardado con éxito"+Fore.RESET)
        with open('output/modelo.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Params', 'Score'])
            for params, score in zip(gs.cv_results_['params'], gs.cv_results_['mean_test_score']):
                writer.writerow([params, score])
    except Exception as e:
        print(Fore.RED+"Error al guardar el modelo"+Fore.RESET)
        print(e)

def mostrar_resultados(gs, x_dev, y_dev):
    """
    Muestra los resultados del clasificador.

    Parámetros:
    - gs: objeto GridSearchCV, el clasificador con la búsqueda de hiperparámetros.
    - x_dev: array-like, las características del conjunto de desarrollo.
    - y_dev: array-like, las etiquetas del conjunto de desarrollo.

    Imprime en la consola los siguientes resultados:
    - Mejores parámetros encontrados por la búsqueda de hiperparámetros.
    - Mejor puntuación obtenida por el clasificador.
    - F1-score micro del clasificador en el conjunto de desarrollo.
    - F1-score macro del clasificador en el conjunto de desarrollo.
    - Informe de clasificación del clasificador en el conjunto de desarrollo.
    - Matriz de confusión del clasificador en el conjunto de desarrollo.
    """
    if args.verbose:
        print(Fore.MAGENTA+"> Mejores parametros:\n"+Fore.RESET, gs.best_params_)
        print(Fore.MAGENTA+"> Mejor puntuacion:\n"+Fore.RESET, gs.best_score_)
        print(Fore.MAGENTA+"> F1-score micro:\n"+Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[0])
        print(Fore.MAGENTA+"> F1-score macro:\n"+Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[1])
        print(Fore.MAGENTA+"> Informe de clasificación:\n"+Fore.RESET, calculate_classification_report(y_dev, gs.predict(x_dev)))
        print(Fore.MAGENTA+"> Matriz de confusión:\n"+Fore.RESET, calculate_confusion_matrix(y_dev, gs.predict(x_dev)))

def kNN():
    """
    Función para implementar el algoritmo kNN.
    Hace un barrido de hiperparametros para encontrar los parametros optimos

    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()
    
    # Hacemos un barrido de hiperparametros

    with tqdm(total=100, desc='Procesando kNN', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(KNeighborsClassifier(), args.kNN, cv=5, n_jobs=args.cpu, scoring=args.estimator)
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  # Esperamos un tiempo aleatorio
            pbar.update(random.random()*2)  # Actualizamos la barra con un valor aleatorio
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
    execution_time = end_time - start_time
    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs)

def decision_tree():
    """
    Función para implementar el algoritmo de árbol de decisión.

    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()

    # 1. Dividimos los datos en entrenamiento y desarrollo
    # Esto nos devuelve las X (pistas) y las y (respuestas)
    x_train, x_dev, y_train, y_dev = divide_data()

    # 2. Preparamos el cronómetro y la barra de progreso
    start_time = time.time()

    try:
        with tqdm(total=100, desc='Entrenando Decision Tree', unit='iter', leave=True) as pbar:
            # 3. Configuramos el barrido de hiperparámetros (GridSearch)
            # - DecisionTreeClassifier(): El modelo base
            # - args.decision_tree: El diccionario de parámetros que lee del JSON
            # - cv=5: Validación cruzada (entrena 5 veces por cada combinación para estar seguro)
            # - n_jobs=args.cpu: Usa los núcleos del procesador en paralelo
            # - scoring=args.estimator: La métrica que elegiste para decidir cuál es el "mejor"
            gs = GridSearchCV(
                DecisionTreeClassifier(random_state=42),
                args.decision_tree,
                cv=5,
                n_jobs=args.cpu,
                scoring=args.estimator
            )

            # 4. ¡¡¡¡A ENTRENAR!!!!
            gs.fit(x_train, y_train)

            # Actualizamos la barra al terminar
            pbar.update(100)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Tiempo de ejecución: {Fore.MAGENTA}{execution_time:.2f}{Fore.RESET} segundos")

        # 5. Mostramos los resultados (usando tus funciones de métricas)
        mostrar_resultados(gs, x_dev, y_dev)

        # 6. Guardamos el modelo para usarlo luego en modo 'test'
        save_model(gs)

    except Exception as e:
        print(Fore.RED + f"Error al ejecutar el Árbol de Decisión: {e}" + Fore.RESET)
        sys.exit(1)
    
def random_forest():
    """
    Función que entrena un modelo de Random Forest utilizando GridSearchCV para encontrar los mejores hiperparámetros.
    Divide los datos en entrenamiento y desarrollo, realiza la búsqueda de hiperparámetros, guarda el modelo entrenado
    utilizando pickle y muestra los resultados utilizando los datos de desarrollo.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
    
    # 1. Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()

    # 2. Iniciamos el cronómetro
    start_time = time.time()

    try:
        # 3. Configuramos el barrido (GridSearch)
        # args.random_forest debe venir del JSON con parámetros como:
        # "n_estimators" (número de árboles), "max_features", etc.
        with tqdm(total=100, desc='Entrenando Random Forest', unit='iter', leave=True) as pbar:
            gs = GridSearchCV(
                RandomForestClassifier(random_state=42),
                args.random_forest,
                cv=5,
                n_jobs=args.cpu,
                scoring=args.estimator
            )

            # 4. Entrenamos el bosque
            gs.fit(x_train, y_train)
            pbar.update(100)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Tiempo de ejecución: {Fore.MAGENTA}{execution_time:.2f}{Fore.RESET} segundos")

        # 5. Mostramos resultados y guardamos el mejor modelo encontrado
        mostrar_resultados(gs, x_dev, y_dev)
        save_model(gs)

    except Exception as e:
        print(f"{Fore.RED}Error en Random Forest: {e}{Fore.RESET}")
        sys.exit(1)

# Funciones para predecir con un modelo

def load_model():
    """
    Carga el modelo desde el archivo 'output/modelo.pkl' y lo devuelve.

    Returns:
        model: El modelo cargado desde el archivo 'output/modelo.pkl'.

    Raises:
        Exception: Si ocurre un error al cargar el modelo.
    """
    try:
        with open('output/modelo.pkl', 'rb') as file:
            model = pickle.load(file)
            print(Fore.GREEN+"Modelo cargado con éxito"+Fore.RESET)
            return model
    except Exception as e:
        print(Fore.RED+"Error al cargar el modelo"+Fore.RESET)
        print(e)
        sys.exit(1)
        
def predict():
    """
    Realiza una predicción utilizando el modelo entrenado y guarda los resultados en un archivo CSV.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
    global data
    # Predecimos
    prediction = model.predict(data)
    
    # Añadimos la prediccion al dataframe data
    data = pd.concat([data, pd.DataFrame(prediction, columns=[args.prediction])], axis=1)
    
# Función principal

if __name__ == "__main__":
    # Fijamos la semilla
    np.random.seed(42)
    print("=== Clasificador ===")
    # Manejamos la señal SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    # Parseamos los argumentos
    args = parse_args()
    # Si la carpeta output no existe la creamos
    print("\n- Creando carpeta output...")
    try:
        os.makedirs('output')
        print(Fore.GREEN+"Carpeta output creada con éxito"+Fore.RESET)
    except FileExistsError:
        print(Fore.GREEN+"La carpeta output ya existe"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al crear la carpeta output"+Fore.RESET)
        print(e)
        sys.exit(1)
    # Cargamos los datos
    print("\n- Cargando datos...")
    data = load_data(args.file)
    # Descargamos los recursos necesarios de nltk
    print("\n- Descargando diccionarios...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    # Preprocesamos los datos
    print("\n- Preprocesando datos...")
    preprocesar_datos()
    if args.debug:
        try:
            print("\n- Guardando datos preprocesados...")
            data.to_csv('output/data-processed.csv', index=False)
            print(Fore.GREEN+"Datos preprocesados guardados con éxito"+Fore.RESET)
        except Exception as e:
            print(Fore.RED+"Error al guardar los datos preprocesados"+Fore.RESET)
    if args.mode == "train":
        # Ejecutamos el algoritmo seleccionado
        print("\n- Ejecutando algoritmo...")
        if args.algorithm == "kNN":
            try:
                kNN()
                print(Fore.GREEN+"Algoritmo kNN ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "decision_tree":
            try:
                decision_tree()
                print(Fore.GREEN+"Algoritmo árbol de decisión ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "random_forest":
            try:
                random_forest()
                print(Fore.GREEN+"Algoritmo random forest ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        else:
            print(Fore.RED+"Algoritmo no soportado"+Fore.RESET)
            sys.exit(1)
    elif args.mode == "test":
        # Cargamos el modelo
        print("\n- Cargando modelo...")
        model = load_model()
        # Predecimos
        print("\n- Prediciendo...")
        try:
            predict()
            print(Fore.GREEN+"Predicción realizada con éxito"+Fore.RESET)
            # Guardamos el dataframe con la prediccion
            data.to_csv('output/data-prediction.csv', index=False)
            print(Fore.GREEN+"Predicción guardada con éxito"+Fore.RESET)
            sys.exit(0)
        except Exception as e:
            print(e)
            sys.exit(1)
    else:
        print(Fore.RED+"Modo no soportado"+Fore.RESET)
        sys.exit(1)