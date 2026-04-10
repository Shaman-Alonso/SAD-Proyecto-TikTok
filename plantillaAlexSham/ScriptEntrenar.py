# -*- coding: utf-8 -*-
"""
Script para la implementación del algoritmo de clasificación
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

import matplotlib.pyplot as plt
import seaborn as sn
from colorama import Fore
# Sklearn
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.impute import SimpleImputer
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

metricas = {
    'accuracy': 'accuracy',
    'precision': 'precision_macro',
    'recall': 'recall_macro',
    'f1_macro': 'f1_macro',
    'f1_micro': 'f1_micro',
    'f1_weighted': 'f1_weighted'
}

# Funciones auxiliares

def signal_handler(sig, frame):
    """
    Función para manejar la señal SIGINT (Ctrl+C)
    :param sig: Señal
    :param frame: Frame
    """
    print("\nSaliendo del programa...")
    sys.exit(0)

def parse_args():
    """
    Función para parsear los argumentos de entrada
    """
    parse = argparse.ArgumentParser(description="Practica de algoritmos de clasificación de datos.")
    parse.add_argument("-f", "--file", help="Fichero csv (/Path_to_file)", required=True)
    parse.add_argument("-p", "--prediction", help="Columna a predecir (Nombre de la columna)", required=True)
    parse.add_argument("-c", "--cpu", help="Número de CPUs a utilizar [-1 para usar todos]", required=False, default=-1,type=int)
    parse.add_argument("-v", "--verbose", help="Muestra las metricas por la terminal", required=False, default=False,action="store_true")
    parse.add_argument("--debug",help="Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]",required=False, default=False, action="store_true")
    # Parseamos los argumentos
    args = parse.parse_args()

    # Leemos los parametros del JSON
    with open('clasificador.json') as json_file:
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
        print("\n- Cargando datos...")
        data = pd.read_csv(file, encoding='utf-8')
        #Para quitar espacios innecesarios en los features y convertir en NaN los vacíos (útil para debugging y también ver el csv más bonitillo)
        data.columns = data.columns.str.strip()
        data = data.map(lambda x: x.strip() if isinstance(x, str) else x).replace(r'^\s*$', np.nan, regex=True)
        print(Fore.GREEN + "Datos cargados con éxito" + Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED + "Error al cargar los datos" + Fore.RESET)
        print(e)
        sys.exit(1)


# Funciones para calcular métricas

def calculate_fscore(y_dev, y_pred):
    """
       Función para calcular el F-score
       :param y_dev: Valores reales
       :param y_pred: Valores predichos
       :return: F-score (micro), F-score (macro)
       """
    fscore_micro = f1_score(y_dev, y_pred, average='micro')
    fscore_macro = f1_score(y_dev, y_pred, average='macro')
    return fscore_micro, fscore_macro

def calculate_classification_report(y_dev, y_pred):
    """
       Función para calcular el informe de clasificación
       :param y_dev: Valores reales
       :param y_pred: Valores predichos
       :return: Informe de clasificación
       """
    report = classification_report(y_dev, y_pred, zero_division=0)
    return report

def calculate_confusion_matrix(y_dev, y_pred):
    """
        Función para calcular la matriz de confusión
        :param y_dev: Valores reales
        :param y_pred: Valores predichos
        :return: Matriz de confusión
        """
    cm = confusion_matrix(y_dev, y_pred)
    return cm

def plot_metricas(rdo_df, cm, cr):
    #Hemos experimentado con estas librerías para mostrar las métricas y demases de manera más visual

    #Creamos el lienzo (ancho x alto)
    fig = plt.figure(figsize=(12,25))

    #=== Matriz de confusión ===
    #Creamos una caja de dos filas para representar los gráficos (fila x col x índice)
    ax1 = fig.add_subplot(3,1,1) #ax1 es la primera fila, donde irá la matriz
    sn.heatmap(cm, annot=True, cmap="Greens", ax=ax1) #Que aparezcan numeros y colorinchis verdes (annot es para que se vea la frec abs)

    #Renombrar filas y columnas
    ax1.set_xlabel('Predicción', fontsize=12)
    ax1.set_ylabel('Real', fontsize=12)
    ax1.set_title('Matriz de Confusión', fontsize=14, pad=20)

    #=== Métricas ===
    ax_txt = fig.add_subplot(3, 1, 2)
    ax_txt.axis('off')
    ax_txt.text(0.5, 0.5, f"Informe de clasificación:\n\n{cr}", fontsize=10, va='center', ha='center', family='monospace')

    #Cogemos el top 10
    metrica_col = f"mean_test_{args.estimator}"
    top_df = rdo_df.sort_values(by=metrica_col, ascending=False).head(10).copy()

    #=== Gráfico Modelos ===
    ax2 = fig.add_subplot(3,1,3) #Segunda fila, el gráfico
    if args.algorithm=="knn":
        top_df['Params'] = top_df.apply(lambda row: f"K{int(row['param_n_neighbors'])}_P{row['param_p']}_{row['param_weights']}", axis=1)
    elif args.algorithm=="decision_tree":
        top_df['Params'] = top_df.apply(lambda row: f"Criterion{row['param_criterion']}_Depth{row['param_max_depth']}_Split{row['param_min_samples_split']}_Leaf{row['param_min_samples_leaf']}", axis=1)
    elif args.algorithm=="random_forest":
        top_df['Params'] = top_df.apply(lambda row: f"N{row['param_n_estimators']}_Depth{row['param_max_depth']}_Split{row['param_min_samples_split']}_Leaf{row['param_min_samples_leaf']}_Bootstrap{row['param_bootstrap']}", axis=1)
    ax2.plot(top_df['Params'], top_df[f"mean_test_{args.estimator}"],marker='o', linestyle='-', color='b', label=args.estimator.upper())

    #Configuración para que se vea bonito

    ax2.set_xticks(range(len(top_df)))
    ax2.set_xticklabels(top_df['Params'], rotation=90, fontsize=8)#Rota los nombres en el eje X
    ax2.set_title(f"Bonanza modelos según {args.estimator.upper()}", fontsize=14, pad=20)
    ax2.set_xlabel('Combinación', fontsize=12)
    ax2.set_ylabel(f"{args.estimator.upper()}", fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.2)

    plt.savefig("Prueba", dpi=300, bbox_inches='tight')
    plt.show()

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
        numerical_feature = data.select_dtypes(include=['int64', 'float64'])  # Columnas numéricas
        if args.prediction in numerical_feature.columns:
            numerical_feature = numerical_feature.drop(columns=[args.prediction])
        # Categorical features
        categorical_feature = data.select_dtypes(include=['object','string'])
        categorical_feature = categorical_feature.loc[:, categorical_feature.nunique() <= args.preprocessing["unique_category_threshold"]]

        # Text features
        text_feature = data.select_dtypes(include='object').drop(columns=categorical_feature.columns)

        print(Fore.GREEN + "Datos separados con éxito" + Fore.RESET)

        if args.debug:
            print(Fore.MAGENTA + "> Columnas numéricas:\n" + Fore.RESET, numerical_feature.columns)
            print(Fore.MAGENTA + "> Columnas de texto:\n" + Fore.RESET, text_feature.columns)
            print(Fore.MAGENTA + "> Columnas categóricas:\n" + Fore.RESET, categorical_feature.columns)
        return numerical_feature, text_feature, categorical_feature
    except Exception as e:
        print(Fore.RED + "Error al separar los datos" + Fore.RESET)
        print(e)
        sys.exit(1)


def process_missing_values(numerical_feature, categorical_feature):
    """
    Procesa los valores faltantes en los datos según la estrategia especificada en los argumentos.

    Args:
        numerical_feature (DataFrame): El DataFrame que contiene las características numéricas.
        categorical_feature (DataFrame): El DataFrame que contiene las características categóricas.

    Returns:
        None

    Raises:
        None
    """
    global data
    try:
        print("\n- Procesando missing values...")
        if args.debug:
            print(f"{Fore.MAGENTA}> Missing values detectados:\n{data.isna().sum().to_string()}{Fore.RESET}")
        modo = args.preprocessing["missing_values"]
        if modo == "impute":
            modo_impute = args.preprocessing["impute_strategy"]
            if not numerical_feature.empty: #Para evitar errores
                data[numerical_feature.columns] = SimpleImputer(strategy=modo_impute).fit_transform(data[numerical_feature.columns])
            if not categorical_feature.empty: #Siempre imputará el más frecuente
                data[categorical_feature.columns] = SimpleImputer(strategy='most_frequent').fit_transform(data[categorical_feature.columns])
            print(Fore.GREEN + f"Missing values imputados con la estrategia {modo_impute} con éxito" + Fore.RESET)
        elif modo == "delete":
            data = data.dropna(subset=list(numerical_feature.columns)+list(categorical_feature.columns))
            print(Fore.GREEN + "Missing values ELIMINADOS con éxito" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No se están procesando los missing values" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al tratar missing values" + Fore.RESET)
        print(e)
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
    global data
    try:
        print("\n- Reescalando datos...")
        #Filtramos el preprocesado elegido
        if not numerical_feature.empty:
            #Diccionario para evitar repetición de muchos ifs y mejorar la escalabilidad (estoy loco, pero qué bonito queda)
            scaling = {
                "maxAbs": MaxAbsScaler,
                "minMax": MinMaxScaler,
                "normalizer": Normalizer,
                "standard": StandardScaler
            }
            modo = args.preprocessing["scaling"]
            if modo in scaling:
                #Aprovechando las clases de la librería sklearn escala
                scaler = scaling[modo]
                data[numerical_feature.columns] = scaler().fit_transform(data[numerical_feature.columns])
                print(Fore.GREEN + "Datos escalados con éxito" + Fore.RESET)
            else:
                print(Fore.YELLOW + "No se están escalando los datos" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No hay datos a escalar" + Fore.RESET)
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
    global data
    try:
        print("\n- Realizando Label Encoding...")
        if not categorical_feature.empty:
            for col in categorical_feature.columns:
                data[col] = LabelEncoder().fit_transform(data[col])
            print(Fore.GREEN + "Label Encoding realizado con éxito" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No se está realizando Label Encoding" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al realizar Label Encoding" + Fore.RESET)
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
    global data
    try:
        print("\n- Simplificando el texto...")
        if not text_feature.empty:
            stop_words = set(stopwords.words('english'))
            stemmer = PorterStemmer()
            for col in text_feature.columns: #Por si hubiera varias
                processed = []
                for text in tqdm(data[col], desc=f"Procesando la columna {col}:"):
                    tokens = word_tokenize(str(text).lower()) #Tokenizado + minúsculas
                    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and t not in set(string.punctuation)] #Lematizar + stop words + signos de puntuación
                    tokens = sorted(tokens) #Ordenado
                    processed.append(" ".join(tokens)) #Lo junta para que no lo identifique como un array
                data[col] = processed

            print(Fore.GREEN + "Texto simplificado con éxito" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No se está simplificando el texto" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al simplificar el texto" + Fore.RESET)
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
        print("\n- Procesando columnas de texto...")
        if text_feature.columns.size > 0:
            if args.preprocessing["text_process"] == "tf-idf":
                tfidf_vectorizer = TfidfVectorizer()
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
                text_features_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
                data = pd.concat([data, text_features_df], axis=1)
                data.drop(text_feature.columns, axis=1, inplace=True)
                print(Fore.GREEN + "Texto tratado usando TF-IDF con éxito" + Fore.RESET)
            elif args.preprocessing["text_process"] == "bow":
                bow_vectorizer = CountVectorizer()
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                bow_matrix = bow_vectorizer.fit_transform(text_data)
                text_features_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vectorizer.get_feature_names_out())
                data = pd.concat([data, text_features_df], axis=1)
                print(Fore.GREEN + "Texto tratado usando BOW con éxito" + Fore.RESET)
            else:
                print(Fore.YELLOW + "No se están tratando los textos" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No se han encontrado columnas de texto a procesar" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al tratar el texto" + Fore.RESET)
        print(e)
        sys.exit(1)

def over_under_sampling(X_train, y_train):
    """
    Realiza oversampling o undersampling en los datos de entrenamiento según la estrategia especificada en args.preprocessing["sampling"].

    Parámetros:
    - X_train (pd.DataFrame): Matriz de características de entrenamiento.
    - y_train (pd.Series): Vector de etiquetas de entrenamiento.

    Retorna:
    - X_train: DataFrame con las características de entrenamiento.
    - y_train: Serie con las etiquetas de entrenamiento.

    Raises:
        Exception: Si ocurre algún error al realizar el oversampling o undersampling.
    """
    try:
        print("\n\t- Realizando Over/Under sampling...")
        sampling = {
            "undersampling": RandomUnderSampler,
            "oversampling": RandomOverSampler
        }
        modo = args.preprocessing["sampling"]
        if modo in sampling:
            #Realizamos over/undersampling teniendo en cuenta qué queremos predecir, la y
            sampler = sampling[modo]
            X_train_resampled, y_train_resampled = sampler(random_state=42).fit_resample(X_train, y_train)
            print(Fore.GREEN + f"\tSe ha realizado {modo} con éxito" + Fore.RESET)
            return X_train_resampled, y_train_resampled
        elif modo == "auto": #Hemos definido un modo automático
            counts = y_train.value_counts()
            ratio_actual = counts.min() / counts.max() #Calcula el ratio de la clase minoritaria
            if ratio_actual < 0.05: #Si es inferior al 5% rebalancea
                if len(y_train) < 10000: #Si es un dataset "pequeño" (menos de 10000 instancias) oversampling
                    modo_auto = "Oversampling (Dataset pequeño)"
                    sampler = RandomOverSampler
                else: #Si es un dataset "grande" undersampling
                    modo_auto = "Undersampling (Dataset grande)"
                    sampler = RandomUnderSampler
                X_train_resampled, y_train_resampled = sampler(random_state=42).fit_resample(X_train, y_train)
                print(Fore.CYAN + f"\tModo Auto: Se ha realizado {modo_auto} de los datos por ratio del {ratio_actual:.2%}" + Fore.RESET)
                return X_train_resampled, y_train_resampled
            else:
                print(Fore.CYAN + f"\tModo Auto: No se están over_under sampling los datos por ratio del {ratio_actual:.2%}" + Fore.RESET)
                return X_train, y_train
        else:
            print(Fore.YELLOW + "\tNo se están over_under sampling los datos" + Fore.RESET)
            return X_train, y_train
    except Exception as e:
        print(Fore.RED + "\tError al realizar el over_under sampling" + Fore.RESET)
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
        print("\n- Eliminando columnas...")
        data = data.drop(columns=args.preprocessing["drop_features"])
        print(Fore.GREEN + "Columnas eliminadas con éxito" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al eliminar columnas" + Fore.RESET)
        print(e)
        sys.exit(1)

def preprocesar_datos():
    """
    Función para preprocesar los datos
        1. Separamos los datos por tipos (Categoriales, numéricos y textos)
        2. Tratamos missing values (Eliminar y imputar)
        3. Pasar los datos de categoriales a numéricos
        4. Reescalamos los datos datos (MinMax, Normalizer, MaxAbsScaler)
        5. Simplificamos el texto (Normalizar, eliminar stopwords, stemming y ordenar alfabéticamente)
        6. Tratamos el texto (TF-IDF, BOW)
        7. Realizamos Oversampling o Undersampling
        8. Borrar columnas no necesarias
    :param data: Datos a preprocesar
    :return: Datos preprocesados y divididos en train y test
    """
    global data

    # Separamos los datos por tipos
    numerical_feature, text_feature, categorical_feature = select_features()

    # Tratamos missing values
    process_missing_values(numerical_feature, categorical_feature)

    # Pasar los datos a categoriales a numéricos
    cat2num(categorical_feature)

    # Reescalamos los datos numéricos
    reescaler(numerical_feature)

    # Simplificamos el texto
    simplify_text(text_feature)

    # Tratamos el texto
    process_text(text_feature)

    # Borrar columnas no necesarias
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
    try:
        print("\n- Dividiendo Train/Dev...")
        X = data.drop(columns=[args.prediction])
        y = data[args.prediction]
        X_train, X_dev, y_train, y_dev = train_test_split(X,y, test_size=float(args.test_size), stratify=y, random_state=42)

        # Realizamos Oversampling o Undersampling
        X_train, y_train = over_under_sampling(X_train, y_train)

        if args.debug:
            train = pd.concat([X_train,y_train], axis=1)
            dev = pd.concat([X_dev,y_dev], axis=1)
            train.to_csv('output/1-train-processed.csv', index=False)
            dev.to_csv('output/2-dev-processed.csv', index=False)
        print(Fore.GREEN + "\nTrain/Dev divididos con éxito" + Fore.RESET)
        return X_train, y_train, X_dev, y_dev
    except Exception as e:
        print(Fore.RED + "Error al realizar la división del train/dev" + Fore.RESET)
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
            print(Fore.CYAN + "Modelo guardado con éxito" + Fore.RESET)
        with open('output/3-modelo.csv', 'w', newline='',encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Combinación', 'Precisión', 'Recall', 'F1_Macro', 'F1_Micro', 'F1_Weighted'])
            rdo_df = pd.DataFrame(gs.cv_results_)
            if args.algorithm == "knn":
                rdo_df['Combinación'] = rdo_df.apply(lambda row: f"kNN_K{row['param_n_neighbors']}_P{row['param_p']}_{row['param_weights']}",axis=1)

            elif args.algorithm == "decision_tree":
                rdo_df['Combinación'] = rdo_df.apply(lambda row: f"DT_{row['param_criterion']}_D{row['param_max_depth']}_S{row['param_min_samples_split']}_L{row['param_min_samples_leaf']}", axis=1)

            elif args.algorithm == "random_forest":
                rdo_df['Combinación'] = rdo_df.apply(lambda row: f"RF_N{row['param_n_estimators']}_D{row['param_max_depth']}_S{row['param_min_samples_split']}_L{row['param_min_samples_leaf']}_B{row['param_bootstrap']}",axis=1)
            cols_salida = ['Combinación','mean_test_precision','mean_test_recall','mean_test_f1_macro','mean_test_f1_micro','mean_test_f1_weighted']

            for fila in rdo_df[cols_salida].itertuples(index=False):
                writer.writerow([fila[0],round(fila[1], 4), round(fila[2], 4), round(fila[3], 4), round(fila[4], 4), round(fila[5], 4)])
            return rdo_df
    except Exception as e:
        print(Fore.RED + "Error al guardar el modelo" + Fore.RESET)
        print(e)

def mostrar_resultados(gs, x_dev, y_dev, rdo_df):
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
        y_pred = gs.predict(x_dev)
        print(Fore.MAGENTA + "> Mejores parametros:\n" + Fore.RESET, gs.best_params_)
        print(Fore.MAGENTA + "> Mejor puntuacion:\n" + Fore.RESET, gs.best_score_)

        #F1-score
        fscore_micro, fscore_macro = calculate_fscore(y_dev, y_pred)
        print(Fore.MAGENTA + "> F1-score micro:\n" + Fore.RESET, fscore_micro)
        print(Fore.MAGENTA + "> F1-score macro:\n" + Fore.RESET, fscore_macro)

        #Informe de clasificación
        cr = calculate_classification_report(y_dev, y_pred)
        print(Fore.MAGENTA + "> Informe de clasificación:\n" + Fore.RESET, cr)

        #La matriz de confusión
        cm = calculate_confusion_matrix(y_dev, y_pred)
        print(Fore.MAGENTA + "> Matriz de confusión:\n" + Fore.RESET, cm)

        plot_metricas(rdo_df, cm, cr)

def calcularIntervalo(intConf):
    if intConf[0] == -1: #Modo intervalo
        limInf = intConf[1]         #Límite inferior
        limSup = intConf[2] + 1     #Límite superior
        step = intConf[3]           #Salto
        if args.algorithm == "knn" and not args.knn_config["pares"]: #Si es knn y NO queremos pares
            if limInf % 2 == 0: #Si es par
                limInf += 1 #Para evitar pares inferiores
            step = step * 2 #Saltos solo a impares
        return [i for i in range(limInf, limSup, step)] #Construir el intervalo
    else: #NO modo intervalo
        return intConf

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
    X_train, y_train, X_dev, y_dev = divide_data()
    # Leemos la configuración del árbol de decisión desde el JSON
    config = args.knn_config
    #Definimos los hiperparámetros
    knn_config = {
        "n_neighbors": calcularIntervalo(config["k"]),
        "p": config["p"],
        "weights": config["weights"]
    }

    with tqdm(total=100, desc='Procesando kNN', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(KNeighborsClassifier(), knn_config, cv=5, n_jobs=args.cpu, scoring=metricas, refit=args.estimator)
        start_time = time.time()
        gs.fit(X_train, y_train)
        end_time = time.time()
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  # Esperamos un tiempo aleatorio
            pbar.update(random.random() * 2)  # Actualizamos la barra con un valor aleatorio
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
    execution_time = end_time - start_time
    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + "segundos")

    # Guardamos el modelo utilizando pickle
    rdo_df = save_model(gs)

    # Mostramos los resultados
    mostrar_resultados(gs, X_dev, y_dev, rdo_df)


def decision_tree():
    """
    Función para entrenar un Decision Tree usando GridSearchCV y sacar el mejor modelo.

    Básicamente: divide datos -> prueba combinaciones -> entrena -> guarda -> muestra resultados.
    """
    # Dividimos los datos en train y dev (lo típico para no evaluarlo con lo mismo que entrenamos)
    X_train, y_train, X_dev, y_dev = divide_data()

    # Pillamos la configuración del JSON (para no hardcodear los hiperparámetros aquí)
    config = args.decision_tree_config
    # Definimos los hiperparámetros que va a probar el GridSearch
    decision_tree_config = {
        "criterion": config["criterion"],                                       # criterio de división (gini o entropy)
        "max_depth": calcularIntervalo(config["max_depth"]),                    # hasta qué profundidad dejamos crecer el árbol
        "min_samples_split": calcularIntervalo(config["min_samples_split"]),    # mínimo de muestras para que un nodo se pueda dividir
        "min_samples_leaf": calcularIntervalo(config["min_samples_leaf"]),      # mínimo de muestras que debe tener una hoja
        "max_features": config["max_features"]                                  # cuántas features puede usar para buscar el mejor split
    }
    # Barra de progreso para que no parezca que el programa se ha muerto mientras entrena xd
    with tqdm(total=100, desc='Procesando decision tree', unit='iter', leave=True) as pbar:

        # Creamos el GridSearchCV para probar todas las combinaciones posibles
        gs = GridSearchCV(
            DecisionTreeClassifier(random_state=42),
            decision_tree_config,
            cv=5,
            n_jobs=args.cpu,
            scoring=metricas,
            refit=args.estimator
        )
        # Medimos cuánto tarda en entrenar (por curiosidad y por comparar modelos)
        start_time = time.time()
        # Entrenamos el modelo con GridSearch (esto ya prueba todo automáticamente)
        gs.fit(X_train, y_train)
        end_time = time.time()

        # Esto es solo para simular progreso en la barra (porque GridSearch no lo actualiza solo)
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))
            pbar.update(random.random() * 2)
        # Forzamos la barra a llegar al 100% sí o sí
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
    # Calculamos el tiempo total de ejecución
    execution_time = end_time - start_time
    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + " segundos")

    # Guardamos el modelo ya entrenado (el mejor que encontró el GridSearch)
    rdo_df = save_model(gs)

    # Mostramos métricas y resultados en el conjunto dev
    mostrar_resultados(gs, X_dev, y_dev, rdo_df)


def random_forest():
    """
    Función que entrena un Random Forest usando GridSearchCV para ajustar hiperparámetros.

    Flujo: divide datos -> prueba configs -> entrena -> guarda -> evalúa.
    """
    # Dividimos los datos en entrenamiento y desarrollo
    X_train, y_train, X_dev, y_dev = divide_data()
    # Leemos la config del JSON (así es más fácil cambiar parámetros sin tocar el código)
    config = args.rf_config
    # Definimos los hiperparámetros que va a probar el GridSearch
    rf_config = {
        "n_estimators": config["n_estimators"],                                 # cuántos árboles va a tener el bosque
        "max_depth": calcularIntervalo(config["max_depth"]),                    # profundidad máxima de cada árbol
        "min_samples_split": calcularIntervalo(config["min_samples_split"]),    # mínimo de muestras para poder dividir un nodo
        "min_samples_leaf": calcularIntervalo(config["min_samples_leaf"]),      # mínimo de muestras que debe tener una hoja
        "bootstrap": config["bootstrap"],                                       # si usa muestreo bootstrap o no
        "max_features": config["max_features"]                                  # nº de features consideradas en cada split
    }
    # Barra de progreso para que se vea bonito mientras se entrena
    with tqdm(total=100, desc='Procesando random forest', unit='iter', leave=True) as pbar:

        # Creamos el GridSearchCV para probar varias combinaciones
        gs = GridSearchCV(
            RandomForestClassifier(random_state=42),
            rf_config,
            cv=5,
            n_jobs=args.cpu,
            scoring=metricas,
            refit=args.estimator
        )
        # Medimos tiempo para ver cuánto tarda el entrenamiento
        start_time = time.time()
        # Entrenamos el Random Forest probando todas las combinaciones posibles
        gs.fit(X_train, y_train)
        end_time = time.time()

        # Simulamos progreso en la barra porque GridSearch no actualiza progreso real
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))
            pbar.update(random.random() * 2)

        # Forzamos la barra a 100% para que quede clean
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
    # Calculamos el tiempo total de ejecución
    execution_time = end_time - start_time
    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + " segundos")
    # Guardamos el modelo entrenado (el mejor que encontró el GridSearch)
    rdo_df = save_model(gs)

    # Mostramos los resultados finales usando el conjunto dev
    mostrar_resultados(gs, X_dev, y_dev, rdo_df)

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
        print(Fore.GREEN + "Carpeta output creada con éxito" + Fore.RESET)
    except FileExistsError:
        print(Fore.GREEN + "La carpeta output ya existe" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al crear la carpeta output" + Fore.RESET)
        print(e)
        sys.exit(1)
    # Cargamos los datos
    print("\n- Cargando datos...")
    data = load_data(args.file)
    # Descargamos los recursos necesarios de nltk
    print("\n- Descargando diccionarios...")
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('wordnet')
    # Preprocesamos los datos
    print("\n- Preprocesando datos...")
    preprocesar_datos()
    if args.debug:
        try:
            print("\n- Guardando datos preprocesados...")
            data.to_csv('output/0-data-processed.csv', index=False)
            print(Fore.GREEN + "Datos preprocesados guardados con éxito" + Fore.RESET)
        except Exception as e:
            print(Fore.RED + "Error al guardar los datos preprocesados" + Fore.RESET)

    # Ejecutamos el algoritmo seleccionado
    print("\n- Ejecutando el algoritmo...")
    try:
        if args.algorithm == "knn":
            kNN()
            print(Fore.GREEN + "Algoritmo knn ejecutado con éxito" + Fore.RESET)
            sys.exit(0)
        elif args.algorithm == "decision_tree":
            decision_tree()
            print(Fore.GREEN + "Algoritmo árbol de decisión ejecutado con éxito" + Fore.RESET)
            sys.exit(0)
        elif args.algorithm == "random_forest":
            random_forest()
            print(Fore.GREEN + "Algoritmo random forest ejecutado con éxito" + Fore.RESET)
            sys.exit(0)
        else:
            print(Fore.RED + "Algoritmo no soportado" + Fore.RESET)
            sys.exit(1)
    except Exception as e:
        print(e)