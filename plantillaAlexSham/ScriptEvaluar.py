# -*- coding: utf-8 -*-
"""
Script para predicción con un modelo ya entrenado
"""

import sys
import signal
import argparse
import pandas as pd
import numpy as np
import pickle
import json
import os

from colorama import Fore

from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def signal_handler(sig, frame):
    print("\nSaliendo del programa...")
    sys.exit(0)


def parse_args():
    parse = argparse.ArgumentParser(description="Predicción con modelo de clasificación.")
    parse.add_argument("-f", "--file", help="Fichero csv (/Path_to_file)", required=True)
    parse.add_argument("-p", "--prediction", help="Columna objetivo (Nombre de la columna)", required=True)
    parse.add_argument("--debug", help="Modo debug", required=False, default=False, action="store_true")

    args = parse.parse_args()

    with open('clasificador.json') as json_file:
        config = json.load(json_file)

    for key, value in config.items():
        setattr(args, key, value)

    return args


def load_data(file):
    try:
        print("\n- Cargando datos...")
        data = pd.read_csv(file, encoding='utf-8')
        data.columns = data.columns.str.strip()
        data = data.map(lambda x: x.strip() if isinstance(x, str) else x).replace(r'^\s*$', np.nan, regex=True)
        print(Fore.GREEN + "Datos cargados con éxito" + Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED + "Error al cargar los datos" + Fore.RESET)
        print(e)
        sys.exit(1)


def select_features():
    try:
        numerical_feature = data.select_dtypes(include=['int64', 'float64'])
        if args.prediction in numerical_feature.columns:
            numerical_feature = numerical_feature.drop(columns=[args.prediction])

        categorical_feature = data.select_dtypes(include=['object', 'string'])
        categorical_feature = categorical_feature.loc[:, categorical_feature.nunique() <= args.preprocessing["unique_category_threshold"]]

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
    global data
    try:
        print("\n- Procesando missing values...")
        if args.debug:
            print(f"{Fore.MAGENTA}> Missing values detectados:\n{data.isna().sum().to_string()}{Fore.RESET}")

        modo = args.preprocessing["missing_values"]
        if modo == "impute":
            modo_impute = args.preprocessing["impute_strategy"]
            if not numerical_feature.empty:
                data[numerical_feature.columns] = SimpleImputer(strategy=modo_impute).fit_transform(data[numerical_feature.columns])
            if not categorical_feature.empty:
                data[categorical_feature.columns] = SimpleImputer(strategy='most_frequent').fit_transform(data[categorical_feature.columns])
            print(Fore.GREEN + f"Missing values imputados con la estrategia {modo_impute} con éxito" + Fore.RESET)

        elif modo == "delete":
            data = data.dropna(subset=list(numerical_feature.columns) + list(categorical_feature.columns))
            print(Fore.GREEN + "Missing values ELIMINADOS con éxito" + Fore.RESET)

        else:
            print(Fore.YELLOW + "No se están procesando los missing values" + Fore.RESET)

    except Exception as e:
        print(Fore.RED + "Error al tratar missing values" + Fore.RESET)
        print(e)
        sys.exit(1)


def reescaler(numerical_feature):
    global data
    try:
        print("\n- Reescalando datos...")
        if not numerical_feature.empty:
            scaling = {
                "maxAbs": MaxAbsScaler,
                "minMax": MinMaxScaler,
                "normalizer": Normalizer,
                "standard": StandardScaler
            }
            modo = args.preprocessing["scaling"]
            if modo in scaling:
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
    global data
    try:
        print("\n- Simplificando el texto...")
        if not text_feature.empty:
            stop_words = set(stopwords.words('english'))
            stemmer = PorterStemmer()

            for col in text_feature.columns:
                processed = []
                for text in tqdm(data[col], desc=f"Procesando la columna {col}:"):
                    tokens = word_tokenize(str(text))
                    tokens = [w.lower() for w in tokens]
                    tokens = [t for t in tokens if t not in stop_words]
                    tokens = [stemmer.stem(t) for t in tokens]
                    tokens = [t for t in tokens if t.isalnum()]
                    tokens = sorted(tokens)
                    processed.append(" ".join(tokens))
                data[col] = processed

            print(Fore.GREEN + "Texto simplificado con éxito" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No se está simplificando el texto" + Fore.RESET)

    except Exception as e:
        print(Fore.RED + "Error al simplificar el texto" + Fore.RESET)
        print(e)
        sys.exit(1)


def process_text(text_feature):
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
                data.drop(text_feature.columns, axis=1, inplace=True)
                print(Fore.GREEN + "Texto tratado usando BOW con éxito" + Fore.RESET)

            else:
                print(Fore.YELLOW + "No se están tratando los textos" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No se han encontrado columnas de texto a procesar" + Fore.RESET)

    except Exception as e:
        print(Fore.RED + "Error al tratar el texto" + Fore.RESET)
        print(e)
        sys.exit(1)


def drop_features():
    global data
    try:
        print("\n- Eliminando columnas...")
        data = data.drop(columns=args.preprocessing["drop_features"], errors="ignore")
        print(Fore.GREEN + "Columnas eliminadas con éxito" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al eliminar columnas" + Fore.RESET)
        print(e)
        sys.exit(1)


def preprocesar_datos():
    global data

    numerical_feature, text_feature, categorical_feature = select_features()
    process_missing_values(numerical_feature, categorical_feature)
    cat2num(categorical_feature)
    reescaler(numerical_feature)
    simplify_text(text_feature)
    process_text(text_feature)
    drop_features()

    return data


def load_model():
    try:
        with open('output/modelo.pkl', 'rb') as file:
            model = pickle.load(file)
            print(Fore.GREEN + "Modelo cargado con éxito" + Fore.RESET)
            return model
    except Exception as e:
        print(Fore.RED + "Error al cargar el modelo" + Fore.RESET)
        print(e)
        sys.exit(1)


def predict(model):
    global data
    X_test = data.drop(columns=[args.prediction], errors="ignore")
    y_pred = model.predict(X_test)

    # Añadimos la prediccion al dataframe data
    data = pd.concat([data, pd.DataFrame(y_pred, columns=[f"Predict_{args.prediction}"])], axis=1)


if __name__ == "__main__":
    np.random.seed(42)
    print("=== Predicción clasificador ===")
    signal.signal(signal.SIGINT, signal_handler)

    args = parse_args()

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

    print("\n- Cargando datos...")
    data = load_data(args.file)

    print("\n- Descargando diccionarios...")
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('wordnet')

    print("\n- Preprocesando datos...")
    preprocesar_datos()

    if args.debug:
        try:
            print("\n- Guardando datos preprocesados...")
            data.to_csv('output/data-processed-predict.csv', index=False)
            print(Fore.GREEN + "Datos preprocesados guardados con éxito" + Fore.RESET)
        except Exception as e:
            print(Fore.RED + "Error al guardar los datos preprocesados" + Fore.RESET)
            print(e)

    print("\n- Cargando modelo...")
    model = load_model()

    print("\n- Prediciendo...")
    try:
        predict(model)
        print(Fore.GREEN + "Predicción realizada con éxito" + Fore.RESET)
        data.to_csv('output/data-prediction.csv', index=False)
        print(Fore.GREEN + "Predicción guardada con éxito" + Fore.RESET)
        sys.exit(0)
    except Exception as e:
        print(e)
        sys.exit(1)