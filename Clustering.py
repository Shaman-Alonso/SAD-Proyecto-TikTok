# -*- coding: utf-8 -*-

import sys
import json
import argparse
import signal
import os
import traceback
import csv
import string
import unicodedata
import numpy as np
import pandas as pd

from colorama import Fore, Style
from tqdm import tqdm

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from gensim.models import Phrases, LdaModel, Nmf, CoherenceModel, TfidfModel
from gensim.corpora import Dictionary

import matplotlib.pyplot as plt


# ===============================
# FUNCIONES AUXILIARES
# ===============================

def signal_handler(sig, frame):
    print("\nSaliendo del programa...")
    sys.exit(0)


def parse_args():
    parse = argparse.ArgumentParser(description="Practica de algoritmos de clustering (LDA / NMF).")
    parse.add_argument("-f", "--file", help="Fichero csv (/Path_to_file)", required=True)
    parse.add_argument("-t", "--textcol", help="Columna de texto a analizar", required=True)
    parse.add_argument("-v", "--verbose", help="Muestra información extra", required=False, default=False, action="store_true")
    parse.add_argument("--debug", help="Modo debug", required=False, default=False, action="store_true")
    parse.add_argument("--config", help="Archivo JSON de configuración", required=False, default="clustering.json")

    parsed_args = parse.parse_args()

    # Leemos config JSON
    with open(parsed_args.config, encoding="utf-8") as json_file:
        config = json.load(json_file)

    for key, value in config.items():
        setattr(parsed_args, key, value)

    return parsed_args


def load_data(file):
    try:
        data = pd.read_csv(file, encoding="utf-8")
        data.columns = data.columns.str.strip()
        data = data.map(lambda x: x.strip() if isinstance(x, str) else x)
        data = data.replace(r'^\s*$', np.nan, regex=True)
        return data
    except Exception as e:
        print(Fore.RED + "Error al cargar los datos" + Fore.RESET)
        print(e)
        sys.exit(1)


def normalize_text(text):
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    return text


def preprocesar_datos(data, args):
    """
    Convierte el texto a lista de tokens (data["text"])
    """
    try:
        print("\n- Preprocesando datos...")

        data = data.dropna(subset=[args.textcol]).copy()

        tokenizer = RegexpTokenizer(r'\w+')
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))

        processed_docs = []

        for doc in tqdm(data[args.textcol], desc="Tokenizando texto"):
            doc = normalize_text(doc).lower()
            tokens = tokenizer.tokenize(doc)

            clean_tokens = []
            for token in tokens:
                if token not in stop_words and token not in string.punctuation:
                    clean_tokens.append(lemmatizer.lemmatize(token))

            processed_docs.append(clean_tokens)

        data["text"] = processed_docs

        print(Fore.GREEN + "Datos preprocesados con éxito" + Fore.RESET)
        return data

    except Exception as e:
        print(Fore.RED + "Error al preprocesar datos" + Fore.RESET)
        print(e)
        traceback.print_exc()
        sys.exit(1)


# ===============================
# LDA
# ===============================

def lda(data, args, safe_folder):
    try:
        bigram = Phrases(data['text'], min_count=20)

        for idx in range(len(data['text'])):
            for token in bigram[data['text'][idx]]:
                if '_' in token:
                    data['text'][idx].append(token)

        dictionary = Dictionary(data['text'])
        dictionary.filter_extremes(no_below=20, no_above=0.5)

        if args.preprocessing["text_process"] == "bow":
            corpus = [dictionary.doc2bow(doc) for doc in data['text']]
        elif args.preprocessing["text_process"] == "tf-idf":
            tfidf = TfidfModel(dictionary=dictionary)
            corpus = [tfidf[dictionary.doc2bow(doc)] for doc in data['text']]
        else:
            corpus = [dictionary.doc2bow(doc) for doc in data['text']]

        id2word = dictionary.id2token

        best_avg_topic_coherence = -999999
        best_model = None
        best_num_topic = None
        best_passes = None
        best_iterations = None

        umass_values = []
        cv_values = []
        num_topics_values = []

        with tqdm(total=len(args.lda["num_topics"]) * len(args.lda["passes"]) * len(args.lda["iterations"])) as pbar:
            with open(safe_folder + '/clustering_results.csv', 'w', newline='', encoding="utf-8") as csvfile:
                fieldnames = ['Num Topics', 'Passes', 'Iterations', 'Coherence']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for num_topic in args.lda["num_topics"]:
                    for passes in args.lda["passes"]:
                        for iterations in args.lda["iterations"]:

                            lda_model = LdaModel(
                                corpus=corpus,
                                id2word=id2word,
                                alpha='auto',
                                eta='auto',
                                iterations=int(iterations),
                                num_topics=int(num_topic),
                                passes=int(passes),
                                random_state=42
                            )

                            avg_topic_coherence = CoherenceModel(
                                model=lda_model,
                                corpus=corpus,
                                dictionary=dictionary,
                                coherence='u_mass'
                            ).get_coherence()

                            writer.writerow({
                                'Num Topics': num_topic,
                                'Passes': passes,
                                'Iterations': iterations,
                                'Coherence': avg_topic_coherence
                            })

                            if avg_topic_coherence > best_avg_topic_coherence:
                                best_model = lda_model
                                best_num_topic = num_topic
                                best_passes = passes
                                best_iterations = iterations
                                best_avg_topic_coherence = avg_topic_coherence

                            umass_values.append(avg_topic_coherence)

                            cv_score = CoherenceModel(
                                model=lda_model,
                                texts=data['text'],
                                dictionary=dictionary,
                                coherence='c_v'
                            ).get_coherence()

                            cv_values.append(cv_score)
                            num_topics_values.append(num_topic)

                            pbar.update(1)

        # Gráficas
        plt.figure()
        plt.plot(num_topics_values, umass_values)
        plt.xlabel('Número de Tópicos')
        plt.ylabel('Coherencia (u_mass)')
        plt.title('Coherencia u_mass vs Número de tópicos (LDA)')
        plt.savefig(safe_folder + '/coherence_umass.png')

        plt.figure()
        plt.plot(num_topics_values, cv_values)
        plt.xlabel('Número de Tópicos')
        plt.ylabel('Coherencia (c_v)')
        plt.title('Coherencia c_v vs Número de tópicos (LDA)')
        plt.savefig(safe_folder + '/coherence_cv.png')

        # Mostrar resultados
        if args.verbose:
            print(f"\nMedia coherencia de tópico: {best_avg_topic_coherence:.4f}")
            print(f"Mejores parámetros: num_topics={best_num_topic}, passes={best_passes}, iterations={best_iterations}")

        # Guardar tópicos
        with open(safe_folder + '/topics.txt', 'w', encoding="utf-8") as f:
            f.write(f"Media coherencia: {best_avg_topic_coherence:.4f}\n")
            f.write(f"Mejores parámetros: num_topics={best_num_topic}, passes={best_passes}, iterations={best_iterations}\n\n")

            i = 0
            for topic in best_model.top_topics(corpus):
                i += 1
                f.write(f"Topic {i}\n")
                f.write(str(topic) + "\n\n")

        best_model.save(safe_folder + '/lda_model')

    except Exception as e:
        print(Fore.RED + "Error al realizar el clustering LDA" + Fore.RESET)
        print(e)
        traceback.print_exc()
        sys.exit(1)


# ===============================
# NMF
# ===============================

def nmf(data, args, safe_folder):
    try:
        bigram = Phrases(data['text'], min_count=20)

        for idx in range(len(data['text'])):
            for token in bigram[data['text'][idx]]:
                if '_' in token:
                    data['text'][idx].append(token)

        dictionary = Dictionary(data['text'])
        dictionary.filter_extremes(no_below=20, no_above=0.5)

        if args.preprocessing["text_process"] == "bow":
            corpus = [dictionary.doc2bow(doc) for doc in data['text']]
        elif args.preprocessing["text_process"] == "tf-idf":
            tfidf = TfidfModel(dictionary=dictionary)
            corpus = [tfidf[dictionary.doc2bow(doc)] for doc in data['text']]
        else:
            corpus = [dictionary.doc2bow(doc) for doc in data['text']]

        id2word = dictionary.id2token

        best_avg_topic_coherence = -999999
        best_model = None
        best_num_topic = None
        best_passes = None
        best_iterations = None

        umass_values = []
        cv_values = []
        num_topics_values = []

        with tqdm(total=len(args.nmf["num_topics"]) * len(args.nmf["passes"]) * len(args.nmf["iterations"])) as pbar:
            with open(safe_folder + '/clustering_results.csv', 'w', newline='', encoding="utf-8") as csvfile:
                fieldnames = ['Num Topics', 'Passes', 'Iterations', 'Coherence']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for num_topic in args.nmf["num_topics"]:
                    for passes in args.nmf["passes"]:
                        for iterations in args.nmf["iterations"]:

                            nmf_model = Nmf(
                                corpus=corpus,
                                id2word=id2word,
                                num_topics=int(num_topic),
                                passes=int(passes),
                                random_state=42
                            )

                            avg_topic_coherence = CoherenceModel(
                                model=nmf_model,
                                corpus=corpus,
                                dictionary=dictionary,
                                coherence='u_mass'
                            ).get_coherence()

                            writer.writerow({
                                'Num Topics': num_topic,
                                'Passes': passes,
                                'Iterations': iterations,
                                'Coherence': avg_topic_coherence
                            })

                            if avg_topic_coherence > best_avg_topic_coherence:
                                best_model = nmf_model
                                best_num_topic = num_topic
                                best_passes = passes
                                best_iterations = iterations
                                best_avg_topic_coherence = avg_topic_coherence

                            umass_values.append(avg_topic_coherence)

                            cv_score = CoherenceModel(
                                model=nmf_model,
                                texts=data['text'],
                                dictionary=dictionary,
                                coherence='c_v'
                            ).get_coherence()

                            cv_values.append(cv_score)
                            num_topics_values.append(num_topic)

                            pbar.update(1)

        # Gráficas
        plt.figure()
        plt.plot(num_topics_values, umass_values)
        plt.xlabel('Número de Tópicos')
        plt.ylabel('Coherencia (u_mass)')
        plt.title('Coherencia u_mass vs Número de tópicos (NMF)')
        plt.savefig(safe_folder + '/coherence_umass.png')

        plt.figure()
        plt.plot(num_topics_values, cv_values)
        plt.xlabel('Número de Tópicos')
        plt.ylabel('Coherencia (c_v)')
        plt.title('Coherencia c_v vs Número de tópicos (NMF)')
        plt.savefig(safe_folder + '/coherence_cv.png')

        if args.verbose:
            print(f"\nMedia coherencia de tópico: {best_avg_topic_coherence:.4f}")
            print(f"Mejores parámetros: num_topics={best_num_topic}, passes={best_passes}, iterations={best_iterations}")

        with open(safe_folder + '/topics.txt', 'w', encoding="utf-8") as f:
            f.write(f"Media coherencia: {best_avg_topic_coherence:.4f}\n")
            f.write(f"Mejores parámetros: num_topics={best_num_topic}, passes={best_passes}, iterations={best_iterations}\n\n")

            i = 0
            for topic in best_model.top_topics(corpus):
                i += 1
                f.write(f"Topic {i}\n")
                f.write(str(topic) + "\n\n")

        best_model.save(safe_folder + '/nmf_model')

    except Exception as e:
        print(Fore.RED + "Error al realizar el clustering NMF" + Fore.RESET)
        print(e)
        traceback.print_exc()
        sys.exit(1)


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    np.random.seed(42)

    print("=== Clustering ===")

    signal.signal(signal.SIGINT, signal_handler)

    args = parse_args()

    # Carpeta output con nombre del CSV
    print("\n- Creando carpeta output...")

    if os.name == 'nt':
        safe_folder = args.file.split('\\')[-1].split('.')[0]
    else:
        safe_folder = args.file.split('/')[-1].split('.')[0]

    try:
        os.makedirs(safe_folder, exist_ok=True)
        print(Fore.GREEN + "Carpeta output creada con éxito" + Fore.RESET)
    except Exception as e:
        print(Fore.RED + "Error al crear la carpeta output" + Fore.RESET)
        print(e)
        sys.exit(1)

    # Cargar datos
    print("\n- Cargando datos...")
    data = load_data(args.file)

    # Descargar recursos nltk
    print("\n- Descargando diccionarios...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

    # Preprocesar
    data = preprocesar_datos(data, args)

    if args.debug:
        try:
            data.to_csv(safe_folder + "/data-processed.csv", index=False)
            print(Fore.GREEN + "Datos preprocesados guardados con éxito" + Fore.RESET)
        except Exception:
            print(Fore.RED + "Error guardando data-processed.csv" + Fore.RESET)

    # Ejecutar algoritmo
    print("\n- Realizando clustering...")

    if args.algorithm == "lda":
        lda(data, args, safe_folder)
    elif args.algorithm == "nmf":
        nmf(data, args, safe_folder)
    else:
        print(Fore.RED + "Algoritmo no soportado. Usa 'lda' o 'nmf'" + Fore.RESET)
        sys.exit(1)

    print(Fore.GREEN + "Clustering realizado con éxito" + Fore.RESET)
    sys.exit(0)