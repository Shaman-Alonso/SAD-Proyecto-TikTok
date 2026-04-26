# -*- coding: utf-8 -*-
import os
import csv

from colorama import Fore
from tqdm import tqdm
import matplotlib.pyplot as plt
# Gensim
from gensim.models import Phrases, LdaModel, Nmf, CoherenceModel, TfidfModel
from gensim.corpora import Dictionary



class ModelClustering:

    def __init__(self, args):
        self.args = args
        self.argsClustering = args.clustering

    def __preparar_corpus(self, data):
        # Por comodidad
        args = self.args
        argsClustering = self.argsClustering
        col_txt = argsClustering["textClustering"]

        # Creamos diccionario (y quitamos las más y menos frecuentes)
        dictionary = Dictionary(data[col_txt])
        dictionary.filter_extremes(no_below=2, no_above=0.5)

        # Calculamos el BoW
        corpus = [dictionary.doc2bow(doc) for doc in data[col_txt]]

        # Si TF-IDF, transformamos
        if args.preprocessing["text_process"] == "tf-idf":
            tfidf = TfidfModel(dictionary=dictionary)
            corpus = [tfidf[doc] for doc in corpus]

        return dictionary, corpus

    @staticmethod
    def __generar_reportes(modo, model, x_vals, umass_y, cv_y, best_score, topics, passes, iters, folder, corpus):
        # Gráficas
        for metric, y_vals in [('u_mass', umass_y), ('c_v', cv_y)]:
            plt.figure()
            plt.plot(x_vals, y_vals, marker='o')
            plt.title(f'Coherencia {metric} vs Tópicos ({modo.upper()})')
            plt.savefig(f"{folder}/coherence_{metric}_{modo}.png")
            plt.close()

        # Guardar Tópicos en TXT
        with open(f"{folder}/topics_{modo}.txt", 'w', encoding="utf-8") as f:
            f.write(f"Mejor Coherencia: {best_score:.4f}\nParams: Topics={topics}, Passes={passes}, Iters={iters}\n\n")
            for i, topic in enumerate(model.top_topics(corpus)):
                f.write(f"Topic {i + 1}\n{topic}\n\n")

        model.save(f"{folder}/{modo}_model")

    def __aplicar_bigramas(self, data):
        col_txt = self.argsClustering["textClustering"]
        # Detectar bigramas frecuentes (que aparezcan >20 veces)
        bigram = Phrases(data[col_txt], min_count=20)

        # Añadir al dataset
        data[col_txt] = data[col_txt].apply(lambda tokens: tokens + [t for t in bigram[tokens] if '_' in t])

    def __ejecutar_cluster(self, modo, cluster, data, safe_folder):
        try:
            # Por comodidad
            args = self.argsClustering
            col_txt = args["textClustering"]

            # Preparar bigramas
            self.__aplicar_bigramas(data)

            # Preparar los datos
            dictionary, corpus = self.__preparar_corpus(data)
            id2word = dictionary.id2token

            # Inicializar variables para el mejor modelo
            best_avg_topic_coherence = -999999
            best_model, best_num_topic, best_passes, best_iterations = None, None, None, None
            umass_values, cv_values, num_topics_values = [], [], []

            # Grid Search
            conf = args[modo]
            combinaciones = len(conf["num_topics"]) * len(conf["passes"]) * len(conf["iterations"])

            with tqdm(total=combinaciones, desc=f"Entrenando {modo}") as pbar:
                with open(f"{safe_folder}/clustering_results_{modo}.csv", 'w', newline='',
                          encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=['Num Topics', 'Passes', 'Iterations', 'Coherence'])
                    writer.writeheader()

                    for num_topic in conf["num_topics"]:
                        for passes in conf["passes"]:
                            for iterations in conf["iterations"]:
                                # Configuración base del modelo
                                model_params = {
                                    'corpus': corpus,
                                    'id2word': id2word,
                                    'num_topics': int(num_topic),
                                    'passes': int(passes),
                                    'random_state': 42
                                }

                                # Añadimos parámetros específicos si es LDA
                                if modo == "lda":
                                    model_params.update({'alpha': 'auto', 'eta': 'auto', 'iterations': int(iterations)})

                                # Instanciamos el modelo dinámicamente (LdaModel(...) o Nmf(...))
                                model = cluster(**model_params) # Los * para que pase de diccionario a params

                                # Cálculo de Coherencia
                                avg_coherence = CoherenceModel(model=model, corpus=corpus, dictionary=dictionary,
                                                               coherence='u_mass').get_coherence()

                                writer.writerow({'Num Topics': num_topic, 'Passes': passes, 'Iterations': iterations,
                                                 'Coherence': avg_coherence})

                                if avg_coherence > best_avg_topic_coherence:
                                    best_avg_topic_coherence, best_model = avg_coherence, model
                                    best_num_topic, best_passes, best_iterations = num_topic, passes, iterations

                                # Para gráficas
                                cv_score = CoherenceModel(model=model, texts=data[col_txt], dictionary=dictionary,
                                                          coherence='c_v').get_coherence()
                                umass_values.append(avg_coherence)
                                cv_values.append(cv_score)
                                num_topics_values.append(num_topic)
                                pbar.update(1)

            self.__generar_reportes(modo, best_model, num_topics_values, umass_values, cv_values,
                                 best_avg_topic_coherence, best_num_topic, best_passes,
                                 best_iterations, safe_folder, corpus)

        except Exception as e:
            raise RuntimeError(f"Error en clustering modo={modo}") from e

    def ejecutar_clustering(self, data):
        try:
            # Por comodidad
            args = self.argsClustering

            clusters = {
                "lda": LdaModel,
                "nmf": Nmf
            }
            modo = args["cluster"]

            if modo in clusters:
                cluster = clusters[modo]

                # Por cada sentimiento ejecuta el clustering
                for sentimiento, data in data.items():
                    # Evitar errores si un grupo se ha quedado vacío
                    if data.empty:
                        print(Fore.YELLOW + f"No hay datos para el sentimiento: {sentimiento}" + Fore.RESET)
                        continue

                    print(Fore.CYAN + f"\n=== Ejecutando Clustering para: {sentimiento.upper()} ===" + Fore.RESET)

                    # Crear carpeta específica para el sentimiento
                    carpeta_analisis = f'./output/sentiment_analysis/{sentimiento}'
                    os.makedirs(carpeta_analisis, exist_ok=True)

                    self.__ejecutar_cluster(modo, cluster, data, carpeta_analisis)
            else:
                print(Fore.RED + "Algoritmo no soportado." + Fore.RESET)

            print(Fore.GREEN + "Clustering realizado con éxito" + Fore.RESET)

        except Exception as e:
            raise RuntimeError("Error en ejecución de clustering") from e