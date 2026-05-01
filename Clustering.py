# -*- coding: utf-8 -*-
import os
import csv

from colorama import Fore
from tqdm import tqdm
import matplotlib.pyplot as plt

from gensim.models import Phrases, LdaModel, Nmf, CoherenceModel, TfidfModel
from gensim.corpora import Dictionary


class ModelClustering:

    def __init__(self, args):
        self.args = args
        self.argsClustering = args.clustering

    # =========================
    # PREPARACIÓN CORPUS
    # =========================
    def __preparar_corpus(self, data):
        args = self.args
        argsClustering = self.argsClustering
        col_txt = argsClustering["textClustering"] + "_Clustering"

        dictionary = Dictionary(data[col_txt])
        dictionary.filter_extremes(no_below=2, no_above=0.5)

        bow_corpus = [dictionary.doc2bow(doc) for doc in data[col_txt]]

        # TF-IDF opcional (solo para modelos que lo soporten bien)
        if args.preprocessing["text_process"] == "tf-idf":
            tfidf = TfidfModel(bow_corpus)
            corpus_model = tfidf[bow_corpus]
        else:
            corpus_model = bow_corpus

        return dictionary, bow_corpus, corpus_model

    # =========================
    # BIGRAMAS
    # =========================
    def __aplicar_bigramas(self, data):
        col_txt = self.argsClustering["textClustering"] + "_Clustering"

        bigram = Phrases(data[col_txt], min_count=20)
        data[col_txt] = data[col_txt].apply(
            lambda tokens: tokens + [t for t in bigram[tokens] if '_' in t]
        )

    # =========================
    # TRIGRAMAS
    # =========================
    def __aplicar_trigramas(self, data):
        col_txt = self.argsClustering["textClustering"] + "_Clustering"

        bigram = Phrases(data[col_txt], min_count=20)
        data[col_txt] = data[col_txt].apply(
            lambda tokens: tokens + [t for t in bigram[tokens] if '_' in t]
        )

        trigram = Phrases(data[col_txt], min_count=20)
        data[col_txt] = data[col_txt].apply(
            lambda tokens: tokens + [t for t in trigram[tokens] if '_' in t]
        )

    # =========================
    # REPORTES
    # =========================
    @staticmethod
    def __generar_reportes(
        modo, model,
        x_vals, umass_y, cv_y,
        best_score, topics, passes, iters,
        folder, corpus_bow, data,
        col_review_id="reviewId"
    ):

        for metric, y_vals in [('u_mass', umass_y), ('c_v', cv_y)]:
            plt.figure()
            plt.plot(x_vals, y_vals, marker='o')
            plt.title(f'Coherencia {metric} vs Tópicos ({modo.upper()})')
            plt.xlabel("Número de tópicos (K)")
            plt.ylabel(f"Coherencia {metric}")
            plt.savefig(f"{folder}/coherence_{metric}_{modo}.png")
            plt.close()

        df_temp = data.copy().reset_index(drop=True)

        topicos_principales = []
        probabilidades = []

        for idx in range(len(corpus_bow)):
            topics_doc = model.get_document_topics(corpus_bow[idx])

            if topics_doc:
                topico, prob = max(topics_doc, key=lambda x: x[1])
                topicos_principales.append(topico)
                probabilidades.append(round(prob, 4))
            else:
                topicos_principales.append(-1)
                probabilidades.append(0.0)

        df_temp['dominante_id'] = topicos_principales
        df_temp['confianza'] = probabilidades

        with open(f"{folder}/topics_{modo}.txt", "w", encoding="utf-8") as f:
            f.write(
                f"Mejor Coherencia: {best_score:.4f}\n"
                f"Params: Topics={topics}, Passes={passes}, Iters={iters}\n\n"
            )

            for i in range(topics):
                f.write(f"TOPIC {i}\n")

                words = model.show_topic(i, topn=10)
                f.write(
                    f"\nPalabras clave:\n\t"
                    f"{', '.join([f'{w} ({p:.4f})' for w, p in words])}\n"
                )

                top_docs = df_temp[df_temp['dominante_id'] == i] \
                    .sort_values(by='confianza', ascending=False).head(10)

                f.write("\nReviews representativas:\n")

                if not top_docs.empty:
                    for idx, (_, row) in enumerate(top_docs.iterrows(), 1):
                        f.write(f"\t{idx}: {row[col_review_id]} (conf={row['confianza']:.4f})\n")
                        f.write(f"\t\t{row['content']}\n\n")
                else:
                    f.write("  - Sin documentos\n")

                f.write("\n" + "-" * 80 + "\n\n")

        model.save(f"{folder}/{modo}_model")

    # =========================
    # ENTRENAMIENTO CLUSTER
    # =========================
    def __ejecutar_cluster(self, modo, cluster, data, safe_folder):
        try:
            args = self.argsClustering
            col_txt = args["textClustering"] + "_Clustering"

            self.__aplicar_bigramas(data)
            self.__aplicar_trigramas(data)

            dictionary, corpus_bow, corpus_model = self.__preparar_corpus(data)

            best_score = -999999
            best_model = None
            best_params = None

            umass_values, cv_values, x_vals = [], [], []

            conf = args[modo]
            total = len(conf["num_topics"]) * len(conf["passes"]) * len(conf["iterations"])

            csv_path = f"{safe_folder}/clustering_results_{modo}.csv"

            with tqdm(total=total, desc=f"Entrenando {modo}") as pbar:
                with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
                    writer = csv.DictWriter(
                        fcsv,
                        fieldnames=["Num Topics", "Passes", "Iterations", "u_mass", "c_v"]
                    )
                    writer.writeheader()

                    for k in conf["num_topics"]:
                        for p in conf["passes"]:
                            for it in conf["iterations"]:

                                params = {
                                    "corpus": corpus_model,
                                    "id2word": dictionary,
                                    "num_topics": int(k),
                                    "passes": int(p),
                                    "random_state": 42
                                }

                                if modo == "lda":
                                    params.update({
                                        "alpha": "symmetric",
                                        "eta": "symmetric",
                                        "iterations": int(it)
                                    })

                                model = cluster(**params)

                                umass = CoherenceModel(
                                    model=model,
                                    corpus=corpus_bow,
                                    dictionary=dictionary,
                                    coherence="u_mass"
                                ).get_coherence()

                                cv = CoherenceModel(
                                    model=model,
                                    texts=data[col_txt],
                                    dictionary=dictionary,
                                    coherence="c_v"
                                ).get_coherence()

                                writer.writerow({
                                    "Num Topics": k,
                                    "Passes": p,
                                    "Iterations": it,
                                    "u_mass": umass,
                                    "c_v": cv
                                })

                                if umass > best_score:
                                    best_score = umass
                                    best_model = model
                                    best_params = (k, p, it)

                                umass_values.append(umass)
                                cv_values.append(cv)
                                x_vals.append(k)

                                pbar.update(1)

            self.__generar_reportes(
                modo,
                best_model,
                x_vals,
                umass_values,
                cv_values,
                best_score,
                best_params[0],
                best_params[1],
                best_params[2],
                safe_folder,
                corpus_bow,
                data,
                "reviewId"
            )

        except Exception as e:
            raise RuntimeError(f"Error en clustering modo={modo}") from e

    # =========================
    # MAIN
    # =========================
    def ejecutar_clustering(self, data):
        try:
            clusters = {
                "lda": LdaModel,
                "nmf": Nmf
            }

            modo = self.argsClustering["cluster"]

            if modo not in clusters:
                print(Fore.RED + "Algoritmo no soportado." + Fore.RESET)
                return

            cluster = clusters[modo]

            for sentimiento, df in data.items():

                if df.empty:
                    print(Fore.YELLOW + f"Sin datos: {sentimiento}" + Fore.RESET)
                    continue

                print(Fore.CYAN + f"\n=== {sentimiento.upper()} ===" + Fore.RESET)

                folder = f"./output/sentiment_analysis/{sentimiento}"
                os.makedirs(folder, exist_ok=True)

                self.__ejecutar_cluster(modo, cluster, df, folder)

            print(Fore.GREEN + "Clustering completado" + Fore.RESET)

        except Exception as e:
            raise RuntimeError("Error en ejecución de clustering") from e