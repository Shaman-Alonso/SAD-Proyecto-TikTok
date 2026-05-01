import pandas as pd
import pickle
import csv

from colorama import Fore
# Sklearn
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

class ModelClassifier:

    def __init__(self, args):
        self.args = args
        self.tools = {}
        self.metricas = {
            'accuracy': 'accuracy',
            'precision': 'precision_macro',
            'recall': 'recall_macro',
            'f1_macro': 'f1_macro',
            'f1_micro': 'f1_micro',
            'f1_weighted': 'f1_weighted'
        }

    #region Funciones para ENTRENAR un modelo

    def __save_model(self, gs):
        """
        Guarda el modelo y los resultados de la búsqueda de hiperparámetros en archivos.

        Parámetros:
        - gs: objeto GridSearchCV, el cual contiene el modelo y los resultados de la búsqueda de hiperparámetros.

        Excepciones:
        - Exception: Si ocurre algún error al guardar el modelo.

        """
        try:
            args = self.args

            with open('./output/modelo.pkl', 'wb') as file:
                pickle.dump(gs, file) # type: ignore para que no de error
                print(Fore.CYAN + "Modelo guardado con éxito" + Fore.RESET)
            with open('./output/4-modelo.csv', 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['Combinación', 'Precisión', 'Recall', 'F1_Macro', 'F1_Micro', 'F1_Weighted'])

                rdo_df = pd.DataFrame(gs.cv_results_)

                if args.algorithm == "knn":
                    rdo_df['Combinación'] = rdo_df.apply(
                        lambda row: f"kNN_K{row['param_n_neighbors']}_P{row['param_p']}_{row['param_weights']}", axis=1)

                elif args.algorithm == "decision_tree":
                    rdo_df['Combinación'] = rdo_df.apply(lambda
                                                             row: f"DT_{row['param_criterion']}_D{row['param_max_depth']}_S{row['param_min_samples_split']}_L{row['param_min_samples_leaf']}",
                                                         axis=1)

                elif args.algorithm == "random_forest":
                    rdo_df['Combinación'] = rdo_df.apply(lambda
                                                             row: f"RF_N{row['param_n_estimators']}_D{row['param_max_depth']}_S{row['param_min_samples_split']}_L{row['param_min_samples_leaf']}_B{row['param_bootstrap']}",
                                                         axis=1)

                elif args.algorithm == "naive_bayes":
                    rdo_df['Combinación'] = rdo_df.apply(
                        lambda row: f"NB_Alpha{row['param_alpha']}_Prior{row['param_fit_prior']}", axis=1)

                elif args.algorithm == "logistic_regression":
                    rdo_df['Combinación'] = rdo_df.apply(
                        lambda row: f"LR_C{row['param_C']}_L1ratio{row['param_l1_ratio']}_Solver{row['param_solver']}_MaxIter{row['param_max_iter']}", axis=1)

                cols_salida = ['Combinación', 'mean_test_precision', 'mean_test_recall', 'mean_test_f1_macro',
                               'mean_test_f1_micro', 'mean_test_f1_weighted']

                for fila in rdo_df[cols_salida].itertuples(index=False):
                    writer.writerow(
                        [fila[0], round(fila[1], 4), round(fila[2], 4), round(fila[3], 4), round(fila[4], 4),
                         round(fila[5], 4)])
                return rdo_df
        except Exception as e:
            raise RuntimeError("Error al guardar el modelo") from e

    #region Funciones para calcular métricas

    @staticmethod
    def __calculate_fscore(y_dev, y_pred):
        """
           Función para calcular el F-score
           :param y_dev: Valores reales
           :param y_pred: Valores predichos
           :return: F-score (micro), F-score (macro)
           """
        fscore_micro = f1_score(y_dev, y_pred, average='micro', zero_division=0)
        fscore_macro = f1_score(y_dev, y_pred, average='macro', zero_division=0)
        return fscore_micro, fscore_macro

    @staticmethod
    def __calculate_classification_report(y_dev, y_pred):
        """
           Función para calcular el informe de clasificación
           :param y_dev: Valores reales
           :param y_pred: Valores predichos
           :return: Informe de clasificación
           """
        report = classification_report(y_dev, y_pred, zero_division=0)
        return report

    @staticmethod
    def __calculate_confusion_matrix(y_dev, y_pred):
        """
            Función para calcular la matriz de confusión
            :param y_dev: Valores reales
            :param y_pred: Valores predichos
            :return: Matriz de confusión
            """
        cm = confusion_matrix(y_dev, y_pred)
        return cm

    #endregion

    def __mostrar_resultados(self, gs, x_dev, y_dev, rdo_df): #TODO rdo_df es para el plot_metricas, se puede quitar
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
        args = self.args

        if args.verbose:
            y_pred = gs.predict(x_dev)
            print(Fore.MAGENTA + "> Mejores parametros:\n" + Fore.RESET, gs.best_params_)
            print(Fore.MAGENTA + "> Mejor puntuacion:\n" + Fore.RESET, gs.best_score_)

            # F1-score
            fscore_micro, fscore_macro = self.__calculate_fscore(y_dev, y_pred)
            print(Fore.MAGENTA + "> F1-score micro:\n" + Fore.RESET, fscore_micro)
            print(Fore.MAGENTA + "> F1-score macro:\n" + Fore.RESET, fscore_macro)

            # Informe de clasificación
            cr = self.__calculate_classification_report(y_dev, y_pred)
            print(Fore.MAGENTA + "> Informe de clasificación:\n" + Fore.RESET, cr)

            # La matriz de confusión
            cm = self.__calculate_confusion_matrix(y_dev, y_pred)
            print(Fore.MAGENTA + "> Matriz de confusión:\n" + Fore.RESET, cm)

            #self.__plot_metricas(rdo_df, cm, cr) #TODO

    def __calcular_intervalo(self, intConf):
        args = self.args
        if intConf[0] == -1:  # Modo intervalo
            limInf = intConf[1]  # Límite inferior
            limSup = intConf[2] + 1  # Límite superior
            step = intConf[3]  # Salto
            if args.algorithm == "knn" and not args.knn_config["pares"]:  # Si es knn y NO queremos pares
                if limInf % 2 == 0:  # Si es par
                    limInf += 1  # Para evitar pares inferiores
                step = step * 2  # Saltos solo a impares
            return [i for i in range(limInf, limSup, step)]  # Construir el intervalo
        else:  # NO modo intervalo
            return intConf

    def __get_config(self, modo):
        args = self.args

        algoritmo_config = {}
        if modo == "knn":
            config = args.knn_config

            algoritmo_config = {
                "n_neighbors": self.__calcular_intervalo(config["k"]),
                "p": config["p"],
                "weights": config["weights"]
            }
        elif modo == "decision_tree":
            config = args.decision_tree_config

            algoritmo_config = {
                "criterion": config["criterion"],  # criterio de división (gini o entropy)
                "max_depth": self.__calcular_intervalo(config["max_depth"]),  # hasta qué profundidad dejamos crecer el árbol
                "min_samples_split": self.__calcular_intervalo(config["min_samples_split"]),
                # mínimo de muestras para que un nodo se pueda dividir
                "min_samples_leaf": self.__calcular_intervalo(config["min_samples_leaf"]),
                # mínimo de muestras que debe tener una hoja
                "max_features": config["max_features"]  # cuántas features puede usar para buscar el mejor split
            }
        elif modo == "random_forest":
            config = args.rf_config

            algoritmo_config = {
                "n_estimators": config["n_estimators"],  # cuántos árboles va a tener el bosque
                "max_depth": self.__calcular_intervalo(config["max_depth"]),  # profundidad máxima de cada árbol
                "min_samples_split": self.__calcular_intervalo(config["min_samples_split"]),
                # mínimo de muestras para poder dividir un nodo
                "min_samples_leaf": self.__calcular_intervalo(config["min_samples_leaf"]),
                # mínimo de muestras que debe tener una hoja
                "bootstrap": config["bootstrap"],  # si usa muestreo bootstrap o no
                "max_features": config["max_features"]  # nº de features consideradas en cada split
            }
        elif modo == "naive_bayes":
            algoritmo_config = {
                "alpha": self.__calcular_intervalo(args.nb_config["alpha"]),
                "fit_prior": args.nb_config["fit_prior"]
            }
        elif modo == "logistic_regression":
            algoritmo_config = {
                "C": self.__calcular_intervalo(args.lr_config["C"]),
                "l1_ratio": args.lr_config["l1_ratio"],
                "solver": args.lr_config["solver"],
                "max_iter": self.__calcular_intervalo(args.lr_config["max_iter"])
            }

        return algoritmo_config

    def __entrenar_algoritmo(self, modo, algoritmo, X_train, y_train, X_dev, y_dev):
        args = self.args

        # Si no son deterministas, establecemos semilla
        if modo == "knn" or modo == "naive_bayes":
            algoritmo_base = algoritmo()
        else:
            algoritmo_base = algoritmo(random_state=42)

        # Definimos los hiperparámetros que va a probar el GridSearch
        algoritmo_config = self.__get_config(modo)
        gs = GridSearchCV(
            algoritmo_base,
            algoritmo_config,
            cv=5,
            n_jobs=args.cpu,
            scoring=self.metricas,
            refit=args.estimator,
            verbose=10
        )

        gs.fit(X_train, y_train)

        # Guardamos el modelo ya entrenado (el mejor que encontró el GridSearch)
        rdo_df = self.__save_model(gs)

        # Mostramos métricas y resultados en el conjunto dev
        self.__mostrar_resultados(gs, X_dev, y_dev, rdo_df)

        return gs

    #endregion

    #region Funciones para EVALUAR un modelo

    @staticmethod
    def __load_model():
        try:
            with open('./output/modelo.pkl', 'rb') as file:
                model = pickle.load(file)
                print(Fore.GREEN + "Modelo cargado con éxito" + Fore.RESET)
                return model
        except Exception as e:
            raise RuntimeError("Error al cargar el modelo") from e

    def __predict(self, model, X_test, y_test):
        args = self.args
        y_pred = model.predict(X_test)

        # Calculamos los resultados
        fmic, fmac = self.__calculate_fscore(y_test, y_pred)
        print(Fore.CYAN + "\n=== RESULTADOS FINALES (TEST) ===" + Fore.RESET)
        print(f"Acierto total (F1-Micro): {round(fmic * 100, 2)}%")
        print(f"F1-Macro: {round(fmac, 4)}")
        print(Fore.CYAN + "=================================\n" + Fore.RESET)

        # Añadimos la prediccion al dataframe data
        test_df = X_test.copy().reset_index(drop=True)
        y_test_df = y_test.reset_index(drop=True)
        pred_df = pd.DataFrame(y_pred, columns=[f"Predict_{args.prediction}"])

        output = pd.concat([test_df, y_test_df, pred_df], axis=1)
        output.to_csv('./output/5-prediction.csv', index=False) # type: ignore para que no de error

        print(Fore.GREEN + "Predicción guardada con éxito" + Fore.RESET)

    def evaluar_algoritmo(self, X_test, y_test):
        print("\n- Prediciendo...")
        modelo = self.__load_model()
        self.__predict(modelo, X_test, y_test)
        print(Fore.GREEN + "Predicción realizada con éxito" + Fore.RESET)

    #endregion

    def ejecutar_algoritmo(self, X_train, y_train, X_dev, y_dev, X_test, y_test):
        try:
            # Por comodidad
            args = self.args

            algorithms = {
                "knn": KNeighborsClassifier,
                "decision_tree": DecisionTreeClassifier,
                "random_forest": RandomForestClassifier,
                "naive_bayes": MultinomialNB,
                "logistic_regression": LogisticRegression
            }
            modo = args.algorithm

            if modo in algorithms:
                algoritmo = algorithms[modo]
                modelo = self.__entrenar_algoritmo(modo, algoritmo, X_train, y_train, X_dev, y_dev)  # Entrena el algoritmo correspondiente
                print(Fore.GREEN + f"Algoritmo {modo} entrenado con éxito" + Fore.RESET)

                #self.__evaluar_algoritmo(modelo, X_test, y_test)
                #print(Fore.GREEN + f"Evaluación realizada con éxito" + Fore.RESET)
            else:
                print(Fore.RED + "Algoritmo no soportado" + Fore.RESET)
        except Exception as e:
            raise RuntimeError("Error al en la ejecución del clasificador") from e