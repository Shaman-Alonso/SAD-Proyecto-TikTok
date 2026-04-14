import sys
import os
import pandas as pd
import numpy as np
import string
from colorama import Fore
# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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


class DataPreprocessor:
    """
    Clase encargada de la limpieza, transformación y balanceo de datos.

    :ivar df_original: Almacena el DataFrame tal cual se carga del archivo CSV.
    :ivar df: DataFrame de trabajo sobre el que se aplican las transformaciones.
    :ivar args: Objeto de configuración con parámetros de preprocesamiento.
    :ivar tools: Diccionario para almacenar objetos ajustados (Scalers, Imputers, Vectorizers).
    """

    def __init__(self, args):
        self.df_original = None
        self.df = None
        self.args = args
        self.tools = {}

    def __load_data(self, file):
        """
        Función para cargar los datos de un fichero CSV

        :param file: Dirección al fichero a leer.
        :type file: str
        :return: El conjunto de datos original.
        :rtype: pandas.DataFrame
        """
        try:
            print("\n- Cargando datos...")
            data = pd.read_csv(file, encoding='utf-8')
            # Para quitar espacios innecesarios en los features y convertir en NaN los vacíos (útil para debugging y también ver el csv más bonitillo)
            data.columns = data.columns.str.strip()
            data = data.map(lambda x: x.strip() if isinstance(x, str) else x).replace(r'^\s*$', np.nan, regex=True)
            print(Fore.GREEN + "Datos cargados con éxito" + Fore.RESET)
            return data
        except Exception as e:
            print(Fore.RED + "Error al cargar los datos" + Fore.RESET)
            print(e)
            sys.exit(1)

    def __select_features(self, data):
        """
        Separa las características del conjunto de datos en características numéricas, de texto y categóricas.

        :param data: El conjunto de datos del que se sacarán sus features.
        :type data: pandas.DataFrame
        :return: Una tupla de 3 elementos:
                (numerical_feature, text_feature, categorical_feature)
        :rtype: tuple
        """
        try:
            args = self.args
            # Numerical features
            numerical_feature = data.select_dtypes(include=['int64', 'float64'])  # Columnas numéricas
            if args.prediction in numerical_feature.columns:
                numerical_feature = numerical_feature.drop(columns=[args.prediction])
            # Categorical features
            categorical_feature = data.select_dtypes(include=['object', 'string'])
            categorical_feature = categorical_feature.loc[
                :, categorical_feature.nunique() <= args.preprocessing["unique_category_threshold"]]

            # Text features
            text_feature = data.select_dtypes(include='object').drop(columns=categorical_feature.columns, errors='ignore')

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

    def __process_missing_values(self, data, numerical_feature, categorical_feature, is_Train):
        """
        Procesa los valores faltantes en los datos según la estrategia especificada en los argumentos.

        :param data: El conjunto de datos que se somete al procesamiento de valores faltantes.
        :type data: pandas.DataFrame
        :param numerical_feature: El conjunto de features de tipo numérico.
        :type numerical_feature: pandas.DataFrame
        :param categorical_feature: El conjunto de features de tipo texto.
        :type categorical_feature: pandas.DataFrame
        :param is_Train: Indica si el bloque corresponde al Train. Evita Data Leakage.
        :type is_Train: bool
        :return: El conjunto de datos original con los valores faltantos tratados (o no).
        :rtype: pandas.DataFrame
        """
        try:
            args = self.args
            print("\n- Procesando missing values...")
            if args.debug:
                print(f"{Fore.MAGENTA}> Missing values detectados:\n{data.isna().sum().to_string()}{Fore.RESET}")

            modo = args.preprocessing["missing_values"]

            if modo == "impute":
                modo_impute = args.preprocessing["impute_strategy"]

                if not numerical_feature.empty:  # Para evitar errores
                    cols_num = numerical_feature.columns
                    if is_Train:
                        self.tools['imputer_num'] = SimpleImputer(strategy=modo_impute) #Guardamos el imputer en el modo elegido
                        data[cols_num] = self.tools['imputer_num'].fit_transform(data[cols_num])
                    else:
                        data[cols_num] = self.tools['imputer_num'].transform(data[cols_num])

                if not categorical_feature.empty:  # Siempre imputará el más frecuente
                    cols_cat = categorical_feature.columns
                    if is_Train:
                        self.tools['imputer_cat'] = SimpleImputer(strategy='most_frequent') #Guardamos el imputer en modo=moda
                        data[cols_cat] = self.tools['imputer_cat'].fit_transform(data[cols_cat])
                    else:
                        data[cols_cat] = self.tools['imputer_cat'].transform(data[cols_cat])
                print(Fore.GREEN + f"Missing values imputados con la estrategia {modo_impute} con éxito" + Fore.RESET)

            elif modo == "delete":
                data = data.dropna(subset=list(numerical_feature.columns) + list(categorical_feature.columns))
                print(Fore.GREEN + "Missing values ELIMINADOS con éxito" + Fore.RESET)

            else:
                print(Fore.YELLOW + "No se están procesando los missing values" + Fore.RESET)

            return data
        except Exception as e:
            print(Fore.RED + "Error al tratar missing values" + Fore.RESET)
            print(e)
            sys.exit(1)

    def __reescaler(self, data, numerical_feature, is_Train):
        """
        Rescala las características numéricas en el conjunto de datos utilizando diferentes métodos de escala.

        :param data: El conjunto de datos que se somete al reescalado de valores numéricos.
        :type data: pandas.DataFrame
        :param numerical_feature: El conjunto de features de tipo numérico.
        :type numerical_feature: pandas.DataFrame
        :param is_Train: Indica si el bloque corresponde al Train. Evita Data Leakage.
        :type is_Train: bool
        :return: El conjunto de datos original con las columnas numéricas reescaladas (o no).
        :rtype: pandas.DataFrame
        """
        try:
            args = self.args
            print("\n- Reescalando datos...")
            # Filtramos el preprocesado elegido
            if not numerical_feature.empty:
                # Diccionario para evitar repetición de muchos ifs y mejorar la escalabilidad (estoy loco, pero qué bonito queda)
                scaling = {
                    "maxAbs": MaxAbsScaler,
                    "minMax": MinMaxScaler,
                    "normalizer": Normalizer,
                    "standard": StandardScaler
                }
                modo = args.preprocessing["scaling"]
                if modo in scaling:
                    cols_num = numerical_feature.columns
                    # Aprovechando las clases de la librería sklearn escala
                    if is_Train:
                        self.tools['scaler'] = scaling[modo]() #Guardamos el Scaler en el modo elegido
                        data[cols_num] = self.tools['scaler'].fit_transform(data[cols_num])
                    else:
                        data[cols_num] = self.tools['scaler'].transform(data[cols_num])
                    print(Fore.GREEN + "Datos escalados con éxito" + Fore.RESET)
                else:
                    print(Fore.YELLOW + "No se están escalando los datos" + Fore.RESET)
            else:
                print(Fore.YELLOW + "No hay datos a escalar" + Fore.RESET)
            return data
        except Exception as e:
            print(Fore.RED + "Error al reescalar los datos" + Fore.RESET)
            print(e)
            sys.exit(1)

    def __cat2num(self, data, categorical_feature, is_Train):
        """
        Convierte las características categóricas en características numéricas utilizando la codificación de etiquetas.

        :param data: El conjunto de datos que se somete a la discretización.
        :type data: pandas.DataFrame
        :param categorical_feature: El conjunto de features de tipo texto.
        :type categorical_feature: pandas.DataFrame
        :param is_Train: Indica si el bloque corresponde al Train. Evita Data Leakage.
        :type is_Train: bool
        :return: El conjunto de datos original con las columnas categoriales discretizadas (o no).
        :rtype: pandas.DataFrame
        """
        try: #TODO LabelEncoder es más para DT, pero OneHot para KNN, revisar muy mucho
            print("\n- Realizando Label Encoding...")
            if not categorical_feature.empty:
                if is_Train and 'label_encoder' not in self.tools: #Para evitar errores, lo inicializamos una vez
                    self.tools['label_encoder'] = {}

                for col in categorical_feature.columns:
                    if is_Train:
                        self.tools['label_encoder'][col] = LabelEncoder()  # Guardamos el LabelEncoder de la columna
                        data[col] = self.tools['label_encoder'][col].fit_transform(data[col])
                    else:
                        data[col] = self.tools['label_encoder'][col].transform(data[col])
                print(Fore.GREEN + "Label Encoding realizado con éxito" + Fore.RESET)
            else:
                print(Fore.YELLOW + "No se está realizando Label Encoding" + Fore.RESET)
            return data
        except Exception as e:
            print(Fore.RED + "Error al realizar Label Encoding" + Fore.RESET)
            print(e)
            sys.exit(1)

    def __simplify_text(self, data, text_feature):
        """
        Función que simplifica el texto de una columna dada en un DataFrame.
            1. Tokenizado
            2. Minúsculas
            3. Lematizar
            4. Eliminar stop words
            5. Eliminar signos de puntuación
            6. Ordenar alfabéticamente

        :param data: El conjunto de datos que se somete al simplificado del texto.
        :type data: pandas.DataFrame
        :param text_feature: El conjunto de features de tipo texto.
        :type text_feature: pandas.DataFrame
        :return: El conjunto de datos original con las columnas de texto simplificadas (o no).
        :rtype: pandas.DataFrame
        """
        try:
            print("\n- Simplificando el texto...")
            if not text_feature.empty:
                stop_words = set(stopwords.words('english'))
                stemmer = PorterStemmer()
                for col in text_feature.columns:  # Por si hubiera varias
                    processed = []
                    for text in tqdm(data[col], desc=f"Procesando la columna {col}:"):
                        tokens = word_tokenize(str(text).lower())  # Tokenizado + minúsculas
                        tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and t not in set(
                            string.punctuation)]  # Lematizar + stop words + signos de puntuación
                        tokens = sorted(tokens)  # Ordenado
                        processed.append(" ".join(tokens))  # Lo junta para que no lo identifique como un array
                    data[col] = processed

                print(Fore.GREEN + "Texto simplificado con éxito" + Fore.RESET)
            else:
                print(Fore.YELLOW + "No se está simplificando el texto" + Fore.RESET)
            return data
        except Exception as e:
            print(Fore.RED + "Error al simplificar el texto" + Fore.RESET)
            print(e)
            sys.exit(1)

    def __process_text(self, data, text_feature, is_Train):
        """
        Procesa las características de texto utilizando técnicas de vectorización como TF-IDF o BOW.

        :param data: El conjunto de datos que se somete al procesamiento del texto.
        :type data: pandas.DataFrame
        :param text_feature: El conjunto de features de tipo texto.
        :type text_feature: pandas.DataFrame
        :param is_Train: Indica si el bloque corresponde al Train. Evita Data Leakage.
        :type is_Train: bool
        :return: El conjunto de datos original con las columnas de texto sustituidas por las nuevas vectorizadas (o no).
        :rtype: pandas.DataFrame
        """
        try:
            args = self.args
            print("\n- Procesando columnas de texto...")
            if not text_feature.empty:
                vectorizers = {
                    "tf-idf": TfidfVectorizer,
                    "bow": CountVectorizer
                }
                modo = args.preprocessing["text_process"]
                if modo in vectorizers:
                    text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                    if is_Train:
                        self.tools['vectorizer'] = vectorizers[modo]()
                        matrix = self.tools['vectorizer'].fit_transform(text_data)
                    else:
                        matrix = self.tools['vectorizer'].transform(text_data)
                    text_features_df = pd.DataFrame(matrix.toarray(),
                                                    columns=self.tools['vectorizer'].get_feature_names_out(),
                                                    index=data.index)
                    data = pd.concat([data, text_features_df], axis=1)
                    data.drop(columns=text_feature.columns, axis=1, inplace=True)
                    print(Fore.GREEN + f"Texto tratado usando {modo.upper()} con éxito" + Fore.RESET)
                else:
                    print(Fore.YELLOW + "No se están tratando los textos (modo no reconocido)" + Fore.RESET)
            else:
                print(Fore.YELLOW + "No se han encontrado columnas de texto a procesar" + Fore.RESET)
            return data
        except Exception as e:
            print(Fore.RED + "Error al tratar el texto" + Fore.RESET)
            print(e)
            sys.exit(1)

    def __over_under_sampling(self, X_train, y_train):
        """
        Realiza oversampling o undersampling en los datos del Train según la estrategia especificada en args.preprocessing["sampling"].

        :param X_train: Los features del Train.
        :type X_train: pandas.DataFrame
        :param y_train: La columna objetivo del Train.
        :type y_train: pandas.DataFrame
        :return: Una tupla de los datos del Train tras balancearlos (o no).
        :rtype: tuple(pandas.DataFrame, pandas.Series)
        """
        try:
            args = self.args
            print("\n\t- Realizando Over/Under sampling...")
            sampling = {
                "undersampling": RandomUnderSampler,
                "oversampling": RandomOverSampler
            }
            modo = args.preprocessing["sampling"]
            if modo in sampling:
                # Realizamos over/undersampling teniendo en cuenta qué queremos predecir, la y
                sampler = sampling[modo]
                X_train_resampled, y_train_resampled = sampler(random_state=42).fit_resample(X_train, y_train)
                print(Fore.GREEN + f"\tSe ha realizado {modo} con éxito" + Fore.RESET)
                return X_train_resampled, y_train_resampled
            elif modo == "auto":  # Hemos definido un modo automático
                counts = y_train.value_counts()
                ratio_actual = counts.min() / counts.max()  # Calcula el ratio de la clase minoritaria
                if ratio_actual < 0.05:  # Si es inferior al 5% rebalancea
                    if len(y_train) < 10000:  # Si es un dataset "pequeño" (menos de 10000 instancias) oversampling
                        modo_auto = "Oversampling (Dataset pequeño)"
                        sampler = RandomOverSampler
                    else:  # Si es un dataset "grande" undersampling
                        modo_auto = "Undersampling (Dataset grande)"
                        sampler = RandomUnderSampler
                    X_train_resampled, y_train_resampled = sampler(random_state=42).fit_resample(X_train, y_train)
                    print(
                        Fore.CYAN + f"\tModo Auto: Se ha realizado {modo_auto} de los datos por ratio del {ratio_actual:.2%}" + Fore.RESET)
                    return X_train_resampled, y_train_resampled
                else:
                    print(
                        Fore.CYAN + f"\tModo Auto: No se están over_under sampling los datos por ratio del {ratio_actual:.2%}" + Fore.RESET)
                    return X_train, y_train
            else:
                print(Fore.YELLOW + "\tNo se están over_under sampling los datos" + Fore.RESET)
                return X_train, y_train
        except Exception as e:
            print(Fore.RED + "\tError al realizar el over_under sampling" + Fore.RESET)
            print(e)
            sys.exit(1)

    def __drop_features(self, data):
        """
        Elimina las columnas especificadas del conjunto de datos.

        :param data: El conjunto de datos que se somete a la eliminación de features.
        :type data: pd.DataFrame
        :return: El conjunto de datos sin las columnas especificadas.
        :rtype: pd.DataFrame
        """
        try:
            args = self.args
            print("\n- Eliminando columnas...")
            data = data.drop(columns=args.preprocessing["drop_features"], errors='ignore')
            print(Fore.GREEN + "Columnas eliminadas con éxito" + Fore.RESET)
            return data
        except Exception as e:
            print(Fore.RED + "Error al eliminar columnas" + Fore.RESET)
            print(e)
            sys.exit(1)

    def __procesar_bloque(self, data, is_Train):
        """
        Función para preprocesar los datos
            1. Borrar columnas no necesarias
            2. Separamos los datos por tipos (Categoriales, numéricos y textos)
            3. Tratamos missing values (Eliminar y imputar)
            4. Pasar los datos de categoriales a numéricos
            5. Reescalamos los datos datos (MinMax, Normalizer, MaxAbsScaler)
            6. Simplificamos el texto (Normalizar, eliminar stopwords, stemming y ordenar alfabéticamente)
            7. Tratamos el texto (TF-IDF, BOW)
            8. Realizamos Oversampling o Undersampling

        :param data: El conjunto de datos que se somete al pipeline de preprocesamiento.
        :type data: pandas.DataFrame
        :param is_Train: Indica si el bloque corresponde al Train. Evita Data Leakage.
        :type is_Train: bool
        :return: Una tupla de los features (X_data) y la columna objetivo (y_data)
        :rtype: tuple(pandas.DataFrame, pandas.Series)
        """
        # Borrar columnas no necesarias
        data = self.__drop_features(data)

        # Separamos los datos por tipos
        numerical_feature, text_feature, categorical_feature = self.__select_features(data)

        # Tratamos missing values
        data = self.__process_missing_values(data, numerical_feature, categorical_feature, is_Train)

        data = data.reset_index(drop=True)

        # Pasar los datos a categoriales a numéricos
        data = self.__cat2num(data, categorical_feature, is_Train)

        # Reescalamos los datos numéricos
        data = self.__reescaler(data, numerical_feature, is_Train)

        # Simplificamos el texto
        data = self.__simplify_text(data, text_feature)

        # Tratamos el texto
        data = self.__process_text(data, text_feature, is_Train)

        X_data = data.drop(columns=self.args.prediction)
        y_data = data[self.args.prediction]
        return X_data, y_data

    # Funciones para entrenar un modelo

    def __divide_data(self, data):
        """
        Función que divide los datos en conjuntos de Train y Dev.

        :param data: El conjunto de datos que se somete a la división Train/Dev
        :type data: pandas.DataFrame
        :return: Una tupla de 4 elementos:
                (X_train, y_train, X_dev, y_dev)
        :rtype: tuple
        """
        # Sacamos la columna a predecir
        try:
            args = self.args
            print("\n- Dividiendo Train/Dev...")
            X = data.drop(columns=[args.prediction])
            y = data[args.prediction]
            X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=float(args.test_size), stratify=y, random_state=42)

            # Realizamos Oversampling o Undersampling
            X_train, y_train = self.__over_under_sampling(X_train, y_train)

            print(Fore.GREEN + "\nTrain/Dev divididos con éxito" + Fore.RESET)
            return X_train, y_train, X_dev, y_dev
        except Exception as e:
            print(Fore.RED + "Error al realizar la división del train/dev" + Fore.RESET)
            print(e)
            sys.exit(1)

    def __save_debug_data(self, X_1, y_1, X_2=None, y_2=None, is_Test=False):
        """
        Exporta los conjuntos de datos procesados a ficheros CSV.
        Detecta automáticamente si se trata de un flujo de Test o de Train/Dev.

        :param X_1: Características del primer bloque (X_train o X_test).
        :type X_1: pandas.DataFrame
        :param y_1: Objetivo del primer bloque (y_train o y_test).
        :type y_1: pandas.Series
        :param X_2: Características del segundo bloque (X_dev), opcional.
        :type X_2: pandas.DataFrame
        :param y_2: Objetivo del segundo bloque (y_dev), opcional.
        :type y_2: pandas.Series
        :param is_Test: Booleano para definir el nombre de los archivos de salida.
        :type is_Test: bool
        """
        try:
            if not os.path.exists('output'):
                os.makedirs('output')

            if is_Test:
                print(Fore.MAGENTA + "- [Debug] Exportando procesado de Test..." + Fore.RESET)
                test = pd.concat([X_1, y_1], axis=1)
                test.to_csv('output/3-test-processed.csv', index=False)
            else:
                print(Fore.MAGENTA + "- [Debug] Exportando procesado de Train y Dev..." + Fore.RESET)
                train = pd.concat([X_1, y_1], axis=1)
                dev = pd.concat([X_2, y_2], axis=1)
                train.to_csv('output/1-train-processed.csv', index=False)
                dev.to_csv('output/2-dev-processed.csv', index=False)

            print(Fore.GREEN + "\tArchivos de debug guardados con éxito" + Fore.RESET)
        except Exception as e:
            print(Fore.RED + f"\tError al guardar archivos de debug: {e}" + Fore.RESET)

    def preprocesar_datos(self, is_TrainDev):
        """
        Maneja el flujo principal de carga y preprocesamiento de los datos.
        :param is_TrainDev: Indica si estamos en Entrenamiento (True) o Evaluación (False)
        :type is_TrainDev: bool
        :return: Si is_TrainDev es True, devuelve una tupla de 4 elementos:
                (X_train, y_train, X_dev, y_dev)
                Si is_TrainDev es False, devuelve una tupla de 2 elementos:
                (X_test, y_test).
        :rtype: tuple
        """

        #Por comodidad, cargamos el df y lo copiamos (más seguro contra Data Leakage)
        args = self.args
        self.df_original = self.__load_data(args.file)
        self.df = self.df_original.copy()

        # Descargamos los recursos necesarios de nltk
        print("\n- Descargando diccionarios...")
        nltk.download('stopwords')
        nltk.download('punkt_tab')
        nltk.download('punkt')
        nltk.download('wordnet')

        #Comprobamos si es ScriptEntrenar o Evaluar
        if is_TrainDev: #TrainDev
            X_train, y_train, X_dev, y_dev = self.__divide_data(self.df) #Divide y balancea Train

            # Juntamos para el procesado (evitamos resetear índices)
            train = pd.concat([X_train, y_train], axis=1)
            dev = pd.concat([X_dev, y_dev], axis=1)

            #Procesamos Train y Dev
            X_train, y_train = self.__procesar_bloque(train, True)
            X_dev, y_dev = self.__procesar_bloque(dev, False)

            #Para comprobar el preproceso
            if args.debug:
                self.__save_debug_data(X_train, y_train, X_dev, y_dev, is_Test=False)

            return X_train, y_train, X_dev, y_dev
        else: #Test
            X_test, y_test = self.__procesar_bloque(self.df, False)

            # Para comprobar el preproceso
            if args.debug:
                self.__save_debug_data(X_test, y_test, is_Test=False)

            return X_test, y_test