import os
import pandas as pd
import numpy as np
import string
from colorama import Fore

import unicodedata
import pickle
# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
# Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
# Imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
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

        # Descargamos los recursos necesarios de nltk
        print("\n- Descargando diccionarios...")
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        print(Fore.GREEN + "Diccionarios descargados con éxito" + Fore.RESET)

    @staticmethod
    def __load_data(file):
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
            for col in data.select_dtypes(include='object'):
                data[col] = data[col].str.strip().replace('', np.nan)
            print(Fore.GREEN + "Datos cargados con éxito" + Fore.RESET)
            return data
        except Exception as e:
            raise RuntimeError(f"Error al cargar los datos desde '{file}'") from e

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

            test_size = args.test_size
            dev_size = args.dev_size
            dev_size = dev_size / (1.0 - test_size) #Para obtener el Dev proporcional

            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

            X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=dev_size, stratify=y_temp, random_state=42)

            print(Fore.GREEN + "\nTrain/Dev divididos con éxito" + Fore.RESET)
            return X_train, y_train, X_dev, y_dev, X_test, y_test
        except Exception as e:
            raise RuntimeError("Error al dividir Train/Dev/Test") from e

    #region Funciones para el PREPROCESADO

    def __select_features(self, data):
        """
        Separa las características del conjunto de datos en características numéricas, de texto y categóricas.

        :param data: El conjunto de datos del que se sacarán sus features.
        :type data: pandas.DataFrame
        :return: Una tupla de 3 elementos:
                (numerical_feature, text_feature, categorical_feature)
        :rtype: tuple
        """
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
            raise RuntimeError("Error al procesar missing values") from e

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
            raise RuntimeError("Error al reescalar datos") from e

    def __encode_target(self, data, target_col, is_Train):
        """
        Función para codificar la variable objetivo (y) de forma aislada.
        """
        if is_Train:
            self.tools['target_encoder'] = LabelEncoder()
            data[target_col] = self.tools['target_encoder'].fit_transform(data[target_col])
        else:
            data[target_col] = self.tools['target_encoder'].transform(data[target_col])
        return data

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
        try:
            args = self.args
            print("\n- Realizando Label Encoding...")

            if not categorical_feature.empty:
                target_col = args.prediction
                modo = args.preprocessing["cat2num"]

                if is_Train:
                    self.tools['cat_encoder'] = {}

                for col in tqdm(categorical_feature.columns, desc="Discretizando datos categoriales"):
                    # Caso extremo: si la columna es el TARGET
                    if col == target_col:
                        data = self.__encode_target(data, col, is_Train)
                        continue # Para que se salte el OneHot/Ordinal

                    if is_Train:
                        if modo == "ordinal":
                            # handle_unknown='use_encoded_value' le asigna -1 a las categorías nuevas (LabelEncoder petaba)
                            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                        else: #elif modo == "oneHot"
                            # handle_unknown='ignore' pone a 0 las desconocidas
                            # sparse_output=False para que no devuelva una matriz comprimida rara
                            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                        resultado = encoder.fit_transform(data[[col]])
                        self.tools['cat_encoder'][col] = encoder
                    else:
                        encoder = self.tools['cat_encoder'][col]
                        resultado = encoder.transform(data[[col]])

                    if modo == "oneHot":
                        col_names = encoder.get_feature_names_out([col])
                        df_ohe = pd.DataFrame(resultado, columns=col_names, index=data.index)
                        data = pd.concat([data.drop(columns=[col]), df_ohe], axis=1)
                    else:
                        # Si es ordinal, simplemente sobreescribimos la columna
                        data[col] = resultado

                print(Fore.GREEN + f"Discretización {modo} realizada con éxito" + Fore.RESET)
            else:
                print(Fore.YELLOW + "No se está realizando discretización" + Fore.RESET)
            return data
        except Exception as e:
            raise RuntimeError("Error en discretización (cat2num)") from e

    # Métodos simplificado texto Clustering
    @staticmethod
    def __normalize_text(text):
        """
        Elimina tildes y caracteres especiales normalizando el formato Unicode.
        """
        text = str(text)
        text = unicodedata.normalize("NFKD", text)
        return text

    def __tokenize_text_to_list(self, data):
        """
        Convierte una columna de texto en una lista de tokens limpios.

        :param data: El conjunto de datos a procesar.
        :type data: pandas.DataFrame
        :return: DataFrame con la columna de texto transformada en listas de tokens.
        """
        try:
            print("\n- Preprocesando texto a lista de tokens...")

            # Por comodidad
            columna_texto = self.args.clustering["textClustering"]

            # Eliminamos filas donde el texto sea nulo
            data = data.dropna(subset=[columna_texto]).copy()

            tokenizer = RegexpTokenizer(r'\w+')
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words("english"))

            processed_docs = []

            for doc in tqdm(data[columna_texto], desc=f"Tokenizando '{columna_texto}'"):
                # Normalizamos y pasamos a minúsculas
                doc = self.__normalize_text(doc).lower()
                tokens = tokenizer.tokenize(doc)

                # Limpieza y Lematización
                clean_tokens = [
                    lemmatizer.lemmatize(token)
                    for token in tokens
                    if token not in stop_words and token not in string.punctuation
                ]

                processed_docs.append(clean_tokens)

            # Sobrescribimos la columna original con las listas de tokens
            columna_texto_modificada = columna_texto + "_Clustering"
            data[columna_texto_modificada] = processed_docs

            print(Fore.GREEN + "Datos tokenizados con éxito" + Fore.RESET)
            return data

        except Exception as e:
            raise RuntimeError("Error al tokenizar texto") from e

    @staticmethod
    def __clean_text_row(text, lemmatizer, stop_words):
        # Limpieza para una sola fila
        tokens = word_tokenize(str(text).lower())
        # Filtramos y aplicamos stemmer sin ordenar alfabéticamente
        cleaned = [lemmatizer(t) for t in tokens
                   if t not in stop_words and t.isalpha()]
        return " ".join(cleaned)

    def __simplify_text(self, data, text_feature):
        """
        Función que simplifica el texto de una columna dada en un DataFrame.
            1. Tokenizado
            2. Minúsculas
            3. Lematizar
            4. Eliminar stop words
            5. Eliminar signos de puntuación
            6. Ordenar alfabéticamente <-- ESTE NO

        :param data: El conjunto de datos que se somete al simplificado del texto.
        :type data: pandas.DataFrame
        :param text_feature: El conjunto de features de tipo texto.
        :type text_feature: pandas.DataFrame
        :return: El conjunto de datos original con las columnas de texto simplificadas (o no).
        :rtype: pandas.DataFrame
        """
        try:
            args = self.args
            print("\n- Simplificando el texto...")
            if not text_feature.empty:
                stop_words = set(stopwords.words('english'))
                lemmatizers = {
                    "stem": PorterStemmer().stem,
                    "lem": WordNetLemmatizer().lemmatize
                }
                modo = args.preprocessing["lemmatization"]

                if modo in lemmatizers:
                    self.tools['lemmatizer'] = lemmatizers[modo]

                    for col in tqdm(text_feature.columns, desc="Simplificando texto"):
                        # Usamos apply para procesar toda la columna de forma optimizada
                        data[col] = data[col].apply(lambda x: self.__clean_text_row(x, self.tools['lemmatizer'], stop_words))

                    print(Fore.GREEN + "Texto simplificado con éxito" + Fore.RESET)
            return data
        except Exception as e:
            raise RuntimeError("Error al simplificar el texto") from e

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
                    "tf-idf": TfidfVectorizer(max_features=5000, min_df=5), #TODO diccionario configurable
                    "bow": CountVectorizer() #TODO meter configuraciones extra
                }
                modo = args.preprocessing["text_process"]
                if modo in vectorizers:
                    text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                    if is_Train:
                        self.tools['vectorizer'] = vectorizers[modo]
                        matrix = self.tools['vectorizer'].fit_transform(text_data)
                    else:
                        matrix = self.tools['vectorizer'].transform(text_data)
                    text_features_df = pd.DataFrame(
                        matrix.toarray(),
                        columns=self.tools['vectorizer'].get_feature_names_out(),
                        index=data.index
                    )
                    data = pd.concat([data, text_features_df], axis=1)
                    data.drop(columns=text_feature.columns, inplace=True)
                    print(Fore.GREEN + f"Texto tratado usando {modo.upper()} con éxito" + Fore.RESET)
                else:
                    print(Fore.YELLOW + "No se están tratando los textos (modo no reconocido)" + Fore.RESET)
            else:
                print(Fore.YELLOW + "No se han encontrado columnas de texto a procesar" + Fore.RESET)
            return data
        except Exception as e:
            raise RuntimeError("Error al procesar el texto") from e

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
                "oversampling": RandomOverSampler,
                "smote": SMOTE
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
            raise RuntimeError("Error al balancear") from e

    def __drop_features(self, data):
        """
        Elimina las columnas especificadas del conjunto de datos.

        :param data: El conjunto de datos que se somete a la eliminación de features.
        :type data: pd.DataFrame
        :return: El conjunto de datos sin las columnas especificadas.
        :rtype: pd.DataFrame
        """
        args = self.args
        print("\n- Eliminando columnas...")
        data = data.drop(columns=args.preprocessing["drop_features"], errors='ignore')
        print(Fore.GREEN + "Columnas eliminadas con éxito" + Fore.RESET)
        return data

    def __actualizar_columnas_especiales(self, data): # Es muy mejorable, pero para este proyecto se qeuda así de momento
        try:
            if self.args.preprocessing["cols_especiales"]:
                if "date" in data: # Guardamos solo el año
                    data['date'] = pd.to_datetime(data['date'], errors='coerce')
                    data['date'] = data['date'].dt.year
                if "location" in data: # Guardamos solo el País
                    data['location'] = data['location'].str.split(",").str[1].str.strip()
            return data
        except Exception as e:
            print(e)

    #endregion

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

        # Columna de fecha y localización
        data = self.__actualizar_columnas_especiales(data)

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

    #region Funciones para el GUARDADO de datos

    @staticmethod
    def __save_debug_data(X_train, y_train, X_dev, y_dev, X_test, y_test):
        """
        Exporta los conjuntos de datos procesados a ficheros CSV.
        Detecta automáticamente si se trata de un flujo de Test o de Train/Dev.

        :param X_train: Características del primer bloque (X_train o X_test).
        :type X_train: pandas.DataFrame
        :param y_train: Objetivo del primer bloque (y_train o y_test).
        :type y_train: pandas.Series
        :param X_dev: Características del segundo bloque (X_dev), opcional.
        :type X_dev: pandas.DataFrame
        :param y_dev: Objetivo del segundo bloque (y_dev), opcional.
        :type y_dev: pandas.Series
        """
        #try: TODO importar el logger etc (NO HACE FALTA)
        if not os.path.exists('./output'):
            os.makedirs('./output')

        # Juntamos para el procesado (evitamos resetear índices)
        train = pd.concat([X_train, y_train], axis=1)
        dev = pd.concat([X_dev, y_dev], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        print(Fore.MAGENTA + "- [Debug] Exportando procesado de Train y Dev..." + Fore.RESET)
        train.to_csv('output/1-train-processed.csv', index=False)
        dev.to_csv('output/2-dev-processed.csv', index=False)

        print(Fore.MAGENTA + "- [Debug] Exportando procesado de Test..." + Fore.RESET)
        test.to_csv('output/3-test-processed.csv', index=False) # type: ignore para que no de error

        print(Fore.GREEN + "\tArchivos de debug guardados con éxito" + Fore.RESET)
        #except Exception as e:
            #logging.warning(f"No se pudo guardar debug; {e}")

    def __save_tools(self):
        print("- Guardando herramientas de preprocesado...")
        with open('./output/preprocessor_tools.pkl', 'wb') as f:
            pickle.dump(self.tools, f)
        print(Fore.GREEN + "Tools guardadas con éxito" + Fore.RESET)

    #endregion

    @staticmethod
    def juntar_data(X_train, y_train, X_dev, y_dev, X_test, y_test):
        train = pd.concat([X_train, y_train], axis=1)
        dev = pd.concat([X_dev, y_dev], axis=1)
        test = pd.concat([X_test, y_test], axis=1)
        return train, dev, test

    def __mapear_target(self, data):
        args = self.args

        # Definimos el diccionario de mapeo
        mapeo = {
            1: 'negativo',
            2: 'negativo',
            3: 'neutro',
            4: 'positivo',
            5: 'positivo'
        }

        # Aplicamos el mapeo a la columna objetivo
        data[args.prediction] = data[args.prediction].map(mapeo)

        # Eliminamos filas que hayan quedado vacías si el rating original no era 1-5
        data = data.dropna(subset=[args.prediction])

        return data

    def preprocesar_datos_clasificador(self):
        """
        Maneja el flujo principal de carga y preprocesamiento de los datos del clasificador.
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

        # Mapeamos en positivos, negativos y neutros
        self.__mapear_target(self.df)

        # Divide en Train/Dev/Test
        X_train, y_train, X_dev, y_dev, X_test, y_test = self.__divide_data(self.df)

        # Juntamos para el procesado (evitamos resetear índices)
        train, dev, test = self.juntar_data(X_train, y_train, X_dev, y_dev, X_test, y_test)

        # Procesamos Train, Dev y Test
        X_train, y_train = self.__procesar_bloque(train, True)
        X_dev, y_dev = self.__procesar_bloque(dev, False)
        X_test, y_test = self.__procesar_bloque(test, False)

        # Balanceamos Train (y solo Train)
        X_train, y_train = self.__over_under_sampling(X_train, y_train)

        # Para comprobar el preproceso
        if args.debug:
            self.__save_debug_data(X_train, y_train, X_dev, y_dev, X_test, y_test)

        # Guardamos herramientas del preproceso
        self.__save_tools()

        return X_train, y_train, X_dev, y_dev, X_test, y_test

    def preprocesar_datos_clustering(self):
        """
        Maneja el flujo principal de carga y preprocesamiento de los datos del clustering.
        """
        args = self.args

        self.df_original = self.__load_data(args.file)
        self.df = self.df_original.copy()

        # Mapeamos en positivos, negativos y neutros
        self.__mapear_target(self.df)

        # Eliminar columnas no deseadas si es necesario
        #self.df = self.__drop_features(self.df)

        # Aplicamos prepro de texto
        self.df = self.__tokenize_text_to_list(self.df)

        datos_separados = {
            'positivo': self.df[self.df[args.prediction] == 'positivo'].copy(),
            'negativo': self.df[self.df[args.prediction] == 'negativo'].copy(),
            'neutro': self.df[self.df[args.prediction] == 'neutro'].copy()
        }

        # Para comprobar el preproceso
        if args.debug:
            datos_separados["positivo"].to_csv('output/7-positivos-processed.csv', index=False)  # type: ignore para que no de error
            datos_separados["negativo"].to_csv('output/8-negativos-processed.csv', index=False)  # type: ignore para que no de error
            datos_separados["neutro"].to_csv('output/9-neutros-processed.csv', index=False)  # type: ignore para que no de error

        return datos_separados

    def preprocesar_datos_generativo(self):
        """
                Maneja el flujo principal de carga y preprocesamiento de los datos del generativo.
        """
        args = self.args

        self.df_original = self.__load_data(args.file)
        self.df = self.df_original.copy()

        # Eliminamos valores que estén fuera de 1,2,3,4,5
        self.df = self.df[self.df[args.prediction].astype(str).isin(['1', '2', '3', '4', '5'])]

        # Divide en Train/Dev/Test
        X_train, y_train, X_dev, y_dev, X_test, y_test = self.__divide_data(self.df)

        # Juntamos para el procesado (evitamos resetear índices)
        train, dev, test = self.juntar_data(X_train, y_train, X_dev, y_dev, X_test, y_test)

        return train, dev, test