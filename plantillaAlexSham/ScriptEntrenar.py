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
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from Preprocesador import DataPreprocessor

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

# Funciones para entrenar un modelo

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

def ejecutar_algoritmo():
    try:
        algorithms = {
            "knn": kNN,
            "decision_tree": decision_tree,
            "random_forest": random_forest
        }
        modo = args.algorithm
        if modo in algorithms:
            algoritmo = algorithms[modo]
            algoritmo() #Ejecuta el algoritmo correspondiente
            print(Fore.GREEN + f"Algoritmo {modo} ejecutado con éxito" + Fore.RESET)
            sys.exit(0)
        else:
            print(Fore.RED + "Algoritmo no soportado" + Fore.RESET)
            sys.exit(1)
    except Exception as e:
        print(e)

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
    # Preprocesamos los datos
    print("\n- Preprocesando datos...")
    prepo = DataPreprocessor(args)
    X_train, y_train, X_dev, y_dev = prepo.preprocesar_datos(True)

    # Ejecutamos el algoritmo seleccionado
    print("\n- Ejecutando el algoritmo...")
    ejecutar_algoritmo()