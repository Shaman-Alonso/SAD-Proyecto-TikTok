# -*- coding: utf-8 -*-
"""
Script para la ejecución principal
"""
import sys
import signal
import argparse
import numpy as np
import json
import os
from colorama import Fore

# TODO Para el log de errores (igual me he motivado con esto; ya miraremos)
"""
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
"""

from Clustering import ModelClustering
from Preprocesador import DataPreprocessor
from Clasificador import ModelClassifier

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
    parse.add_argument("-m", "--mode", help="Modo: Train, Test o Cluster", required=True)
    parse.add_argument("-p", "--prediction", help="Columna a predecir (Nombre de la columna)", required=True)
    parse.add_argument("-c", "--cpu", help="Número de CPUs a utilizar [-1 para usar todos]", required=False, default=-1,type=int)
    parse.add_argument("-v", "--verbose", help="Muestra las metricas por la terminal", required=False, default=False,action="store_true")
    parse.add_argument("--debug",help="Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]",required=False, default=False, action="store_true")
    parse.add_argument("--config", help="Archivo JSON de configuración", required=False, default='clasificador.json')

    # Parseamos los argumentos
    parsed_args = parse.parse_args()

    # Leemos los parametros del JSON
    with open(parsed_args.config) as json_file:
        config = json.load(json_file)

    # Juntamos lo anterior en una variable
    for key, value in config.items():
        setattr(parsed_args, key, value)

    # Parseamos los argumentos
    return parsed_args

# Función principal

if __name__ == "__main__":
    # Fijamos la semilla
    np.random.seed(42)

    print("=== Clasificador ===")

    # Manejamos la señal SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Parseamos los argumentos
        args = parse_args()

        # Si la carpeta output no existe la creamos
        print("\n- Creando carpeta output...")
        os.makedirs('./output', exist_ok=True)
        print(Fore.GREEN + "Carpeta output creada con éxito" + Fore.RESET)

        # Preprocesamos los datos
        print("\n- Preprocesando datos...")
        prepro = DataPreprocessor(args)
        if args.mode == "Train":
            X_train, y_train, X_dev, y_dev, X_test, y_test = prepro.preprocesar_datos_clasificador()

            # Ejecutamos el algoritmo seleccionado
            print("\n- Ejecutando el algoritmo...")
            modelo = ModelClassifier(args)
            modelo.ejecutar_algoritmo(X_train, y_train, X_dev, y_dev, X_test, y_test)
        elif args.mode == "Cluster":
            # Preparamos la data para el Clustering
            print("\n- Realizando clustering...")
            data_clustering = prepro.preprocesar_datos_clustering()

            cluster = ModelClustering(args)
            cluster.ejecutar_clustering(data_clustering)
        elif args.mode == "Test":
            X_test, y_test = prepro.preprocesar_soloTest()

            # Ejecutamos el algoritmo seleccionado
            print("\n- Ejecutando el algoritmo...")
            modelo = ModelClassifier(args)
            modelo.evaluar_algoritmo(X_test, y_test)
        else:
            print(Fore.RED + "Modo no soportado. Escribe: Train, Test o Cluster" + Fore.RESET)

        sys.exit(0)
    except Exception as e:
        print("Error: " + e)
        #logging.exception("Error durante la ejecución")
        sys.exit(1)