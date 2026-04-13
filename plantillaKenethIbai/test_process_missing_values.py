import sys
import pandas as pd
from colorama import Fore
from plantillaPosibleDecisionTreesParaEGela import process_missing_values, select_features, parse_args
import plantillaPosibleDecisionTreesParaEGela

def probar_processor_drop():
    print(f"\n{Fore.CYAN}--- Iniciando prueba del proceso de missing values. DROP ---{Fore.RESET}")

    # --- TRUCO PARA EL TEST: Simulamos la entrada de terminal ---

    sys.argv = [
        'test_process_missing_values.py',
        '-m', 'train',
        '-f', 'tu_archivo.csv',
        '-a', 'kNN',
        '-p', 'cuerpoMensaje'
    ]

    # 1. CARGAR ARGS CORRECTAMENTE
    # Guardamos el resultado de parse_args() en el módulo de la plantilla
    plantillaPosibleDecisionTreesParaEGela.args = parse_args()

    # Seleccionamos que se hará con los missing_values
    plantillaPosibleDecisionTreesParaEGela.args.preprocessing["missing_values"]="drop"

    print(Fore.GREEN + "Configuración cargada con éxito" + Fore.RESET)

    # 2. Crear datos de prueba
    data_prueba = {
        'cuerpoMensaje': ["Texto A", "Texto B", "Texto B", None],
        'edad': [20, 40, None, 10]
    }
    plantillaPosibleDecisionTreesParaEGela.data = pd.DataFrame(data_prueba)

    # 3. EJECUTAR SELECT_FEATURES
    numerical_feature, text_feature, categorical_feature = select_features()

    # 4. Ejecutar la función de limpieza de nulos
    try:
        process_missing_values(numerical_feature, categorical_feature)

        print(f"\n{Fore.GREEN}Resultado final del DataFrame:{Fore.RESET}")
        # IMPORTANTE: Imprimimos la variable global del módulo, que es la que se modifica
        print(plantillaPosibleDecisionTreesParaEGela.data)

    except Exception as e:
        print(f"{Fore.RED}Error en la prueba: {e}{Fore.RESET}")

def probar_processor_impute_mean():
    print(f"\n{Fore.CYAN}--- Iniciando prueba del proceso de missing values. Impute - Strategy: mean ---{Fore.RESET}")

    # --- TRUCO PARA EL TEST: Simulamos la entrada de terminal ---

    sys.argv = [
        'test_process_missing_values.py',
        '-m', 'train',
        '-f', 'tu_archivo.csv',
        '-a', 'kNN',
        '-p', 'cuerpoMensaje'
    ]

    # 1. CARGAR ARGS CORRECTAMENTE
    # Guardamos el resultado de parse_args() en el módulo de la plantilla
    plantillaPosibleDecisionTreesParaEGela.args = parse_args()

    # Seleccionamos que se hará con los missing_values
    plantillaPosibleDecisionTreesParaEGela.args.preprocessing["missing_values"] = "impute"

    # Seleccionamos la estrategia de imputación en caso de missing_values
    plantillaPosibleDecisionTreesParaEGela.args.preprocessing["impute_strategy"] = "mean"

    print(Fore.GREEN + "Configuración cargada con éxito" + Fore.RESET)

    # 2. Crear datos de prueba
    data_prueba = {
        'cuerpoMensaje': ["Texto A", "Texto B", "Texto B", None],
        'edad': [20, 40, None, 10]
    }
    plantillaPosibleDecisionTreesParaEGela.data = pd.DataFrame(data_prueba)

    # 3. EJECUTAR SELECT_FEATURES
    numerical_feature, text_feature, categorical_feature = select_features()

    # 4. Ejecutar la función de limpieza de nulos
    try:
        process_missing_values(numerical_feature, categorical_feature)

        print(f"\n{Fore.GREEN}Resultado final del DataFrame:{Fore.RESET}")
        # IMPORTANTE: Imprimimos la variable global del módulo, que es la que se modifica
        print(plantillaPosibleDecisionTreesParaEGela.data)

    except Exception as e:
        print(f"{Fore.RED}Error en la prueba: {e}{Fore.RESET}")


def probar_processor_impute_median():
    print(f"\n{Fore.CYAN}--- Iniciando prueba del proceso de missing values. Impute - Strategy: median ---{Fore.RESET}")

    # --- TRUCO PARA EL TEST: Simulamos la entrada de terminal ---

    sys.argv = [
        'test_process_missing_values.py',
        '-m', 'train',
        '-f', 'tu_archivo.csv',
        '-a', 'kNN',
        '-p', 'cuerpoMensaje'
    ]

    # 1. CARGAR ARGS CORRECTAMENTE
    # Guardamos el resultado de parse_args() en el módulo de la plantilla
    plantillaPosibleDecisionTreesParaEGela.args = parse_args()

    # Seleccionamos que se hará con los missing_values
    plantillaPosibleDecisionTreesParaEGela.args.preprocessing["missing_values"] = "impute"

    # Seleccionamos la estrategia de imputación en caso de missing_values
    plantillaPosibleDecisionTreesParaEGela.args.preprocessing["impute_strategy"] = "median"

    print(Fore.GREEN + "Configuración cargada con éxito" + Fore.RESET)

    # 2. Crear datos de prueba
    data_prueba = {
        'cuerpoMensaje': ["Texto A", "Texto B", "Texto B", None],
        'edad': [20, 40, None, 10]
    }
    plantillaPosibleDecisionTreesParaEGela.data = pd.DataFrame(data_prueba)

    # 3. EJECUTAR SELECT_FEATURES
    # Ahora que 'args' ya existe en el módulo, no dará error
    numerical_feature, text_feature, categorical_feature = select_features()

    # 4. Ejecutar la función de limpieza de nulos
    try:
        process_missing_values(numerical_feature, categorical_feature)

        print(f"\n{Fore.GREEN}Resultado final del DataFrame:{Fore.RESET}")
        # IMPORTANTE: Imprimimos la variable global del módulo, que es la que se modifica
        print(plantillaPosibleDecisionTreesParaEGela.data)

    except Exception as e:
        print(f"{Fore.RED}Error en la prueba: {e}{Fore.RESET}")


if __name__ == "__main__":
    probar_processor_drop()
    probar_processor_impute_mean()
    probar_processor_impute_median()