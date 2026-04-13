import sys
import pandas as pd
from colorama import Fore
from plantillaPosibleDecisionTreesParaEGela import reescaler, select_features, parse_args
import plantillaPosibleDecisionTreesParaEGela

def probar_reescaler_standard():
    print(f"\n{Fore.CYAN}--- Iniciando prueba del reescalado standard ---{Fore.RESET}")

    # --- TRUCO PARA EL TEST: Simulamos la entrada de terminal ---
    sys.argv = [
        'test_reescaler.py',
        '-m', 'train',
        '-f', 'tu_archivo.csv',
        '-a', 'kNN',
        '-p', 'cuerpoMensaje'
    ]

    # 1. CARGAR ARGS CORRECTAMENTE
    # Guardamos el resultado de parse_args() en el módulo de la plantilla
    plantillaPosibleDecisionTreesParaEGela.args = parse_args()
    print(Fore.GREEN + "Configuración ficticia cargada con éxito" + Fore.RESET)

    # 2. Crear datos de prueba
    data_prueba = {
        'cuerpoMensaje': ["Texto A", "Texto B", "Texto C", "Texto D"],
        'edad': [20, 40, 30, 10]
    }
    plantillaPosibleDecisionTreesParaEGela.data = pd.DataFrame(data_prueba)

    # 3. EJECUTAR SELECT_FEATURES
    numerical_feature, text_feature, categorical_feature = select_features()

    # 4. Ejecutar la función de reescalado
    try:
        reescaler(numerical_feature)

        print(f"\n{Fore.GREEN}Resultado final del DataFrame:{Fore.RESET}")
        # IMPORTANTE: Imprimimos la variable global del módulo, que es la que se modifica
        print(plantillaPosibleDecisionTreesParaEGela.data)

    except Exception as e:
        print(f"{Fore.RED}Error en la prueba: {e}{Fore.RESET}")

def probar_reescaler_minmax():
    print(f"\n{Fore.CYAN}--- Iniciando prueba del reescalado min-max ---{Fore.RESET}")
    sys.argv = [
        'test_reescaler.py',
        '-m', 'train',
        '-f', 'tu_archivo.csv',
        '-a', 'kNN',
        '-p', 'cuerpoMensaje'
    ]

    # 1. CARGAR ARGS CORRECTAMENTE
    # Guardamos el resultado de parse_args() en el módulo de la plantilla
    plantillaPosibleDecisionTreesParaEGela.args = parse_args()
    print(Fore.GREEN + "Configuración cargada con éxito" + Fore.RESET)
    plantillaPosibleDecisionTreesParaEGela.args.preprocessing["scaling"]="min-max"
    # 2. Crear datos de prueba
    data_prueba = {
        'cuerpoMensaje': ["Texto A", "Texto B", "Texto C", "Texto D"],
        'edad': [20, 40, 30, 10]
    }
    plantillaPosibleDecisionTreesParaEGela.data = pd.DataFrame(data_prueba)

    # 3. EJECUTAR SELECT_FEATURES
    numerical_feature, text_feature, categorical_feature = select_features()

    # 4. Ejecutar la función de reescalado
    try:
        reescaler(numerical_feature)

        print(f"\n{Fore.GREEN}Resultado final del DataFrame:{Fore.RESET}")
        # IMPORTANTE: Imprimimos la variable global del módulo, que es la que se modifica
        print(plantillaPosibleDecisionTreesParaEGela.data)

    except Exception as e:
        print(f"{Fore.RED}Error en la prueba: {e}{Fore.RESET}")

def probar_reescaler_maxabs():
    print(f"\n{Fore.CYAN}--- Iniciando prueba del reescalado max-abs ---{Fore.RESET}")

    sys.argv = [
        'test_reescaler.py',
        '-m', 'train',
        '-f', 'tu_archivo.csv',
        '-a', 'kNN',
        '-p', 'cuerpoMensaje'
    ]

    # 1. CARGAR ARGS CORRECTAMENTE
    # Guardamos el resultado de parse_args() en el módulo de la plantilla
    plantillaPosibleDecisionTreesParaEGela.args = parse_args()
    print(Fore.GREEN + "Configuración cargada con éxito" + Fore.RESET)
    plantillaPosibleDecisionTreesParaEGela.args.preprocessing["scaling"]="max-abs"
    # 2. Crear datos de prueba
    data_prueba = {
        'cuerpoMensaje': ["Texto A", "Texto B", "Texto C", "Texto D"],
        'edad': [20, 40, 30, 10]
    }
    plantillaPosibleDecisionTreesParaEGela.data = pd.DataFrame(data_prueba)

    # 3. EJECUTAR SELECT_FEATURES
    numerical_feature, text_feature, categorical_feature = select_features()

    # 4. Ejecutar la función de reescalado
    try:
        reescaler(numerical_feature)

        print(f"\n{Fore.GREEN}Resultado final del DataFrame:{Fore.RESET}")
        # IMPORTANTE: Imprimimos la variable global del módulo, que es la que se modifica
        print(plantillaPosibleDecisionTreesParaEGela.data)

    except Exception as e:
        print(f"{Fore.RED}Error en la prueba: {e}{Fore.RESET}")

def probar_reescaler_normalizer():
    print(f"\n{Fore.CYAN}--- Iniciando prueba del reescalado normalizer ---{Fore.RESET}")

    sys.argv = [
        'test_reescaler.py',
        '-m', 'train',
        '-f', 'tu_archivo.csv',
        '-a', 'kNN',
        '-p', 'cuerpoMensaje'
    ]

    # 1. CARGAR ARGS CORRECTAMENTE
    # Guardamos el resultado de parse_args() en el módulo de la plantilla
    plantillaPosibleDecisionTreesParaEGela.args = parse_args()
    print(Fore.GREEN + "Configuración cargada con éxito" + Fore.RESET)
    plantillaPosibleDecisionTreesParaEGela.args.preprocessing["scaling"]="normalizer"
    # 2. Crear datos de prueba
    data_prueba = {
        'cuerpoMensaje': ["Texto A", "Texto B", "Texto C", "Texto D"],
        'edad': [20, 40, 30, 10],
        'peso': [49, 52, 60, 40],
        'altura': [150, 172, 186, 123]
    }
    plantillaPosibleDecisionTreesParaEGela.data = pd.DataFrame(data_prueba)

    # 3. EJECUTAR SELECT_FEATURES
    numerical_feature, text_feature, categorical_feature = select_features()

    # 4. Ejecutar la función de reescalado
    try:
        reescaler(numerical_feature)

        print(f"\n{Fore.GREEN}Resultado final del DataFrame:{Fore.RESET}")
        # IMPORTANTE: Imprimimos la variable global del módulo, que es la que se modifica
        print(plantillaPosibleDecisionTreesParaEGela.data)

    except Exception as e:
        print(f"{Fore.RED}Error en la prueba: {e}{Fore.RESET}")


if __name__ == "__main__":
    probar_reescaler_standard()
    probar_reescaler_minmax()
    probar_reescaler_maxabs()
    probar_reescaler_normalizer()
