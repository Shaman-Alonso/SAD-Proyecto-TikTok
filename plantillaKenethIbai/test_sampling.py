import sys
import pandas as pd
from colorama import Fore

# Importamos lo necesario de la plantilla
import plantillaPosibleDecisionTreesParaEGela
from plantillaPosibleDecisionTreesParaEGela import over_under_sampling, parse_args


def probar_sampling():
    print(f"\n{Fore.CYAN}--- Iniciando prueba de Over/Under Sampling ---{Fore.RESET}")

    # 1. Simulamos argumentos de terminal
    sys.argv = [
        'test_sampling.py',
        '-m', 'train',
        '-f', 'datos.csv',
        '-a', 'kNN',
        '-p', 'clase_objetivo'  # Columna a predecir
    ]

    plantillaPosibleDecisionTreesParaEGela.args = parse_args()

    # 2. Creamos un dataset MUY desequilibrado
    # 8 filas de 'Normal' y solo 2 filas de 'Spam'
    data_dict = {
        'valor': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'clase_objetivo': ['Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Normal', 'Spam',
                           'Spam']
    }
    df_inicial = pd.DataFrame(data_dict)

    print(f"\n{Fore.YELLOW}Distribución inicial de clases:{Fore.RESET}")
    print(df_inicial['clase_objetivo'].value_counts())

    # --- TEST 1: OVERSAMPLING ---
    print(f"\n{Fore.MAGENTA}>> Probando OVERSAMPLING (Igualar por arriba)...{Fore.RESET}")
    plantillaPosibleDecisionTreesParaEGela.data = df_inicial.copy()
    plantillaPosibleDecisionTreesParaEGela.args.preprocessing["sampling"] = "over"

    over_under_sampling()

    print(f"{Fore.GREEN}Distribución tras OVERSAMPLING:{Fore.RESET}")
    print(plantillaPosibleDecisionTreesParaEGela.data['clase_objetivo'].value_counts())
    # Deberías ver 8 y 8

    # --- TEST 2: UNDERSAMPLING ---
    print(f"\n{Fore.MAGENTA}>> Probando UNDERSAMPLING (Igualar por abajo)...{Fore.RESET}")
    plantillaPosibleDecisionTreesParaEGela.data = df_inicial.copy()
    plantillaPosibleDecisionTreesParaEGela.args.preprocessing["sampling"] = "under"

    over_under_sampling()

    print(f"{Fore.GREEN}Distribución tras UNDERSAMPLING:{Fore.RESET}")
    print(plantillaPosibleDecisionTreesParaEGela.data['clase_objetivo'].value_counts())
    # Deberías ver 2 y 2


if __name__ == "__main__":
    probar_sampling()