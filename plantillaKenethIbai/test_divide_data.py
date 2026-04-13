import sys
import pandas as pd
import nltk
from colorama import Fore
from plantillaPosibleDecisionTreesParaEGela import divide_data, select_features, parse_args
import plantillaPosibleDecisionTreesParaEGela
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string

def probar_divider():
    print(f"\n{Fore.CYAN}--- Iniciando prueba del reescalado standard ---{Fore.RESET}")

    # --- TRUCO PARA EL TEST: Simulamos la entrada de terminal ---
    sys.argv = [
        'test_divide_data.py',
        '-m', 'train',
        '-f', 'tu_archivo.csv',
        '-a', 'kNN',
        '-p', 'edad'
    ]

    # 1. CARGAR ARGS CORRECTAMENTE
    # Guardamos el resultado de parse_args() en el módulo de la plantilla
    plantillaPosibleDecisionTreesParaEGela.args = parse_args()
    print(Fore.GREEN + "Configuración ficticia cargada con éxito" + Fore.RESET)

    # 2. Crear datos de prueba
    data_prueba = {
        'cuerpoMensaje': ["Texto A", "Texto B", "Texto C", "Texto D", "Texto E", "Texto F", "Texto G", "Texto H",
                          "Texto I", "Texto J"],
        'edad': [20, 40, 30, 10, 20, 40, 30, 10, 60, 70]
    }
    plantillaPosibleDecisionTreesParaEGela.data = pd.DataFrame(data_prueba)

    # 3. Ejecutar la función de division
    try:
        x_train, x_dev, y_train, y_dev = divide_data()
        print(Fore.GREEN + f"Datos divididos: Entrenamiento ({len(x_train)}), Desarrollo ({len(x_dev)})" + Fore.RESET)
        print(f"\n{Fore.GREEN}Resultado de x_train:{Fore.RESET}")
        print(x_train)
        print(f"\n{Fore.GREEN}Resultado de x_dev:{Fore.RESET}")
        print(x_dev)
        print(f"\n{Fore.GREEN}Resultado de y_train:{Fore.RESET}")
        print(y_train)
        print(f"\n{Fore.GREEN}Resultado de y_dev:{Fore.RESET}")
        print(y_dev)

    except Exception as e:
        print(f"{Fore.RED}Error en la prueba: {e}{Fore.RESET}")

if __name__ == "__main__":
    probar_divider()
