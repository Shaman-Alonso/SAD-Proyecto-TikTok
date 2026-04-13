import sys
import pandas as pd
from colorama import Fore
from plantillaPosibleDecisionTreesParaEGela import cat2num, select_features, parse_args
import plantillaPosibleDecisionTreesParaEGela

def probar_cat2num():
    print(f"\n{Fore.CYAN}--- Iniciando prueba de la funcion cat2num ---{Fore.RESET}")

    # --- TRUCO PARA EL TEST: Simulamos la entrada de terminal ---
    sys.argv = [
        'test_cat2num.py',
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
        'cuerpoMensaje': ["Texto A", "Texto A", "Texto C", "Texto D", "Texto E", "Texto E", "Texto G", "Texto H",
                          "Texto I", "Texto G"],
        'edad': [20, 40, 30, 10, 20, 40, 30, 10, 60, 70]
    }
    plantillaPosibleDecisionTreesParaEGela.data = pd.DataFrame(data_prueba)

    # Obtenemos las categorical_features con select_features()
    numerical_feature, text_feature, categorical_feature = select_features()

    # 3. Ejecutar la función cat2num
    try:
        cat2num(categorical_feature)
        print(f"\n{Fore.GREEN}Resultado del DataFrame tras cat2num:{Fore.RESET}")
        print(plantillaPosibleDecisionTreesParaEGela.data)

    except Exception as e:
        print(f"{Fore.RED}Error en la prueba: {e}{Fore.RESET}")

if __name__ == "__main__":
    probar_cat2num()
