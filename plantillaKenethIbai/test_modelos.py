import sys
import pandas as pd
import numpy as np
import os
from colorama import Fore

# Importamos el script principal (ajusta el nombre si es necesario)
import plantillaPosibleDecisionTreesParaEGela
from plantillaPosibleDecisionTreesParaEGela import (
    decision_tree,
    random_forest,
    parse_args
)


def preparar_entorno_test(algoritmo):
    print(f"\n{Fore.CYAN}--- Configurando prueba para: {algoritmo} ---{Fore.RESET}")

    # 1. Simulamos la entrada por terminal
    sys.argv = [
        'test_modelos.py',
        '-m', 'train',
        '-f', 'fichero_ficticio.csv',
        '-a', algoritmo,
        '-p', 'objetivo',  # La columna que queremos predecir
        '--cpu', '1',  # Usamos 1 CPU para que el test sea ligero
        '-v'
    ]

    # 2. Cargamos los argumentos
    plantillaPosibleDecisionTreesParaEGela.args = parse_args()

    # 3. Creamos un dataset
    # 20 filas, 3 columnas numéricas y 1 objetivo
    np.random.seed(42)
    X_ficticio = np.random.rand(20, 3)
    y_ficticio = np.random.randint(0, 2, 20)

    df = pd.DataFrame(X_ficticio, columns=['feature1', 'feature2', 'feature3'])
    df['objetivo'] = y_ficticio

    plantillaPosibleDecisionTreesParaEGela.data = df

    # Aseguramos que existan los parámetros en el objeto args para el test
    # (Por si el JSON está vacío o no tiene estas claves)
    if algoritmo == "decision_tree":
        plantillaPosibleDecisionTreesParaEGela.args.decision_tree = {
            "max_depth": [3, 5],
            "criterion": ["gini", "entropy"]
        }
    elif algoritmo == "random_forest":
        plantillaPosibleDecisionTreesParaEGela.args.random_forest = {
            "n_estimators": [10, 20],
            "max_depth": [5]
        }


def probar_modelos():
    # Aseguramos que la carpeta output existe para que save_model no falle
    if not os.path.exists('output'):
        os.makedirs('output')

    try:
        # --- TEST 1: Decision Tree ---
        preparar_entorno_test("decision_tree")
        decision_tree()
        print(f"{Fore.GREEN}✔ Test Decision Tree completado.{Fore.RESET}")

        print("-" * 50)

        # --- TEST 2: Random Forest ---
        preparar_entorno_test("random_forest")
        random_forest()
        print(f"{Fore.GREEN}✔ Test Random Forest completado.{Fore.RESET}")

    except Exception as e:
        print(f"{Fore.RED}❌ Error durante los tests: {e}{Fore.RESET}")


if __name__ == "__main__":
    probar_modelos()