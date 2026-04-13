import sys
from colorama import Fore

# Importamos las funciones desde el script principal
from plantillaPosibleDecisionTreesParaEGela import (
    calculate_fscore,
    calculate_classification_report,
    calculate_confusion_matrix
)

def probar_metricas():
    print(f"\n{Fore.CYAN}--- Iniciando prueba de las funciones de métricas ---{Fore.RESET}")

    # 1. Crear datos de prueba (Etiquetas reales vs Predicciones del modelo)
    # Imaginemos que 0 = "Texto A", 1 = "Texto B", 2 = "Texto C"
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    # En las predicciones, vamos a fallar intencionadamente en dos casos para ver la magia:
    # - En el índice 2, era 2 y predecimos 1
    # - En el índice 7, era 1 y predecimos 0
    y_pred = [0, 1, 1, 0, 1, 2, 0, 0, 2]

    print(f"{Fore.YELLOW}Etiquetas Reales (y_true):{Fore.RESET} {y_true}")
    print(f"{Fore.YELLOW}Predicciones   (y_pred):{Fore.RESET} {y_pred}")
    print("-" * 50)

    try:
        # 2. Probar F1-Score
        f1_micro, f1_macro = calculate_fscore(y_true, y_pred)
        print(f"{Fore.GREEN}> Resultado calculate_fscore:{Fore.RESET}")
        print(f"  F1-Score Micro: {f1_micro:.4f}")
        print(f"  F1-Score Macro: {f1_macro:.4f}\n")

        # 3. Probar Informe de Clasificación
        report = calculate_classification_report(y_true, y_pred)
        print(f"{Fore.GREEN}> Resultado calculate_classification_report:{Fore.RESET}\n{report}")

        # 4. Probar Matriz de Confusión
        matrix = calculate_confusion_matrix(y_true, y_pred)
        print(f"{Fore.GREEN}> Resultado calculate_confusion_matrix:{Fore.RESET}\n{matrix}\n")

        print(Fore.CYAN + "--- Prueba finalizada con éxito ---" + Fore.RESET)

    except Exception as e:
        print(f"{Fore.RED}Error en la prueba: {e}{Fore.RESET}")

if __name__ == "__main__":
    probar_metricas()