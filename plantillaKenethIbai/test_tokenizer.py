import pandas as pd
import nltk
from colorama import Fore
from plantillaPosibleDecisionTreesParaEGela import simplify_text

def probar_mi_tokenizer():
    print(f"\n{Fore.CYAN}--- Iniciando prueba del Tokenizer ---{Fore.RESET}")

    # 1. Crear un pequeño set de datos de prueba
    data_prueba = {
        'cuerpoMensaje': [
            "The quick brown fox is running over the lazy dog!!",
            "I love processing data with Python and Scikit-learn",
            "Text mining is amazing",
            None  # Para ver si aguanta nulos
        ]
    }

    # Creamos un dataframe global porque la función usa 'global data'
    import plantillaPosibleDecisionTreesParaEGela
    plantillaPosibleDecisionTreesParaEGela.data = pd.DataFrame(data_prueba)

    # 2. Simular el objeto text_feature que espera la función
    # Seleccionamos solo la columna de texto
    text_features = plantillaPosibleDecisionTreesParaEGela.data[['cuerpoMensaje']]

    # 3. Ejecutar la función
    try:
        simplify_text(text_features)

        # 4. Ver los resultados
        df_resultado = plantillaPosibleDecisionTreesParaEGela.data
        print(f"\n{Fore.GREEN}Resultado final:{Fore.RESET}")
        print(df_resultado[['cuerpoMensaje']])

    except Exception as e:
        print(f"{Fore.RED}Error en la prueba: {e}{Fore.RESET}")


if __name__ == "__main__":
    # Aseguramos que los diccionarios estén listos
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    probar_mi_tokenizer()