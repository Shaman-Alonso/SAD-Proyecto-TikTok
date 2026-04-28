import numpy as np
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_ollama.llms import OllamaLLM
from datasets import load_dataset
import argparse
import pandas as pd
import random
import json
import os
import sys
import subprocess

from marshmallow import RAISE
from numpy.f2py.auxfuncs import throw_error
from sklearn.metrics import f1_score,classification_report
from preproceso_generativa import PreprocesadoGenerativo
from tqdm import tqdm

from Preprocesador import DataPreprocessor

#run "ollama pull gemma2:2b" in your terminal before running this script

parser=argparse.ArgumentParser(description='Clasificacion Ollama')
parser.add_argument('--config', type=str, default='generativa.json', help='Ruta al archivo JSON con los parametros del modelo')
parser.add_argument('--sample', type=int, default=-1, help='Numero de filas a evaluar') #Numero de instancias a evaluar. Por defecto todas (-1)
parser.add_argument('--shots',type=int,default=0,help='Numero de ejemplos')
parser.add_argument('--data',type=str,help='Ruta al archivo CSV')
parser.add_argument('--mode',type=str,help='Modos: clasificacion o generacion')
args=parser.parse_args()

def main():
    # Cargamos los ajustes del JSON
    with open(args.config, 'r') as f:
        config = json.load(f)
    parametros = config['model']
    print(f"Cargando configuración desde {args.config}: {config['model']}")
    asegurar_modelo_ollama(parametros['model'])
    modo = args.mode
    if modo == 'classify':
        clasificar(config)
    if modo == 'data_augmentation':
        aumento_datos(config)
    else: raise ValueError("Introduzca uno de los dos modos del scritpt 'classify|data_augmentation'")

def clasificar(config):
    with open("clasificador.json", "r") as f:
        config_global = json.load(f)

    for clave,valor in config_global.items():
        setattr(args, clave, valor)
    args.prediction = "score"
    if not hasattr(args,"file"):
        args.file = args.data

    preprocesador = PreprocesadoGenerativo(args)
    train,dev,test = preprocesador.obtener_datos()

    #Ejemplos de shots
    ejemplos_seleccionados = ""
    ejemplos_pool = []

    train_mezclado = train.sample(frac=1, random_state=42).reset_index(drop=True)

    for _, row in train_mezclado.iterrows():
        texto = str(row['content']).strip()
        valor = str(row[args.prediction]).strip().lower()
        #print(valor)
        if texto and texto.lower() != 'nan':
            # Mapeamos a palabra para enseñarle al LLM
            if valor in ['1', '2']:
                categoria = 'negative'
            elif valor == '3':
                categoria = 'neutral'
            elif valor in ['4', '5']:
                categoria = 'positive'
            else:
                continue

            ejemplos_pool.append({"review": texto, "categoria": categoria})

        if len(ejemplos_pool) >= 150:
            break

    #Elegimos el tipo de prompt
    if args.shots == 0:
        template_str=config['settings'].get('prompt_oneshot')
        prompt = PromptTemplate(template=template_str, input_variables=["texto_nuevo"])

    else:
        ejemplos_seleccionados = obtener_ejemplos_balanceados(ejemplos_pool.copy(), args.shots)

        plantilla_ejemplo = PromptTemplate(
            input_variables=["review","categoria"],
            template="Review: {review}\nClassification: {categoria}"
        )

        prompt=FewShotPromptTemplate(
            examples=ejemplos_seleccionados,
            example_prompt=plantilla_ejemplo,
            prefix=config['settings'].get('prompt_fewshot_prefix'),
            suffix=config['settings'].get('prompt_fewshot_suffix'),
            input_variables=["texto_nuevo"]
        )

    #Inicializar el modelo
    parametros = config['model']
    model = OllamaLLM(model=parametros['model'],temperature=parametros['temperature'],num_predict=parametros['num_predict'],top_k=parametros['top_k'],top_p=parametros['top_p'],stop=parametros['stop']) #determinista
    chain = prompt | model

    #Configurar evaluacion
    ok = 0
    wrongOut = 0
    etiquetas_validas=['positive','neutral','negative']
    y_true = []
    y_pred = []
    log_predicciones = []
    total_instancias = len(dev) if args.sample == -1 else min(args.sample, len(dev))

    #Bucle de inferencia y evaluacion
    for n, row in tqdm(dev.iterrows(), total=total_instancias, desc="Evaluando modelo", unit="res"):
        if n==args.sample:
            break

        #Extraemos el texto de la review
        texto_entrada=str(row['content'])
        #Extraemos el valor numerico y lo convertimos a etiqueta
        valor = str(row[args.prediction]).strip().lower()
        if valor in ['1','2']: etiqueta_real = "negative"
        elif valor in ['3']: etiqueta_real = "neutral"
        elif valor in ['4','5']: etiqueta_real = "positive"
        else: etiqueta_real = "neutral"
        #Invocamos al modelo
        ans_raw = chain.invoke({'texto_nuevo': texto_entrada}).strip().lower()

        #Verificamos si el modelo ha seguido las instrucciones
        ans = "error"
        if "positive" in ans_raw:
            ans = "positive"
        elif "negative" in ans_raw:
            ans = "negative"
        elif "neutral" in ans_raw:
            ans = "neutral"
        else:
            wrongOut += 1
            ans = "neutral"

        # Evaluación
        if ans == etiqueta_real: ok += 1
        acc = round(100 * ok / (n + 1), 2)
        y_true.append(etiqueta_real)
        y_pred.append(ans)
        log_predicciones.append({
            "Texto de entrada": texto_entrada,
            "Texto de salida": ans,
            "Valor real": etiqueta_real,
            "Respuesta_cruda": ans_raw,
        })

        tqdm.write(
            f"| N: {n + 1} | Acc: {acc}% | Out: {wrongOut} | Pred: {ans} | Real: {etiqueta_real} | Raw: '{ans_raw}' |")

    print("-" * 80)
    print("¡Proceso finalizado!")

    #Calculo de metricas
    f1_macro = f1_score(y_true, y_pred, average='macro',zero_division=0)
    f1_macro_pct = round(f1_macro * 100, 2)

    print(f"Accuracy final: {acc}")
    print(f"F1-macro final: {f1_macro_pct}")
    print(f"Reporte detallado por clase: ")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("-" * 80)

    #Preparamos los datos para guardarlos
    prompt_final = prompt.format(texto_nuevo="[DATOS EVALUACION]")

    resultados_experimento = {
        "Modelo": parametros.get('model'),
        "Parametros":json.dumps(parametros),
        "Aciertos": f"{acc}%",
        "Total instancias": n,
        "No validas": wrongOut,
        "Shots": args.shots,
        "Ejemplos seleccionados": ejemplos_seleccionados,
        "Prompt template": prompt_final,
    }

    #Guardamos los resultados
    df_resultado = pd.DataFrame([resultados_experimento])
    carpeta_salida = 'output'
    archivo_reporte = os.path.join(carpeta_salida, 'reporte_generativo.csv')

    os.makedirs(carpeta_salida, exist_ok=True)
    es_nuevo_archivo = not os.path.exists(archivo_reporte)

    df_resultado.to_csv(archivo_reporte, mode='a', index=False, header=es_nuevo_archivo, encoding='utf-8')
    print(f"Resultados guardados con exito en {archivo_reporte}")

    if log_predicciones:
        df_logs = pd.DataFrame(log_predicciones)

        #Limpiamos el nombre del modelo
        nombre_modelo = parametros.get('model','modelo').replace('/','_').replace(':','_')
        temp = parametros.get('temperature','temperature')
        top_k = parametros.get('top_k','top_k')
        num_predict = parametros.get('num_predict','num_predict')
        top_p = parametros.get('top_p','top_p')
        stop = parametros.get('stop','stop')

        nombre_archivo_logs = f"logs_{nombre_modelo}_t{temp}_k{top_k}_{args.shots}shots.csv"
        ruta_logs = os.path.join(carpeta_salida, nombre_archivo_logs)

        df_logs.to_csv(ruta_logs, index=False, encoding='utf-8')
        print(f"CSV de detalle guardado con éxito en: {ruta_logs}")
def aumento_datos(config):
    print("\n--- Iniciando aumento de datos (Parafraseos) ---")
    with open("clasificador.json", "r") as f:
        config_global = json.load(f)

    for clave, valor in config_global.items():
        setattr(args, clave, valor)
    args.prediction = "score"
    if not hasattr(args, "file"):
        args.file = args.data

    preprocesador = PreprocesadoGenerativo(args)
    train, _, _ = preprocesador.obtener_datos()
    train['clase_final'] = train[args.prediction].astype(str).map(
        {'1': 'negative', '2': 'negative', '3': 'neutral', '4': 'positive', '5': 'positive'}
    )

    conteo_clases = train['clase_final'].value_counts()
    clase_mayoritaria = conteo_clases.idxmax()
    max_ejemplos = conteo_clases.max()

    print(f"\n--- Análisis de distribución ---")
    print(conteo_clases)
    print(f"La clase mayoritaria es: '{clase_mayoritaria}' con {max_ejemplos} ejemplos.")

    clases_a_aumentar = conteo_clases[conteo_clases<max_ejemplos].index.tolist()
    print(f"Clases a aumentar: {clases_a_aumentar}")

    parametros = config['model']
    llm = OllamaLLM(model=parametros.get('model'),temperature=parametros.get('temperature'),top_k=parametros.get('top_k'),top_p=parametros.get('top_p'),stop=parametros.get('stop'))

    template_str = config['settings']['prompt_augmentation']
    prompt = PromptTemplate(template=template_str, input_variables=["texto", "etiqueta"])
    chain = prompt | llm

    nuevas_filas = []

    #Iteramos sobre la clase
    for cat in clases_a_aumentar:
        n_necesarios = max_ejemplos-conteo_clases[cat]
        if args.sample != -1 and n_necesarios > args.sample:
            n_necesarios = args.sample
        print(f"Generando {n_necesarios} nuevas reseñas para la clase: {cat}...")

        subset_cat=train[train['clase_final']==cat]

        #Generamos hasta equilibrar o hasta el limite de sample
        for j in tqdm(range(n_necesarios), desc=f"Generando {cat}", unit="res"):            #if args.sample != -1 and len(nuevas_filas)>=args.sample:break

            #Cogemos una fila aleatoria de esta categoria para parafrasearla
            fila_original = subset_cat.sample(1).iloc[0]
            parrafos = ""
            intentos = 0
            max_intentos = 3
            exito = False
            while intentos<max_intentos and not exito:
                try:
                    parrafo = chain.invoke({
                        "texto": fila_original['content'],
                        "etiqueta": cat
                    }).strip()
                    #print(fila_original['content'])
                    if parrafo.lower() != fila_original['content'].lower() and len(parrafo)>5:
                        exito = True
                    else:
                        intentos += 1
                        tqdm.write(f"[!] Intento {intentos}: Texto identico generado, reintentando...")
                except Exception as e:
                    tqdm.write(f"Error generando: {e}")
            if not exito:
                tqdm.write(f"Se omitio una fila tras {max_intentos} intentos fallidos.")
                continue

            # Creamos la nueva fila mantenida TODA la estructura original
            nueva_fila = fila_original.copy()
            nueva_fila['content'] = parrafo
            # ID unico sintetico
            nueva_fila['reviewId'] = f"gen_{cat}_{j}"
            if 'clase_final' in nueva_fila:
                del nueva_fila['clase_final']
            nuevas_filas.append(nueva_fila)
    #Guardamos el resultado
    if nuevas_filas:
        df_final = pd.DataFrame(nuevas_filas)

        #Definimos el orden exacto
        orden_oficial = ['reviewId','content','score','gender','location','date','anonymous']
        df_final = df_final.loc[:, ~df_final.columns.str.contains('^Unnamed')]
        for col in orden_oficial:
            if col not in df_final.columns:
                df_final[col] = ""

        df_final = df_final[orden_oficial]

        os.makedirs('output', exist_ok=True)
        ruta = os.path.join('output',f"augmented.csv")
        df_final.to_csv(ruta, index=False, encoding='utf-8')
        print(f"Proceso finalizado. Archivo guardado con {len(df_final)} filas sinteticas.")
    return

#Funciones auxiliares
def obtener_ejemplos_balanceados(pool,nums_shots):
    if nums_shots==0:return
    categorias = {'positive': [], 'neutral': [], 'negative': []}
    for ej in pool: categorias[ej['categoria']].append(ej)

    seleccionados = []
    lista_categorias = list(categorias.keys())
    idx = 0
    while len(seleccionados)<nums_shots:
        cat_actual = lista_categorias[idx%len(lista_categorias)]
        if categorias[cat_actual]:
            ej_elegido = random.choice(categorias[cat_actual])
            seleccionados.append(ej_elegido)
            categorias[cat_actual].remove(ej_elegido)
        idx+=1
        if sum(len(v) for v in categorias.values())==0: break
    random.shuffle(seleccionados)
    return seleccionados


def asegurar_modelo_ollama(nombre_modelo):
    """Comprueba si el modelo de Ollama está descargado y, si no, lo descarga."""
    modelo_limpio = nombre_modelo.split('/')[-1]

    print(f"\n- [Sistema] Comprobando disponibilidad del modelo: '{modelo_limpio}'...")

    try:
        # Ejecutamos 'ollama list' para leer qué modelos hay instalados
        resultado = subprocess.run(["ollama", "list"], capture_output=True, text=True)

        if modelo_limpio in resultado.stdout:
            print("El modelo ya está instalado y listo para usarse.")
        else:
            print(f"El modelo no está instalado. Iniciando descarga ('ollama pull {modelo_limpio}')...")
            subprocess.run(["ollama", "pull", modelo_limpio], check=True)
            print("\n¡Modelo descargado con éxito!")

    except FileNotFoundError:
        print("\nERROR CRÍTICO: No se ha detectado 'ollama' en el sistema.")
        print("Por favor, asegúrate de tener Ollama instalado y corriendo (https://ollama.com/)")
        sys.exit(1)  # Detenemos la ejecución porque sin Ollama no podemos hacer nada
    except subprocess.CalledProcessError:
        print(f"\nERROR: Fallo al intentar descargar el modelo '{modelo_limpio}'.")
        print("Comprueba tu conexión a internet o si el nombre del modelo es correcto.")
        sys.exit(1)

if __name__ == "__main__":
    main()
