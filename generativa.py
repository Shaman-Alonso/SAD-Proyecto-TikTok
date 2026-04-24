from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_ollama.llms import OllamaLLM
from datasets import load_dataset
import argparse
import pandas as pd
import random
import json
import os

#run "ollama pull gemma2:2b" in your terminal before running this script

parser=argparse.ArgumentParser(description='Clasificacion Ollama')
parser.add_argument('--config', type=str, default='generativa.json', help='Ruta al archivo JSON con los parametros del modelo')
parser.add_argument('--sample', type=int, default=-1, help='Numero de filas a evaluar') #Numero de instancias a evaluar. Por defecto todas (-1)
parser.add_argument('--shots',type=int,default=0,help='Numero de ejemplos')
args=parser.parse_args()

#Cargamos los ajustes del JSON
with open(args.config,'r') as f:
    parametros=json.load(f)

print(f"Cargando configuración desde {args.config}: {parametros['model']}")

#Cargamos los datos a un diccionario
df= pd.read_csv("datos/TikTok.csv")

#Ejemplos de shots
ejemplos_pool = [
    {"review": "The app crashes every time I try to open it. Completely useless.", "categoria": "negative"},
    {"review": "Terrible UI and way too many ads. I'm uninstalling right now.", "categoria": "negative"},
    {"review": "It does the job, but it's nothing special.", "categoria": "neutral"},
    {"review": "Standard app. Needs more features but works okay for now.", "categoria": "neutral"},
    {"review": "Amazing app! Very intuitive and helpful for my daily tasks.", "categoria": "positive"},
    {"review": "Great experience, smooth performance and no bugs so far.", "categoria": "positive"}
]

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

#Elegimos el tipo de prompt
if args.shots == 0:
    template_str="""Classify the following review about TIKTOK with only one of the following words [positive, neutral, negative],
Text:{texto_nuevo}
Classification:"""
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
        prefix="Classify the following app review into [positive, neutral, negative]. Look at these examples:\n",
        suffix="\nReview: {texto_nuevo}\nRespond with ONLY ONE WORD (positive, neutral, or negative): ",
        input_variables=["texto_nuevo"]
    )

#Inicializar el modelo
model = OllamaLLM(model=parametros['model'],temperature=parametros['temperature'],num_predict=parametros['num_predict'],top_k=parametros['top_k'],top_p=parametros['top_p'],stop=parametros['stop']) #determinista
chain = prompt | model

#Configurar evaluacion
ok = 0
wrongOut = 0
etiquetas_validas=['positive','neutral','negative']

#Bucle de inferencia y evaluacion
for n,row in df.iterrows():
    if n==args.sample:
        break

    #Extraemos el texto de la review
    texto_entrada=str(row['content'])
    #Extraemos el score en el .csv
    score_numerico = int(str(row['score']).split('.')[0].strip())

    #Mapeamos los valores del score como un el sentimiento
    if score_numerico in [1,2]:
        etiqueta_real = "negative"
    elif score_numerico == 3:
        etiqueta_real = "neutral"
    elif score_numerico in [4,5]:
        etiqueta_real = "positive"
    else:
        etiqueta_real = "error"

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
        wrongOut += 1  # Si no dijo ninguna de las 3, contamos un Out of format

    # Evaluación
    if ans == etiqueta_real: ok += 1
    acc = round(100 * ok / (n + 1), 2)

    print(
        f"| N: {n + 1} | Acc: {acc}% | Out: {wrongOut} | Pred: {ans} | Real: {etiqueta_real} | Raw: '{ans_raw}' |")

print("-" * 80)
print("¡Proceso finalizado!")

#Preparamos los datos para guardarlos
prompt_final = prompt.format(texto_nuevo="[DATOS EVALUACION]")

resultados_experimento = {
    "Modelo": parametros.get('model'),
    "Parametros":json.dumps(parametros),
    "Aciertos": f"{acc}%",
    "Total instancias": n + 1,
    "No validas": wrongOut,
    "Shots": args.shots,
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