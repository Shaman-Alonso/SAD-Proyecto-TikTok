import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

#Cargar datos
df=pd.read_csv('train.csv')



# Cargar configuracion
with open('config.json','r') as f:
    config = json.load(f)
    f.close()

method = config['train']['method']
params = config[method]

#Comprobar si los datos estan balanceados
target = config['train']['target']
umbral = config['train']['balance_threshold']
target_counts = df[target].value_counts(normalize=True)*100
if target_counts.min() < umbral:
    print(f"Los datos estan desbalanceados")
    debe_balancear=True
else:
    print(f"Los datos estan balanceados")
    debe_balancear=False





# Selector de modelo
models={
    "KNN":KNeighborsClassifier(),
    "DecisionTree":DecisionTreeClassifier(),
    "RandomForest":RandomForestClassifier(),
    "NaiveBayes":GaussianNB(),
}

if method not in models:
    raise ValueError("Method does not exist")

#Entrenamiento
