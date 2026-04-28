import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class PreprocesadoGenerativo:
    def __init__(self,args):
        self.args = args

    def load_data(self):
        print("\n- [Generativa] Cargando datos RAW...")
        data = pd.read_csv(self.args.file, encoding='utf-8')
        # Limpieza básica de columnas para evitar fallos tontos de espacios
        data.columns = data.columns.str.strip()
        return data

    def divide_data(self, data):
        print("- [Generativa] Dividiendo en Train/Dev/Test (random_state=42)...")
        X = data.drop(columns=[self.args.prediction])
        y = data[self.args.prediction]

        test_size = self.args.test_size
        dev_size = self.args.dev_size
        dev_size_adj = dev_size / (1.0 - test_size)

        # IMPORTANTE: random_state=42 garantiza que sean las mismas filas que tu compañero
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
        X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=dev_size_adj, stratify=y_temp,
                                                          random_state=42)

        # Juntamos de nuevo para que sea fácil iterar en el modelo generativo
        train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
        dev = pd.concat([X_dev, y_dev], axis=1).reset_index(drop=True)
        test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

        return train, dev, test

    def obtener_datos(self):
        """Función principal para llamar desde fuera"""
        data = self.load_data()
        data = data[data[self.args.prediction].astype(str).isin(['1', '2', '3', '4', '5'])]
        train, dev, test = self.divide_data(data)

        print(f"  -> Train: {len(train)} | Dev: {len(dev)} | Test: {len(test)}")
        return train, dev, test
