## Configuración del Experimento (`config.json`)

El archivo `config.json` centraliza la lógica del preprocesamiento y los rangos de búsqueda para el entrenamiento. El script está diseñado para leer este archivo y ejecutar el pipeline de forma automática.

### Estructura y Valores Permitidos

#### 1. Bloque `preprocessing`
Define cómo se limpian y transforman los datos antes de entrar al modelo.

| Campo | Valores Permitidos | Descripción |
| :--- | :--- | :--- |
| `missing_values` | `"drop"`, `"impute"` | **drop**: Elimina filas con nulos. <br> **impute**: Rellena huecos según la estrategia. |
| `impute_strategy` | `"mean"`, `"median"` | Estrategia de relleno para variables numéricas (Media o Mediana). |
| `unique_category_threshold`| `int` (ej. `10`) | Máximo de valores únicos para considerar una columna como categórica. |
| `scaling` | `"standard"`, `"min-max"`, `"max-abs"`, `"normalizer"` | Método de escalado/normalización para variables numéricas. |
| `text_process` | `"tf-idf"`, `"bow"`, `"none"` | Técnica para convertir texto en vectores numéricos. |
| `sampling` | `"oversampling"`, `"undersampling"`, `"none"` | Gestión de clases desbalanceadas (usando `imblearn`). |
| `drop_features` | `["list", "of", "strings"]` | Lista de nombres de columnas que se eliminarán del dataset. |



#### 2. Bloque de Algoritmos (Hiperparámetros)
Cada algoritmo acepta una lista de valores para realizar un **barrido de hiperparámetros (GridSearchCV)**.

**kNN (k-Nearest Neighbors):**
* `n_neighbors`: Lista de enteros (ej. `[1, 3, 5]`).
* `weights`: `["uniform", "distance"]`.
* `p`: `[1, 2]` (1: Manhattan, 2: Euclídea).

**decision_tree (Árbol de Decisión):**
* `max_depth`: Lista de profundidades (ej. `[3, 10, None]`).
* `criterion`: `["gini", "entropy"]`.
* `min_samples_split`: Lista de enteros (ej. `[2, 5, 10]`).

---

### Ejemplo de archivo `config.json` completo:

```json
{
  "preprocessing": {
    "missing_values": "impute",
    "impute_strategy": "mean",
    "unique_category_threshold": 10,
    "scaling": "standard",
    "text_process": "tf-idf",
    "sampling": "none",
    "drop_features": ["ID", "Name", "Ticket"]
  },
  "kNN": {
    "n_neighbors": [1, 3, 5, 7],
    "weights": ["uniform", "distance"],
    "p": [1, 2]
  },
  "decision_tree": {
    "max_depth": [3, 5, 7, 10],
    "criterion": ["gini", "entropy"],
    "min_samples_split": [2, 5]
  }
}
