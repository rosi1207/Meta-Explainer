import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer 

from sklearn.datasets import load_breast_cancer
import pandas as pd

def prepare_breast_cancer_dataset():
    """
    Prepara el conjunto de datos de cáncer de mama proveniente de scikit-learn para su análisis.

    Descripción:
    - Esta función carga el conjunto de datos de cáncer de mama, lo convierte en un DataFrame de pandas
      y añade una columna de objetivo (`target`) que contiene las etiquetas de clase (0 para maligno y 1 para benigno).
    
    Retorna:
    df : DataFrame
        El DataFrame que contiene las características del conjunto de datos junto con la columna de objetivo `target`.
    target_column : str
        El nombre de la columna objetivo (`target`), que se usa para el modelado y análisis posterior.
    """
    
    # Cargar el conjunto de datos de cáncer de mama desde scikit-learn
    data = load_breast_cancer()
    
    # Convertir los datos a un DataFrame de pandas con las características como columnas
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    # Añadir la columna de objetivo (0 para maligno, 1 para benigno)
    df['target'] = data.target

    return df, 'target'

def preprocess_dataset(df, target_column):
    """
    Función de preprocesamiento que realiza las siguientes tareas en el DataFrame:
    - Elimina filas con valores nulos.
    - Obtiene las columnas numéricas.
    - Aplica codificación one-hot a las columnas categóricas (excluyendo la columna objetivo).
    - Identifica las características originales en el DataFrame.
    - Filtra las columnas necesarias para mantener solo las características relevantes y la columna objetivo.
    - Crea un mapa de características que asocia las columnas originales con sus derivadas codificadas.

    Parámetros:
    df : DataFrame
        El DataFrame de entrada que contiene todas las características y la columna objetivo.
    target_column : str o list
        Nombre de la columna objetivo o lista de nombres si hay múltiples columnas objetivo.

    Retorna:
    df : DataFrame
        El DataFrame procesado, con las columnas categóricas codificadas en formato one-hot.
    feature_names : list
        Lista de nombres de las columnas de características codificadas.
    class_values : list
        Lista de valores de clase únicos en la columna objetivo.
    numeric_columns : list
        Lista de nombres de las columnas numéricas originales.
    rdf : DataFrame
        DataFrame filtrado que incluye solo las características y la columna objetivo necesarias.
    real_feature_names : list
        Lista de nombres de las características originales.
    features_map : dict
        Diccionario que mapea cada característica original a sus columnas codificadas.
    """

    # Remover filas con valores nulos en el DataFrame
    df = df.dropna()
    
    # Obtener columnas numéricas usando select_dtypes
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    print(f"COLUMNAS NUMERICAS:\n{numeric_columns}")

    # Crear una copia del DataFrame original para preservar la estructura antes de la codificación
    rdf = df.copy()

    # Aplicar codificación one-hot a todas las columnas categóricas excepto la columna objetivo
    df, feature_names, class_values = one_hot_encoding(df, target_column)
    print(f"FEATURES NAMES:\n{feature_names}\nCLASS VALUES: {class_values}")

    # Obtener nombres de características reales (antes de la codificación one-hot)
    real_feature_names = get_real_feature_names(rdf, numeric_columns, target_column)
    print(f"REAL FEATURES NAMES:\n{real_feature_names}")

    # Obtener nombres de características discretas
    categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()
    print(f"CATEGORICAL FEATURES NAMES:\n{categorical_columns}")

    # Filtrar el DataFrame original para mantener solo las columnas de características reales y la columna objetivo
    rdf = rdf[real_feature_names + (class_values if isinstance(target_column, list) else [target_column])]

    # Crear el mapa de características, relacionando las columnas originales con sus versiones codificadas
    features_map = get_features_map(feature_names, real_feature_names)
    print(f"FEATURES MAP: {features_map}")

    # Retornar todos los elementos procesados necesarios para el flujo de trabajo posterior
    return df, feature_names, class_values, numeric_columns, rdf, real_feature_names, categorical_columns, features_map

#Función para hacer la codificar a one_hot_encoding
def one_hot_encoding(df, target_column):
    """
    Realiza la codificación one-hot de todas las columnas categóricas en el DataFrame excepto la columna objetivo.
    
    Parámetros:
    df : DataFrame
        El DataFrame que contiene los datos a codificar.
    target_column : str
        El nombre de la columna objetivo (la cual no se incluirá en la codificación one-hot).
    
    Retorna:
    df : DataFrame
        El DataFrame resultante después de aplicar la codificación one-hot y el mapeo de la columna objetivo.
    feature_names : list
        Lista de nombres de las columnas de características codificadas.
    class_values : list
        Lista de los valores de clase únicos en la columna objetivo (antes del mapeo a valores numéricos).
    
    Descripción:
    Este método aplica codificación one-hot a todas las columnas categóricas del DataFrame, excluyendo la columna objetivo.
    Además, convierte los valores de la columna objetivo a valores numéricos, lo cual facilita su uso en algoritmos de machine learning.
    """

    # Aplicar codificación one-hot a todas las columnas categóricas excepto la columna objetivo
    dfX = pd.get_dummies(df.drop(columns=[target_column]), prefix_sep='=')

    # Mapear la columna de clase a valores numéricos
    class_name_map = {v: k for k, v in enumerate(sorted(df[target_column].unique()))}
    dfY = df[target_column].map(class_name_map)

    # Combinar las características codificadas con la columna de clase mapeada
    df = pd.concat([dfX, dfY], axis=1)

    # Obtener los nombres de las columnas de características
    feature_names = list(dfX.columns)

    # Obtener los valores de clase únicos en la columna objetivo (antes del mapeo)
    class_values = sorted(class_name_map)

    return df, feature_names, class_values

# Función para obtener los nombres de características reales
def get_real_feature_names(rdf, numeric_columns, target_column):
    """
    Obtiene una lista ordenada de los nombres de las columnas de características "reales" en el DataFrame,
    excluyendo la columna objetivo. Esta función ayuda a diferenciar entre columnas de características
    numéricas y categóricas, manteniendo el orden de tipos (numéricas primero, luego categóricas) 
    y excluyendo la columna objetivo para evitar duplicación en el modelado.

    Parámetros:
    rdf : DataFrame
        El DataFrame original que contiene todas las columnas, incluidas las categóricas y la columna objetivo.
    numeric_columns : list
        Lista de nombres de las columnas numéricas del DataFrame.
    target_column : str o list
        Nombre de la columna objetivo, o lista de nombres si existen múltiples columnas de clase.

    Retorna:
    real_feature_names : list
        Lista de los nombres de las columnas de características (excluyendo la columna objetivo),
        con las columnas numéricas primero, seguidas de las categóricas.
    """
    
    # Obtener columnas numéricas excluyendo la columna objetivo
    real_feature_names = [col for col in numeric_columns if col not in (target_column if isinstance(target_column, list) else [target_column])]
    
    # Agregar las columnas no numéricas excluyendo la columna objetivo
    real_feature_names += [col for col in rdf.columns if col not in numeric_columns and col not in (target_column if isinstance(target_column, list) else [target_column])]
    
    return real_feature_names


from collections import defaultdict

def get_features_map(feature_names, real_feature_names):
    """
    Crea un mapa de características que asocia cada columna original (real_feature_names)
    con sus columnas derivadas codificadas (feature_names), si aplica.

    Parámetros:
    feature_names : list
        Lista de nombres de las columnas codificadas (como en one-hot encoding).
    real_feature_names : list
        Lista de nombres de las columnas originales.

    Retorna:
    features_map : dict
        Un diccionario que mapea cada característica original a sus columnas codificadas, 
        representado como un defaultdict con diccionarios internos.
    """
    
    features_map = defaultdict(dict)
    j = 0

    for i, feature in enumerate(feature_names):
        # Verifica si la característica codificada corresponde directamente a una original
        if j < len(real_feature_names) and feature.startswith(real_feature_names[j]):
            # Extrae el valor codificado después del nombre de la característica original
            encoded_value = feature.replace(f'{real_feature_names[j]}=', '')
            features_map[j][encoded_value] = i
            
            # Si es una coincidencia exacta, pasa a la siguiente característica real
            if feature == real_feature_names[j]:
                j += 1
        elif j < len(real_feature_names):
            # Mueve al siguiente índice de características reales si no hay coincidencia parcial
            j += 1

    return features_map
