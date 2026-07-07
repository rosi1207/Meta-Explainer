from xml.parsers.expat import model

import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
import torch
from Metrics.metric_Quantus import faithfulnessEstimate, pixelFlipping,  complexity, sparseness


# Funcion para preprocesar la salida de lime
def procesar_salida_lime(salida_lime, num_caracteristicas=6):
    # Crear dos arreglos de ceros del tamaño del número de características
    relevancias_clase_0 = [0] * num_caracteristicas
    relevancias_clase_1 = [0] * num_caracteristicas

    # Recorrer las relevancias de ambas clases simultáneamente
    for (indice_0, relevancia_0), (indice_1, relevancia_1) in zip(salida_lime[0],
                                                                  salida_lime[1]):
        relevancias_clase_0[indice_0] = relevancia_0
        relevancias_clase_1[indice_1] = relevancia_1

    # Devolver las relevancias para ambas clases
    return [relevancias_clase_0, relevancias_clase_1]

def save_explanations_to_pkl(explanations, filename):
    """
    Saves explanations to a .pkl file.

    Args:
        explanations (list): List of integrated explanations to save.
        filename (str): Name of the output .pkl file.
    """
    with open(filename, 'wb') as file:
        pickle.dump(explanations, file)
    print(f"Explanations saved to {filename}")
    
def load_explanations_from_pkl(filename):
    """
    Loads explanations from a .pkl file.

    Args:
        filename (str): Name of the input .pkl file.

    Returns:
        list: List of loaded integrated explanations.
    """
    with open(filename, 'rb') as file:
        explanations = pickle.load(file)
    print(f"Explanations loaded from {filename}")
    return explanations

def separar_arreglo_entrada(arreglo):
    """
    Separa un arreglo NumPy 2D en dos arreglos:
    - El primero contiene todos los valores de la columna 1 (valores normales).
    - El segundo contiene todos los valores de la columna 0 (valores opuestos).

    Parámetros:
        arreglo (numpy.ndarray): Arreglo de entrada, de forma (n, 2).

    Retorna:
        numpy.ndarray: Un arreglo de NumPy con dos elementos: [valores_normales, valores_opuestos].
    """
    if not isinstance(arreglo, np.ndarray) or arreglo.shape[1] != 2:
        raise ValueError("La entrada debe ser un arreglo NumPy de forma (n, 2).")

    # Separar las columnas
    valores_opuestos = arreglo[:, 0]
    valores_normales = arreglo[:, 1]

    # Crear el resultado como un arreglo NumPy
    resultado = np.array([valores_opuestos, valores_normales], dtype=object)

    return resultado

def transfer_to_pytorch(model):
    """
    Transfiere MLPClassifier de sklearn a PyTorch.
    Para arquitectura: input (6) → hidden (50) → output (2)
    """
    
    # Obtener pesos y sesgos
    w1 = model.coefs_[0]  # (6, 50)
    b1 = model.intercepts_[0]  # (50,)
    w2 = model.coefs_[1]  # (50, 2)
    b2 = model.intercepts_[1]  # (2,)
    
    # Crear modelo PyTorch
    model_pt = nn.Sequential(
        nn.Linear(w1.shape[0], w1.shape[1]),  # Capa oculta (6 → 50)
        nn.Tanh(),                            # Activación
        nn.Linear(w2.shape[0], w2.shape[1]),  # Capa salida (50 → 2)
        nn.Softmax(dim=1)                     # Softmax para probabilidades
    )
    
    # Transferir pesos
    with torch.no_grad():
        # Capa oculta
        model_pt[0].weight = nn.Parameter(torch.tensor(w1.T, dtype=torch.float32))
        model_pt[0].bias = nn.Parameter(torch.tensor(b1, dtype=torch.float32))
        
        # Capa de salida
        model_pt[2].weight = nn.Parameter(torch.tensor(w2.T, dtype=torch.float32))
        model_pt[2].bias = nn.Parameter(torch.tensor(b2, dtype=torch.float32))
    
    return model_pt

def crear_arreglo_opuesto_organizado_por_prediccion(model, arreglo, instancia):
    """
    Crea un arreglo NumPy de dos elementos donde el arreglo original y su opuesto
    se colocan según la clase predicha por el modelo:
        - Si predice clase 0: [original, opuesto]
        - Si predice clase 1: [opuesto, original]

    Parámetros:
        model: Modelo de PyTorch entrenado.
        arreglo (array-like): La instancia de entrada (list o np.ndarray).

    Retorna:
        numpy.ndarray: [instancia para clase 0, instancia para clase 1]
    """
    # Dentro de la función
    if torch.is_tensor(arreglo):
        arreglo = arreglo.cpu().detach().numpy()
    # Convertir a tensor
    if not isinstance(arreglo, np.ndarray):
        arreglo = np.array(arreglo)
    # tensor = torch.tensor(arreglo, dtype=torch.float32).unsqueeze(0)  # (1, input_size)

    # Predecir la clase
    model.eval()
    with torch.no_grad():
        salida = model(instancia)
        print(salida)
        clase_predicha = torch.argmax(salida, dim=1).item()
    # Crear opuesto
    opuesto = -arreglo

    # Organizar según la clase predicha
    resultado = np.empty(2, dtype=object)
    resultado[clase_predicha] = arreglo[0]
    resultado[1 - clase_predicha] = opuesto[0]

    return resultado

def evaluate_explanations(explanations, metrics, model, instances_to_explain, X, y, **kwargs):
    """
    Evaluate the relevances using the specified metrics.
    
    Args:
        relevances: List of explanations by instance and class.
                       Shape: [[[clase0_instance1], [clase1_instance1]], [[clase0_instance2], ...]]
        metrics: List of metric functions to apply.
        **kwargs: Additional arguments for the metrics (ej: filter, y_true, etc.)
    
    Returns:
        dict: {metric_name: [valor_imagen1, valor_imagen2, ...]}
    """
    direcciones ={
        'fidelity': 'menor_mejor', 
        'faithfulnessEstimate': 'mayor_mejor',
        'pixelFlipping': 'menor_mejor',
        'complexity': 'menor_mejor',
        'sparseness': 'mayor_mejor',
        'lse': 'mayor_mejor',
    }
    
    kwargs['feature_stds'] = X.std(axis=0)
    
    x_sel = X.iloc[instances_to_explain].values
    y_sel = y.iloc[instances_to_explain].values
    
    metrics_dic = {metric.__name__: [] for metric in metrics}
    
    for metric in metrics:
        valor = metric( explanations, y_sel, x_sel, model, **kwargs)
        #if metric.__name__ == 'lse_calculate':
        #    valor = [min(v, 1000) for v in valor]
        metrics_dic[metric.__name__] = valor
                
    # Mostrar       
    for name, values in metrics_dic.items():
        print(f"{name}: {values}")
        print()
    
    # Normalizar
    metrics_norm = {}
    for nombre, valores in metrics_dic.items():
        vmin = min(valores)
        vmax = max(valores)
        direccion = direcciones[nombre]
    
        norm = []
        for v in valores:
            # Evitar división por cero si todos son iguales
            if vmax == vmin:
                n = 1.0
            else:
                n = (v - vmin) / (vmax - vmin)
            # Invertir si menor_mejor
            if direccion == 'menor_mejor':
                n = 1 - n
            norm.append(n)
        metrics_norm[nombre] = norm
    # Promedio de metricas por instancia 
    promedio_por_instance = []
    num_instances = len(instances_to_explain)
    if len(metrics) == 1:
        promedio_por_instance = list(metrics_norm.values())[0]
    else:
        for i in range(num_instances):
            valores_instance = [metrics_norm[metrica][i] for metrica in metrics_norm]
            promedio_por_instance.append(np.mean(valores_instance))
    print("Promedio por instancia:", promedio_por_instance)
    return promedio_por_instance
 
def normalize_to_minus1_1(x):
    """Normaliza un array al rango [-1, 1]."""
    x_min = np.min(x)
    x_max = np.max(x)
    x_norm = (x - x_min) / (x_max - x_min)  # Normalizado a [0, 1]
    return 2 * x_norm - 1  # Escalar a [-1, 1]

def normalize_signed(x):
    x = np.array(x, dtype=float)
    max_abs = np.max(np.abs(x))
    return x / max_abs if max_abs != 0 else x

def normalize_to_minus1_1_nonzero(x):
    """
    Normaliza los valores distintos de 0 de un array al rango [-1, 1],
    dejando los ceros sin modificar.
    """
    x = np.array(x, dtype=float)  # asegurar tipo float
    mask = x != 0                  # máscara de elementos a normalizar
    if not np.any(mask):
        return x                   # si todos son 0, no hay nada que hacer

    # extraer solo los elementos no cero
    x_nonzero = x[mask]
    x_min = x_nonzero.min()
    x_max = x_nonzero.max()

    # evitar división por cero si todos los no‑ceros son iguales
    if x_max == x_min:
        # en ese caso, asignamos 0 al rango [-1,1]
        x[mask] = 0
    else:
        # normalizar a [0,1] y luego escalar a [-1,1]
        x_norm = (x_nonzero - x_min) / (x_max - x_min)
        x[mask] = 2 * x_norm - 1

    return x

def normalize_map(map_):
    """
    Normalizes a relevance map to values between 0 and 1.
    If the range is zero (min == max), returns a map filled with zeros.

    Args:
        map_ (numpy.ndarray): Relevance map to normalize.

    Returns:
        numpy.ndarray: Normalized relevance map.
    """
    min_val = np.min(map_)
    max_val = np.max(map_)

    # If the range is zero (all values are equal), return a map filled with zeros
    if min_val == max_val:
        return np.zeros_like(map_)

    # Standard normalization
    return (map_ - min_val) / (max_val - min_val)
    # return map_
    
def normalize_explanation(explanation):
    """
    Normaliza una explicación que contiene múltiples instancias.
    
    Args:
        explanation: Lista/array de explicaciones por instancia
                    Formato: [instancia1, instancia2, ...]
    
    Returns:
        np.ndarray: Explicación normalizada como array float32
    """
    try:
        # Convertir a array
        arr = np.array(explanation, dtype=object)
        
        # Para cada instancia, normalizar sus valores
        normalized_instances = []
        for instancia in arr:
            if isinstance(instancia, (list, np.ndarray)):
                # Convertir esta instancia a float32
                inst_norm = np.array(instancia, dtype=np.float32)
                normalized_instances.append(inst_norm)
            else:
                normalized_instances.append(instancia)
        
        # Stackear todas las instancias
        result = np.stack(normalized_instances)
        return result.astype(np.float32)
    
    except Exception as e:
        print(f"Error al normalizar explicación múltiple: {e}")
        return None
    
def visualizar_explicacion(relevances, feature_names):
    """
    REGLA VISUAL:
    - IZQUIERDA (negativo) = Apoya a CLASE 0
    - DERECHA (positivo) = Apoya a CLASE 1
    
    reducir_barras: 1=normal, 2=mitad, 3=tercio, 4=cuarto, etc.
    """
    clase0 = relevances[0]
    clase1 = relevances[1]
    clase0 = np.array(clase0)
    clase1 = np.array(clase1)
    n = len(clase0)
    
    escala_barras = 1 / 6
    
    # Valores transformados y escalados
    valores0 = clase0 * escala_barras
    valores1 = clase1 * escala_barras
    abs0 = np.abs(clase0)
    abs1 = np.abs(clase1)
    # Límites del eje (ajustados a los valores escalados)
    max_valor = max(abs(valores0).max(), abs(valores1).max()) + 0.02
    
    # Crear figura
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Quitar ejes
    ax0.set_axis_off()
    ax1.set_axis_off()
    
    # Configurar límites
    ax0.set_xlim(-max_valor, max_valor)
    ax1.set_xlim(-max_valor, max_valor)
    
    # ========== CLASE 0 ==========
    colores0 = ['blue' if v < 0 else 'orange' for v in valores0]
    barras0 = ax0.barh(range(n), valores0, color=colores0, alpha=0.7, edgecolor='black', height=0.5)
    ax0.axvline(x=0, color='black', linewidth=1)
    ax0.text(0, n + 0.3, 'PREDICCIÓN: CLASE 0', ha='center', fontsize=11, fontweight='bold')
    
    # "0" y "1"
    ax0.text(-max_valor * 0.6, n - 0.2, '0', fontsize=12, fontweight='bold', ha='center')
    ax0.text(max_valor * 0.6, n - 0.2, '1', fontsize=12, fontweight='bold', ha='center')
    
    # Nombres y valores
    for i, (v, v_orig) in enumerate(zip(valores0, abs0)):
        ax0.text(-max_valor - 0.02, i, feature_names[i], va='center', ha='right', fontsize=8)
        if v < 0:
            ax0.text(v - 0.01, i, f'{v_orig:.3f}', va='center', ha='right', fontsize=7)
        else:
            ax0.text(v + 0.01, i, f'{v_orig:.3f}', va='center', ha='left', fontsize=7)
    
    # ========== CLASE 1 ==========
    colores1 = ['blue' if v < 0 else 'orange' for v in valores1]
    barras1 = ax1.barh(range(n), valores1, color=colores1, alpha=0.7, edgecolor='black', height=0.5)
    ax1.axvline(x=0, color='black', linewidth=1)
    ax1.text(0, n + 0.3, 'PREDICCIÓN: CLASE 1', ha='center', fontsize=11, fontweight='bold')
    
    ax1.text(-max_valor * 0.6, n - 0.2, '0', fontsize=12, fontweight='bold', ha='center')
    ax1.text(max_valor * 0.6, n - 0.2, '1', fontsize=12, fontweight='bold', ha='center')
    
    for i, (v, v_orig) in enumerate(zip(valores1, abs1)):
        ax1.text(-max_valor - 0.02, i, feature_names[i], va='center', ha='right', fontsize=8)
        if v < 0:
            ax1.text(v - 0.01, i, f'{v_orig:.3f}', va='center', ha='right', fontsize=7)
        else:
            ax1.text(v + 0.01, i, f'{v_orig:.3f}', va='center', ha='left', fontsize=7)
    
    # Leyenda
    # ========== BARRA DE PORCENTAJE ==========
    suma_clase0 = np.sum(clase0)
    suma_clase1 = np.sum(clase1)
    print(suma_clase0, suma_clase1)
    total = abs(suma_clase0) + abs(suma_clase1)
    porcentaje0 = (abs(suma_clase0) / total) * 100
    porcentaje1 = (abs(suma_clase1) / total) * 100

    clase_dominante = "CLASE 0" if porcentaje0 > porcentaje1 else "CLASE 1"
    color_dominante = 'blue' if porcentaje0 > porcentaje1 else 'orange'

    fig.subplots_adjust(bottom=0.40)
    ax_bar = fig.add_axes([0.12, 0.01, 0.76, 0.1])
    ax_bar.set_axis_off()
    ax_bar.set_xlim(0, 100)

    ax_bar.barh(0, 100, color='#E8E8E8', alpha=0.5, height=0.55)
    ax_bar.barh(0, porcentaje0, color='#1f77b4', alpha=0.85, edgecolor='#999999', linewidth=1, height=0.55)
    ax_bar.barh(0, porcentaje1, left=porcentaje0, color='#ff7f0e', alpha=0.85, edgecolor='#999999', linewidth=1, height=0.55)

    ax_bar.text(porcentaje0/2, 0, f'{porcentaje0:.1f}%', va='center', ha='center', fontsize=9, fontweight='bold', color='white')
    ax_bar.text(porcentaje0 + porcentaje1/2, 0, f'{porcentaje1:.1f}%', va='center', ha='center', fontsize=9, fontweight='bold', color='white')
    ax_bar.text(-2, 0, 'Clase 0', va='center', ha='right', fontsize=9, fontweight='bold', color='#1f77b4')
    ax_bar.text(102, 0, 'Clase 1', va='center', ha='left', fontsize=9, fontweight='bold', color='#ff7f0e')
    
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.show()


def visualizar_explicaciones(all_explanations, feature_names, y_true=None, model=None, x_sel=None, n_clase0=4, n_clase1=4):
    """
    Visualiza explicaciones con MÉTODOS como FILAS e INSTANCIAS como COLUMNAS.

    Args:
        all_explanations (dict): {nombre_metodo: [exp_inst1, exp_inst2, ...]}
        feature_names (list): Nombres de las características
        y_true (array): Etiquetas verdaderas de las instancias (8)
        model: Modelo para predecir
        x_sel: Datos de las instancias (8, n_features)
        n_clase0 (int): Número de instancias de clase 0 (primeras 4)
        n_clase1 (int): Número de instancias de clase 1 (últimas 4)
    """
        # Función para convertir etiqueta a texto
    def etiqueta_a_texto(valor):
        if valor == 0:
            return "No fallece"
        elif valor == 1:
            return "Fallece"
        else:
            return "?"

    metodos = list(all_explanations.keys())
    n_metodos = len(metodos)
    n_instancias = n_clase0 + n_clase1
    n_features = len(feature_names)

    # Obtener predicciones del modelo
    predicciones = []
    if model is not None and x_sel is not None:
        with torch.no_grad():
            input_tensor = torch.tensor(x_sel, dtype=torch.float32)
            output = model(input_tensor)
            predicciones = torch.argmax(output, dim=1).numpy()

    # Crear figura: FILAS = métodos, COLUMNAS = instancias
    fig, axes = plt.subplots(n_metodos, n_instancias, figsize=(n_instancias * 3.5, n_metodos * 2.5))

    if n_metodos == 1:
        axes = axes.reshape(1, -1)
    if n_instancias == 1:
        axes = axes.reshape(-1, 1)

    for i, metodo in enumerate(metodos):          # i = fila (método)
        for j in range(n_instancias):              # j = columna (instancia)
            exp = all_explanations[metodo][j]

            # Determinar clase predicha para esta instancia
            if model is not None and predicciones is not None:
                clase_predicha = predicciones[j]
            else:
                clase_predicha = 1

            # Extraer explicación de la clase predicha
            if isinstance(exp, (list, np.ndarray)):
                if len(exp) == 2:
                    exp = exp[clase_predicha]
                elif hasattr(exp, 'ndim') and exp.ndim == 2 and exp.shape[0] == 2:
                    exp = exp[clase_predicha]

            exp = np.array(exp).flatten()[:n_features]

            # ✅ CORREGIDO: Colores consistentes con visualizar_explicacion
            # AZUL = negativo (apoya CLASE 0)
            # NARANJA = positivo (apoya CLASE 1)
            colors = ['orange' if x > 0 else 'blue' for x in exp]

            ax = axes[i, j]
            ax.barh(range(n_features), exp, color=colors, alpha=0.7, edgecolor='black', height=0.6)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

            # Título de la columna (solo en la primera fila)
            if i == 0:
                clase_real = y_true[j] if y_true is not None else "?"
                pred = predicciones[j] if predicciones is not None else "?"
                acierto = "✓" if (y_true is not None and pred == y_true[j]) else ""
                tipo = etiqueta_a_texto(clase_real)
                ax.set_title(f'Caso{j+1}\n{tipo}', fontsize=9)

            # Etiqueta del método (solo en la primera columna)
            if j == 0:
                ax.set_ylabel(metodo, fontsize=10, fontweight='bold')

            # Configurar eje Y (solo en primera columna)
            if j == 0:
                ax.set_yticks(range(n_features))
                ax.set_yticklabels(feature_names, fontsize=7)
            else:
                ax.set_yticks([])



    plt.suptitle('META-EXPLICACIÓN', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()