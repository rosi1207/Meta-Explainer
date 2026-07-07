import numpy as np

def sum_n_val(index, analysis):
    """ Suma los valores de los índices relevantes para la clasificación utilizando NumPy. """
    # Convertir a arrays de NumPy si no lo son
    analysis = np.asarray(analysis)
    index = np.asarray(index)
    # Seleccionar los valores directamente en lugar de usar bucles
    selected_values = analysis[:, index]

    return selected_values.sum(axis=1)

def lse(relevances, y_sel ,x_sel, model, filter=0.15, **kwargs):
    """
    Calcula el IFI usando el array de relevancias (ex. LIME) ya generado.

    Parámetros:
    - relevances: Array de relevancias (ex. generado por LIME) que contiene la relevancia de las neuronas de salida.
    - predict: Predicción del modelo (salida de la red para la entrada dada).
    - class_index: Índice de la neurona que tiene la clasificación más probable.
    - filter: Umbral de filtro para los valores de relevancia.

    Retorna:
    - LCE: Valor LCE calculado.
    """
    filter = kwargs.get('filter', filter)
 
    lse_for_instance = []
    for i in range(len(relevances)):
        # f_analysis = relevances
        f_analysis = np.array(relevances[i]).astype(float)
        # Obtener el vector de relevancia aplanado para la neurona de la clase predicha
        f_analysis_class = np.asarray(f_analysis[y_sel[i]])

        # Verificar si los valores están fuera del rango [0,1]
        max_val = np.max(f_analysis_class)
        min_val = np.min(f_analysis_class)

        # Normalizar solo si es necesario
        if max_val > 1 or min_val < 0:
            # Normalización manual al rango [0,1]
            norm_class = (f_analysis_class - min_val) / (max_val - min_val)
            f_analysis_normalized = (f_analysis - np.min(f_analysis)) / (np.max(f_analysis) - np.min(f_analysis))
        else:

            norm_class = f_analysis_class
            f_analysis_normalized = f_analysis

        # Usar np.where para encontrar los índices más relevantes por encima del filtro
        index_max = np.where(norm_class >= filter)[0]

        # Sumar los n-valores de mayor relevancia para la clasificación (por cada neurona de salida)
        sum_vals = sum_n_val(index=index_max, analysis=f_analysis_normalized)

        # Calcular el valor de relevancia de la neurona de clasificación frente al resto
        val_class = sum_vals[y_sel[i]]
        val_other = sum(sum_vals[j] for j in range(len(sum_vals)) if j != y_sel[i])

        # Calcular el LsE
        if val_other > 0:
            lse = (len(relevances) * val_class) / val_other
        elif val_other == 0:
            lse = 0
        else:
            lse = float('inf')

        lse_for_instance.append(lse)
         
    return lse_for_instance