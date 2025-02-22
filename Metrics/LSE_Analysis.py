import numpy as np
import matplotlib.pyplot as plt
from aux_functions import ensure_single_channel, is_valid_map, normalize_map

def sum_n_val(index, analysis):
    """ Suma los valores de los índices relevantes para la clasificación utilizando NumPy. """
    # Convertir a arrays de NumPy si no lo son
    analysis = np.asarray(analysis)
    index = np.asarray(index)
    # Seleccionar los valores directamente en lugar de usar bucles
    selected_values = analysis[:, index]
    # Sumar a lo largo de la segunda dimensión (por los índices)
    return selected_values.sum(axis=1)

def lse_calculate(relevances, class_index, filter=0):
    """
    Calcula el IFI utilizando mapas de calor de relevancias.

    Args:
        relevances (list of np.ndarray): Lista de mapas de calor de relevancias.
        class_index (int): Índice de la clase objetivo para el cálculo.
        filter (float): Umbral mínimo para considerar un valor relevante.

    Returns:
        float: Valor IFI calculado.
    """
    # Configuración de la visualización de los mapas de calor
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Mapas de Calor', fontsize=16)

    # Procesar y almacenar mapas válidos y mapas inválidos como ceros
    valid_relevances = []
    invalid_count = 0  # Contador de mapas inválidos

    for i, relevance in enumerate(relevances):
        row, col = divmod(i, 5)
        ax = axs[row, col]
        ax.imshow(relevance, cmap='seismic')
        ax.axis('off')

        # Si el mapa es válido, procesarlo y añadirlo; si no, añadir un mapa de ceros y contar como inválido
        if is_valid_map(relevance):
            processed_map = ensure_single_channel(relevance)
            processed_map = normalize_map(processed_map)
            valid_relevances.append(processed_map)
        else:
            valid_relevances.append(np.zeros_like(relevance))
            invalid_count += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Imprimir información sobre los mapas válidos
    print(f"Forma original de relevances: {np.array(relevances).shape}")
    print(f"Mapas válidos (incluyendo ceros para inválidos): {len(valid_relevances)}")
    print(f"Cantidad de mapas inválidos: {invalid_count}")
    print(f"Forma mapas válidos: {np.array(valid_relevances).shape}")

    # Aplanar mapas para análisis
    f_analysis = [arr.flatten() for arr in valid_relevances]
    print(f"Forma de relevancias aplanadas: {np.array(f_analysis).shape}")
    
    # Obtener el vector de relevancia de la clase objetivo
    f_analysis_class = np.asarray(f_analysis[class_index])
    print(f"Forma de relevancia clase: {np.array(f_analysis_class).shape}")

    # Imprimir máximos y mínimos de cada mapa aplanado para verificación
    for idx, selected in enumerate(f_analysis):
        print(f"Mapa {idx}: Max: {np.max(selected)}, Min: {np.min(selected)}")

    # Seleccionar los índices con valores mayores o iguales al filtro
    index_max = np.where(f_analysis_class >= filter)[0]
    print(f"Cantidad de rasgos mayores que {filter}: {len(index_max)}")
    print(f"Indices mayores que {filter}: {index_max}")

    # Calcular la suma de valores relevantes para cada clase
    sum_vals = sum_n_val(index=index_max, analysis=f_analysis)

    # Relevancia de la clase objetivo vs. el resto de clases
    val_class = sum_vals[class_index]
    val_other = sum(sum_vals[i] for i in range(len(sum_vals)) if i != class_index)
    print(f"Suma clase objetivo: {val_class}\nSuma de otras clases: {val_other}")


    if val_other > 0:
        ifi = ((len(valid_relevances)-1) * val_class) / val_other
    elif val_class > 0 and val_other == 0:
        ifi = (len(valid_relevances)-1) * val_class
    else:
        ifi = 0

    # Imprimir resultado de IFI
    print(f"IFI calculado: {ifi}")
    
    return ifi