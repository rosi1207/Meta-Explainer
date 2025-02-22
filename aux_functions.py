import matplotlib.pyplot as plt
import numpy as np
import pickle


# Visualización de mapas de calor para las 10 clases
def visualize_all_classes(explanations, selected_images, selected_labels):
    """
    Muestra los mapas de calor para todas las clases (10 clases) para cada imagen.

    Args:
        explanations (list): Lista de mapas de calor (10 imágenes, 10 clases, 28, 28, 3).
        selected_images (list): Lista de imágenes originales.
        selected_labels (list): Etiquetas reales de las imágenes.
    """
    fig, axs = plt.subplots(10, 11, figsize=(25, 25))

    for i in range(10):  # Iterar sobre las 10 imágenes seleccionadas
        # Mostrar la imagen original en la primera columna
        axs[i, 0].imshow(np.squeeze(selected_images[i]), cmap='gray')
        axs[i, 0].set_title(f'Original: {selected_labels[i]}')
        axs[i, 0].axis('off')

        # Mostrar las explicaciones para las 10 clases en las siguientes columnas
        for class_idx in range(10):
            explanation = explanations[i][class_idx]  # Mapa para la clase actual
            axs[i, class_idx + 1].imshow(explanation, cmap='seismic')
            axs[i, class_idx + 1].set_title(f'Clase {class_idx}')
            axs[i, class_idx + 1].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_explanations_horizontal_with_original(selected_images, selected_labels, predictions, explanations_dict, x_selected, y_selected):
    """
    Visualiza las imágenes originales junto con un número variable de explicaciones en una cuadrícula horizontal.

    Args:
        selected_images: Lista de imágenes originales seleccionadas.
        selected_labels: Lista de etiquetas reales para las imágenes.
        predictions: Lista de predicciones del modelo.
        explanations_dict: Diccionario con el nombre del método como clave y las explicaciones como valor.
        x_selected: Lista de imágenes originales en escala de grises.
        y_selected: Lista de clases de interés para las explicaciones.
    """
    num_images = len(selected_images)
    num_methods = len(explanations_dict) + 1  # +1 para la imagen original

    # Crear figura y configuración de tamaño
    fig, axs = plt.subplots(num_methods, num_images, figsize=(30, 20), gridspec_kw={'wspace': 0.02, 'hspace': 0.05})
    fig.suptitle("META-EXPLANATION", fontsize=24, fontweight='bold', ha='center')  # Título centrado

    # Asignar nombres de métodos
    method_names = ['Digit'] + list(explanations_dict.keys())

    for i in range(num_images):
        # Imagen original en la primera fila
        axs[0, i].imshow(np.squeeze(selected_images[i]), cmap='gray')
        axs[0, i].set_title(f"{selected_labels[i]}", fontsize=16, fontweight='bold')  # Mostrar solo la etiqueta
        axs[0, i].axis('off')

        # Iterar sobre el diccionario de explicaciones
        for j, (method_name, explanation) in enumerate(explanations_dict.items(), start=1):
            if method_name == 'LIME':
                lime_mask = np.squeeze(explanation[i][y_selected[i]])
                axs[j, i].imshow(x_selected[i], cmap='gray')  # Imagen original en gris
                axs[j, i].imshow(lime_mask, cmap='seismic', alpha=0.5)  # Superposición de LIME
            else:
                axs[j, i].imshow(explanation[i][y_selected[i]], cmap='seismic')
            
            axs[j, i].axis('off')

            # Etiquetar la fila en la primera columna
            if i == 0:
                axs[j, 0].text(-0.5, 0.5, f"({chr(96+j)}) {method_name}", fontsize=16, ha='right', va='center', fontweight='bold',
                               transform=axs[j, 0].transAxes, rotation=0, color='black')

    # Ajustar el espacio entre las subparcelas y reducir la separación horizontal
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.1)
    
    plt.show()




# Lista de explicaciones integradas para guardar (ejemplo: integrated_explanations)
def save_explanations_to_pkl(explanations, filename):
    """
    Guarda las explicaciones en un archivo .pkl.

    Args:
        explanations (list): Lista de explicaciones integradas a guardar.
        filename (str): Nombre del archivo .pkl de salida.
    """
    with open(filename, 'wb') as file:
        pickle.dump(explanations, file)
    print(f"Explicaciones guardadas en {filename}")

def load_explanations_from_pkl(filename):
    """
    Carga las explicaciones desde un archivo .pkl.

    Args:
        filename (str): Nombre del archivo .pkl de entrada.

    Returns:
        list: Lista de explicaciones integradas cargadas.
    """
    with open(filename, 'rb') as file:
        explanations = pickle.load(file)
    print(f"Explicaciones cargadas desde {filename}")
    return explanations


# Función para convertir a formato (28, 28, 1)
def ensure_single_channel(image):
    """
    Asegura que la imagen tenga la forma (28, 28).
    Convierte imágenes RGB (28, 28, 3) a escala de grises sin agregar dimensión de canal adicional.
    """
    if image.ndim == 3:
        # Convertir RGB a escala de grises
        image = image.mean(axis=2)  # (28, 28, 3) -> (28, 28)
    # Normalizar la imagen después de la conversión
    # image = normalize_map(image)
    
    return image


# def ensure_single_channel(image):
#     """
#     Asegura que la imagen tenga la forma (28, 28, 1).
#     Convierte las imágenes RGB (28, 28, 3) a escala de grises y les agrega una dimensión de canal.
#     """
#     if image.ndim == 3 and image.shape[-1] == 3:
#         # Convertir RGB a escala de grises
#         image = np.mean(image, axis=-1, keepdims=True)  # (28, 28, 3) -> (28, 28, 1)
#     elif image.ndim == 2:
#         # Si la imagen no tiene la dimensión del canal, agregarla
#         image = np.expand_dims(image, axis=-1)  # (28, 28) -> (28, 28, 1)
#     image = normalize_map(image)
    
#     return image

def normalize_map(map_):
    """
    Normaliza los valores de un mapa de calor entre 0 y 1, solo si están fuera de este rango.

    Args:
        map_ (numpy.ndarray): Mapa de calor a normalizar.

    Returns:
        numpy.ndarray: Mapa de calor normalizado (si es necesario).
    """
    map_min = np.min(map_)
    map_max = np.max(map_)

    # Evitar división por cero si todos los valores son iguales
    if map_max == map_min:
        return np.zeros_like(map_)

    # Verificar si el mapa ya está dentro del rango [0, 1]
    if 0 <= map_min and map_max <= 1:
        return map_

    # Normalizar al rango [0, 1] si es necesario
    normalized_map = (map_ - map_min) / (map_max - map_min)
    return normalized_map


#Función para validar mapa de calor
def is_valid_map(map_data):
    """ Verifica si el mapa de calor es válido (no contiene NaN y tiene valores no uniformes). """
    if np.isnan(map_data).any() or np.all(map_data == map_data[0]):
        return False
    return True