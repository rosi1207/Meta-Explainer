import matplotlib.pyplot as plt
import numpy as np
import pickle
from Metrics.metrics_Quantus import pixelFlipping, faithfulnessEstimate, complexity, sparseness
# Heatmap visualization for all 10 classes
def visualize_all_classes(explanations, selected_images, selected_labels):
    """
    Displays heatmaps for all classes (2 classes) for each image.

    Args:
        explanations (list): List of heatmaps (8 images, 2 classes, 224, 224, 1).
        selected_images (list): List of original images.
        selected_labels (list): True labels of the images.
    """
    class_names = ['nrm', 'pat']
    fig, axs = plt.subplots(8, 3, figsize=(15, 40))

    for i in range(8):
        # Imagen original
        axs[i, 0].imshow(selected_images[i]/255, cmap='gray')
        axs[i, 0].set_title(f'Class: "{class_names[selected_labels[i]]}"')     
        axs[i, 0].axis('off')
    
        # Mapas de cada clase
        for class_idx in range(2):
            explanation = explanations[i][class_idx]
            explanation = normalize_map(explanation)
            axs[i, class_idx + 1].imshow(explanation, cmap='seismic')
            axs[i, class_idx + 1].set_title(f'Class: {class_names[class_idx]}') 
            axs[i, class_idx + 1].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_explanations_horizontal_with_original(selected_images, selected_labels, predictions, explanations_dict, x_selected, y_selected):
    """
    Visualizes original images alongside a variable number of explanations in a horizontal grid.

    Args:
        selected_images: List of selected original images.
        selected_labels: List of true labels for the images.
        predictions: List of model predictions.
        explanations_dict: Dictionary with the method name as key and explanations as value.
        x_selected: List of original grayscale images.
        y_selected: List of classes of interest for explanations.
    """
    num_images = len(selected_images)
    num_methods = len(explanations_dict) + 1  # +1 for the original image

    # Create figure and configure size
    fig, axs = plt.subplots(num_methods, num_images, figsize=(30, 20), gridspec_kw={'wspace': 0.02, 'hspace': 0.05})
    fig.suptitle("META-EXPLICACIÓN", fontsize=24, fontweight='bold', ha='center')  # Centered title

    # Assign method names
    method_names = ['Digit'] + list(explanations_dict.keys())

    for i in range(num_images):
        # Original image in the first row
        axs[0, i].imshow(np.squeeze(selected_images[i]), cmap='gray')
        axs[0, i].set_title(f"{selected_labels[i]}", fontsize=16, fontweight='bold')  # Display only the label
        axs[0, i].axis('off')

        # Iterate over the explanations dictionary
        for j, (method_name, explanation) in enumerate(explanations_dict.items(), start=1):
            if method_name == 'LIME':
                lime_mask = np.squeeze(explanation[i][y_selected[i]])
                axs[j, i].imshow(x_selected[i], cmap='gray')  # Original image in grayscale
                axs[j, i].imshow(lime_mask, cmap='seismic', alpha=0.5)  # LIME overlay
            else:
                axs[j, i].imshow(explanation[i][y_selected[i]], cmap='seismic')

            axs[j, i].axis('off')

            # Label the row in the first column
            if i == 0:
                axs[j, 0].text(-0.5, 0.5, f"{method_name}", fontsize=16, ha='right', va='center', fontweight='bold',
                               transform=axs[j, 0].transAxes, rotation=0, color='black')

    # Adjust spacing between subplots and reduce horizontal separation
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.1)

    plt.show()

# List of integrated explanations to save (example: integrated_explanations)
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

# Function to convert to (28, 28, 1) format
def ensure_single_channel(image):
    """
    Ensures the image has the shape (28, 28).
    Converts RGB images (28, 28, 3) to grayscale without adding an additional channel dimension.
    """
    if image.ndim == 3:
        # Convert RGB to grayscale
        image = image.mean(axis=2)  # (28, 28, 3) -> (28, 28)
    # Normalize the image after conversion
    # image = normalize_map(image)
    
    return image

# def ensure_single_channel(image):
#     """
#     Ensures the image has the shape (28, 28, 1).
#     Converts RGB images (28, 28, 3) to grayscale and adds a channel dimension.
#     """
#     if image.ndim == 3 and image.shape[-1] == 3:
#         # Convert RGB to grayscale
#         image = np.mean(image, axis=-1, keepdims=True)  # (28, 28, 3) -> (28, 28, 1)
#     elif image.ndim == 2:
#         # If the image lacks a channel dimension, add it
#         image = np.expand_dims(image, axis=-1)  # (28, 28) -> (28, 28, 1)
#     image = normalize_map(image)
    
#     return image

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

# Function to validate heatmap
def is_valid_map(map_data):
    """ Checks if the heatmap is valid (does not contain NaN and has non-uniform values). """
    if np.isnan(map_data).any() or np.all(map_data == map_data[0]):
        return False
    return True

# Function to apply replacements to the lse_value
def transform_lse_value(lse_value):
    # Replace values equal to 1 with 12.5
    if lse_value == 1:
        return 12.5
    # Replace values greater than 12 with 10.5
    elif lse_value > 12:
        return 10.5
    return lse_value  # Return the value unchanged if it doesn't meet any conditions

# Funcion para visualizar cada mapa normalizado
def visualizate_image(maps):
    # Configure heatmap visualization (only for images)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Heatmaps', fontsize=16)

    for i, map in enumerate(maps):
        map = np.array(map)
        row, col = divmod(i, 2)
        ax = axs[col]
        # Normalizar SOLO para visualización
        relevance_visualization = normalize_map(map)
        ax.imshow(relevance_visualization, cmap='seismic')
        ax.axis('off')
        
        total = np.prod(map.shape)
        print("\n")
        print(f"Statistics for the class {i} map:")
        print(f"Min: {np.min(map):.6f}")
        print(f"Max: {np.max(map):.6f}")
        print(f"Media: {np.mean(map):.6f}")
        print(f"Desviación: {np.std(map):.6f}")
        print(f"Píxeles positivos: {np.sum(map > 0)} / {total}")
        print(f"Píxeles negativos: {np.sum(map < 0)} / {total}")
        print(f"Píxeles ~cero: {np.sum(np.abs(map) < 1e-6)} / {total}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()  
 
def evaluate_explanations(explanations, metrics, model, x_selected, y_selected, **kwargs):
    """
    Evaluate the explanations using the specified metrics.
    
    Args:
        explanations: List of explanations by image and class.
                       Shape: [[[clase0_img1], [clase1_img1]], [[clase0_img2], ...]]
        metrics: List of metric functions to apply.
        **kwargs: Additional arguments for the metrics (ej: filter, y_true, etc.)
    
    Returns:
        dict: {metric_name: [valor_imagen1, valor_imagen2, ...]}
    """
    direcciones ={
        'pixelFlipping': 'menor_mejor',
        'regionPerturbation':  'menor_mejor',
        'continuity': 'menor_mejor',
        'maxSensitivity': 'menor_mejor',
        'lse_calculate': 'mayor_mejor',
        'complexity': 'menor_mejor',
        'sparseness': 'mayor_mejor',
    }
    
    
    metrics_dic = {metric.__name__: [] for metric in metrics}
    
    for metric in metrics:
        valor = metric( explanations, y_selected, x_selected, model, **kwargs)
        if metric.__name__ == 'lse_calculate':
            valor = [min(v, 1000) for v in valor]
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
    # Promedio de metricas por imagen 
    promedio_por_imagen = []
    num_imagenes = len(explanations)
    if len(metrics) == 1:
        promedio_por_imagen = list(metrics_norm.values())[0]
    else:
        for i in range(num_imagenes):
            valores_imagen = [metrics_norm[metrica][i] for metrica in metrics_norm]
            promedio_por_imagen.append(np.mean(valores_imagen))
    print("Promedio por imagen:", promedio_por_imagen)
    return promedio_por_imagen