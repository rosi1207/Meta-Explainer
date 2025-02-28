import matplotlib.pyplot as plt
import numpy as np
import pickle

# Heatmap visualization for all 10 classes
def visualize_all_classes(explanations, selected_images, selected_labels):
    """
    Displays heatmaps for all classes (10 classes) for each image.

    Args:
        explanations (list): List of heatmaps (10 images, 10 classes, 28, 28, 3).
        selected_images (list): List of original images.
        selected_labels (list): True labels of the images.
    """
    fig, axs = plt.subplots(10, 11, figsize=(25, 25))

    for i in range(10):  # Iterate over the 10 selected images
        # Display the original image in the first column
        axs[i, 0].imshow(np.squeeze(selected_images[i]), cmap='gray')
        axs[i, 0].set_title(f'Original: {selected_labels[i]}')
        axs[i, 0].axis('off')

        # Display explanations for the 10 classes in the following columns
        for class_idx in range(10):
            explanation = explanations[i][class_idx]  # Heatmap for the current class
            axs[i, class_idx + 1].imshow(explanation, cmap='seismic')
            axs[i, class_idx + 1].set_title(f'Class {class_idx}')
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
    fig.suptitle("META-EXPLANATION", fontsize=24, fontweight='bold', ha='center')  # Centered title

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
                axs[j, 0].text(-0.5, 0.5, f"({chr(96+j)}) {method_name}", fontsize=16, ha='right', va='center', fontweight='bold',
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