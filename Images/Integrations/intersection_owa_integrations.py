import numpy as np
import matplotlib.pyplot as plt
from aux_functions import ensure_single_channel, normalize_map, is_valid_map
from Metrics.LSE_Analysis import lse_calculate

import numpy as np
import matplotlib.pyplot as plt
from aux_functions import ensure_single_channel, normalize_map, is_valid_map
from Metrics.LSE_Analysis import lse_calculate

def intersection_operator_class_by_class(maps, weights, threshold=0.85, operator='sum'):
    """
    Applies an intersection operator at the class level to combine heatmaps from different methods into a single map per class.
    This operator can either use a weighted sum or select the pixel value from the map with the highest weight for shared relevant pixels.

    Args:
        maps (list of numpy.ndarray): List of heatmaps. Shape: (n_methods, n_classes, 28, 28, 1)
        weights (list of float): List of weights applied in the combination of relevant pixels shared by all methods.
        threshold (float): Threshold to determine relevant pixels for intersection.
        operator (str): Operator to combine shared pixels. 'sum' for weighted sum, 'max' to select value from highest-weight map.

    Returns:
        numpy.ndarray: Integrated heatmap per class (n_classes, 28, 28, 1), where values reflect the combined relevance.
    """
    num_classes = np.array(maps[0]).shape[0]
    print(f"Total classes: {num_classes}")
    integrated_maps_per_class = []

    for class_idx in range(num_classes):
        integrated_map = np.zeros_like(maps[0][class_idx], dtype=np.float64)
        integrated_map = ensure_single_channel(integrated_map)  # Ensure shape (28, 28, 1)

        # Initialize an intersection mask that initially marks all pixels as relevant
        intersection_mask = np.ones_like(integrated_map, dtype=bool).squeeze()  # (28, 28)

        valid_maps = []
        valid_weights = []

        # Filter valid maps before processing the intersection
        for map_, weight in zip(maps, weights):
            map_class = ensure_single_channel(map_[class_idx])
            if is_valid_map(map_class):  # Check if the map is valid
                valid_maps.append(normalize_map(map_class))
                valid_weights.append(weight)

        # Generate masks for each method and update the intersection mask
        method_masks = []
        for map_class in valid_maps:
            mask = map_class > threshold
            intersection_mask &= mask.squeeze()
            method_masks.append(mask.squeeze())

        if operator == 'sum':
            # Sum values for pixels in the intersection
            for map_class in valid_maps:
                integrated_map[intersection_mask] += map_class[intersection_mask]
        elif operator == 'max':
            # Select the value from the map with the highest weight for pixels in the intersection
            max_value_map = np.zeros_like(integrated_map)
            for map_class, weight in zip(valid_maps, valid_weights):
                # Only update with the map value if the weight is the highest so far
                update_mask = (map_class > threshold) & (weight >= np.max(valid_weights))
                max_value_map[update_mask] = map_class[update_mask]
            integrated_map[intersection_mask] = max_value_map[intersection_mask]

        integrated_maps_per_class.append(integrated_map)

        # Visualization of each method's mask and the intersection mask in a single row
        plt.figure(figsize=(18, 4))
        for idx, method_mask in enumerate(method_masks):
            plt.subplot(1, len(method_masks) + 2, idx + 1)
            plt.imshow(method_mask, cmap='gray')
            plt.title(f'Method {idx + 1} Mask')
            plt.axis('off')

        # Show the intersection mask
        plt.subplot(1, len(method_masks) + 2, len(method_masks) + 1)
        plt.imshow(intersection_mask, cmap='gray')
        plt.title('Intersection Mask')
        plt.axis('off')

        # Show the integrated map
        plt.subplot(1, len(method_masks) + 2, len(method_masks) + 2)
        plt.imshow(integrated_map, cmap='jet')
        plt.title('Integrated Map')
        plt.axis('off')

        plt.suptitle(f'Class {class_idx} (Operator: {operator})')
        plt.tight_layout()
        plt.show()

    return np.array(integrated_maps_per_class)

# Function to calculate integrated explanations and integrated LSE for each image
def calculate_lse_intersection_integrated_explanations(explanations_data, y_selected, threshold=0.85, operator='sum'):
    """
    Calculates integrated explanations and Localized Sum of Errors (LSE) for each image using the intersection operator.
    The function applies the intersection operator per class to merge heatmaps, then calculates the LSE in the integrated explanation.

    Args:
        explanations_data (list of dict): List of dictionaries, where each dictionary contains:
                                          - 'explanations' (list of numpy.ndarray): heatmaps per method for each image.
                                          - 'lse_value' (list of float): LSE for each image in the corresponding method.
        y_selected (list): List of selected classes for each image, used when calculating the LSE.
        threshold (float): Threshold to filter LSE calculation in the integrated explanation.
        operator (str): Operator for combining shared pixels in the intersection operator.

    Returns:
        tuple: A tuple with two elements:
            - integrated_explanations (list of numpy.ndarray): List of integrated heatmaps, one per image.
            - lse_integrated (list of float): List of integrated LSE values, one per image.
    """
    lse_integrated = []
    integrated_explanations = []  # Stores the integrated explanations

    for i in range(10):  # For each image
        maps = [data['explanations'][i] for data in explanations_data]
        lse_values = [data['lse_value'][i] for data in explanations_data]

        integrated_explanation = intersection_operator_class_by_class(maps, lse_values, threshold, operator=operator)
        integrated_explanations.append(integrated_explanation)

        lse_value = lse_calculate(integrated_explanation, y_selected[i], filter=threshold)
        lse_integrated.append(lse_value)

    return integrated_explanations, lse_integrated