import numpy as np
import matplotlib.pyplot as plt
from aux_functions import ensure_single_channel, normalize_map, is_valid_map
from Metrics.LSE_Analysis import lse_calculate

def union_operator_class_by_class(maps, weights, threshold=0.8, operator='sum'):
    """
    Applies a union operator independently for each class. Depending on the selected operator,
    it either sums (OWA) the relevant pixels across methods with weights, or selects the
    highest-weighted pixel value when multiple methods detect the pixel as relevant.

    Args:
        maps (list of numpy.ndarray): List of heatmaps from different explanation methods.
                                      Shape: (n_methods, n_classes, 28, 28, 1)
        weights (list of float): Weights for each method.
        threshold (float): Threshold to determine relevant pixels.
        operator (str): Operator to use for combining maps. 'sum' for weighted sum, 'max' for maximum selection.

    Returns:
        numpy.ndarray: Integrated heatmap per class. Shape: (n_classes, 28, 28, 1)
    """
    num_classes = np.array(maps[0]).shape[0]
    integrated_maps_per_class = []

    for class_idx in range(num_classes):
        map_shape = maps[0][class_idx].shape
        integrated_map = np.zeros(map_shape, dtype=np.float64)
        integrated_map = ensure_single_channel(integrated_map)
        
        valid_maps = []
        valid_weights = []

        # Filter valid maps and their corresponding weights
        for map_, weight in zip(maps, weights):
            map_class = ensure_single_channel(map_[class_idx])
            map_class = normalize_map(map_class)
            if is_valid_map(map_class):
                valid_maps.append(map_class)
                valid_weights.append(weight)
        
        if not valid_maps:
            integrated_maps_per_class.append(integrated_map)
            continue

        valid_maps = np.array(valid_maps)
        valid_weights = np.array(valid_weights)
        
        masks = valid_maps > threshold

        if operator == 'sum':
            # Perform weighted sum
            for map_class, weight in zip(valid_maps, valid_weights):
                mask = map_class > threshold
                integrated_map[mask] += map_class[mask] * weight

        elif operator == 'max':
            # Perform max-weight selection per pixel
            for x in range(map_shape[0]):
                for y in range(map_shape[1]):
                    relevant_methods = [i for i, mask in enumerate(masks) if mask[x, y]]
                    if len(relevant_methods) == 1:
                        method_idx = relevant_methods[0]
                        integrated_map[x, y] = valid_maps[method_idx, x, y]
                    elif len(relevant_methods) > 1:
                        max_weight_idx = max(relevant_methods, key=lambda idx: valid_weights[idx])
                        integrated_map[x, y] = valid_maps[max_weight_idx, x, y]

        integrated_maps_per_class.append(integrated_map)

        # Plot the integrated map for the class
        print(f"\nClass {class_idx}: Plotting integrated map after processing with operator '{operator}'...")
        plt.imshow(integrated_map.squeeze(), cmap='seismic')
        plt.title(f'Integrated Map for Class {class_idx} (Operator: {operator})')
        plt.colorbar()
        plt.show()

    return np.array(integrated_maps_per_class)

def calculate_lse_union_integrated_explanations(explanations_data, y_selected, threshold=0.85, operator='sum'):
    """
    Calculates integrated explanations and integrated LSE (Localized Sum of Errors) for each image using the union of heatmaps from different methods.
    The function applies the union operator per class to merge heatmaps and then calculates the LSE in the integrated explanation.

    Args:
        explanations_data (list of dict): List of dictionaries, where each dictionary contains:
                                          - 'explanations' (list of numpy.ndarray): heatmaps per method for each image.
                                          - 'lse_value' (list of float): LSE for each image in the corresponding method.
        y_selected (list): List of selected classes for each image, used when calculating the LSE.
        threshold (float): Threshold for filtering LSE calculation in the integrated explanation.
        operator (str): Operator to use for combining maps. 'sum' for weighted sum, 'max' for maximum selection.

    Returns:
        tuple: A tuple with two elements:
            - integrated_explanations (list of numpy.ndarray): List of integrated heatmaps, one per image.
            - lse_integrated (list of float): List of integrated LSE values, one per image.
    """
    lse_integrated = []
    integrated_explanations = []  # Stores the integrated explanations

    for i in range(10):  # For each image
        # Collect explanations and LSE values for the 4 methods of the current image
        maps = [data['explanations'][i] for data in explanations_data]
        lse_values = [data['lse_value'][i] for data in explanations_data]

        # Apply the union operator per class to get the integrated explanation using the weights of this image
        integrated_explanation = union_operator_class_by_class(maps, lse_values, threshold, operator=operator)
        integrated_explanations.append(integrated_explanation)

        # Calculate LSE for the integrated explanation using the selected class and filter threshold
        lse_value = lse_calculate(integrated_explanation, y_selected[i], filter=threshold)
        lse_integrated.append(lse_value)

    return integrated_explanations, lse_integrated