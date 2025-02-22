import numpy as np
from aux_functions import ensure_single_channel, is_valid_map, normalize_map
from Metrics.LSE_Analysis import lse_calculate

def owa_operator_class_by_class(maps, weights, method="weighted_sum"):
    """
    Applies the OWA operator to integrate heatmaps from different methods by class, using the specified combination method.

    Args:
        maps (list of numpy.ndarray): List of heatmaps from different methods, with shape (n_methods, n_classes, 28, 28, 1).
        weights (list of float): Weights to apply in the OWA operator, derived from LSE values.
        method (str): OWA method to use, can be "weighted_sum", "weighted_average", or "simple_average".

    Returns:
        numpy.ndarray: Integrated heatmap per class with shape (n_classes, 28, 28, 1).
    """
    if method == "weighted_sum":
        return owa_weighted_sum(maps, weights)
    elif method == "weighted_average":
        return owa_weighted_average(maps, weights)
    elif method == "simple_average":
        return owa_simple_average(maps)
    else:
        raise ValueError("The specified method is not valid. Use 'weighted_sum', 'weighted_average', or 'simple_average'.")

def owa_weighted_sum(maps, weights):
    """
    Applies the OWA operator using a weighted sum to integrate heatmaps.

    Args:
        maps (list of numpy.ndarray): List of heatmaps per class from different methods.
        weights (list of float): Weights for the weighted sum of each method.

    Returns:
        numpy.ndarray: Integrated heatmap per class with the weighted sum applied.
    """
    num_classes = np.array(maps[0]).shape[0]
    integrated_maps_per_class = []

    for class_idx in range(num_classes):
        processed_maps = [normalize_map(ensure_single_channel(method_maps[class_idx])) for method_maps in maps]

        # Filter valid maps
        valid_maps = [map_ for map_ in processed_maps if is_valid_map(map_)]
        valid_weights = [weights[i] for i, map_ in enumerate(processed_maps) if is_valid_map(map_)]

        if not valid_maps:
            integrated_maps_per_class.append(np.zeros_like(processed_maps[0]))
            continue

        # Calculate the weighted sum of the heatmaps
        integrated_map = np.zeros_like(processed_maps[0], dtype=np.float64)
        for map_, weight in zip(processed_maps, weights):
            integrated_map += weight * map_
        
        integrated_maps_per_class.append(integrated_map)

    return np.array(integrated_maps_per_class)

def owa_weighted_average(maps, weights):
    """
    Applies the OWA operator using a weighted average to integrate heatmaps.

    Args:
        maps (list of numpy.ndarray): List of heatmaps per class from different methods.
        weights (list of float): Weights to calculate the weighted average of each method.

    Returns:
        numpy.ndarray: Integrated heatmap per class with the weighted average applied.
    """
    num_classes = np.array(maps[0]).shape[0]
    integrated_maps_per_class = []

    for class_idx in range(num_classes):
        processed_maps = [ensure_single_channel(method_maps[class_idx]) for method_maps in maps]

        # Filter valid maps
        valid_maps = [map_ for map_ in processed_maps if is_valid_map(map_)]
        valid_weights = [weights[i] for i, map_ in enumerate(processed_maps) if is_valid_map(map_)]

        if not valid_maps:
            integrated_maps_per_class.append(np.zeros_like(processed_maps[0]))
            continue

        # Calculate the weighted average
        integrated_map = np.zeros_like(processed_maps[0], dtype=np.float64)
        for map_, weight in zip(processed_maps, weights):
            integrated_map += weight * map_

        integrated_map *= 1 / len(maps)
        integrated_maps_per_class.append(integrated_map)

    return np.array(integrated_maps_per_class)

def owa_simple_average(maps):
    """
    Applies the OWA operator using a simple average to integrate heatmaps.

    Args:
        maps (list of numpy.ndarray): List of heatmaps per class from different methods.

    Returns:
        numpy.ndarray: Integrated heatmap per class with the simple average applied.
    """
    num_classes = np.array(maps[0]).shape[0]
    integrated_maps_per_class = []

    for class_idx in range(num_classes):
        processed_maps = [ensure_single_channel(method_maps[class_idx]) for method_maps in maps]

        # Filter valid maps
        valid_maps = [map_ for map_ in processed_maps if is_valid_map(map_)]

        if not valid_maps:
            integrated_maps_per_class.append(np.zeros_like(processed_maps[0]))
            continue

        # Calculate the simple average of the heatmaps
        integrated_map = np.zeros_like(processed_maps[0], dtype=np.float64)
        for map_ in processed_maps:
            integrated_map += map_

        integrated_map /= len(maps)
        integrated_maps_per_class.append(integrated_map)

    return np.array(integrated_maps_per_class)

def calculate_lse_integrated_explanations(explanations_data, y_selected, threshold_lse=0.85, method="weighted_sum"):
    """
    Calculates integrated explanations and integrated LSE for each image using the OWA operator and LSE values as weights.

    Args:
        explanations_data (list of dict): List of dictionaries with explanations from different methods and their LSE values.
                                          Each dictionary must contain:
                                          - 'explanations' (list of numpy.ndarray): heatmaps for each image.
                                          - 'lse_value' (list of float): LSE for each image in the corresponding method.
        y_selected (list): Classes of interest to calculate LSE.
        threshold_lse (float): Threshold for LSE calculation in the integrated explanation.
        method (str): OWA method to use, can be "weighted_sum", "weighted_average", or "simple_average".

    Returns:
        tuple: 
            - integrated_explanations (list): List of integrated heatmaps.
            - lse_integrated (list): List of integrated LSE values.
    """
    lse_integrated = []
    integrated_explanations = []  # Stores the integrated explanations

    for i in range(10):  # For each image
        # Collect explanations and LSE values for the 4 methods
        maps = [data['explanations'][i] for data in explanations_data]
        weights = [data['lse_value'][i] for data in explanations_data]  # Use LSE values directly as weights

        # Apply the OWA operator to get the integrated explanation using the selected method
        integrated_explanation = owa_operator_class_by_class(maps, weights, method=method)
        integrated_explanations.append(integrated_explanation)
        
        # Calculate LSE for the integrated explanation using the specified threshold
        lse_value = lse_calculate(integrated_explanation, y_selected[i], filter=threshold_lse)
        lse_integrated.append(lse_value)

    return integrated_explanations, lse_integrated