import numpy as np
from aux_functions import normalize_signed, evaluate_explanations, normalize_signed, visualizar_explicacion

def owa_operator_class_by_class(maps, weights, method="weighted_sum"):
    """
    Applies the OWA operator to integrate tabular explanations from different methods by class.

    Args:
        maps (list of np.ndarray): List of relevance matrices with dimensions (n_methods, n_classes, n_features).
        weights (list of float): Weights to apply in the OWA operator, derived from LSE values.
        method (str): OWA method to use: "weighted_sum", "weighted_average", or "simple_average".

    Returns:
        np.ndarray: Integrated relevance matrix per class with dimensions (n_classes, n_features).
    """
    print(f"Weights: {weights}")
    num_classes, num_features = np.array(maps[0]).shape
    integrated_maps_per_class = []

    for class_idx in range(num_classes):
        # Extract maps from each method for the specific class and normalize them
        processed_maps = [method_maps[class_idx] for method_maps in maps]

        # Filter valid maps (those without NaN or Inf)
        valid_maps = [normalize_signed(map_) for map_ in processed_maps if np.all(np.isfinite(map_))]
        valid_weights = [weights[i] for i, map_ in enumerate(processed_maps) if np.all(np.isfinite(map_))]

        if not valid_maps:
            integrated_maps_per_class.append(np.zeros(num_features))
            continue

        if method == "weighted_sum":
            integrated_map = np.zeros(num_features, dtype=np.float64)
            for map_, weight in zip(valid_maps, valid_weights):
                integrated_map += weight * map_
            integrated_maps_per_class.append(integrated_map)

        elif method == "weighted_average":
            integrated_map = np.zeros(num_features, dtype=np.float64)
            for map_, weight in zip(valid_maps, valid_weights):
                integrated_map += weight * map_
            integrated_map /= sum(valid_weights) if sum(valid_weights) != 0 else 1
            integrated_maps_per_class.append(integrated_map)

        elif method == "simple_average":
            integrated_map = np.mean(valid_maps, axis=0)
            integrated_maps_per_class.append(integrated_map)

        else:
            raise ValueError("Invalid OWA method. Use 'weighted_sum', 'weighted_average', or 'simple_average'.")

    return np.array(integrated_maps_per_class)

def calculate_metric_integrated_explanations(explanations_data, instances_to_explain, metrics, model, X, y, method="weighted_sum", **kwargs):
    """
    Calculates integrated explanations and integrated LSE for each image using the OWA operator and LSE values as weights.

    Args:
        explanations_data (list of dict): List of dictionaries with explanations from different methods and their LSE values.
                                          Each dictionary must contain:
                                          - 'explanations' (list of numpy.ndarray): heatmaps for each image.
                                          - 'metric_value' (list of float): Metric for each image in the corresponding method.
        y_selected (list): Classes of interest to calculate metrics.
        method (str): OWA method to use, can be "weighted_sum", "weighted_average", or "simple_average".

    Returns:
        tuple:
            - integrated_explanations (list): List of integrated heatmaps.
            - metric_integrated (list): List of integrated metric values.
    """
    metric_integrated = []
    integrated_explanations = []  # Stores the integrated explanations
    x_selected = X.iloc[instances_to_explain].values
    y_selected = y.iloc[instances_to_explain].values
    feature_names = X.columns.tolist()
    for i in range(len(x_selected)):  # For each image
        # Collect explanations and LSE values for the 4 methods
        maps = [data['explanations'][i] for data in explanations_data]
        weights = [data['metric'][i] for data in explanations_data]  # Use LSE values directly as weights

        # Apply the OWA operator to get the integrated explanation using the selected method
        integrated_explanation = owa_operator_class_by_class(maps, weights, method=method)
        visualizar_explicacion(integrated_explanation, feature_names)
        integrated_explanations.append(integrated_explanation)

        # Calculate metric for the integrated explanation
    metric_value = evaluate_explanations(integrated_explanations, metrics, model, instances_to_explain, X, y, **kwargs)
    metric_integrated.append(metric_value)

    return integrated_explanations, metric_integrated

