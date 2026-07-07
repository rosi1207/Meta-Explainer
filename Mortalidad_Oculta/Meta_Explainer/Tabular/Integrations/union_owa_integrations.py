import numpy as np
from aux_functions import normalize_signed, evaluate_explanations, visualizar_explicacion


def union_operator_class_by_class(maps, weights, threshold=0.8, operator='sum'):
    """
    Union operator usando magnitud (valor absoluto) para determinar relevancia.
    """
    print(f"Weights: {weights}")
    num_classes, num_features = np.array(maps[0]).shape
    integrated_maps_per_class = []

    for class_idx in range(num_classes):
        integrated_map = np.zeros(num_features, dtype=np.float64)

        valid_maps = []
        valid_weights = []
        masks = []

        for map_, weight in zip(maps, weights):
            map_class = map_[class_idx]
            map_class_normalized = normalize_signed(map_class)  # Rango [-1, 1]

            if np.all(np.isfinite(map_class_normalized)):
                valid_maps.append(map_class_normalized)
                valid_weights.append(weight)

                # ✅ CORREGIDO: Usar VALOR ABSOLUTO para la máscara
                mask = np.abs(map_class_normalized) > threshold
                masks.append(mask)

                print(f"Class {class_idx}: {np.sum(mask)} features with |value| > {threshold}")

        if not valid_maps:
            integrated_maps_per_class.append(integrated_map)
            continue

        valid_maps = np.array(valid_maps)
        valid_weights = np.array(valid_weights)
        masks = np.array(masks)

        if operator == 'sum':
            for map_class, weight, mask in zip(valid_maps, valid_weights, masks):
                integrated_map += (map_class * mask) * weight  # Mantiene el signo original

        elif operator == 'max':
            for feature_idx in range(num_features):
                relevant_methods = [i for i in range(len(valid_maps)) if masks[i][feature_idx]]
                if relevant_methods:
                    # Elegir el método con mayor peso, pero mantener el valor original (con signo)
                    max_weight_idx = max(relevant_methods, key=lambda idx: valid_weights[idx])
                    integrated_map[feature_idx] = valid_maps[max_weight_idx][feature_idx]

        integrated_maps_per_class.append(integrated_map)

    return np.array(integrated_maps_per_class)

def calculate_metric_union_integrated_explanations(explanations_data, instances_to_explain, metrics, model,X ,y, threshold=0.85, operator='sum', **kwargs):
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
    metric_integrated = []
    integrated_explanations = []  # Stores the integrated explanations
    x_selected = X.iloc[instances_to_explain].values
    feature_names = X.columns.tolist()
    for i in range(len(x_selected)):  # For each image
        # Collect explanations and LSE values for the 4 methods of the current image
        maps = [data['explanations'][i] for data in explanations_data]
        metric_values = [data['metric'][i] for data in explanations_data]

        # Apply the union operator per class to get the integrated explanation using the weights of this image
        integrated_explanation = union_operator_class_by_class(maps, metric_values, threshold, operator=operator)
        visualizar_explicacion(integrated_explanation, feature_names)
        integrated_explanations.append(integrated_explanation)

        # Calculate LSE for the integrated explanation using the selected class and filter threshold
    metric_value = evaluate_explanations(integrated_explanations, metrics, model,instances_to_explain, X, y, **kwargs)
    metric_integrated.extend(metric_value)

    return integrated_explanations, metric_integrated