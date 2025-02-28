import numpy as np
from aux_functions import normalize_map

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
        valid_maps = [normalize_map(map_) for map_ in processed_maps if np.all(np.isfinite(map_))]
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

def union_operator_class_by_class(maps, weights, threshold=0.8, operator='sum'):
    """
    Applies the union operator to integrate tabular explanations from different methods by class.
    Creates a binary mask for each map, where values greater than the threshold are relevant
    and are summed or the maximum value is selected.

    Args:
        maps (list of np.ndarray): List of relevance matrices with dimensions (n_methods, n_classes, n_features).
        weights (list of float): Weights to apply in the union operator.
        threshold (float): Threshold to determine which values are relevant (between 0 and 1).
        operator (str): Operator to use for combining explanations: 'sum' for weighted sum, 'max' for maximum selection.

    Returns:
        np.ndarray: Integrated relevance matrix per class with dimensions (n_classes, n_features).
    """
    print(f"Weights: {weights}")
    num_classes, num_features = np.array(maps[0]).shape  # Get the number of classes and features
    integrated_maps_per_class = []

    for class_idx in range(num_classes):
        integrated_map = np.zeros(num_features, dtype=np.float64)

        valid_maps = []
        valid_weights = []
        masks = []

        # Filter valid maps (those without NaN or Inf) and their corresponding weights
        for map_, weight in zip(maps, weights):
            map_class = map_[class_idx]
            # Normalize values between 0 and 1 (min-max scaling)
            map_class_normalized = normalize_map(map_class)
            if np.all(np.isfinite(map_class_normalized)):  # Ensure the map is valid
                valid_maps.append(map_class_normalized)
                valid_weights.append(weight)
                # Create a binary mask (1 if the value exceeds the threshold, 0 otherwise)
                mask = map_class_normalized > threshold
                masks.append(mask)

                # Print the number of features greater than the threshold in this map
                print(f"Class {class_idx}: Map with {np.sum(mask)} features greater than the threshold ({threshold})")

        if not valid_maps:
            integrated_maps_per_class.append(integrated_map)
            continue

        valid_maps = np.array(valid_maps)
        valid_weights = np.array(valid_weights)
        masks = np.array(masks)

        if operator == 'sum':
            # Weighted sum of explanations, only where the mask is relevant (greater than the threshold)
            for map_class, weight, mask in zip(valid_maps, valid_weights, masks):
                integrated_map += (map_class * mask) * weight

        elif operator == 'max':
            # Maximum selection per feature (based on the relevance threshold)
            for feature_idx in range(num_features):
                relevant_methods = [i for i in range(len(valid_maps)) if masks[i][feature_idx]]
                if relevant_methods:
                    max_weight_idx = max(relevant_methods, key=lambda idx: valid_weights[idx])
                    integrated_map[feature_idx] = valid_maps[max_weight_idx][feature_idx]

        integrated_maps_per_class.append(integrated_map)

    return np.array(integrated_maps_per_class)

def intersection_operator_class_by_class(maps, weights, threshold=0.8):
    """
    Applies the intersection operator to combine tabular explanations from different methods by class.
    Only features that are relevant (greater than the threshold) in all methods are summed.

    Args:
        maps (list of np.ndarray): List of relevance matrices with dimensions (n_methods, n_classes, n_features).
        weights (list of float): Weights to apply in the intersection operator.
        threshold (float): Threshold to determine which values are relevant (between 0 and 1).

    Returns:
        np.ndarray: Integrated relevance matrix per class with dimensions (n_classes, n_features).
    """
    num_classes, num_features = np.array(maps[0]).shape  # Get the number of classes and features
    integrated_maps_per_class = []

    for class_idx in range(num_classes):
        # Initialize an empty map to accumulate relevant values per class
        integrated_map = np.zeros(num_features, dtype=np.float64)

        # Create an intersection mask that initially marks all features as relevant
        intersection_mask = np.ones(num_features, dtype=bool)  # (n_features,)

        total_relevant_elements = 0
        valid_maps = []
        valid_weights = []

        # Filter valid maps before processing the intersection
        for map_, weight in zip(maps, weights):
            map_class = map_[class_idx]
            normalized_map_class = normalize_map(map_class)
            if np.all(np.isfinite(normalized_map_class)):  # Ensure the map is valid
                valid_maps.append(normalized_map_class)
                valid_weights.append(weight)
            else:
                # If the map has invalid values, ignore it
                valid_maps.append(np.zeros_like(map_class))
                valid_weights.append(weight)

        # Apply the intersection operator only if there are valid maps
        for map_class in valid_maps:
            # Create a binary mask for each method (1 if the value exceeds the threshold, 0 otherwise)
            mask = map_class > threshold
            print(f'Class {class_idx}: {np.sum(mask)} elements in the map')

            # Update the intersection mask considering only features relevant in all methods
            intersection_mask &= mask  # Keep relevant only where all methods agree

        print(f'Class {class_idx}: {np.sum(intersection_mask)} Total intersected elements')

        # Accumulate weighted values in the integrated map for features in the intersection
        for map_class, weight in zip(valid_maps, valid_weights):
            integrated_map[intersection_mask] += map_class[intersection_mask] * weight

        # Show the total number of relevant elements in the intersection of all methods
        intersection_relevant_elements = np.sum(intersection_mask)
        # print(f'Class {class_idx}: {intersection_relevant_elements} elements in the intersection of all methods')

        integrated_maps_per_class.append(integrated_map)

    return np.array(integrated_maps_per_class)