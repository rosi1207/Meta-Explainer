import numpy as np
from aux_functions import normalize_signed, evaluate_explanations, visualizar_explicacion

def intersection_operator_class_by_class(maps, weights, threshold=0.8):
    """
    Intersection operator: Only features that are relevant (|value| > threshold)
    in ALL methods are kept.

    Args:
        maps (list): List of relevance matrices (n_methods, n_classes, n_features)
        weights (list): Weights for each method
        threshold (float): Threshold for relevance (0 to 1)

    Returns:
        np.ndarray: Integrated matrix (n_classes, n_features)
    """
    num_classes, num_features = np.array(maps[0]).shape
    integrated_maps_per_class = []

    for class_idx in range(num_classes):
        # Inicializar máscara de intersección (todos True inicialmente)
        intersection_mask = np.ones(num_features, dtype=bool)

        # Guardar mapas y pesos válidos
        valid_maps = []      # Normalizados a [0,1] para la máscara
        valid_maps_signed = []  # Normalizados a [-1,1] para el valor final
        valid_weights = []

        # Primer paso: recolectar mapas válidos
        for map_, weight in zip(maps, weights):
            map_class = map_[class_idx]

            # Verificar si el mapa es válido
            if not np.all(np.isfinite(map_class)):
                print(f"Warning: Invalid map for class {class_idx}, skipping...")
                continue

            # Normalizar de dos formas diferentes
            # Para máscara: usar valor absoluto normalizado a [0,1]
            abs_map = np.abs(map_class)
            max_abs = abs_map.max()
            if max_abs > 0:
                map_for_mask = abs_map / max_abs  # Rango [0,1]
            else:
                map_for_mask = abs_map

            # Para valor final: preservar signo, normalizar a [-1,1]
            map_signed = normalize_signed(map_class)  # Rango [-1,1]

            valid_maps.append(map_for_mask)
            valid_maps_signed.append(map_signed)
            valid_weights.append(weight)

        # Si no hay mapas válidos, devolver ceros
        if not valid_maps:
            integrated_maps_per_class.append(np.zeros(num_features))
            continue

        # Segundo paso: calcular intersección (solo features relevantes en TODOS)
        for map_for_mask in valid_maps:
            mask = map_for_mask > threshold
            intersection_mask &= mask

        print(f"Class {class_idx}: {np.sum(intersection_mask)} features in intersection (threshold={threshold})")

        # Tercer paso: sumar ponderadamente SOLO las features en intersección
        integrated_map = np.zeros(num_features, dtype=np.float64)

        for map_signed, weight in zip(valid_maps_signed, valid_weights):
            # Solo sumar donde hay intersección
            integrated_map[intersection_mask] += map_signed[intersection_mask] * weight

        # Opcional: normalizar por suma de pesos
        # integrated_map[intersection_mask] /= sum(valid_weights)

        integrated_maps_per_class.append(integrated_map)

    return np.array(integrated_maps_per_class)


def calculate_metric_intersection_integrated_explanations(explanations_data, instances_to_explain, metrics, model, X, y, threshold=0.85, **kwargs):
    """
    Calculates integrated explanations using intersection operator.
    """
    metric_integrated = []
    integrated_explanations = []
    x_selected = X.iloc[instances_to_explain].values
    feature_names = X.columns.tolist()

    for i in range(len(x_selected)):
        # Recolectar explicaciones de TODOS los métodos para esta instancia
        maps = [data['explanations'][i] for data in explanations_data]
        metric_values = [data['metric'][i] for data in explanations_data]

        # Aplicar intersección
        integrated_explanation = intersection_operator_class_by_class(
            maps, metric_values, threshold=threshold
        )

        # Visualizar
        visualizar_explicacion(integrated_explanation, feature_names)
        integrated_explanations.append(integrated_explanation)

    # Evaluar métricas
    metric_value = evaluate_explanations(
        integrated_explanations, metrics, model, instances_to_explain, X, y, **kwargs
    )
    metric_integrated.extend(metric_value)

    return integrated_explanations, metric_integrated