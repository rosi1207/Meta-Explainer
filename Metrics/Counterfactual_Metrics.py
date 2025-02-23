from scipy.spatial.distance import cdist
import numpy as np

def calculate_implausibility(counterfactuals, reference_data, reference_labels, model_predict, metric='euclidean', input_shape=(28, 28, 1)):
    """
    Calculate normalized minimum distances of counterfactuals to reference data.
    Only considers reference instances with the same classification as the model's prediction.

    Parameters:
        - counterfactuals: List or array of counterfactuals (each counterfactual should be an array).
        - reference_data: Reference data for distance calculation (feature matrix).
        - reference_labels: Labels corresponding to the reference dataset.
        - model_predict: Function to predict class for each counterfactual.
        - metric: Distance metric to use (default 'euclidean').
        - input_shape: Expected model input shape.

    Returns:
        - List of normalized minimum distances for each counterfactual.
    """
    distances = []
    print("Calculating implausibility...")
    for idx, cf in enumerate(counterfactuals):
        print(f"\nCounterfactual #{idx + 1}")

        # Reshape counterfactual to match model input
        cf_reshaped = cf.reshape((1,) + input_shape)

        # Get predicted class for the counterfactual
        probabilities = model_predict(cf_reshaped)
        cf_label = np.argmax(probabilities, axis=1)[0]
        print(f"Predicted class for counterfactual: {cf_label}")

        # Filter reference instances with the same class
        same_class_instances = reference_data[reference_labels == cf_label]
        print(f"Number of instances in the same class ({cf_label}): {len(same_class_instances)}")

        # If no instances in the same class, assign max penalization
        if len(same_class_instances) == 0:
            print("No instances in the same class. Penalizing with 1.")
            distances.append(1.0)
            continue

        # Flatten images for distance calculation
        cf_flattened = cf.flatten()
        same_class_instances_flattened = same_class_instances.reshape(len(same_class_instances), -1)

        # Calculate distances between counterfactual and same-class instances
        raw_distances = cdist([cf_flattened], same_class_instances_flattened, metric=metric)

        # Normalize by finding the max distance among same-class instances
        max_distance = np.max(cdist(same_class_instances_flattened, same_class_instances_flattened, metric=metric))

        # Ensure max_distance is not zero to avoid division by zero
        if max_distance == 0:
            print("All instances are identical, setting distance to 0.")
            normalized_distance = 0.0
        else:
            # Normalize to [0, 1] range
            min_distance = np.min(raw_distances)
            normalized_distance = min_distance / max_distance

        print(f"Minimum normalized distance: {normalized_distance}")
        distances.append(normalized_distance)

    # Return normalized distances
    print(f"\nNormalized minimum distances for all counterfactuals: {distances}")
    return distances

def calculate_feasibility(counterfactuals, original_instances, reference_data, k=5, metric='euclidean'):
    """
    Calculate feasibility by averaging the distance between counterfactuals and the k nearest neighbors
    of the original instances.

    Parameters:
        - counterfactuals: List or array of counterfactuals.
        - original_instances: List or array of original instances.
        - reference_data: Reference data.
        - k: Number of nearest neighbors.
        - metric: Distance metric.

    Returns:
        - List of feasibility scores for each counterfactual.
    """
    feasibility_scores = []
    print("Calculating feasibility...")

    # Ensure reference_data is two-dimensional
    if len(reference_data.shape) == 3 or len(reference_data.shape) == 4:
        reference_data = reference_data.reshape(len(reference_data), -1)

    for idx, (cf, orig) in enumerate(zip(counterfactuals, original_instances)):
        print(f"\nCounterfactual #{idx + 1}")

        # Flatten the original instance
        orig_flat = orig.flatten()

        # Calculate distances between the original instance and all reference data
        distances = cdist([orig_flat], reference_data, metric=metric).flatten()

        # Get the k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]
        nearest_distances = cdist([cf], reference_data[nearest_indices], metric=metric).flatten()

        # Calculate the average distance to the k nearest neighbors
        feasibility = np.mean(nearest_distances)
        print(f"Feasibility: {feasibility:.4f}")
        feasibility_scores.append(feasibility)

    return feasibility_scores

def calculate_discriminative_power(counterfactuals, reference_data, reference_labels, model_predict, k=5, metric='euclidean', input_shape=(28, 28, 1)):
    """
    Calculate the discriminative power of counterfactuals.
    """
    discriminative_power_scores = []
    print("Calculating discriminative power...")

    # Ensure reference_data is two-dimensional
    if len(reference_data.shape) == 3 or len(reference_data.shape) == 4:
        reference_data = reference_data.reshape(len(reference_data), -1)

    for idx, cf in enumerate(counterfactuals):
        print(f"\nCounterfactual #{idx + 1}")

        # Reshape the counterfactual to match the model's expected input format
        cf_reshaped = cf.reshape((1,) + input_shape)

        # Get the predicted class for the counterfactual
        cf_label = np.argmax(model_predict(cf_reshaped), axis=1)[0]
        print(f"Predicted class for counterfactual: {cf_label}")

        # Filter reference instances that belong to the same class as the counterfactual
        same_class_instances = reference_data[reference_labels == cf_label]

        if len(same_class_instances) == 0:
            print("No instances in the same class. Assigning maximum distance.")
            discriminative_power_scores.append(1.0)
            continue

        # Calculate distances and select the k nearest neighbors
        cf_flattened = cf.flatten()
        same_class_instances_flat = same_class_instances.reshape(len(same_class_instances), -1)
        distances = cdist([cf_flattened], same_class_instances_flat, metric=metric).flatten()
        nearest_distances = np.sort(distances)[:k]

        # Calculate the average distance
        dp_score = np.mean(nearest_distances)
        print(f"Discriminative power: {dp_score:.4f}")
        discriminative_power_scores.append(dp_score)

    return discriminative_power_scores

def calculate_proximity(counterfactuals, originals, metric='euclidean'):
    """
    Calculate normalized distances between counterfactuals and their original instances.

    Parameters:
        - counterfactuals: List or array of counterfactuals.
        - originals: List or array of original instances.
        - metric: Distance metric to use (default 'euclidean').

    Returns:
        - List of normalized distances for each counterfactual relative to its original instance.
    """
    distances = []
    print("Calculating proximity...")
    for idx, (cf, orig) in enumerate(zip(counterfactuals, originals)):
        print(f"\nCounterfactual #{idx + 1}")
        print(f"Original instance #{idx + 1}")

        # Calculate the distance between the counterfactual and the original instance
        raw_distance = cdist([cf], [orig], metric=metric)[0][0]

        # Normalize by the norm of the original instance
        norm_factor = np.linalg.norm(orig)
        if norm_factor == 0:
            print("The original instance has zero norm, which may cause issues in normalization.")
            norm_factor = 1  # Avoid division by zero

        normalized_distance = raw_distance / norm_factor
        print(f"Normalized distance between counterfactual and original instance: {normalized_distance}")
        distances.append(normalized_distance)

    # Return normalized distances
    print(f"\nNormalized distances for all counterfactuals relative to originals: {distances}")
    return distances

def calculate_sparsity(image1, image2, threshold=0.1):
    """
    Calculate the changes between two images and return the total number of changes
    and the number of changes above a specified threshold.

    Parameters:
        - image1: Original image.
        - image2: Perturbed image.
        - threshold: Threshold for considering a significant change.

    Returns:
        - total_changes: Total number of changes in the image.
        - significant_changes: Number of changes above the threshold.
    """
    # Calculate the difference between the two images
    diff = np.abs(image1 - image2)
    total_changes = np.sum(diff > 0)  # Total number of changes
    significant_changes = np.sum(diff > threshold)  # Changes above the threshold

    return total_changes, significant_changes