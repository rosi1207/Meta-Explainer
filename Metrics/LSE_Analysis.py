import numpy as np
import matplotlib.pyplot as plt
from aux_functions import ensure_single_channel, is_valid_map, normalize_map

def sum_n_val(index, analysis):
    """ Sums the values of the relevant indices for classification using NumPy. """
    analysis = np.asarray(analysis)
    index = np.asarray(index)
    selected_values = analysis[:, index]  # Select relevant features
    return selected_values.sum(axis=1)  # Sum relevances for the classes

def lse_calculate(relevances, class_index, filter=0, data_type="image"):
    """
    Calculates the IFI using heatmaps of relevances or tabular relevances.

    Args:
        relevances (list of np.ndarray): List of heatmaps of relevances or tabular relevances.
        class_index (int): Index of the target class for the calculation.
        filter (float): Minimum threshold to consider a value relevant.
        data_type (str): Type of data, can be "image" or "tabular".

    Returns:
        float: Calculated IFI value.
    """
    # Configure heatmap visualization (only for images)
    if data_type == "image":
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle('Heatmaps', fontsize=16)

    # Process and store valid maps and invalid maps as zeros
    valid_relevances = []
    invalid_count = 0  # Counter for invalid maps

    for i, relevance in enumerate(relevances):
        if data_type == "image":
            row, col = divmod(i, 5)
            ax = axs[row, col]
            ax.imshow(relevance, cmap='seismic')
            ax.axis('off')

        # Process the relevance map based on the data type
        if data_type == "image":
            if is_valid_map(relevance):
                processed_map = ensure_single_channel(relevance)
                processed_map = normalize_map(processed_map)
                valid_relevances.append(processed_map)
            else:
                valid_relevances.append(np.zeros_like(relevance))
                invalid_count += 1
        elif data_type == "tabular":
            # Normalize and verify tabular relevances
            normalized_relevance = normalize_map(relevance)
            if np.all(np.isfinite(normalized_relevance)):  # Verify there are no NaN or infinite values
                valid_relevances.append(normalized_relevance)
            else:
                invalid_count += 1

    print(f"Normalized Maps: {valid_relevances}")

    if data_type == "image":
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # Print information about valid maps
    print(f"Original shape of relevances: {np.array(relevances).shape}")
    print(f"Valid maps (including zeros for invalid ones): {len(valid_relevances)}")
    print(f"Number of invalid maps: {invalid_count}")
    print(f"Shape of valid maps: {np.array(valid_relevances).shape}")

    # Flatten maps for analysis (only necessary for images)
    if data_type == "image":
        f_analysis = [arr.flatten() for arr in valid_relevances]
    else:
        f_analysis = valid_relevances  # For tabular data, they are already in the correct format

    print(f"Shape of flattened relevances: {np.array(f_analysis).shape}")

    # Get the relevance vector for the target class
    f_analysis_class = np.asarray(f_analysis[class_index])
    print(f"Shape of class relevance: {np.array(f_analysis_class).shape}")

    # Print maximum and minimum values of each flattened map for verification
    for idx, selected in enumerate(f_analysis):
        print(f"Map {idx}: Max: {np.max(selected)}, Min: {np.min(selected)}")

    # Select indices with values greater than or equal to the filter
    index_max = np.where(f_analysis_class >= filter)[0]
    print(f"Number of features greater than {filter}: {len(index_max)}")
    print(f"Indices greater than {filter}: {index_max}")

    # Calculate the sum of relevant values for each class
    sum_vals = sum_n_val(index=index_max, analysis=f_analysis)

    # Relevance of the target class vs. the other classes
    val_class = sum_vals[class_index]
    val_other = sum(sum_vals[i] for i in range(len(sum_vals)) if i != class_index)
    print(f"Sum of target class: {val_class}\nSum of other classes: {val_other}")

    # Calculate the IFI
    if val_other > 0:
        ifi = ((len(valid_relevances)-1) * val_class) / val_other
    elif val_class > 0 and val_other == 0:
        ifi = (len(valid_relevances)-1) * val_class
    else:
        ifi = 0

    # Print the IFI result
    print(f"Calculated IFI: {ifi}\n")

    return ifi