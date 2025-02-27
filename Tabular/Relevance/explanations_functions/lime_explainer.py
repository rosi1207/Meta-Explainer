from lime import lime_tabular
import numpy as np

def lime_explanations(model, instance, X_train, feature_names, Y_train):
    """
    Generates LIME explanations for a given instance using the provided model and training data.

    Args:
        model: The trained machine learning model.
        instance: The input instance to explain (1D array-like).
        X_train: The training data used to train the model (2D array-like).
        feature_names: List of feature names corresponding to the columns in X_train.
        Y_train: The target labels used to train the model (1D array-like).

    Returns:
        explanations_data: A list of explanations for each class, where each explanation
                           is a list of tuples (feature, relevance value).
    """
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=[str(cls) for cls in np.unique(Y_train)],  # Ensure correct class names
        discretize_continuous=True
    )
    pred_proba = model.predict(instance.reshape(1, -1))

    # Get explanations for all classes
    exp = explainer.explain_instance(instance, model.predict, num_features=len(feature_names), top_labels=pred_proba.shape[1])

    explanations_data = []

    for class_idx in range(pred_proba.shape[1]):
        explanations_data.append(exp.as_list(label=class_idx))

    return explanations_data


def extract_relevances(lime_explanations_df):
    """
    Extracts only the relevance values from the LIME explanations.

    Args:
        lime_explanations_df: A list of LIME explanations, where each explanation is a list of tuples 
                              (feature, relevance value) for each class.

    Returns:
        relevance_array: A numpy array containing only the relevance values for each feature and class.
    """
    # Initialize a list to store the new data
    relevance_data = []

    # Iterate over all rows in the DataFrame
    for row in lime_explanations_df:
        # Iterate over the explanations for each class
        relevances_per_class = []

        # For each class, extract only the relevance values from the explanations
        for explanation in row:
            # Extract the relevance value (second value in the tuple)
            relevance_value = explanation[1]
            relevances_per_class.append(relevance_value)

        # Add the row with the relevance values
        relevance_data.append(relevances_per_class)

    # Convert the list of relevance values to a numpy array
    relevance_array = np.array(relevance_data)
    return relevance_array