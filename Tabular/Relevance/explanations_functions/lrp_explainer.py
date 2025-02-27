import innvestigate
import numpy as np

def iNN_explanations(model, instance, method):
    """
    Generates explanations for a given instance using the provided model and the specified iNNvestigate method.

    Args:
        model: The trained machine learning model with a softmax output layer.
        instance: The input instance to explain (1D array-like).
        method: A string specifying the iNNvestigate method to use for generating explanations 
                (e.g., "lrp.alpha_1_beta_0", "integrated_gradients").

    Returns:
        explanations_data: A list of explanations for each class, where each explanation is a 2D array 
                           representing the relevance values for the input features.
    """
    try:
        # Remove the softmax layer from the model for analysis
        model_wo_softmax_ = innvestigate.model_wo_softmax(model)
    except Exception as e:
        print(f"Error removing Softmax: {e}")
        model_wo_softmax_ = model  # Use the original model if an error occurs

    # Create the analyzer with the specified method
    analyzer = innvestigate.create_analyzer(method, model_wo_softmax_, neuron_selection_mode="index")

    explanations_data = []

    # Iterate over all classes (assuming binary classification for this example)
    for class_idx in range(2):
        # Analyze the instance for the selected class
        a = analyzer.analyze(instance.reshape(1, -1), neuron_selection=class_idx)
        explanations_data.append(a.tolist())

    return explanations_data


def extract_iNN_relevances(explanations_df):
    """
    Extracts and processes the relevance values from iNNvestigate explanations.

    Args:
        explanations_df: A list of explanations, where each explanation is a 2D array 
                             representing the relevance values for the input features.

    Returns:
        final_relevance_array: A numpy array containing the relevance values for each class and feature.
    """
    # Initialize a list to store the processed relevance values
    relevance_data = explanations_df

    all_relevances = []

    # Iterate over each class's relevance values
    for class_idx in range(len(relevance_data)):
        # Convert the explanation for the class into a numpy array
        class_relevances = np.array(relevance_data[class_idx][0])  # Convert the explanation to an array
        all_relevances.append(class_relevances)

    # Convert the list of relevance values into a single numpy array
    final_relevance_array = np.array(all_relevances)

    return final_relevance_array