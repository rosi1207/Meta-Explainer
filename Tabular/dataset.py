import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer 

from sklearn.datasets import load_breast_cancer
import pandas as pd

def prepare_breast_cancer_dataset():
    """
    Prepares the breast cancer dataset from scikit-learn for analysis.

    Description:
    - This function loads the breast cancer dataset, converts it into a pandas DataFrame,
      and adds a target column (`target`) containing the class labels (0 for malignant and 1 for benign).
    
    Returns:
    df : DataFrame
        The DataFrame containing the dataset features along with the target column `target`.
    target_column : str
        The name of the target column (`target`), which is used for modeling and further analysis.
    """
    
    # Load the breast cancer dataset from scikit-learn
    data = load_breast_cancer()
    
    # Convert the data into a pandas DataFrame with features as columns
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    # Add the target column (0 for malignant, 1 for benign)
    df['target'] = data.target

    return df, 'target'

def preprocess_dataset(df, target_column):
    """
    Preprocessing function that performs the following tasks on the DataFrame:
    - Removes rows with null values.
    - Identifies numeric columns.
    - Applies one-hot encoding to categorical columns (excluding the target column).
    - Identifies the original features in the DataFrame.
    - Filters the necessary columns to keep only relevant features and the target column.
    - Creates a feature map that associates original columns with their encoded derivatives.

    Parameters:
    df : DataFrame
        The input DataFrame containing all features and the target column.
    target_column : str or list
        Name of the target column or list of names if there are multiple target columns.

    Returns:
    df : DataFrame
        The processed DataFrame with categorical columns encoded in one-hot format.
    feature_names : list
        List of names of the encoded feature columns.
    class_values : list
        List of unique class values in the target column.
    numeric_columns : list
        List of names of the original numeric columns.
    rdf : DataFrame
        Filtered DataFrame including only the necessary features and the target column.
    real_feature_names : list
        List of names of the original features.
    features_map : dict
        Dictionary mapping each original feature to its encoded columns.
    """

    # Remove rows with null values in the DataFrame
    df = df.dropna()
    
    # Get numeric columns using select_dtypes
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    print(f"NUMERIC COLUMNS:\n{numeric_columns}")

    # Create a copy of the original DataFrame to preserve the structure before encoding
    rdf = df.copy()

    # Apply one-hot encoding to all categorical columns except the target column
    df, feature_names, class_values = one_hot_encoding(df, target_column)
    print(f"FEATURE NAMES:\n{feature_names}\nCLASS VALUES: {class_values}")

    # Get names of real features (before one-hot encoding)
    real_feature_names = get_real_feature_names(rdf, numeric_columns, target_column)
    print(f"REAL FEATURE NAMES:\n{real_feature_names}")

    # Get names of categorical columns
    categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()
    print(f"CATEGORICAL FEATURE NAMES:\n{categorical_columns}")

    # Filter the original DataFrame to keep only real feature columns and the target column
    rdf = rdf[real_feature_names + (class_values if isinstance(target_column, list) else [target_column])]

    # Create the feature map, associating original columns with their encoded versions
    features_map = get_features_map(feature_names, real_feature_names)
    print(f"FEATURE MAP: {features_map}")

    # Return all processed elements needed for the subsequent workflow
    return df, feature_names, class_values, numeric_columns, rdf, real_feature_names, categorical_columns, features_map

# Function to perform one-hot encoding
def one_hot_encoding(df, target_column):
    """
    Performs one-hot encoding on all categorical columns in the DataFrame except the target column.
    
    Parameters:
    df : DataFrame
        The DataFrame containing the data to be encoded.
    target_column : str
        The name of the target column (which is not included in the one-hot encoding).
    
    Returns:
    df : DataFrame
        The resulting DataFrame after applying one-hot encoding and mapping the target column.
    feature_names : list
        List of names of the encoded feature columns.
    class_values : list
        List of unique class values in the target column (before mapping to numeric values).
    
    Description:
    This method applies one-hot encoding to all categorical columns in the DataFrame, excluding the target column.
    Additionally, it maps the target column values to numeric values, making it easier to use in machine learning algorithms.
    """

    # Apply one-hot encoding to all categorical columns except the target column
    dfX = pd.get_dummies(df.drop(columns=[target_column]), prefix_sep='=')

    # Map the target column to numeric values
    class_name_map = {v: k for k, v in enumerate(sorted(df[target_column].unique()))}
    dfY = df[target_column].map(class_name_map)

    # Combine the encoded features with the mapped target column
    df = pd.concat([dfX, dfY], axis=1)

    # Get the names of the feature columns
    feature_names = list(dfX.columns)

    # Get the unique class values in the target column (before mapping)
    class_values = sorted(class_name_map)

    return df, feature_names, class_values

# Function to get real feature names
def get_real_feature_names(rdf, numeric_columns, target_column):
    """
    Retrieves a sorted list of the names of "real" feature columns in the DataFrame,
    excluding the target column. This function helps differentiate between numeric
    and categorical feature columns, maintaining the order of types (numeric first,
    then categorical) and excluding the target column to avoid duplication in modeling.

    Parameters:
    rdf : DataFrame
        The original DataFrame containing all columns, including categorical and target columns.
    numeric_columns : list
        List of names of numeric columns in the DataFrame.
    target_column : str or list
        Name of the target column, or list of names if there are multiple target columns.

    Returns:
    real_feature_names : list
        List of the names of feature columns (excluding the target column),
        with numeric columns first, followed by categorical columns.
    """
    
    # Get numeric columns excluding the target column
    real_feature_names = [col for col in numeric_columns if col not in (target_column if isinstance(target_column, list) else [target_column])]
    
    # Add non-numeric columns excluding the target column
    real_feature_names += [col for col in rdf.columns if col not in numeric_columns and col not in (target_column if isinstance(target_column, list) else [target_column])]
    
    return real_feature_names


from collections import defaultdict

def get_features_map(feature_names, real_feature_names):
    """
    Creates a feature map that associates each original column (real_feature_names)
    with its derived encoded columns (feature_names), if applicable.

    Parameters:
    feature_names : list
        List of names of the encoded columns (e.g., from one-hot encoding).
    real_feature_names : list
        List of names of the original columns.

    Returns:
    features_map : dict
        A dictionary mapping each original feature to its encoded columns,
        represented as a defaultdict with internal dictionaries.
    """
    
    features_map = defaultdict(dict)
    j = 0

    for i, feature in enumerate(feature_names):
        # Check if the encoded feature directly corresponds to an original one
        if j < len(real_feature_names) and feature.startswith(real_feature_names[j]):
            # Extract the encoded value after the original feature name
            encoded_value = feature.replace(f'{real_feature_names[j]}=', '')
            features_map[j][encoded_value] = i
            
            # If it's an exact match, move to the next real feature
            if feature == real_feature_names[j]:
                j += 1
        elif j < len(real_feature_names):
            # Move to the next index of real features if there's no partial match
            j += 1

    return features_map