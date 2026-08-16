import pandas as pd
import numpy as np

def oversample_undersample(sampled_data, y, mlb):
    """
    Perform oversampling and undersampling on the dataset.
    
    Args:
        sampled_data (pd.DataFrame): DataFrame containing 'Sequence' and 'Labels'.
        y (np.ndarray): Multi-hot encoded labels.
        mlb (MultiLabelBinarizer): Fitted MultiLabelBinarizer used for encoding the labels.
    
    Returns:
        pd.DataFrame: Balanced DataFrame with oversampled and undersampled sequences.
        np.ndarray: Balanced multi-hot encoded labels.
    """
    # Combine sequences and labels into a DataFrame
    data = pd.DataFrame({'Sequence': sampled_data['Sequence'], 'Labels': list(y)})

    # Convert labels back to tuple for easier manipulation
    data['Labels'] = data['Labels'].apply(lambda x: tuple(x))  # Convert numpy.ndarray to tuple

    # Calculate label counts
    label_counts = np.sum(y, axis=0)
    max_count = np.max(label_counts)
    min_count = np.min(label_counts)

    # Create a list to store the new balanced data
    new_data_list = []

    for idx, label in enumerate(mlb.classes_):
        # Get the samples with the current label
        label_data = data[data['Labels'].apply(lambda x: x[idx] == 1)]
        count = label_counts[idx]
        
        # Oversample if count < max_count
        if count < max_count:
            num_samples_to_add = max_count - count
            sampled_data_to_add = label_data.sample(num_samples_to_add, replace=True, random_state=42)
            new_data_list.append(sampled_data_to_add)
        # Only undersample if the label is not the one with max_count
        elif count > min_count and count != max_count:
            num_samples_to_remove = count - min_count
            sampled_data = label_data.sample(count - num_samples_to_remove, random_state=42)
            new_data_list.append(sampled_data)
        else:
            new_data_list.append(label_data)

    # Combine all the processed data
    balanced_data = pd.concat(new_data_list).drop_duplicates().reset_index(drop=True)

    # Update sequences and labels
    X_balanced = balanced_data['Sequence']
    # Convert tuple labels back to numpy array
    y_balanced = np.array([np.array(label) for label in balanced_data['Labels']])

    return X_balanced, y_balanced
