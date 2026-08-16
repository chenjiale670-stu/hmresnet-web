import random
import pandas as pd
import pickle

# **Remove low-frequency amino acids**
unwanted_amino_acids = set(['X', 'U', 'B', 'J', 'Z'])

def remove_unwanted_amino_acids(sequence, unwanted_aa):
    """Remove unwanted amino acids from the sequence."""
    return ''.join([aa for aa in sequence if aa not in unwanted_aa])

# **Define the 20 standard amino acids**
standard_amino_acids = list('ACDEFGHIKLMNPQRSTVWY')

def filter_standard_amino_acids(sequence, standard_aa):
    """Filter out non-standard amino acids from the sequence."""
    return ''.join([aa for aa in sequence if aa in standard_aa])

# **Data Augmentation Function**
def random_substitution(sequence, n=1):
    """Randomly substitute characters in the sequence n times."""
    sequence = list(sequence)
    length = len(sequence)
    for _ in range(n):
        idx = random.randint(0, length-1)
        amino_acid = random.choice(standard_amino_acids)
        sequence[idx] = amino_acid
    return ''.join(sequence)

def augment_sequence(sequence):
    """Apply random substitution augmentation to the sequence."""
    return random_substitution(sequence)

# **Clean Labels Function for Multi-label Classification**
def clean_labels(compound):
    """Clean and split labels for multi-label classification."""
    labels = [label.strip() for label in compound.split(',')]
    return [label for label in labels if label]

def preprocess_data(sampled_data):
    """Main function to preprocess the data."""
    
    # Remove unwanted amino acids
    sampled_data['Sequence'] = sampled_data['Sequence'].apply(lambda seq: remove_unwanted_amino_acids(seq, unwanted_amino_acids))

    # Remove sequences that become empty after removing unwanted amino acids
    sampled_data = sampled_data[sampled_data['Sequence'].str.len() > 0]

    # Ensure sequences only contain standard amino acids
    sampled_data['Sequence'] = sampled_data['Sequence'].apply(lambda seq: filter_standard_amino_acids(seq, standard_amino_acids))

    # Remove sequences that become empty after filtering standard amino acids
    sampled_data = sampled_data[sampled_data['Sequence'].str.len() > 0]

    # Count unique amino acids in the sequences and create an index
    amino_acids = set(''.join(sampled_data['Sequence']))
    amino_acid_to_index = {aa: idx for idx, aa in enumerate(sorted(amino_acids), start=1)}  # Start indexing from 1
    vocab_size = len(amino_acid_to_index) + 1  # Plus 1 for padding index
    with open('amino_acid_to_index.pkl', 'wb') as f:
        pickle.dump(amino_acid_to_index, f)
    print("amino_acid_to_index has been saved.")

    # Apply data augmentation (random substitution)
    augmented_data = sampled_data.copy()
    augmented_data['Sequence'] = augmented_data['Sequence'].apply(augment_sequence)

    # Concatenate the original and augmented data
    sampled_data = pd.concat([sampled_data, augmented_data]).reset_index(drop=True)

    # Clean the 'Compound' column for multi-label classification
    sampled_data['Compound'] = sampled_data['Compound'].apply(clean_labels)

    return sampled_data, amino_acid_to_index, vocab_size
