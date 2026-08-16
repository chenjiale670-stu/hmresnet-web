import torch
from torch.utils.data import Dataset

class AminoAcidDataset(Dataset):
    def __init__(self, sequences, labels, aa_to_index, max_length=512):
        """
        Custom Dataset for amino acid sequences with embedding.

        Args:
            sequences (pd.Series): Series containing amino acid sequences.
            labels (np.ndarray): Multi-labels for each sequence.
            aa_to_index (dict): Dictionary mapping amino acids to indices.
            max_length (int): Maximum length for sequence padding/truncation. Default is 512.
        """
        self.sequences = sequences.reset_index(drop=True)  # Reset index to avoid potential issues
        self.labels = labels
        self.aa_to_index = aa_to_index
        self.max_length = max_length

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx):
        """Fetch a sequence and its label for a given index."""
        sequence = self.sequences.iloc[idx]  # Get the amino acid sequence
        # Encode sequence: use 0 for unknown amino acids
        encoded_seq = [self.aa_to_index.get(aa, 0) for aa in sequence]  
        # Pad or truncate the sequence to the maximum length
        padded_seq = encoded_seq + [0] * (self.max_length - len(encoded_seq))  
        padded_seq = padded_seq[:self.max_length]  # Truncate if sequence is longer than max_length
        
        # Return the sequence as tensor and the corresponding label as tensor
        return torch.tensor(padded_seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)
