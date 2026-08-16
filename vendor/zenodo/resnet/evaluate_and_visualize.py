import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, hamming_loss, jaccard_score
import time
import os
import random  # Import random module for data augmentation
import matplotlib.pyplot as plt  # For plotting metrics
import pickle
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard monitoring
from preprocess_aa_data import preprocess_data
from oversample_undersample import oversample_undersample
from amino_acid_dataset_ed import AminoAcidDataset
from resnet1d import ResNet1DEmbedding
from train_model import train_model
from test_model import test_model

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)  # Ensure random operations are reproducible

# Define paths
sampled_data_path = '../Sequence_HM_clean.txt'  # Path to the sampled data
results_path = 'resnet18_main.txt'  # Path to save the results
model_path = 'resnet18_main7_model.pth'  # Path to save the trained model
plots_dir = 'plots/'  # Directory to save plots
metrics_path = 'best_model_resnet18_main.txt'  # File to save best model's metrics

# Create plots directory if it doesn't exist
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Load the sampled data
sampled_data = pd.read_csv(sampled_data_path, sep='\t')

sampled_data, amino_acid_to_index, vocab_size = preprocess_data(sampled_data)

# Use MultiLabelBinarizer to transform multi-label Compound column
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(sampled_data['Compound'])
# Save mlb
with open('mlb.pkl', 'wb') as f:
    pickle.dump(mlb, f)
print("MultiLabelBinarizer has been saved.")

# Output actual label classes and their count to check for errors
print(f"Number of classes: {len(mlb.classes_)}")
print(f"Classes: {mlb.classes_}")

# Assume that sampled_data and y, as well as mlb (MultiLabelBinarizer), are prepared
X, y = oversample_undersample(sampled_data, y, mlb)

# Split data into train, validation, and test sets (80-10-10 split)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)

# Define DataLoader parameters
batch_size = 128
max_length = 1024  # Adjust max length according to sequence data and GPU memory

# Create DataLoader for training, validation, and testing
train_dataset = AminoAcidDataset(X_train, y_train, amino_acid_to_index, max_length)
val_dataset = AminoAcidDataset(X_val, y_val, amino_acid_to_index, max_length)
test_dataset = AminoAcidDataset(X_test, y_test, amino_acid_to_index, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
embedding_dim = 128  # Set embedding dimension size
num_classes = len(mlb.classes_)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet1DEmbedding(vocab_size, embedding_dim, num_classes=num_classes).to(device)

# **Handle class imbalance: compute the weight for each class**
# Compute label counts in the training data
label_counts = np.sum(y_train, axis=0)
# Avoid division by zero and compute inverse frequency
label_weights = 1.0 / (label_counts + 1e-6)
# Normalize weights
label_weights = label_weights / np.sum(label_weights) * len(label_counts)
# Convert to tensor and move to device
class_weights = torch.tensor(label_weights, dtype=torch.float32).to(device)

# Define loss function with class weights
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

# **Add weight_decay parameter**
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

# Early stopping parameters
early_stopping_patience = 20

# **Global initialization of tracking variables**
best_jaccard = 0  # Initialize best Jaccard score
early_stopping_counter = 0  # Initialize early stopping counter
best_epoch = 0  # Initialize best epoch
best_train_metrics = {}  # Initialize best training metrics
best_val_metrics = {}  # Initialize best validation metrics

# Train the model
start_time = time.time()
best_epoch, best_train_metrics, best_val_metrics, train_times = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=200, device=device 
)

end_time = time.time()

# Use test_model function
test_model(
    model=model,
    test_loader=test_loader,
    model_path='best_model.pth',
    results_path='test_results.txt',
    metrics_path='metrics.txt',
    best_epoch=best_epoch,
    best_train_metrics=best_train_metrics,
    best_val_metrics=best_val_metrics,
    device=device
)
