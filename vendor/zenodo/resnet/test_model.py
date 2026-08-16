# test_model.py

import time
import numpy as np
from sklearn.metrics import hamming_loss, jaccard_score, f1_score, precision_score, recall_score
import torch

def test_model(model, test_loader, model_path, results_path, metrics_path, best_epoch, best_train_metrics, best_val_metrics, device='cpu'):
    # Load the best model for testing
    model.load_state_dict(torch.load(model_path))

    # Initialize lists to store test labels and predictions
    test_labels = []
    test_preds = []
    
    # Start the timer for overall test duration
    start_time = time.time()

    # Initialize a variable to track total sequence length in base pairs
    total_length_bp = 0.0

    # Evaluate on the test set
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5  # Apply sigmoid and threshold at 0.5

            # Add inputs' lengths to the total length counter (assuming inputs are sequences of amino acids)
            total_length_bp += inputs.size(0)  # Adjust if necessary for sequence representation

            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())

    # Calculate overall test metrics
    test_labels = np.array(test_labels)
    test_preds = np.array(test_preds)

    test_hamming_loss = hamming_loss(test_labels, test_preds)
    test_jaccard = jaccard_score(test_labels, test_preds, average='samples')
    test_f1_macro = f1_score(test_labels, test_preds, average='macro')
    test_f1_weighted = f1_score(test_labels, test_preds, average='weighted')
    test_precision = precision_score(test_labels, test_preds, average='micro')
    test_recall = recall_score(test_labels, test_preds, average='micro')

    # Calculate test duration and per 1000bp time
    end_time = time.time()
    total_test_time = end_time - start_time
    time_per_1000bp = total_test_time / (total_length_bp / 1000)

    # Save the results to the results file
    with open(results_path, 'w') as results_file:
        results_file.write("ResNet-1D with Embedding Layer Multi-Label Model Evaluation Results:\n")
        results_file.write(f"Test Hamming Loss: {test_hamming_loss:.4f}\n")
        results_file.write(f"Test Jaccard Score: {test_jaccard:.4f}\n")
        results_file.write(f"Test F1 Macro: {test_f1_macro:.4f}\n")
        results_file.write(f"Test F1 Weighted: {test_f1_weighted:.4f}\n")
        results_file.write(f"Test Precision: {test_precision:.4f}\n")
        results_file.write(f"Test Recall: {test_recall:.4f}\n")
        results_file.write(f"Total Test Time: {total_test_time:.2f} seconds\n")
        results_file.write(f"Time per 1000bp: {time_per_1000bp:.4f} seconds\n")
        
        # Include best model's epoch and metrics
        results_file.write(f"\nBest Model Epoch: {best_epoch}\n")
        results_file.write("Best Training Metrics:\n")
        for key, value in best_train_metrics.items():
            results_file.write(f"{key}: {value:.4f}\n")
        results_file.write("Best Validation Metrics:\n")
        for key, value in best_val_metrics.items():
            results_file.write(f"{key}: {value:.4f}\n")

    # Save the best model's epoch and metrics
    with open(metrics_path, 'w') as metrics_file:
        metrics_file.write(f"Best Model Epoch: {best_epoch}\n")
        metrics_file.write("Best Training Metrics:\n")
        for key, value in best_train_metrics.items():
            metrics_file.write(f"{key}: {value:.4f}\n")
        metrics_file.write("Best Validation Metrics:\n")
        for key, value in best_val_metrics.items():
            metrics_file.write(f"{key}: {value:.4f}\n")

    # Display the results
    print("ResNet-1D with Embedding Layer Multi-Label Model Evaluation Results:")
    print(f"Test Hamming Loss: {test_hamming_loss:.4f}")
    print(f"Test Jaccard Score: {test_jaccard:.4f}")
    print(f"Test F1 Macro: {test_f1_macro:.4f}")
    print(f"Test F1 Weighted: {test_f1_weighted:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Total Test Time: {total_test_time:.2f} seconds")
    print(f"Time per 1000bp: {time_per_1000bp:.4f} seconds")
    print(f"Best Model Epoch: {best_epoch}")
    print("Best Training Metrics:")
    for key, value in best_train_metrics.items():
        print(f"{key}: {value:.4f}")
    print("Best Validation Metrics:")
    for key, value in best_val_metrics.items():
        print(f"{key}: {value:.4f}")

