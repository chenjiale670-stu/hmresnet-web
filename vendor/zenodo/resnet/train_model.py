import time
import torch
import numpy as np
from sklearn.metrics import hamming_loss, jaccard_score, f1_score, precision_score, recall_score, roc_curve, auc
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, model_path='best_model.pth', early_stopping_patience=20, device='cpu'):
    best_jaccard = -float('inf')
    early_stopping_counter = 0
    train_times = []

    # Open txt files to save the logs
    with open('train_metrics.txt', 'w') as f_train, open('val_metrics.txt', 'w') as f_val, open('train_times.txt', 'w') as f_times, open('roc_auc_train.txt', 'w') as f_roc_train, open('roc_auc_val.txt', 'w') as f_roc_val:
        f_train.write('Epoch,Loss,Hamming_Loss,Jaccard,F1_Macro,F1_Weighted,Precision,Recall\n')
        f_val.write('Epoch,Loss,Hamming_Loss,Jaccard,F1_Macro,F1_Weighted,Precision,Recall\n')
        f_roc_train.write('Epoch,FPR,TPR,AUC\n')  # FPR: False Positive Rate, TPR: True Positive Rate
        f_roc_val.write('Epoch,FPR,TPR,AUC\n')

        for epoch in range(num_epochs):
            start_time = time.time()

            # Training phase
            model.train()
            running_loss = 0.0
            train_labels = []
            train_preds = []
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # Gradient clipping
                optimizer.step()
                running_loss += loss.item()

                preds = torch.sigmoid(outputs) > 0.5  # Apply sigmoid and threshold at 0.5
                train_labels.extend(labels.cpu().numpy())
                train_preds.extend(preds.cpu().numpy())

            # Compute average training loss
            average_train_loss = running_loss / len(train_loader)

            # Compute training metrics
            train_labels = np.array(train_labels)
            train_preds = np.array(train_preds)

            train_hamming_loss = hamming_loss(train_labels, train_preds)
            train_jaccard = jaccard_score(train_labels, train_preds, average='samples')
            train_f1_macro = f1_score(train_labels, train_preds, average='macro')
            train_f1_weighted = f1_score(train_labels, train_preds, average='weighted')
            train_precision = precision_score(train_labels, train_preds, average='micro')
            train_recall = recall_score(train_labels, train_preds, average='micro')

            # Save training metrics to file
            f_train.write(f"{epoch+1},{average_train_loss:.4f},{train_hamming_loss:.4f},{train_jaccard:.4f},{train_f1_macro:.4f},{train_f1_weighted:.4f},{train_precision:.4f},{train_recall:.4f}\n")

            # Compute ROC and AUC for training
            fpr_train, tpr_train, _ = roc_curve(train_labels.ravel(), train_preds.ravel())
            auc_train = auc(fpr_train, tpr_train)
            for fpr, tpr in zip(fpr_train, tpr_train):
                f_roc_train.write(f"{epoch+1},{fpr:.4f},{tpr:.4f},{auc_train:.4f}\n")

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_labels = []
            val_preds = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    preds = torch.sigmoid(outputs) > 0.5  # Apply sigmoid and threshold at 0.5
                    val_labels.extend(labels.cpu().numpy())
                    val_preds.extend(preds.cpu().numpy())

            # Compute average validation loss
            average_val_loss = val_loss / len(val_loader)

            # Calculate validation metrics
            val_labels = np.array(val_labels)
            val_preds = np.array(val_preds)

            val_hamming_loss = hamming_loss(val_labels, val_preds)
            val_jaccard = jaccard_score(val_labels, val_preds, average='samples')
            val_f1_macro = f1_score(val_labels, val_preds, average='macro')
            val_f1_weighted = f1_score(val_labels, val_preds, average='weighted')
            val_precision = precision_score(val_labels, val_preds, average='micro')
            val_recall = recall_score(val_labels, val_preds, average='micro')

            # Save validation metrics to file
            f_val.write(f"{epoch+1},{average_val_loss:.4f},{val_hamming_loss:.4f},{val_jaccard:.4f},{val_f1_macro:.4f},{val_f1_weighted:.4f},{val_precision:.4f},{val_recall:.4f}\n")

            # Compute ROC and AUC for validation
            fpr_val, tpr_val, _ = roc_curve(val_labels.ravel(), val_preds.ravel())
            auc_val = auc(fpr_val, tpr_val)
            for fpr, tpr in zip(fpr_val, tpr_val):
                f_roc_val.write(f"{epoch+1},{fpr:.4f},{tpr:.4f},{auc_val:.4f}\n")

            # Step the scheduler based on validation Jaccard Score
            scheduler.step(val_jaccard)

            # Print current metrics for epoch
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_train_loss:.4f}, "
                  f"Train Hamming Loss: {train_hamming_loss:.4f}, Train Jaccard: {train_jaccard:.4f}, "
                  f"Train F1 Macro: {train_f1_macro:.4f}, Train F1 Weighted: {train_f1_weighted:.4f}, "
                  f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}")
            print(f"Val Loss: {average_val_loss:.4f}, Val Hamming Loss: {val_hamming_loss:.4f}, "
                  f"Val Jaccard: {val_jaccard:.4f}, Val F1 Macro: {val_f1_macro:.4f}, "
                  f"Val F1 Weighted: {val_f1_weighted:.4f}, Val Precision: {val_precision:.4f}, "
                  f"Val Recall: {val_recall:.4f}")

            # Check for best model based on Jaccard Score
            if val_jaccard > best_jaccard:
                best_jaccard = val_jaccard
                early_stopping_counter = 0
                torch.save(model.state_dict(), model_path)
                print("Best model saved.")
                best_epoch = epoch + 1
                best_train_metrics = {
                    'Loss': average_train_loss,
                    'Hamming_Loss': train_hamming_loss,
                    'Jaccard_Score': train_jaccard,
                    'F1_Macro': train_f1_macro,
                    'F1_Weighted': train_f1_weighted,
                    'Precision': train_precision,
                    'Recall': train_recall
                }
                best_val_metrics = {
                    'Loss': average_val_loss,
                    'Hamming_Loss': val_hamming_loss,
                    'Jaccard_Score': val_jaccard,
                    'F1_Macro': val_f1_macro,
                    'F1_Weighted': val_f1_weighted,
                    'Precision': val_precision,
                    'Recall': val_recall
                }
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break

            # Record training time for the epoch
            epoch_time = time.time() - start_time
            train_times.append(epoch_time)
            f_times.write(f"{epoch+1},{epoch_time:.2f}\n")  # Save epoch time to file
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds.")

        print("Training complete.")

    return best_epoch, best_train_metrics, best_val_metrics, train_times
