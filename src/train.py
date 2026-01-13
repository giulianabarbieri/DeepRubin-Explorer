import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from dataset import LightCurveDataset
from model import LightCurveCNN1D


def train_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_x.size(0)
    
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """Validate the model and compute loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Compute loss
            running_loss += loss.item() * batch_x.size(0)
            
            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    epoch_loss = running_loss / len(loader.dataset)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy


def main():
    # Configuration
    data_dir = Path("data/processed")
    x_path = data_dir / "X_lightcurves.npy"
    y_path = data_dir / "y_labels.csv"
    model_save_path = Path("../models/rubin_cnn_model.pth")
    
    # Hyperparameters
    batch_size = 16
    learning_rate = 0.001
    epochs = 30
    train_split = 0.8
    
    # Set MLflow experiment
    mlflow.set_experiment("Rubin_LightCurve_Classification")
    
    # Device configuration (CPU only)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = LightCurveDataset(x_path, y_path)
    print(f"Total samples: {len(dataset)}")
    
    # Get number of classes from dataset
    if dataset.class_to_idx is not None:
        n_classes = len(dataset.class_to_idx)
        print(f"Number of classes: {n_classes}")
        print(f"Class mapping: {dataset.class_to_idx}")
    else:
        n_classes = len(set(dataset.y))
        print(f"Number of classes: {n_classes}")
    
    # Train/validation split
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train samples: {train_size}, Validation samples: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = LightCurveCNN1D(n_channels=2, n_times=100, n_classes=n_classes)
    model = model.to(device)
    print(f"Model initialized: {model.__class__.__name__}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Start MLflow run
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("train_split", train_split)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("loss_function", "CrossEntropyLoss")
        mlflow.log_param("model_architecture", "LightCurveCNN1D")
        mlflow.log_param("device", str(device))
        mlflow.log_param("n_channels", 2)
        mlflow.log_param("n_times", 100)
        mlflow.log_param("n_classes", n_classes)
        
        # Log dataset information
        mlflow.log_param("x_data_path", str(x_path.absolute()))
        mlflow.log_param("y_data_path", str(y_path.absolute()))
        mlflow.log_param("total_samples", len(dataset))
        mlflow.log_param("train_samples", train_size)
        mlflow.log_param("val_samples", val_size)
        
        # Log dataset using mlflow.data
        try:
            from mlflow.data.pandas_dataset import from_pandas
            import pandas as pd
            import numpy as np
            
            # Create a summary DataFrame for dataset tracking
            X_data = np.load(x_path)
            y_data = pd.read_csv(y_path)
            
            dataset_summary = pd.DataFrame({
                'num_samples': [len(dataset)],
                'num_features': [X_data.shape[1] * X_data.shape[2]],
                'num_classes': [n_classes],
                'train_samples': [train_size],
                'val_samples': [val_size]
            })
            
            mlflow_dataset = from_pandas(
                dataset_summary,
                source=str(x_path.absolute()),
                name="lightcurve_dataset"
            )
            mlflow.log_input(mlflow_dataset, context="training")
        except Exception as e:
            print(f"Could not log dataset with mlflow.data: {e}")
        
        # Training loop
        print(f"\nStarting training for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            
            # Print progress
            print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
        
        # Save model
        print(f"\nSaving model to {model_save_path}...")
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        
        # Log model as artifact
        mlflow.log_artifact(str(model_save_path), artifact_path="model")
        
        # Log model using MLflow's pytorch autologging (optional)
        mlflow.pytorch.log_model(model, "pytorch_model")
        
        print("Training complete!")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
