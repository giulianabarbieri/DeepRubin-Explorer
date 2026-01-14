import torch
import torch.nn.functional as F
from torch.utils.data import random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import mlflow
import argparse

from dataset import LightCurveDataset
from model import LightCurveTCN


def load_model_and_data(model_path, x_path, y_path, n_classes):
    """Load trained model and dataset."""
    # Load dataset
    dataset = LightCurveDataset(x_path, y_path)
    
    # Initialize model with same architecture
    model = LightCurveTCN(n_channels=2, n_times=100, n_classes=n_classes)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set to evaluation mode
    
    return model, dataset


def get_validation_split(dataset, train_split=0.8, seed=42):
    """Reproduce the train/validation split from training."""
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    # Split dataset (same way as in training)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    return val_dataset


def evaluate_model(model, val_dataset, device):
    """Run inference on validation set and collect predictions."""
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for i in range(len(val_dataset)):
            x, y = val_dataset[i]
            x = x.unsqueeze(0).to(device)  # Add batch dimension
            
            # Forward pass
            output = model(x)
            probs = F.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1)
            
            all_preds.append(pred.cpu().item())
            all_labels.append(y.item())
            all_probs.append(probs.cpu().numpy()[0])
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Generate and plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - LightCurveTCN', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    return cm


def visualize_sample_prediction(model, val_dataset, dataset, class_names, device, save_path=None):
    """Visualize a random sample with model prediction."""
    # Select random sample from validation set
    idx = np.random.randint(0, len(val_dataset))
    x, y_true = val_dataset[idx]
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        x_batch = x.unsqueeze(0).to(device)
        output = model(x_batch)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
        y_pred = np.argmax(probs)
    
    # Get class names
    true_class = class_names[y_true]
    pred_class = class_names[y_pred]
    confidence = probs[y_pred] * 100
    
    # Extract light curve data (convert back from (2, 100) to (100, 2))
    x_np = x.numpy().T  # Shape: (100, 2)
    times = np.arange(len(x_np))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # g-band (filter 1)
    ax1.plot(times, x_np[:, 0], 'o-', color='green', alpha=0.7, label='g-band')
    ax1.set_ylabel('Magnitude', fontsize=11)
    ax1.set_title(f'Light Curve Sample - True: {true_class} | Predicted: {pred_class} ({confidence:.1f}%)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.invert_yaxis()  # Astronomical convention: brighter = lower magnitude
    
    # r-band (filter 2)
    ax2.plot(times, x_np[:, 1], 'o-', color='red', alpha=0.7, label='r-band')
    ax2.set_xlabel('Relative Time (days)', fontsize=11)
    ax2.set_ylabel('Magnitude', fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sample prediction saved to {save_path}")
        plt.close()  # Close figure to prevent display when saving
    else:
        plt.show()
    
    # Print probabilities for all classes
    print("\nClass Probabilities:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {probs[i]*100:.2f}%")


def log_to_mlflow(cm_path, metrics_dict, experiment_name="Rubin_LightCurve_Classification"):
    """Log evaluation metrics and confusion matrix to MLflow."""
    mlflow.set_experiment(experiment_name)
    
    # Start a new run or get the latest run
    with mlflow.start_run(run_name="evaluation"):
        # Log metrics
        for metric_name, value in metrics_dict.items():
            mlflow.log_metric(metric_name, value)
        
        # Log confusion matrix as artifact
        if cm_path and Path(cm_path).exists():
            mlflow.log_artifact(str(cm_path), artifact_path="evaluation")
            print(f"Logged confusion matrix to MLflow")
        
        print(f"MLflow evaluation run ID: {mlflow.active_run().info.run_id}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate LightCurveTCN model')
    parser.add_argument('--save-plots', action='store_true', 
                        help='Save prediction plots in addition to confusion matrix')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to save evaluation outputs (default: models)')
    args = parser.parse_args()
    
    # Configuration
    script_dir = Path(__file__).parent
    data_dir = Path("data/processed")
    x_path = data_dir / "X_lightcurves.npy"
    y_path = data_dir / "y_labels.csv"
    model_path = script_dir.parent / "models" / "rubin_tcn_model.pth"
    
    # Setup output directory
    output_dir = script_dir.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cm_save_path = output_dir / "confusion_matrix.png"
    
    # Device configuration (CPU only)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load data and model
    print("\nLoading dataset and model...")
    dataset = LightCurveDataset(x_path, y_path)
    
    # Get class names
    if dataset.class_to_idx is not None:
        n_classes = len(dataset.class_to_idx)
        class_names = [dataset.idx_to_class[i] for i in range(n_classes)]
        print(f"Classes: {class_names}")
    else:
        print("Error: Could not determine class names from dataset")
        return
    
    # Load model
    model, _ = load_model_and_data(model_path, x_path, y_path, n_classes)
    model = model.to(device)
    print(f"Model loaded from {model_path}")
    
    # Get validation split (same as training)
    print("\nReproducing train/validation split...")
    val_dataset = get_validation_split(dataset, train_split=0.8, seed=42)
    print(f"Validation samples: {len(val_dataset)}")
    
    # Evaluate model
    print("\nEvaluating model on validation set...")
    y_pred, y_true, y_probs = evaluate_model(model, val_dataset, device)
    
    # Calculate accuracy
    accuracy = (y_pred == y_true).mean() * 100
    print(f"Validation Accuracy: {accuracy:.2f}%")
    
    # Generate classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(report)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    cm = plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_save_path)
    
    # Visualize sample prediction
    print("\nVisualizing random sample prediction...")
    if args.save_plots:
        sample_save_path = output_dir / "sample_prediction.png"
        visualize_sample_prediction(model, val_dataset, dataset, class_names, device, save_path=sample_save_path)
    else:
        visualize_sample_prediction(model, val_dataset, dataset, class_names, device)
    
    # Log to MLflow
    print("\nLogging results to MLflow...")
    metrics = {
        "eval_accuracy": accuracy,
        "eval_samples": len(val_dataset)
    }
    log_to_mlflow(cm_save_path, metrics)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
