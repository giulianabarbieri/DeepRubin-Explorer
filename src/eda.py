#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for Astronomical Light Curve Classification
==========================================================================

This script performs comprehensive exploratory data analysis on processed 
light curve data from the DeepRubin-Explorer project, providing scientific 
insights into the temporal and photometric characteristics of different 
astronomical transient classes.

Author: Giuliana Barbieri
Project: DeepRubin-Explorer - Real-time Transient Classification & Astrobiological Target Selection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import mlflow
import mlflow.artifacts
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configure scientific plotting style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Define astronomical class mapping and colors
CLASS_MAPPING = {
    'CEP': 'Cepheid Variables',
    'QSO': 'Quasi-Stellar Objects',
    'SNII': 'Type II Supernovae', 
    'SNIa': 'Type Ia Supernovae'
}

CLASS_COLORS = {
    'CEP': '#1f77b4',    # Blue - for periodic variables
    'QSO': '#ff7f0e',    # Orange - for stochastic sources
    'SNII': '#d62728',   # Red - for core-collapse SNe
    'SNIa': '#2ca02c'    # Green - for thermonuclear SNe
}

def load_data(data_path):
    """
    Load processed light curve data and labels.
    
    Parameters:
    -----------
    data_path : str or Path
        Path to the data directory containing processed files
        
    Returns:
    --------
    X : np.ndarray
        Light curve tensor of shape (N, 2, 100) - N samples, 2 bands (g,r), 100 timesteps
    y : np.ndarray
        Class labels as strings
    """
    data_dir = Path(data_path)
    
    # Load light curve tensor and labels
    X = np.load(data_dir / "processed" / "X_lightcurves.npy")
    y_df = pd.read_csv(data_dir / "processed" / "y_labels.csv", header=None)
    y = y_df.values.squeeze()
    
    # Check and transpose if necessary - we expect shape (N, 2, 100)
    if X.shape[1] != 2:
        # Data is likely in shape (N, 100, 2) - transpose to (N, 2, 100)
        X = np.transpose(X, (0, 2, 1))
    
    # Handle dimension mismatch - ensure X and y have same number samples
    min_samples = min(X.shape[0], len(y))
    X = X[:min_samples]
    y = y[:min_samples]
    
    # Filter out invalid labels (like '0' which seems to be an error)
    # Only keep the 4 expected classes
    valid_classes = ['CEP', 'QSO', 'SNII', 'SNIa']
    valid_mask = np.isin(y, valid_classes)
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"📊 Loaded dataset:")
    print(f"   • Light curves shape: {X.shape}")
    print(f"   • Labels shape: {y.shape}")
    print(f"   • Bands: g (index 0), r (index 1)")
    print(f"   • Timesteps: {X.shape[2]} days")
    
    return X, y

def plot_class_balance(y, output_dir):
    """
    Create a bar plot showing the distribution of samples across classes.
    
    Parameters:
    -----------
    y : np.ndarray
        Array of class labels
    output_dir : Path
        Directory to save the plot
    """
    class_counts = Counter(y)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for plotting
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = [CLASS_COLORS.get(cls, '#888888') for cls in classes]
    
    # Create bar plot
    bars = ax.bar(classes, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_xlabel('Astronomical Object Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax.set_title('Dataset Class Distribution\nDeepRubin-Explorer Light Curve Classification', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Add class descriptions as legend
    legend_labels = [f"{cls}: {CLASS_MAPPING.get(cls, cls)}" for cls in classes]
    ax.legend(bars, legend_labels, loc='upper right', frameon=True, 
             fancybox=True, shadow=True, fontsize=10)
    
    # Format plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(fontsize=11, fontweight='bold')
    plt.yticks(fontsize=11)
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / "eda_class_balance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Class balance plot saved: {output_path}")
    return str(output_path)

def plot_mean_light_curves(X, y, output_dir):
    """
    Create a 2x2 panel showing mean light curves for each class.
    
    Parameters:
    -----------
    X : np.ndarray
        Light curve tensor of shape (N, 2, 100)
    y : np.ndarray
        Array of class labels
    output_dir : Path
        Directory to save the plot
    """
    unique_classes = sorted(set(y))
    n_classes = len(unique_classes)
    
    # Create time array (assuming 100 days observation window)
    time = np.linspace(0, 100, X.shape[2])
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, cls in enumerate(unique_classes):
        ax = axes[i]
        
        # Extract samples for this class
        class_mask = (y == cls)
        class_data = X[class_mask]  # Shape: (N_class, 2, 100)
        
        if len(class_data) == 0:
            continue
            
        # Calculate mean and std for each band
        mean_g = np.mean(class_data[:, 0, :], axis=0)  # g-band average
        std_g = np.std(class_data[:, 0, :], axis=0)    # g-band std
        mean_r = np.mean(class_data[:, 1, :], axis=0)  # r-band average  
        std_r = np.std(class_data[:, 1, :], axis=0)    # r-band std
        
        # Plot mean curves with error bands
        color = CLASS_COLORS.get(cls, '#888888')
        
        # g-band (primary)
        ax.plot(time, mean_g, color=color, linewidth=2.5, label='g-band', alpha=0.9)
        ax.fill_between(time, mean_g - std_g, mean_g + std_g, 
                       color=color, alpha=0.2)
        
        # r-band (secondary)
        ax.plot(time, mean_r, color=color, linestyle='--', linewidth=2.5, 
               label='r-band', alpha=0.8)
        ax.fill_between(time, mean_r - std_r, mean_r + std_r, 
                       color=color, alpha=0.15)
        
        # Customize subplot
        ax.set_title(f'{cls}: {CLASS_MAPPING.get(cls, cls)}\n(N = {len(class_data)} samples)', 
                    fontsize=12, fontweight='bold', color=color)
        ax.set_xlabel('Time [days]', fontsize=11)
        ax.set_ylabel('Normalized Flux', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # Set reasonable y-limits
        all_data = np.concatenate([mean_g, mean_r])
        y_min, y_max = np.percentile(all_data, [5, 95])
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - margin, y_max + margin)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Overall title
    fig.suptitle('Mean Light Curve Profiles by Astronomical Class\nDeepRubin-Explorer: Temporal Signatures of Transients', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save plot
    output_path = output_dir / "eda_mean_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Mean light curves plot saved: {output_path}")
    return str(output_path)

def plot_color_evolution(X, y, output_dir):
    """
    Plot color evolution (g - r) over time for each class.
    
    Parameters:
    -----------
    X : np.ndarray
        Light curve tensor of shape (N, 2, 100)
    y : np.ndarray  
        Array of class labels
    output_dir : Path
        Directory to save the plot
    """
    unique_classes = sorted(set(y))
    time = np.linspace(0, 100, X.shape[2])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for cls in unique_classes:
        # Extract samples for this class
        class_mask = (y == cls)
        class_data = X[class_mask]  # Shape: (N_class, 2, 100)
        
        if len(class_data) == 0:
            continue
            
        # Calculate color (g - r) for each object and timestep
        colors_gr = class_data[:, 0, :] - class_data[:, 1, :]  # Shape: (N_class, 100)
        
        # Calculate mean and std color evolution
        mean_color = np.mean(colors_gr, axis=0)
        std_color = np.std(colors_gr, axis=0)
        
        # Plot with confidence interval
        color = CLASS_COLORS.get(cls, '#888888')
        ax.plot(time, mean_color, color=color, linewidth=3, 
               label=f'{cls}: {CLASS_MAPPING.get(cls, cls)}', alpha=0.9)
        ax.fill_between(time, mean_color - std_color, mean_color + std_color,
                       color=color, alpha=0.2)
    
    # Customize plot
    ax.set_xlabel('Time [days]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Color Index (g - r) [mag]', fontsize=12, fontweight='bold')
    ax.set_title('Color Evolution of Astronomical Transients\nPhotometric Signature Analysis: g - r Band Difference', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add horizontal line at zero for reference
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.5, linewidth=1)
    
    # Legend and formatting
    ax.legend(loc='best', fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / "eda_color_evolution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Color evolution plot saved: {output_path}")
    return str(output_path)

def log_to_mlflow(artifacts, dataset_info):
    """
    Log EDA artifacts and metrics to MLflow.
    
    Parameters:
    -----------
    artifacts : list
        List of paths to generated plots
    dataset_info : dict
        Dictionary containing dataset statistics
    """
    # Start MLflow run
    with mlflow.start_run(run_name="Exploratory Data Analysis") as run:
        # Log dataset parameters
        mlflow.log_param("total_samples", dataset_info['total_samples'])
        mlflow.log_param("n_classes", dataset_info['n_classes'])
        mlflow.log_param("n_timesteps", dataset_info['n_timesteps'])
        mlflow.log_param("n_bands", dataset_info['n_bands'])
        
        # Log class distribution
        for cls, count in dataset_info['class_counts'].items():
            mlflow.log_metric(f"samples_{cls}", count)
        
        # Log artifacts (plots)
        for artifact_path in artifacts:
            mlflow.log_artifact(artifact_path)
            
        print(f"✅ MLflow tracking completed. Run ID: {run.info.run_id}")

def main():
    """Main function to orchestrate EDA analysis."""
    parser = argparse.ArgumentParser(description='Astronomical Light Curve EDA')
    parser.add_argument('--data-path', type=str, default='../data',
                      help='Path to data directory (default: ../data)')
    parser.add_argument('--output-dir', type=str, default='../assets',
                      help='Output directory for plots (default: ../assets)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("🌌 DeepRubin-Explorer: Exploratory Data Analysis")
    print("=" * 55)
    
    # Load data
    X, y = load_data(args.data_path)
    
    # Calculate dataset statistics
    dataset_info = {
        'total_samples': len(y),
        'n_classes': len(set(y)),
        'n_timesteps': X.shape[2],
        'n_bands': X.shape[1],
        'class_counts': dict(Counter(y))
    }
    
    print("\n📈 Generating visualizations...")
    
    # Generate plots
    artifacts = []
    
    # 1. Class balance
    artifact_path = plot_class_balance(y, output_dir)
    artifacts.append(artifact_path)
    
    # 2. Mean light curves
    artifact_path = plot_mean_light_curves(X, y, output_dir)
    artifacts.append(artifact_path)
    
    # 3. Color evolution
    artifact_path = plot_color_evolution(X, y, output_dir)
    artifacts.append(artifact_path)
    
    print("\n🔬 Dataset Summary:")
    print(f"   • Total samples: {dataset_info['total_samples']}")
    print(f"   • Classes: {dataset_info['n_classes']}")
    print(f"   • Class distribution:")
    for cls, count in dataset_info['class_counts'].items():
        percentage = (count / dataset_info['total_samples']) * 100
        print(f"     - {cls}: {count} samples ({percentage:.1f}%)")
    
    # Log to MLflow
    print("\n📊 Logging to MLflow...")
    log_to_mlflow(artifacts, dataset_info)
    
    print(f"\n✨ EDA completed successfully!")
    print(f"📁 Plots saved in: {output_dir}")
    print("🚀 Run 'mlflow ui' to view experiment tracking dashboard")

if __name__ == "__main__":
    main()