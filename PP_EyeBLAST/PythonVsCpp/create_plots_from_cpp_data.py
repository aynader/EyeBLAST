"""
Create plots from C++ FastEyeBLAST results data
This script reads the exported raw data from the C++ version and creates
identical plots to the Python version.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import argparse

def load_plot_data(data_dir):
    """
    Load plot data from the C++ export
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the exported data files
        
    Returns:
    --------
    dict
        Dictionary containing all plot data
    """
    plot_data_file = os.path.join(data_dir, 'plot_data.json')
    
    if not os.path.exists(plot_data_file):
        raise FileNotFoundError(f"Plot data file not found: {plot_data_file}")
    
    with open(plot_data_file, 'r') as f:
        data = json.load(f)
    
    return data

def plot_confusion_matrix(y_true, y_pred, output_dir, title="Confusion Matrix"):
    """
    Plot and save confusion matrix
    
    Parameters:
    -----------
    y_true : list
        True labels
    y_pred : list
        Predicted labels
    output_dir : str
        Directory to save plot
    title : str
        Plot title
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = sorted(set(y_true))
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_normalized_confusion_matrix(y_true, y_pred, output_dir, title="Normalized Confusion Matrix"):
    """
    Plot and save normalized confusion matrix
    
    Parameters:
    -----------
    y_true : list
        True labels
    y_pred : list
        Predicted labels
    output_dir : str
        Directory to save plot
    title : str
        Plot title
    """
    # Create normalized confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    classes = sorted(set(y_true))
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Proportion'})
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_performance(class_data, output_dir):
    """
    Plot per-class performance metrics
    
    Parameters:
    -----------
    class_data : dict
        Dictionary containing class metrics
    output_dir : str
        Directory to save plot
    """
    classes = class_data['classes']
    precision = class_data['precision']
    recall = class_data['recall']
    f1 = class_data['f1']
    support = class_data['support']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot for precision, recall, F1
    x = np.arange(len(classes))
    width = 0.25
    
    ax1.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax1.bar(x, recall, width, label='Recall', alpha=0.8)
    ax1.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax1.set_xlabel('Task Class')
    ax1.set_ylabel('Score')
    ax1.set_title('Per-Class Performance Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Support (number of samples per class)
    ax2.bar(classes, support, alpha=0.7, color='green')
    ax2.set_xlabel('Task Class')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Class Distribution (Support)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_timing_analysis(timing_data, output_dir):
    """
    Plot timing analysis
    
    Parameters:
    -----------
    timing_data : dict
        Dictionary with timing information
    output_dir : str
        Directory to save plot
    """
    # Extract timing data
    phases = list(timing_data.keys())
    times = list(timing_data.values())
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot
    colors = plt.cm.Set3(np.linspace(0, 1, len(phases)))
    bars = ax1.bar(phases, times, color=colors, alpha=0.8)
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Execution Time by Phase')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time_val:.2f}s', ha='center', va='bottom')
    
    # Pie plot
    ax2.pie(times, labels=phases, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Time Distribution by Phase')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timing_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_vs_parameters(parameter_sweep_data, output_dir):
    """
    Plot accuracy vs different k values from parameter sweep
    
    Parameters:
    -----------
    parameter_sweep_data : dict
        Dictionary containing parameter sweep results
    output_dir : str
        Directory to save plot
    """
    k_values = parameter_sweep_data['k_values']
    accuracies = parameter_sweep_data['accuracies']
    f1_scores = parameter_sweep_data['f1_scores']
    
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(k_values, accuracies, 'bo-', linewidth=2, markersize=8, label='Accuracy')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Neighbors')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # F1 score plot
    plt.subplot(1, 2, 2)
    plt.plot(k_values, f1_scores, 'ro-', linewidth=2, markersize=8, label='Macro F1')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Number of Neighbors')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_k.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_summary(plot_data, output_dir):
    """
    Create a comprehensive performance summary report
    
    Parameters:
    -----------
    plot_data : dict
        Dictionary containing all plot data
    output_dir : str
        Directory to save report
    """
    # Create summary figure
    fig = plt.figure(figsize=(16, 12))
    
    # Create a 3x2 grid
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
    
    # Overall metrics (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        plot_data['accuracy'],
        plot_data['macro_avg']['precision'],
        plot_data['macro_avg']['recall'],
        plot_data['macro_avg']['f1']
    ]
    
    bars = ax1.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax1.set_ylim(0, 1)
    ax1.set_title('Overall Performance Metrics', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Timing breakdown (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    timing_data = plot_data['timing_data']
    phases = list(timing_data.keys())
    times = list(timing_data.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(phases)))
    
    wedges, texts, autotexts = ax2.pie(times, labels=phases, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax2.set_title('Execution Time Distribution', fontweight='bold')
    
    # Per-class performance (middle row)
    ax3 = fig.add_subplot(gs[1, :])
    class_data = plot_data['class_data']
    classes = class_data['classes']
    precision = class_data['precision']
    recall = class_data['recall']
    f1 = class_data['f1']
    
    x = np.arange(len(classes))
    width = 0.25
    
    ax3.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax3.bar(x, recall, width, label='Recall', alpha=0.8)
    ax3.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax3.set_xlabel('Task Class')
    ax3.set_ylabel('Score')
    ax3.set_title('Per-Class Performance Metrics', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)
    
    # System info and summary (bottom)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Create summary text
    total_time = sum(timing_data.values())
    total_samples = plot_data['total_samples']
    
    summary_text = f"""
    FastEyeBLAST C++ Performance Summary
    ===================================
    
    Dataset Information:
    • Total samples processed: {total_samples}
    • Number of classes: {len(classes)}
    • Classes: {', '.join(map(str, classes))}
    
    Performance Metrics:
    • Overall Accuracy: {plot_data['accuracy']:.4f} ({plot_data['accuracy']*100:.2f}%)
    • Macro-averaged F1: {plot_data['macro_avg']['f1']:.4f}
    • Macro-averaged Precision: {plot_data['macro_avg']['precision']:.4f}
    • Macro-averaged Recall: {plot_data['macro_avg']['recall']:.4f}
    
    Timing Analysis:
    • Total execution time: {total_time:.2f} seconds
    • Training time: {timing_data.get('train_time', 0):.2f} seconds
    • Prediction time: {timing_data.get('predict_time', 0):.2f} seconds
    • Average prediction per sample: {timing_data.get('predict_time', 0)*1000/total_samples:.2f} ms
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_all_plots(data_dir, output_dir=None):
    """
    Create all plots from C++ exported data
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the exported data files
    output_dir : str, optional
        Directory to save plots (defaults to same as data_dir)
    """
    if output_dir is None:
        output_dir = data_dir
    
    print(f"Loading plot data from: {data_dir}")
    plot_data = load_plot_data(data_dir)
    
    print("Creating plots...")
    
    # Extract prediction data
    y_true = plot_data['predictions']['y_true']
    y_pred = plot_data['predictions']['y_pred']
    
    # Create confusion matrices
    print("Creating confusion matrices...")
    plot_confusion_matrix(y_true, y_pred, output_dir)
    plot_normalized_confusion_matrix(y_true, y_pred, output_dir)
    
    # Create class performance plots
    print("Creating class performance plots...")
    plot_class_performance(plot_data['class_data'], output_dir)
    
    # Create timing analysis
    print("Creating timing analysis plots...")
    plot_timing_analysis(plot_data['timing_data'], output_dir)
    
    # Create parameter sweep plots if available
    if 'parameter_sweep' in plot_data:
        print("Creating parameter analysis plots...")
        plot_accuracy_vs_parameters(plot_data['parameter_sweep'], output_dir)
    
    # Create comprehensive summary
    print("Creating performance summary...")
    create_performance_summary(plot_data, output_dir)
    
    print(f"All plots saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Create plots from C++ FastEyeBLAST results data"
    )
    parser.add_argument("--data_dir", default="FastEyeBLAST_Results_Cpp_Compare",
                       help="Directory containing the exported C++ data files")
    parser.add_argument("--output_dir", default=None,
                       help="Directory to save plots (defaults to same as data_dir)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        print("Make sure to run the C++ FastEyeBLAST first to generate the data.")
        return
    
    print("=" * 60)
    print("Creating plots from C++ FastEyeBLAST results")
    print("=" * 60)
    
    try:
        create_all_plots(args.data_dir, args.output_dir)
        print("\n" + "=" * 60)
        print("Plot creation completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"Error creating plots: {e}")
        raise

if __name__ == "__main__":
    main() 