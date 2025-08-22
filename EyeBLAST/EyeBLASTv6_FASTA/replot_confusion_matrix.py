#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def plot_confusion_matrix(cm, class_names, output_path, title=None):
    """Plot confusion matrix and save to file"""
    plt.figure(figsize=(12, 10))
    
    # Check if confusion matrix contains float values (from averaging)
    # Use 'd' format for integers, '.1f' for floats
    fmt = '.1f' if np.issubdtype(cm.dtype, np.floating) else 'd'
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title or 'Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved confusion matrix to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Replot confusion matrix with correct labels')
    parser.add_argument('--results_file', required=True, 
                        help='Path to the JSON results file')
    parser.add_argument('--output_dir', default=None,
                        help='Directory to save the plot (defaults to same directory as results)')
    parser.add_argument('--algorithm', default='FastEyeBLAST_DNA',
                        help='Algorithm to plot (default: FastEyeBLAST_DNA)')
    args = parser.parse_args()
    
    # Load results file
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    
    if args.algorithm not in results:
        print(f"Error: Algorithm '{args.algorithm}' not found in results file")
        return
    
    # Get confusion matrix
    cm = np.array(results[args.algorithm]['confusion_matrix'])
    
    # Get class names from the report
    class_names = []
    for class_name in results[args.algorithm]['report']:
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            class_names.append(class_name)
    
    # Sort class names (they might be in a different order in the JSON)
    class_names = sorted(class_names)
    
    # Set output directory
    output_dir = args.output_dir or os.path.dirname(args.results_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set output path
    output_path = os.path.join(output_dir, f'{args.algorithm}_confusion_matrix_fixed.png')
    
    # Get accuracy and number of runs (if available)
    accuracy = results[args.algorithm]['accuracy']
    num_runs = results[args.algorithm].get('num_runs', 1)
    
    # Create title with accuracy information
#    if num_runs > 1:
 #       std = results[args.algorithm].get('accuracy_std', 0)
 #       title = f"{args.algorithm} Confusion Matrix\nAccuracy: {accuracy:.4f} ± {std:.4f} (Average of {num_runs} runs)"
 ##       title = f"{args.algorithm} Confusion Matrix\nAccuracy: {accuracy:.4f}"
    title = f"Confusion Matrix"
    # Plot and save
    plot_confusion_matrix(cm, class_names, output_path, title)

if __name__ == "__main__":
    main() 