import os
import time
import random
import argparse
import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from Fast_EyeBLAST_DNA import FastEyeBLASTDNA, load_fasta_dir

def run_experiment(params, X_train, y_train, X_test, y_test, log_file):
    """Run a single experiment with given parameters and log results"""
    
    start_time = time.time()
    
    # Create classifier with parameters
    clf = FastEyeBLASTDNA(
        seed_length=params['seed_length'],
        band_width=params['band_width'],
        num_neighbors=params['num_neighbors'],
        num_hash_tables=params['num_hash_tables'],
        num_hash_bits=params['num_hash_bits'],
        max_candidates=params['max_candidates'],
        lsh_hash_size=params['lsh_hash_size']
    )
    
    # Train
    print(f"Training with parameters: {params}")
    clf.fit(X_train, y_train)
    
    # Predict
    print("Predicting...")
    preds = clf.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, preds)
    conf_matrix = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    
    # Execution time
    train_time = clf.train_time if hasattr(clf, 'train_time') else time.time() - start_time
    predict_time = clf.predict_time if hasattr(clf, 'predict_time') else 0
    
    # Log results
    with open(log_file, 'a') as f:
        f.write(f"Parameters: {params}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Training time: {train_time:.2f}s\n")
        f.write(f"Prediction time: {predict_time:.2f}s\n")
        f.write("Classification Report:\n")
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                f.write(f"  {label}: precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1-score={metrics['f1-score']:.4f}\n")
        f.write("\n" + "-"*80 + "\n\n")
    
    return {
        'params': params,
        'accuracy': accuracy,
        'train_time': train_time,
        'predict_time': predict_time,
        'report': report,
        'conf_matrix': conf_matrix
    }

def visualize_results(results, output_dir):
    """Create visualizations of parameter search results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrame for easier visualization
    df_results = []
    for res in results:
        row = {
            'accuracy': res['accuracy'],
            'train_time': res['train_time'],
            'predict_time': res['predict_time']
        }
        row.update(res['params'])
        df_results.append(row)
    
    df = pd.DataFrame(df_results)
    
    # 1. Accuracy vs Parameters
    param_cols = list(results[0]['params'].keys())
    for param in param_cols:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=param, y='accuracy', data=df, s=100, alpha=0.7)
        plt.title(f'Accuracy vs {param}')
        plt.savefig(os.path.join(output_dir, f'accuracy_vs_{param}.png'))
        plt.close()
    
    # 2. Heatmaps for parameter pairs
    for param1, param2 in itertools.combinations(param_cols, 2):
        plt.figure(figsize=(10, 8))
        pivot = df.pivot_table(
            values='accuracy', 
            index=param1,
            columns=param2,
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
        plt.title(f'Accuracy Heatmap: {param1} vs {param2}')
        plt.savefig(os.path.join(output_dir, f'heatmap_{param1}_vs_{param2}.png'))
        plt.close()
    
    # 3. Correlation between parameters and metrics
    plt.figure(figsize=(12, 10))
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation between Parameters and Metrics')
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()
    
    # 4. Training time vs Accuracy
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='train_time', y='accuracy', data=df, s=100, alpha=0.7)
    for i, row in df.iterrows():
        plt.annotate(i, (row['train_time'], row['accuracy']), 
                     fontsize=8, alpha=0.7)
    plt.title('Training Time vs Accuracy')
    plt.savefig(os.path.join(output_dir, 'train_time_vs_accuracy.png'))
    plt.close()
    
    # 5. Prediction time vs Accuracy
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='predict_time', y='accuracy', data=df, s=100, alpha=0.7)
    for i, row in df.iterrows():
        plt.annotate(i, (row['predict_time'], row['accuracy']), 
                     fontsize=8, alpha=0.7)
    plt.title('Prediction Time vs Accuracy')
    plt.savefig(os.path.join(output_dir, 'predict_time_vs_accuracy.png'))
    plt.close()
    
    # 6. Best parameters summary
    best_idx = df['accuracy'].idxmax()
    best_params = df.loc[best_idx, param_cols].to_dict()
    
    with open(os.path.join(output_dir, 'best_parameters.txt'), 'w') as f:
        f.write("Best Parameters:\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
        f.write(f"\nBest Accuracy: {df.loc[best_idx, 'accuracy']:.4f}\n")
        f.write(f"Training Time: {df.loc[best_idx, 'train_time']:.2f}s\n")
        f.write(f"Prediction Time: {df.loc[best_idx, 'predict_time']:.2f}s\n")

def main():
    parser = argparse.ArgumentParser(description='Parameter search for FastEyeBLASTDNA')
    parser.add_argument('--input_dir', required=True, help='Directory of FASTA files')
    parser.add_argument('--output_dir', default='parameter_search_results', help='Output directory for results')
    parser.add_argument('--sample_percentage', type=int, default=10, help='Percentage of data to use (default: 10)')
    parser.add_argument('--balanced', action='store_true', help='Balance class distribution')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test split size (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'parameter_search_log.txt')
    
    # Initialize log file
    with open(log_file, 'w') as f:
        f.write(f"FastEyeBLASTDNA Parameter Search\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input directory: {args.input_dir}\n")
        f.write(f"Sample percentage: {args.sample_percentage}%\n")
        f.write(f"Balanced sampling: {args.balanced}\n")
        f.write(f"Test size: {args.test_size}\n")
        f.write(f"Random seed: {args.seed}\n\n")
        f.write("-"*80 + "\n\n")
    
    # Load data
    print(f"Loading data from {args.input_dir} ({args.sample_percentage}% sample, balanced={args.balanced})...")
    seqs, labels = load_fasta_dir(args.input_dir, args.sample_percentage, args.balanced)
    
    # Split data
    indices = list(range(len(seqs)))
    random.shuffle(indices)
    split = int(len(seqs) * (1 - args.test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train = [seqs[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_test = [seqs[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]
    
    print(f"Training set: {len(X_train)} sequences")
    print(f"Test set: {len(X_test)} sequences")
    
    # Parameter grid
    param_grid = {
        'seed_length': [9, 11, 13, 15],
        'band_width': [12, 16, 20, 24],
        'num_neighbors': [3, 5, 7, 9],
        'num_hash_tables': [8, 12, 16, 20],
        'num_hash_bits': [12, 16, 20],
        'max_candidates': [20, 30, 50, 70],
        'lsh_hash_size': [48, 64, 80]
    }
    
    # Generate parameter combinations
    # Use a subset of combinations to keep runtime reasonable
    param_keys = list(param_grid.keys())
    
    # First, test each parameter individually while keeping others at default
    default_params = {
        'seed_length': 11,
        'band_width': 16,
        'num_neighbors': 5,
        'num_hash_tables': 12,
        'num_hash_bits': 16,
        'max_candidates': 30,
        'lsh_hash_size': 64
    }
    
    combinations = []
    
    # Add default parameters as baseline
    combinations.append(default_params.copy())
    
    # Test each parameter individually
    for param in param_keys:
        for value in param_grid[param]:
            if value != default_params[param]:
                params = default_params.copy()
                params[param] = value
                combinations.append(params)
    
    # Add some combinations of two parameters
    key_pairs = [
        ('seed_length', 'band_width'),
        ('seed_length', 'max_candidates'),
        ('num_hash_tables', 'num_hash_bits'),
        ('num_neighbors', 'max_candidates')
    ]
    
    for param1, param2 in key_pairs:
        for val1 in param_grid[param1]:
            for val2 in param_grid[param2]:
                if val1 != default_params[param1] or val2 != default_params[param2]:
                    params = default_params.copy()
                    params[param1] = val1
                    params[param2] = val2
                    combinations.append(params)
    
    # Run experiments
    results = []
    for i, params in enumerate(combinations):
        print(f"\nExperiment {i+1}/{len(combinations)}")
        result = run_experiment(params, X_train, y_train, X_test, y_test, log_file)
        results.append(result)
    
    # Find best parameters
    best_result = max(results, key=lambda x: x['accuracy'])
    print("\nBest parameters:")
    for param, value in best_result['params'].items():
        print(f"  {param}: {value}")
    print(f"Best accuracy: {best_result['accuracy']:.4f}")
    
    # Save best parameters to file
    with open(os.path.join(args.output_dir, 'best_parameters.txt'), 'w') as f:
        f.write("Best Parameters:\n")
        for param, value in best_result['params'].items():
            f.write(f"{param}: {value}\n")
        f.write(f"\nBest Accuracy: {best_result['accuracy']:.4f}\n")
        f.write(f"Training Time: {best_result['train_time']:.2f}s\n")
        f.write(f"Prediction Time: {best_result['predict_time']:.2f}s\n")
    
    # Create visualizations
    visualize_results(results, args.output_dir)
    
    print(f"\nParameter search complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main() 