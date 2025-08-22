import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Analyze parameter search results for GazeBase")
    parser.add_argument("--results-dir", default="parameter_search_results_GazeBase",
                        help="Directory containing parameter search results")
    parser.add_argument("--summary-file", default=None,
                        help="Specific summary CSV file to analyze (default: most recent)")
    args = parser.parse_args()

    # Find summary CSV file
    if args.summary_file:
        summary_file = args.summary_file
    else:
        summary_files = glob.glob(os.path.join(args.results_dir, "summary_*.csv"))
        if not summary_files:
            print(f"No summary files found in {args.results_dir}")
            return
        summary_file = max(summary_files, key=os.path.getmtime)
    
    print(f"Analyzing summary file: {summary_file}")
    
    # Load summary data
    df = pd.read_csv(summary_file)
    
    # Create output directory for visualizations
    vis_dir = os.path.join(args.results_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. Heatmap of fixation percentages by parameter combinations
    plt.figure(figsize=(12, 8))
    heatmap_data = df.pivot_table(
        values='fix_token_pct', 
        index='thresh_fix', 
        columns=['thresh_ms', 'thresh_sacc']
    )
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.1f')
    plt.title('GazeBase: Fixation Percentage by Parameter Combination')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'fixation_heatmap.png'))
    
    # 2. Parallel coordinates plot to visualize relationships across all parameters
    plt.figure(figsize=(12, 8))
    pd.plotting.parallel_coordinates(
        df, 'set', 
        cols=['thresh_fix', 'thresh_ms', 'thresh_sacc', 'fix_token_pct', 'ms_token_pct', 'sacc_token_pct']
    )
    plt.title('GazeBase: Parameter Set Relationships')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'parameter_relationships.png'))
    
    # 3. Stacked bar chart showing distribution of eye movement types
    plt.figure(figsize=(15, 8))
    
    # Token count percentages
    token_data = df[['set', 'fix_token_pct', 'ms_token_pct', 'sacc_token_pct']]
    token_data = token_data.set_index('set')
    token_data.plot(kind='bar', stacked=True, ax=plt.gca(), 
                 color=['#3274A1', '#E1812C', '#3A923A'])
    
    plt.title('GazeBase: Eye Movement Type Distribution by Parameter Set (Token Count)')
    plt.xlabel('Parameter Set')
    plt.ylabel('Percentage')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.legend(['Fixations', 'Microsaccades', 'Saccades'])
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'token_distribution.png'))
    
    # 4. Scatter plot to visualize trade-offs
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        df['fix_token_pct'], 
        df['ms_token_pct'], 
        c=df['thresh_fix'],
        s=df['thresh_sacc']*5,
        alpha=0.7,
        cmap='viridis'
    )
    
    # Add set numbers as labels
    for i, row in df.iterrows():
        plt.annotate(f"Set {row['set']}", 
                    (row['fix_token_pct'], row['ms_token_pct']),
                    xytext=(5, 5),
                    textcoords='offset points')
    
    plt.colorbar(scatter, label='Fixation Threshold (deg/s)')
    plt.title('GazeBase: Trade-off between Fixation and Microsaccade Detection')
    plt.xlabel('Fixation Percentage')
    plt.ylabel('Microsaccade Percentage')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'fix_ms_tradeoff.png'))
    
    # 5. Raw token count comparison
    plt.figure(figsize=(15, 8))
    raw_counts = df[['set', 'fix_tokens', 'ms_tokens', 'sacc_tokens']].set_index('set')
    raw_counts.plot(kind='bar', ax=plt.gca())
    plt.title('GazeBase: Raw Token Counts by Parameter Set')
    plt.xlabel('Parameter Set')
    plt.ylabel('Number of Tokens')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.legend(['Fixations', 'Microsaccades', 'Saccades'])
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'raw_token_counts.png'))
    
    # 6. Summary table
    fig, ax = plt.figure(figsize=(15, 6)), plt.subplot(111)
    ax.axis('off')
    table_data = df[['set', 'thresh_fix', 'thresh_ms', 'thresh_sacc', 
                     'fix_token_pct', 'ms_token_pct', 'sacc_token_pct']].copy()
    
    # Format percentages
    for col in ['fix_token_pct', 'ms_token_pct', 'sacc_token_pct']:
        table_data[col] = table_data[col].apply(lambda x: f"{x:.2f}%")
    
    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    plt.title('GazeBase Parameter Search Summary', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'summary_table.png'))
    
    print(f"Visualizations saved to {vis_dir}/")
    
    # Suggest optimal parameter set based on highest fixation percentage
    best_fix_set = df.loc[df['fix_token_pct'].idxmax()]
    print(f"\nParameter set with highest fixation percentage: Set {best_fix_set['set']}")
    print(f"  thresh_fix={best_fix_set['thresh_fix']}, thresh_ms={best_fix_set['thresh_ms']}, thresh_sacc={best_fix_set['thresh_sacc']}")
    print(f"  fixations: {best_fix_set['fix_token_pct']:.2f}%, microsaccades: {best_fix_set['ms_token_pct']:.2f}%, saccades: {best_fix_set['sacc_token_pct']:.2f}%")
    
    # Alternative criteria could be balance between fixations and microsaccades
    # Calculate the most balanced distribution (closest to ideal ratio like 70% fix, 20% ms, 10% sacc)
    ideal = np.array([70, 20, 10])
    df['balance_score'] = df.apply(
        lambda row: np.sqrt(((np.array([row['fix_token_pct'], row['ms_token_pct'], row['sacc_token_pct']]) - ideal)**2).sum()),
        axis=1
    )
    best_balanced_set = df.loc[df['balance_score'].idxmin()]
    
    print(f"\nMost balanced parameter set: Set {best_balanced_set['set']}")
    print(f"  thresh_fix={best_balanced_set['thresh_fix']}, thresh_ms={best_balanced_set['thresh_ms']}, thresh_sacc={best_balanced_set['thresh_sacc']}")
    print(f"  fixations: {best_balanced_set['fix_token_pct']:.2f}%, microsaccades: {best_balanced_set['ms_token_pct']:.2f}%, saccades: {best_balanced_set['sacc_token_pct']:.2f}%")
    
    # 7. Bonus: Plot token length distribution for the most balanced parameter set
    try:
        # Find one of the token files from the best balanced set
        best_set_folder = best_balanced_set['output_folder']
        token_files = glob.glob(os.path.join(best_set_folder, "IVT_output", "*_tokens.json"))
        
        if token_files:
            # Analyze token lengths for the first file
            sample_file = token_files[0]
            print(f"\nAnalyzing token length distribution for {os.path.basename(sample_file)}")
            
            with open(sample_file, 'r') as f:
                tokens = json.load(f)
            
            # Extract token types and lengths
            token_lengths = {'F': [], 'M': [], 'S': []}
            
            for token in tokens:
                if token.startswith('F'):
                    try:
                        length = int(token[1:])
                        token_lengths['F'].append(length)
                    except:
                        pass
                elif token.startswith('M'):
                    try:
                        length = int(token[1:])
                        token_lengths['M'].append(length)
                    except:
                        pass
                elif token.startswith('S'):
                    try:
                        length = int(token[1:])
                        token_lengths['S'].append(length)
                    except:
                        pass
            
            # Plot distributions
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.hist(token_lengths['F'], bins=30, alpha=0.7, color='#3274A1')
            plt.title('Fixation Lengths')
            plt.xlabel('Length (samples)')
            plt.ylabel('Count')
            
            plt.subplot(132)
            plt.hist(token_lengths['M'], bins=30, alpha=0.7, color='#E1812C')
            plt.title('Microsaccade Lengths')
            plt.xlabel('Length (samples)')
            
            plt.subplot(133)
            plt.hist(token_lengths['S'], bins=30, alpha=0.7, color='#3A923A')
            plt.title('Saccade Lengths')
            plt.xlabel('Length (samples)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'token_length_distribution.png'))
            
            # Stats about token lengths
            print("\nToken length statistics:")
            for token_type, lengths in token_lengths.items():
                if lengths:
                    print(f"  {token_type}: mean={np.mean(lengths):.1f}, median={np.median(lengths):.1f}, min={np.min(lengths)}, max={np.max(lengths)}")
                else:
                    print(f"  {token_type}: no tokens of this type")
            
    except Exception as e:
        print(f"Could not analyze token lengths: {e}")

if __name__ == "__main__":
    main() 