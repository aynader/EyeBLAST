import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Analyze MPIIDPEye parameter search results")
    parser.add_argument("--results-dir", default="IVT Thresholds/MPIIDPEye_parameter_search_results",
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
    plt.title('MPIIDPEye: Fixation Percentage by Parameter Combination')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'fixation_heatmap.png'))
    
    # 2. Parallel coordinates plot to visualize relationships across all parameters
    plt.figure(figsize=(12, 8))
    pd.plotting.parallel_coordinates(
        df, 'set', 
        cols=['thresh_fix', 'thresh_ms', 'thresh_sacc', 'fix_token_pct', 'ms_token_pct', 'sacc_token_pct']
    )
    plt.title('MPIIDPEye: Parameter Set Relationships')
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
    
    plt.title('MPIIDPEye: Eye Movement Type Distribution by Parameter Set (Token Count)')
    plt.xlabel('Parameter Set')
    plt.ylabel('Percentage')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.legend(['Fixations', 'Microsaccades', 'Saccades'])
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'token_distribution.png'))
    
    # Time percentages if available
    if 'fix_time_pct' in df.columns and not df['fix_time_pct'].isna().all():
        plt.figure(figsize=(15, 8))
        time_data = df[['set', 'fix_time_pct', 'ms_time_pct', 'sacc_time_pct']]
        time_data = time_data.set_index('set')
        time_data.plot(kind='bar', stacked=True, ax=plt.gca(), 
                    color=['#3274A1', '#E1812C', '#3A923A'])
        
        plt.title('MPIIDPEye: Eye Movement Type Distribution by Parameter Set (Time Spent)')
        plt.xlabel('Parameter Set')
        plt.ylabel('Percentage')
        plt.xticks(rotation=0)
        plt.grid(axis='y')
        plt.legend(['Fixations', 'Microsaccades', 'Saccades'])
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'time_distribution.png'))

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
    plt.title('MPIIDPEye: Trade-off between Fixation and Microsaccade Detection')
    plt.xlabel('Fixation Percentage')
    plt.ylabel('Microsaccade Percentage')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'fix_ms_tradeoff.png'))
    
    # 5. Summary table
    plt.figure(figsize=(15, 6))
    ax = plt.subplot(111)
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
    plt.title('MPIIDPEye: Parameter Search Summary', fontsize=16, pad=20)
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
    
    # Also provide ratios that maximize time spent in fixation
    if 'fix_time_pct' in df.columns and not df['fix_time_pct'].isna().all():
        best_fix_time_set = df.loc[df['fix_time_pct'].idxmax()]
        print(f"\nParameter set with highest fixation time percentage: Set {best_fix_time_set['set']}")
        print(f"  thresh_fix={best_fix_time_set['thresh_fix']}, thresh_ms={best_fix_time_set['thresh_ms']}, thresh_sacc={best_fix_time_set['thresh_sacc']}")
        print(f"  fixations: {best_fix_time_set['fix_time_pct']:.2f}%, microsaccades: {best_fix_time_set['ms_time_pct']:.2f}%, saccades: {best_fix_time_set['sacc_time_pct']:.2f}%")

if __name__ == "__main__":
    main() 