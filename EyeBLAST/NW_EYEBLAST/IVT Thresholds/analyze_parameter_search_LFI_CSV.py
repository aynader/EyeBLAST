import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Analyze parameter search results for LFI_CSV dataset")
    parser.add_argument("--results-dir", default="parameter_search_results_LFI_CSV",
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
    plt.title('Fixation Percentage by Parameter Combination (LFI_CSV)')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'fixation_heatmap.png'))
    plt.close()
    
    # 2. Parallel coordinates plot to visualize relationships across all parameters
    plt.figure(figsize=(12, 8))
    pd.plotting.parallel_coordinates(
        df, 'set', 
        cols=['thresh_fix', 'thresh_ms', 'thresh_sacc', 'fix_token_pct', 'ms_token_pct', 'sacc_token_pct']
    )
    plt.title('Parameter Set Relationships (LFI_CSV)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'parameter_relationships.png'))
    plt.close()
    
    # 3. Stacked bar chart showing distribution of eye movement types
    plt.figure(figsize=(15, 8))
    
    # Token count percentages
    token_data = df[['set', 'fix_token_pct', 'ms_token_pct', 'sacc_token_pct']]
    token_data = token_data.set_index('set')
    token_data.plot(kind='bar', stacked=True, ax=plt.gca(), 
                 color=['#3274A1', '#E1812C', '#3A923A'])
    
    plt.title('Eye Movement Type Distribution by Parameter Set (Token Count) - LFI_CSV')
    plt.xlabel('Parameter Set')
    plt.ylabel('Percentage')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.legend(['Fixations', 'Microsaccades', 'Saccades'])
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'token_distribution.png'))
    plt.close()
    
    # Time percentages if available
    if 'fix_time_pct' in df.columns and not df['fix_time_pct'].isna().all():
        plt.figure(figsize=(15, 8))
        time_data = df[['set', 'fix_time_pct', 'ms_time_pct', 'sacc_time_pct']]
        time_data = time_data.set_index('set')
        time_data.plot(kind='bar', stacked=True, ax=plt.gca(), 
                    color=['#3274A1', '#E1812C', '#3A923A'])
        
        plt.title('Eye Movement Type Distribution by Parameter Set (Time Spent) - LFI_CSV')
        plt.xlabel('Parameter Set')
        plt.ylabel('Percentage')
        plt.xticks(rotation=0)
        plt.grid(axis='y')
        plt.legend(['Fixations', 'Microsaccades', 'Saccades'])
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'time_distribution.png'))
        plt.close()

    # 4. Scatter plot to visualize trade-offs
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        df['fix_token_pct'], 
        df['ms_token_pct'], 
        c=df['thresh_fix'],
        s=df['thresh_sacc']*10,  # Adjusted scaling for LFI values
        alpha=0.7,
        cmap='viridis'
    )
    
    # Add set numbers as labels
    for i, row in df.iterrows():
        plt.annotate(f"Set {row['set']}", 
                    (row['fix_token_pct'], row['ms_token_pct']),
                    xytext=(5, 5),
                    textcoords='offset points')
    
    plt.colorbar(scatter, label='Fixation Threshold (units/s)')
    plt.title('Trade-off between Fixation and Microsaccade Detection (LFI_CSV)')
    plt.xlabel('Fixation Percentage')
    plt.ylabel('Microsaccade Percentage')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'fix_ms_tradeoff.png'))
    plt.close()
    
    # 5. Threshold comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Fixation threshold vs fixation percentage
    axes[0].scatter(df['thresh_fix'], df['fix_token_pct'], alpha=0.7, s=100)
    for i, row in df.iterrows():
        axes[0].annotate(f"Set {row['set']}", 
                        (row['thresh_fix'], row['fix_token_pct']),
                        xytext=(5, 5), textcoords='offset points')
    axes[0].set_xlabel('Fixation Threshold (units/s)')
    axes[0].set_ylabel('Fixation Percentage')
    axes[0].set_title('Fixation Threshold vs Fixation Detection')
    axes[0].grid(True)
    
    # Microsaccade threshold vs microsaccade percentage
    axes[1].scatter(df['thresh_ms'], df['ms_token_pct'], alpha=0.7, s=100, color='orange')
    for i, row in df.iterrows():
        axes[1].annotate(f"Set {row['set']}", 
                        (row['thresh_ms'], row['ms_token_pct']),
                        xytext=(5, 5), textcoords='offset points')
    axes[1].set_xlabel('Microsaccade Threshold (units/s)')
    axes[1].set_ylabel('Microsaccade Percentage')
    axes[1].set_title('Microsaccade Threshold vs Detection')
    axes[1].grid(True)
    
    # Saccade threshold vs saccade percentage
    axes[2].scatter(df['thresh_sacc'], df['sacc_token_pct'], alpha=0.7, s=100, color='green')
    for i, row in df.iterrows():
        axes[2].annotate(f"Set {row['set']}", 
                        (row['thresh_sacc'], row['sacc_token_pct']),
                        xytext=(5, 5), textcoords='offset points')
    axes[2].set_xlabel('Saccade Threshold (units/s)')
    axes[2].set_ylabel('Saccade Percentage')
    axes[2].set_title('Saccade Threshold vs Detection')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'threshold_comparisons.png'))
    plt.close()
    
    # 6. Summary table
    fig, ax = plt.subplots(figsize=(15, 6))
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
    plt.title('Parameter Search Summary (LFI_CSV)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'summary_table.png'))
    plt.close()
    
    print(f"Visualizations saved to {vis_dir}/")
    
    # Suggest optimal parameter set based on highest fixation percentage
    best_fix_set = df.loc[df['fix_token_pct'].idxmax()]
    print(f"\nParameter set with highest fixation percentage: Set {best_fix_set['set']}")
    print(f"  thresh_fix={best_fix_set['thresh_fix']}, thresh_ms={best_fix_set['thresh_ms']}, thresh_sacc={best_fix_set['thresh_sacc']}")
    print(f"  fixations: {best_fix_set['fix_token_pct']:.2f}%, microsaccades: {best_fix_set['ms_token_pct']:.2f}%, saccades: {best_fix_set['sacc_token_pct']:.2f}%")
    
    # Alternative criteria could be balance between fixations and microsaccades
    # Calculate the most balanced distribution (closest to ideal ratio like 60% fix, 25% ms, 15% sacc)
    # Adjusted for LFI dataset characteristics
    ideal = np.array([60, 25, 15])
    df['balance_score'] = df.apply(
        lambda row: np.sqrt(((np.array([row['fix_token_pct'], row['ms_token_pct'], row['sacc_token_pct']]) - ideal)**2).sum()),
        axis=1
    )
    best_balanced_set = df.loc[df['balance_score'].idxmin()]
    
    print(f"\nMost balanced parameter set: Set {best_balanced_set['set']}")
    print(f"  thresh_fix={best_balanced_set['thresh_fix']}, thresh_ms={best_balanced_set['thresh_ms']}, thresh_sacc={best_balanced_set['thresh_sacc']}")
    print(f"  fixations: {best_balanced_set['fix_token_pct']:.2f}%, microsaccades: {best_balanced_set['ms_token_pct']:.2f}%, saccades: {best_balanced_set['sacc_token_pct']:.2f}%")
    
    # LFI_CSV specific recommendations
    print(f"\n--- LFI_CSV Dataset Specific Analysis ---")
    print(f"Dataset characteristics:")
    print(f"  - Laser interferometry measurements")
    print(f"  - Cube velocities as spatial proxies")
    print(f"  - Multiple task types (TALK, READ, VIDEO, WALK, TYPE, SOLVE, CYCLE)")
    print(f"  - Continuous monitoring tasks")
    
    # Find parameter set that balances precision and recall for diverse tasks
    # For LFI, we want good fixation detection while maintaining sensitivity to different movement types across tasks
    df['diversity_score'] = (df['fix_token_pct'] * 0.5 + 
                            df['ms_token_pct'] * 0.3 + 
                            df['sacc_token_pct'] * 0.2)
    best_diversity_set = df.loc[df['diversity_score'].idxmax()]
    
    print(f"\nRecommended for diverse task analysis: Set {best_diversity_set['set']}")
    print(f"  thresh_fix={best_diversity_set['thresh_fix']}, thresh_ms={best_diversity_set['thresh_ms']}, thresh_sacc={best_diversity_set['thresh_sacc']}")
    print(f"  fixations: {best_diversity_set['fix_token_pct']:.2f}%, microsaccades: {best_diversity_set['ms_token_pct']:.2f}%, saccades: {best_diversity_set['sacc_token_pct']:.2f}%")
    
    # Find conservative parameter set (high fixation detection, low false positives)
    df['conservative_score'] = df['fix_token_pct'] - (df['ms_token_pct'] + df['sacc_token_pct']) * 0.1
    best_conservative_set = df.loc[df['conservative_score'].idxmax()]
    
    print(f"\nConservative parameter set (high fixation detection): Set {best_conservative_set['set']}")
    print(f"  thresh_fix={best_conservative_set['thresh_fix']}, thresh_ms={best_conservative_set['thresh_ms']}, thresh_sacc={best_conservative_set['thresh_sacc']}")
    print(f"  fixations: {best_conservative_set['fix_token_pct']:.2f}%, microsaccades: {best_conservative_set['ms_token_pct']:.2f}%, saccades: {best_conservative_set['sacc_token_pct']:.2f}%")
    
    # Print parameter recommendations based on velocity ranges
    print(f"\n--- Parameter Recommendations for LFI_CSV ---")
    print(f"Based on laser interferometry characteristics:")
    print(f"  - Very low thresholds (0.1-0.5) for high sensitivity")
    print(f"  - Medium thresholds (0.5-1.5) for balanced detection")
    print(f"  - High thresholds (1.5-2.5) for conservative detection")
    print(f"\nFor different applications:")
    print(f"  - Precision tasks (READ, TYPE, SOLVE): Use lower thresholds (Set 1-3)")
    print(f"  - Movement tasks (WALK, CYCLE): Use medium thresholds (Set 4-6)")
    print(f"  - Communication tasks (TALK, VIDEO): Use higher thresholds (Set 7-9)")

if __name__ == "__main__":
    main() 