import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Analyze parameter search results for SynchronEyes")
    parser.add_argument("--results-dir", default=None,
                        help="Directory containing parameter search results (default: most recent parameter_search_results_* directory)")
    parser.add_argument("--summary-file", default=None,
                        help="Specific summary CSV file to analyze (default: most recent)")
    parser.add_argument("--target-fix", type=float, default=89.03,
                        help="Target percentage for fixations (default: 89.03)")
    parser.add_argument("--target-ms", type=float, default=7.37,
                        help="Target percentage for microsaccades (default: 7.37)")
    parser.add_argument("--target-sacc", type=float, default=3.60,
                        help="Target percentage for saccades (default: 3.60)")
    parser.add_argument("--top-n", type=int, default=3,
                        help="Number of top parameter sets to visualize in detail (default: 3)")
    args = parser.parse_args()

    # Find results directory if not specified
    if args.results_dir is None:
        results_dirs = glob.glob("parameter_search_results_*")
        if not results_dirs:
            print("No parameter search results directories found!")
            return
        args.results_dir = max(results_dirs, key=os.path.getmtime)
    
    print(f"Analyzing results in: {args.results_dir}")
    
    # Find summary CSV file if not specified
    if args.summary_file is None:
        summary_files = glob.glob(os.path.join(args.results_dir, "summary_*.csv"))
        if not summary_files:
            print(f"No summary files found in {args.results_dir}")
            return
        args.summary_file = max(summary_files, key=os.path.getmtime)
    
    print(f"Analyzing summary file: {args.summary_file}")
    
    # Set target distribution
    target_fix = args.target_fix
    target_ms = args.target_ms
    target_sacc = args.target_sacc
    
    print(f"Target distribution: Fixations: {target_fix:.2f}%, Microsaccades: {target_ms:.2f}%, Saccades: {target_sacc:.2f}%")
    
    # Load summary data
    df = pd.read_csv(args.summary_file)
    
    # Create output directory for visualizations
    vis_dir = os.path.join(args.results_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. Scatter plot comparing MSE (error to target) vs parameters
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # MSE vs thresh_fix
    sns.scatterplot(x='thresh_fix', y='mse', hue='scale_factor', size='smooth_window',
                   data=df, ax=axs[0, 0])
    axs[0, 0].set_title('MSE vs Fixation Threshold')
    axs[0, 0].set_yscale('log')  # Log scale for better visibility
    
    # MSE vs thresh_ms
    sns.scatterplot(x='thresh_ms', y='mse', hue='scale_factor', size='smooth_window',
                   data=df, ax=axs[0, 1])
    axs[0, 1].set_title('MSE vs Microsaccade Threshold')
    axs[0, 1].set_yscale('log')
    
    # MSE vs thresh_sacc
    sns.scatterplot(x='thresh_sacc', y='mse', hue='scale_factor', size='smooth_window',
                   data=df, ax=axs[1, 0])
    axs[1, 0].set_title('MSE vs Saccade Threshold')
    axs[1, 0].set_yscale('log')
    
    # MSE vs scale_factor
    sns.scatterplot(x='scale_factor', y='mse', hue='thresh_fix', size='smooth_window',
                   data=df, ax=axs[1, 1])
    axs[1, 1].set_title('MSE vs Scale Factor')
    axs[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'mse_vs_parameters.png'))
    
    # 2. Bar chart showing distribution vs target for the best parameter sets
    # Sort by MSE (lowest first)
    df_sorted = df.sort_values('mse')
    top_n = min(args.top_n, len(df))
    
    # Create subplot for each of the top parameter sets
    fig, axs = plt.subplots(top_n, 1, figsize=(12, 4 * top_n))
    if top_n == 1:
        axs = [axs]  # Make axs a list if only one subplot
    
    for i in range(top_n):
        row = df_sorted.iloc[i]
        set_num = row['set']
        
        # Prepare data for plotting
        categories = ['Fixations', 'Microsaccades', 'Saccades']
        actual_values = [row['fix_pct'], row['ms_pct'], row['sacc_pct']]
        target_values = [target_fix, target_ms, target_sacc]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axs[i].bar(x - width/2, actual_values, width, label='Actual', color='skyblue')
        axs[i].bar(x + width/2, target_values, width, label='Target', color='lightgreen')
        
        axs[i].set_title(f'Set {set_num}: fix={row["thresh_fix"]}, ms={row["thresh_ms"]}, sacc={row["thresh_sacc"]}, '
                        f'scale={row["scale_factor"]}, smooth={row["smooth_window"]} (MSE={row["mse"]:.4f})')
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(categories)
        axs[i].set_ylabel('Percentage (%)')
        axs[i].set_ylim(0, max(max(actual_values), max(target_values)) * 1.1)
        axs[i].legend()
        
        # Add value labels
        for j, v in enumerate(actual_values):
            axs[i].text(j - width/2, v + 0.5, f'{v:.2f}%', ha='center')
        for j, v in enumerate(target_values):
            axs[i].text(j + width/2, v + 0.5, f'{v:.2f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'top_distributions.png'))
    
    # 3. Heatmap of fixation percentages
    # Create pivot tables for each metric
    plt.figure(figsize=(10, 8))
    pivot = df.pivot_table(
        values='fix_pct',
        index='thresh_fix',
        columns='thresh_ms',
        aggfunc='mean'
    )
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlGnBu')
    plt.title('Fixation Percentage vs Thresholds')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'fixation_heatmap.png'))
    
    # 4. 3D visualization of the parameter space
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            df['thresh_fix'],
            df['thresh_ms'],
            df['thresh_sacc'],
            c=df['mse'],
            s=df['scale_factor'] * 5,
            alpha=0.6,
            cmap='viridis'
        )
        
        # Add set numbers as labels for the top parameter sets
        for i in range(min(5, len(df_sorted))):
            row = df_sorted.iloc[i]
            ax.text(
                row['thresh_fix'],
                row['thresh_ms'],
                row['thresh_sacc'],
                f"Set {int(row['set'])}",
                color='red'
            )
        
        ax.set_xlabel('Fixation Threshold')
        ax.set_ylabel('Microsaccade Threshold')
        ax.set_zlabel('Saccade Threshold')
        plt.colorbar(scatter, label='MSE (lower is better)', shrink=0.5)
        plt.title('Parameter Space Exploration')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'parameter_space_3d.png'))
    except Exception as e:
        print(f"Skipping 3D visualization due to error: {e}")
    
    # 5. Distribution vs MSE scatter plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['fix_pct', 'ms_pct', 'sacc_pct']
    titles = ['Fixation %', 'Microsaccade %', 'Saccade %']
    target_vals = [target_fix, target_ms, target_sacc]
    
    for i, (metric, title, target) in enumerate(zip(metrics, titles, target_vals)):
        sns.scatterplot(
            x=metric,
            y='mse',
            hue='thresh_fix',
            size='scale_factor',
            data=df,
            ax=axs[i]
        )
        
        # Add vertical line for target
        axs[i].axvline(x=target, color='red', linestyle='--', alpha=0.7)
        axs[i].text(target, df['mse'].max() * 0.9, f'Target: {target:.2f}%', 
                   rotation=90, verticalalignment='top')
        
        axs[i].set_title(f'{title} vs MSE')
        axs[i].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'distributions_vs_mse.png'))
    
    # 6. Create summary table
    fig, ax = plt.figure(figsize=(15, len(df_sorted) * 0.4 + 2)), plt.subplot(111)
    ax.axis('off')
    
    table_data = df_sorted[['set', 'thresh_fix', 'thresh_ms', 'thresh_sacc', 'scale_factor', 'smooth_window',
                           'fix_pct', 'ms_pct', 'sacc_pct', 'mse']].copy()
    
    # Format percentages and MSE
    for col in ['fix_pct', 'ms_pct', 'sacc_pct']:
        table_data[col] = table_data[col].apply(lambda x: f"{x:.2f}%")
    
    table_data['mse'] = table_data['mse'].apply(lambda x: f"{x:.4f}")
    
    # Create table
    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title('Parameter Search Results (sorted by MSE)', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'results_table.png'))
    
    # 7. Generate result report in text file
    report_path = os.path.join(vis_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write("SynchronEyes IVT Parameter Search Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Target Distribution:\n")
        f.write(f"  - Fixations: {target_fix:.2f}%\n")
        f.write(f"  - Microsaccades: {target_ms:.2f}%\n")
        f.write(f"  - Saccades: {target_sacc:.2f}%\n\n")
        
        f.write("Top 5 Parameter Sets:\n")
        for i in range(min(5, len(df_sorted))):
            row = df_sorted.iloc[i]
            f.write(f"\n{i+1}. Set {int(row['set'])} (MSE: {row['mse']:.4f})\n")
            f.write(f"   - thresh_fix: {row['thresh_fix']}\n")
            f.write(f"   - thresh_ms: {row['thresh_ms']}\n")
            f.write(f"   - thresh_sacc: {row['thresh_sacc']}\n")
            f.write(f"   - scale_factor: {row['scale_factor']}\n")
            f.write(f"   - smooth_window: {int(row['smooth_window'])}\n")
            f.write(f"   - Distribution: Fixations={row['fix_pct']:.2f}%, Microsaccades={row['ms_pct']:.2f}%, Saccades={row['sacc_pct']:.2f}%\n")
        
        f.write("\n\nRecommendation:\n")
        best = df_sorted.iloc[0]
        f.write(f"The optimal parameters for achieving the target distribution are:\n")
        f.write(f"  - thresh_fix: {best['thresh_fix']}\n")
        f.write(f"  - thresh_ms: {best['thresh_ms']}\n")
        f.write(f"  - thresh_sacc: {best['thresh_sacc']}\n")
        f.write(f"  - scale_factor: {best['scale_factor']}\n")
        f.write(f"  - smooth_window: {int(best['smooth_window'])}\n\n")
        
        f.write(f"With these parameters, the distribution would be:\n")
        f.write(f"  - Fixations: {best['fix_pct']:.2f}% (target: {target_fix:.2f}%)\n")
        f.write(f"  - Microsaccades: {best['ms_pct']:.2f}% (target: {target_ms:.2f}%)\n")
        f.write(f"  - Saccades: {best['sacc_pct']:.2f}% (target: {target_sacc:.2f}%)\n")
        
        f.write(f"\nTo use these parameters, run the SynchronEyes_IVT.py script with:\n")
        f.write(f"python SynchronEyes_IVT.py --thresh-fix {best['thresh_fix']} --thresh-ms {best['thresh_ms']} --thresh-sacc {best['thresh_sacc']} --smooth-window {int(best['smooth_window'])}\n")
        f.write(f"\nand update the scale_factor value in the script to {best['scale_factor']}\n")
    
    print(f"Analysis complete! Visualizations saved to {vis_dir}/")
    print(f"Report saved to {report_path}")
    
    # Final summary output
    print("\nTop 3 parameter sets:")
    for i in range(min(3, len(df_sorted))):
        row = df_sorted.iloc[i]
        print(f"\n{i+1}. Set {int(row['set'])} (MSE: {row['mse']:.4f})")
        print(f"   - thresh_fix: {row['thresh_fix']}")
        print(f"   - thresh_ms: {row['thresh_ms']}")
        print(f"   - thresh_sacc: {row['thresh_sacc']}")
        print(f"   - scale_factor: {row['scale_factor']}")
        print(f"   - smooth_window: {int(row['smooth_window'])}")
        print(f"   - Distribution: Fixations={row['fix_pct']:.2f}%, Microsaccades={row['ms_pct']:.2f}%, Saccades={row['sacc_pct']:.2f}%")
    
    # Command to use
    best = df_sorted.iloc[0]
    print("\nRecommended command:")
    print(f"python SynchronEyes_IVT.py --thresh-fix {best['thresh_fix']} --thresh-ms {best['thresh_ms']} --thresh-sacc {best['thresh_sacc']} --smooth-window {int(best['smooth_window'])}")
    print(f"Remember to update the scale_factor in the script to {best['scale_factor']}")

if __name__ == "__main__":
    main() 