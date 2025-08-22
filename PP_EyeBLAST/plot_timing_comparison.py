#!/usr/bin/env python3
"""
Plot Timing Comparison: Plain EyeBLAST vs PP_EyeBLAST

This script creates matplotlib.pyplot graphs comparing performance between:
1. Plain EyeBLAST 
2. PP_EyeBLAST (Privacy-Preserving)

Generates separate graphs for 1:1 and 1:N comparison scenarios.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class TimingPlotter:
    """Create timing comparison plots"""
    
    def __init__(self, results_file: str):
        """
        Initialize plotter with results data
        
        Args:
            results_file: Path to JSON results file from benchmark_plain_vs_pp.py
        """
        self.results_file = results_file
        self.data = self._load_results()
        
        # Set up matplotlib style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def _load_results(self) -> Dict:
        """Load results from JSON file"""
        if not Path(self.results_file).exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        with open(self.results_file, 'r') as f:
            data = json.load(f)
        
        return data
    
    def plot_1_to_1_comparison(self, save_path: Optional[str] = None) -> None:
        """
        Create 1:1 timing comparison plot
        
        Args:
            save_path: Optional path to save the plot
        """
        if "1_to_1" not in self.data.get("plain_eyeblast", {}):
            print("No 1:1 comparison data found")
            return
        
        result = self.data["plain_eyeblast"]["1_to_1"]
        
        # Extract data
        plain_times = result["plain_times"]
        pp_times = result["pp_times"]
        avg_plain = result["avg_plain"]
        avg_pp = result["avg_pp"]
        speedup_factor = result["speedup_factor"]
        runs = result["runs"]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Individual run times
        x_pos = np.arange(runs)
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, plain_times, width, 
                       label='Plain EyeBLAST', color='skyblue', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, pp_times, width,
                       label='PP_EyeBLAST', color='lightcoral', alpha=0.8)
        
        ax1.set_xlabel('Run Number')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('1:1 Comparison - Individual Run Times')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'Run {i+1}' for i in range(runs)])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.4f}s', ha='center', va='bottom', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.4f}s', ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Average comparison
        methods = ['Plain EyeBLAST', 'PP_EyeBLAST']
        avg_times = [avg_plain, avg_pp]
        colors = ['skyblue', 'lightcoral']
        
        bars = ax2.bar(methods, avg_times, color=colors, alpha=0.8, width=0.6)
        ax2.set_ylabel('Average Execution Time (seconds)')
        ax2.set_title('1:1 Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels and speedup info
        for i, (bar, time_val) in enumerate(zip(bars, avg_times)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{time_val:.4f}s', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
        
        # Add speedup annotation
        ax2.text(0.5, max(avg_times) * 0.8, 
                f'{speedup_factor:.2f}x slower\n({runs} runs averaged)',
                ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"1:1 comparison plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_1_to_n_comparison(self, save_path: Optional[str] = None) -> None:
        """
        Create 1:N timing comparison plot
        
        Args:
            save_path: Optional path to save the plot
        """
        if "1_to_n" not in self.data.get("plain_eyeblast", {}):
            print("No 1:N comparison data found")
            return
        
        result = self.data["plain_eyeblast"]["1_to_n"]
        
        # Extract data
        database_size = result["database_size"]
        plain_times = result["plain_times"]
        pp_times = result["pp_times"]
        avg_plain = result["avg_plain"]
        avg_pp = result["avg_pp"]
        speedup_factor = result["speedup_factor"]
        runs = result["runs"]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Total time comparison
        methods = ['Plain EyeBLAST', 'PP_EyeBLAST']
        avg_times = [avg_plain, avg_pp]
        colors = ['skyblue', 'lightcoral']
        
        bars = ax1.bar(methods, avg_times, color=colors, alpha=0.8, width=0.6)
        ax1.set_ylabel('Total Execution Time (seconds)')
        ax1.set_title(f'1:N Comparison \n(Database size: {database_size} files)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, time_val in zip(bars, avg_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{time_val:.4f}s', ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
        
        # Add speedup annotation
        ax1.text(0.5, max(avg_times) * 0.8,
                f'{speedup_factor:.2f}x slower\n({runs} runs averaged)',
                ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        # Plot 2: Per-file average time
        per_file_plain = avg_plain / database_size
        per_file_pp = avg_pp / database_size
        per_file_times = [per_file_plain, per_file_pp]
        
        bars2 = ax2.bar(methods, per_file_times, color=colors, alpha=0.8, width=0.6)
        ax2.set_ylabel('Average Time per File (seconds)')
        ax2.set_title('1:N Comparison - Per-File Average')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, time_val in zip(bars2, per_file_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{time_val:.4f}s', ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"1:N comparison plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_scaling_analysis(self, save_path: Optional[str] = None) -> None:
        """
        Create scaling analysis plot comparing both methods
        
        Args:
            save_path: Optional path to save the plot
        """
        # This would be used if we had multiple database sizes
        # For now, create a simulated scaling plot
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Simulated data points based on current measurements
        if "1_to_1" in self.data.get("plain_eyeblast", {}):
            result_1_to_1 = self.data["plain_eyeblast"]["1_to_1"]
            plain_1_to_1 = result_1_to_1["avg_plain"]
            pp_1_to_1 = result_1_to_1["avg_pp"]
        else:
            plain_1_to_1, pp_1_to_1 = 0.001, 0.1  # Default values
        
        if "1_to_n" in self.data.get("plain_eyeblast", {}):
            result_1_to_n = self.data["plain_eyeblast"]["1_to_n"]
            database_size = result_1_to_n["database_size"]
            plain_total = result_1_to_n["avg_plain"]
            pp_total = result_1_to_n["avg_pp"]
        else:
            database_size = 10
            plain_total = plain_1_to_1 * database_size
            pp_total = pp_1_to_1 * database_size
        
        # Create scaling data
        sizes = [1, database_size // 2, database_size] if database_size > 1 else [1]
        plain_times = [plain_1_to_1 * size for size in sizes]
        pp_times = [pp_1_to_1 * size for size in sizes]
        
        # Plot scaling lines
        ax.plot(sizes, plain_times, 'o-', color='skyblue', linewidth=3, 
               markersize=8, label='Plain EyeBLAST', markerfacecolor='blue')
        ax.plot(sizes, pp_times, 's-', color='lightcoral', linewidth=3,
               markersize=8, label='PP_EyeBLAST', markerfacecolor='red')
        
        ax.set_xlabel('Database Size (number of scanpaths)')
        ax.set_ylabel('Total Execution Time (seconds)')
        ax.set_title('Scaling Analysis: Plain vs PP_EyeBLAST')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add value annotations
        for i, (size, plain_time, pp_time) in enumerate(zip(sizes, plain_times, pp_times)):
            ax.annotate(f'{plain_time:.4f}s', 
                       (size, plain_time), textcoords="offset points",
                       xytext=(0,10), ha='center', fontsize=10)
            ax.annotate(f'{pp_time:.4f}s',
                       (size, pp_time), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scaling analysis plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_summary_dashboard(self, save_path: Optional[str] = None) -> None:
        """
        Create comprehensive summary dashboard
        
        Args:
            save_path: Optional path to save the plot
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
        
        # Title
        fig.suptitle('Plain EyeBLAST vs PP_EyeBLAST Performance Summary', 
                    fontsize=16, fontweight='bold')
        
        # 1:1 Comparison (top left)
        if "1_to_1" in self.data.get("plain_eyeblast", {}):
            ax1 = fig.add_subplot(gs[0, 0])
            result = self.data["plain_eyeblast"]["1_to_1"]
            
            methods = ['Plain', 'PP']
            times = [result["avg_plain"], result["avg_pp"]]
            colors = ['skyblue', 'lightcoral']
            
            bars = ax1.bar(methods, times, color=colors, alpha=0.8)
            ax1.set_ylabel('Time (seconds)')
            ax1.set_title('1:1 Comparison')
            ax1.grid(True, alpha=0.3)
            
            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{time_val:.4f}s', ha='center', va='bottom', fontsize=10)
        
        # 1:N Comparison (top right)
        if "1_to_n" in self.data.get("plain_eyeblast", {}):
            ax2 = fig.add_subplot(gs[0, 1])
            result = self.data["plain_eyeblast"]["1_to_n"]
            
            methods = ['Plain', 'PP']
            times = [result["avg_plain"], result["avg_pp"]]
            colors = ['skyblue', 'lightcoral']
            
            bars = ax2.bar(methods, times, color=colors, alpha=0.8)
            ax2.set_ylabel('Time (seconds)')
            ax2.set_title(f'1:N Comparison\n({result["database_size"]} files)')
            ax2.grid(True, alpha=0.3)
            
            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{time_val:.4f}s', ha='center', va='bottom', fontsize=10)
        
        # Speedup factors (middle)
        ax3 = fig.add_subplot(gs[1, :])
        
        scenarios = []
        speedups = []
        
        if "1_to_1" in self.data.get("plain_eyeblast", {}):
            scenarios.append('1:1 Comparison')
            speedups.append(self.data["plain_eyeblast"]["1_to_1"]["speedup_factor"])
        
        if "1_to_n" in self.data.get("plain_eyeblast", {}):
            scenarios.append('1:N Comparison')
            speedups.append(self.data["plain_eyeblast"]["1_to_n"]["speedup_factor"])
        
        if scenarios:
            bars = ax3.bar(scenarios, speedups, color='orange', alpha=0.8)
            ax3.set_ylabel('Slowdown Factor (PP/Plain)')
            ax3.set_title('PP_EyeBLAST Slowdown vs Plain EyeBLAST')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal Performance')
            ax3.legend()
            
            for bar, speedup in zip(bars, speedups):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{speedup:.2f}x', ha='center', va='bottom', 
                        fontsize=12, fontweight='bold')
        
        # Summary text (bottom)
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Generate summary text
        summary_lines = ["Performance Summary:", "=" * 50]
        
        if "metadata" in self.data:
            metadata = self.data["metadata"]
            summary_lines.extend([
                f"• Test Configuration:",
                f"  - Alice file: {Path(metadata.get('alice_file', '')).name}",
                f"  - Bob single: {Path(metadata.get('bob_single', '')).name}",
                f"  - Bob directory: {metadata.get('bob_dir', 'N/A')}",
                f"  - Runs per test: {metadata.get('runs', 'N/A')}",
                ""
            ])
        
        if "1_to_1" in self.data.get("plain_eyeblast", {}):
            result = self.data["plain_eyeblast"]["1_to_1"]
            summary_lines.extend([
                f"• 1:1 Comparison Results:",
                f"  - Plain EyeBLAST: {result['avg_plain']:.4f}s average",
                f"  - PP_EyeBLAST: {result['avg_pp']:.4f}s average", 
                f"  - PP is {result['speedup_factor']:.2f}x slower",
                ""
            ])
        
        if "1_to_n" in self.data.get("plain_eyeblast", {}):
            result = self.data["plain_eyeblast"]["1_to_n"]
            summary_lines.extend([
                f"• 1:N Comparison Results ({result['database_size']} files):",
                f"  - Plain EyeBLAST: {result['avg_plain']:.4f}s total",
                f"  - PP_EyeBLAST: {result['avg_pp']:.4f}s total",
                f"  - PP is {result['speedup_factor']:.2f}x slower",
                f"  - Per-file: {result['avg_plain']/result['database_size']:.4f}s vs {result['avg_pp']/result['database_size']:.4f}s",
                ""
            ])
        
        summary_text = '\n'.join(summary_lines)
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Summary dashboard saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_performance_summary_box(self, save_path: Optional[str] = None) -> None:
        """
        Create a standalone performance summary box
        
        Args:
            save_path: Optional path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Generate summary text
        summary_lines = ["🏆 Plain EyeBLAST vs PP_EyeBLAST Performance Summary", "=" * 60]
        
        if "metadata" in self.data:
            metadata = self.data["metadata"]
            summary_lines.extend([
                "",
                "📋 Test Configuration:",
                f"  • Alice file: {Path(metadata.get('alice_file', '')).name}",
                f"  • Bob single: {Path(metadata.get('bob_single', '')).name}",
                f"  • Bob directory: {metadata.get('bob_dir', 'N/A')}",
                f"  • Runs per test: {metadata.get('runs', 'N/A')}",
                ""
            ])
        
        if "1_to_1" in self.data.get("plain_eyeblast", {}):
            result = self.data["plain_eyeblast"]["1_to_1"]
            summary_lines.extend([
                "🎯 1:1 Comparison Results:",
                f"  • Plain EyeBLAST: {result['avg_plain']:.4f}s average ({result['avg_plain']*1000:.1f}ms)",
                f"  • PP_EyeBLAST: {result['avg_pp']:.4f}s average ({result['avg_pp']*1000:.1f}ms)", 
                f"  • Privacy Cost: {result['speedup_factor']:.0f}x slower",
                f"  • Individual runs: {[f'{t:.3f}s' for t in result['pp_times']]}",
                ""
            ])
        
        if "1_to_n" in self.data.get("plain_eyeblast", {}):
            result = self.data["plain_eyeblast"]["1_to_n"]
            summary_lines.extend([
                f"🔍 1:N Database Search Results ({result['database_size']} files):",
                f"  • Plain EyeBLAST: {result['avg_plain']:.4f}s total ({result['avg_plain']*1000:.1f}ms)",
                f"  • PP_EyeBLAST: {result['avg_pp']:.3f}s total ({result['avg_pp']:.1f}s)",
                f"  • Privacy Cost: {result['speedup_factor']:.0f}x slower",
                f"  • Per-file average: {result['avg_plain']/result['database_size']*1000:.2f}ms vs {result['avg_pp']/result['database_size']*1000:.0f}ms",
                f"  • Individual runs: {[f'{t:.1f}s' for t in result['pp_times']]}",
                ""
            ])
        
        # Add analysis
        if "1_to_1" in self.data.get("plain_eyeblast", {}) and "1_to_n" in self.data.get("plain_eyeblast", {}):
            result_1to1 = self.data["plain_eyeblast"]["1_to_1"]
            result_1ton = self.data["plain_eyeblast"]["1_to_n"]
            
        
        summary_text = '\n'.join(summary_lines)
        
        # Create text box with improved styling
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1.5', facecolor='lightblue', 
                         alpha=0.9, edgecolor='navy', linewidth=2))
        
        # Add title
        fig.suptitle('Privacy-Preserving EyeBLAST Performance Analysis', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance summary box saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_all_plots(self, output_dir: str = "plots") -> None:
        """
        Create all timing comparison plots
        
        Args:
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Creating timing comparison plots...")
        print("-" * 40)
        
        # Create individual plots
        if "1_to_1" in self.data.get("plain_eyeblast", {}):
            self.plot_1_to_1_comparison(str(output_path / "1_to_1_comparison.png"))
        
        if "1_to_n" in self.data.get("plain_eyeblast", {}):
            self.plot_1_to_n_comparison(str(output_path / "1_to_n_comparison.png"))
        
        # Create scaling analysis
        self.plot_scaling_analysis(str(output_path / "scaling_analysis.png"))
        
        # Create summary dashboard
        self.plot_summary_dashboard(str(output_path / "summary_dashboard.png"))
        
        # Create standalone performance summary box
        self.plot_performance_summary_box(str(output_path / "performance_summary_box.png"))
        
        print(f"\nAll plots saved to: {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot timing comparison between Plain and PP_EyeBLAST"
    )
    parser.add_argument("--results", default="timing_results.json",
                       help="JSON results file from benchmark_plain_vs_pp.py")
    parser.add_argument("--output_dir", default="plots",
                       help="Directory to save plots")
    parser.add_argument("--show", action="store_true",
                       help="Show plots instead of saving")
    parser.add_argument("--plot_type", choices=["1_to_1", "1_to_n", "scaling", "summary", "summary_box", "all"],
                       default="all", help="Type of plot to create")
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not Path(args.results).exists():
        print(f"Error: Results file not found: {args.results}")
        print("Run benchmark_plain_vs_pp.py first to generate timing data")
        return
    
    print("=" * 60)
    print("Creating Plain vs PP_EyeBLAST Timing Comparison Plots")
    print("=" * 60)
    
    try:
        plotter = TimingPlotter(args.results)
        
        if args.show:
            # Show plots interactively
            if args.plot_type in ["1_to_1", "all"]:
                plotter.plot_1_to_1_comparison()
            if args.plot_type in ["1_to_n", "all"]:
                plotter.plot_1_to_n_comparison()
            if args.plot_type in ["scaling", "all"]:
                plotter.plot_scaling_analysis()
            if args.plot_type in ["summary", "all"]:
                plotter.plot_summary_dashboard()
            if args.plot_type in ["summary_box", "all"]:
                plotter.plot_performance_summary_box()
        else:
            # Save plots to files
            if args.plot_type == "all":
                plotter.create_all_plots(args.output_dir)
            else:
                output_path = Path(args.output_dir)
                output_path.mkdir(exist_ok=True)
                
                if args.plot_type == "1_to_1":
                    plotter.plot_1_to_1_comparison(str(output_path / "1_to_1_comparison.png"))
                elif args.plot_type == "1_to_n":
                    plotter.plot_1_to_n_comparison(str(output_path / "1_to_n_comparison.png"))
                elif args.plot_type == "scaling":
                    plotter.plot_scaling_analysis(str(output_path / "scaling_analysis.png"))
                elif args.plot_type == "summary":
                    plotter.plot_summary_dashboard(str(output_path / "summary_dashboard.png"))
                elif args.plot_type == "summary_box":
                    plotter.plot_performance_summary_box(str(output_path / "performance_summary_box.png"))
        
        print("\n" + "=" * 60)
        print("Plot creation completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        raise


if __name__ == "__main__":
    main()