import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from typing import Dict, Any, List

class Visualizer:
    def __init__(self, reports_dir="reports"):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
        # Apply seaborn style
        sns.set_theme(style="whitegrid")
        
    def plot_throughput_comparison(self, results: Dict[str, Dict[str, Any]]):
        """
        results format:
        {
            "Scenario A": {"AI": metrics, "Baseline": metrics},
            ...
        }
        """
        data = []
        for scenario, controllers in results.items():
            for ctrl, metrics in controllers.items():
                data.append({
                    "Scenario": scenario,
                    "Controller": ctrl,
                    "Throughput": metrics["throughput_per_hr"]
                })
                
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df, x="Scenario", y="Throughput", hue="Controller", palette=["#1890ff", "#ff8c00"])
        plt.title("Section Throughput Comparison (AI vs Baseline)")
        plt.ylabel("Trains Cleared per Hour")
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, "throughput_comparison.png"), dpi=300)
        plt.close()
        
    def plot_delay_distribution(self, ai_delays: List[float], baseline_delays: List[float]):
        """
        Plot KDE/Histogram of delays
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(baseline_delays, color="#ff8c00", label="Baseline", kde=True, stat="density", bins=20, alpha=0.5)
        sns.histplot(ai_delays, color="#1890ff", label="AI Controller", kde=True, stat="density", bins=20, alpha=0.5)
        
        plt.title("Delay Distribution (AI vs Baseline)")
        plt.xlabel("Delay (Minutes)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, "delay_distribution.png"), dpi=300)
        plt.close()
        
    def plot_stringline_chart(self, ai_trajectories: Dict[str, List[float]], baseline_trajectories: Dict[str, List[float]]):
        """
        Time-Distance trajectory graph
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
        
        # Plot AI
        for t_id, trajectory in ai_trajectories.items():
            ax1.plot(range(len(trajectory)), trajectory, marker='', linewidth=2, alpha=0.7)
        ax1.set_title("AI Controller Trajectories")
        ax1.set_xlabel("Time (Steps/Min)")
        ax1.set_ylabel("Distance (km)")
        ax1.grid(True)
        
        # Plot Baseline
        for t_id, trajectory in baseline_trajectories.items():
            ax2.plot(range(len(trajectory)), trajectory, marker='', linewidth=2, alpha=0.7)
        ax2.set_title("Baseline Fixed-Block Trajectories")
        ax2.set_xlabel("Time (Steps/Min)")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.reports_dir, "stringline_chart.png"), dpi=300)
        plt.close()
