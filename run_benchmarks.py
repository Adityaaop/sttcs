import argparse
import json
import os
from benchmarks.evaluator import Evaluator
from benchmarks.visualizer import Visualizer

def main():
    parser = argparse.ArgumentParser(description="Benchmarking Suite: AI vs Baseline")
    parser.add_argument('--scenarios', type=str, default='all', help="Comma-separated scenarios (A,B,C) or 'all'")
    parser.add_argument('--episodes', type=int, default=10, help="Number of episodes (steps per scenario) for testing")
    parser.add_argument('--export-report', action='store_true', help="Export summary JSON and plots")
    args = parser.parse_args()
    
    if args.scenarios == 'all':
        scenarios = ['A', 'B', 'C']
    else:
        scenarios = args.scenarios.split(',')
        
    evaluator = Evaluator(num_stations=4)
    visualizer = Visualizer(reports_dir="reports")
    
    results = {}
    summary = {}
    
    # Track distributions for visualizer
    all_ai_delays = []
    all_baseline_delays = []
    
    for scenario in scenarios:
        print(f"Running Scenario {scenario}...")
        results[scenario] = {}
        summary[scenario] = {}
        
        for controller in ["Baseline", "AI"]:
            print(f"  > Evaluating {controller} Controller")
            metrics = evaluator.evaluate(scenario, controller, steps=args.episodes)
            results[scenario][controller] = metrics
            
            # Populate summary for JSON
            summary[scenario][controller] = {
                "Throughput (trains/hr)": float(metrics["throughput_per_hr"]),
                "Average Delay (min)": float(metrics["avg_delay_per_train"]),
                "Safety Violations": int(metrics["safety_violations"]),
                "Min Headway (km)": float(metrics["min_headway_recorded"]),
                "Avg Latency (ms)": float(metrics["avg_latency_ms"])
            }
            
            if controller == "AI":
                all_ai_delays.extend(metrics["delays_list"])
            else:
                all_baseline_delays.extend(metrics["delays_list"])
                
    if args.export_report:
        print("Generating visualization reports...")
        visualizer.plot_throughput_comparison(results)
        
        if all_ai_delays and all_baseline_delays:
            # We filter out exactly 0 delays to see the distribution better, or leave as is
            visualizer.plot_delay_distribution(all_ai_delays, all_baseline_delays)
            
        # For stringline, we take the trajectories from Scenario A (or the last run scenario)
        last_scen = scenarios[-1]
        visualizer.plot_stringline_chart(
            results[last_scen]["AI"]["trajectories"], 
            results[last_scen]["Baseline"]["trajectories"]
        )
        
        # Save JSON
        with open("reports/kpi_validation_summary.json", "w") as f:
            json.dump(summary, f, indent=4)
            
        print("Reports saved to reports/")

if __name__ == "__main__":
    main()
