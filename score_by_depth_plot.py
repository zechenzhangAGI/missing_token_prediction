#!/usr/bin/env python3
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re

class ScoreByDepthAnalyzer:
    """Tool for analyzing scores by depth level across multiple tree experiment results."""
    
    def __init__(self, results_dir):
        """Initialize with the results directory containing experiment files."""
        self.results_dir = results_dir
        self.depth_patterns = {}  # Will store results by depth
        
    def load_experiment_files(self, min_depth=3, max_depth=7):
        """Load all experiment files within the specified depth range."""
        for depth in range(min_depth, max_depth + 1):
            # Find all files matching the pattern for this depth
            pattern = os.path.join(self.results_dir, f"experiment_depth{depth}_*.json")
            matching_files = glob.glob(pattern)
            
            if matching_files:
                print(f"Found {len(matching_files)} files for depth {depth}")
                self.depth_patterns[depth] = matching_files
            else:
                print(f"No files found for depth {depth}")
                
        if not self.depth_patterns:
            print(f"No experiment files found in {self.results_dir} for depths {min_depth}-{max_depth}")
            return False
            
        return True
    
    def extract_scores_by_level(self, tree, level=0, level_scores=None):
        """
        Recursively extract scores at each level of the tree.
        
        Args:
            tree: The tree node dictionary
            level: Current level in the tree
            level_scores: Dictionary to store scores by level
            
        Returns:
            Dictionary mapping level -> list of scores
        """
        if level_scores is None:
            level_scores = {}
            
        # Initialize level if not exists
        if level not in level_scores:
            level_scores[level] = []
            
        # Add scores from current node
        scores = tree.get("judgeScores", [])
        if scores:
            level_scores[level].extend(scores)
            
        # Process children recursively
        for child in tree.get("children", []):
            self.extract_scores_by_level(child, level + 1, level_scores)
            
        return level_scores
    
    def analyze_depth_performance(self):
        """Analyze the performance at each level for each tree depth."""
        # Dictionary to store average scores by depth and level
        depth_level_scores = {}
        
        for depth, file_list in self.depth_patterns.items():
            depth_level_scores[depth] = {}
            
            # Process all files for this depth
            for file_path in file_list:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                tree = data.get("tree", {})
                level_scores = self.extract_scores_by_level(tree)
                
                # Aggregate scores for each level
                for level, scores in level_scores.items():
                    if level not in depth_level_scores[depth]:
                        depth_level_scores[depth][level] = []
                    depth_level_scores[depth][level].extend(scores)
        
        # Calculate averages
        avg_scores_by_depth = {}
        for depth, level_data in depth_level_scores.items():
            avg_scores_by_depth[depth] = {
                level: np.mean(scores) if scores else 0
                for level, scores in level_data.items()
            }
            
        return avg_scores_by_depth
    
    def plot_scores_by_depth(self, output_file="depth_level_scores.png"):
        """
        Create a plot showing average scores at each level for different tree depths.
        
        Args:
            output_file: Path to save the output image file
        """
        avg_scores = self.analyze_depth_performance()
        
        if not avg_scores:
            print("No data available for plotting")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Plot line for each depth
        colors = plt.cm.viridis(np.linspace(0, 1, len(avg_scores)))
        
        for i, (depth, level_data) in enumerate(sorted(avg_scores.items())):
            levels = sorted(level_data.keys())
            scores = [level_data[level] for level in levels]
            
            plt.plot(levels, scores, marker='o', linewidth=2, markersize=8, 
                    label=f"Depth {depth}", color=colors[i])
            
        plt.xlabel("Tree Level", fontsize=14)
        plt.ylabel("Average Score", fontsize=14)
        plt.title("Average Scores by Tree Level for Different Maximum Depths", fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # Set integer ticks for levels
        max_level = max(max(level_data.keys()) for level_data in avg_scores.values())
        plt.xticks(range(max_level + 1))
        
        # Set y-axis limits
        plt.ylim(0, 10)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to {output_file}")
        return output_file

def main():
    parser = argparse.ArgumentParser(description="Analyze scores by depth level across tree experiments")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing experiment result files")
    parser.add_argument("--min-depth", type=int, default=3, help="Minimum tree depth to analyze")
    parser.add_argument("--max-depth", type=int, default=7, help="Maximum tree depth to analyze")
    parser.add_argument("--output", type=str, default="depth_level_scores.png", help="Output image filename")
    
    args = parser.parse_args()
    
    analyzer = ScoreByDepthAnalyzer(args.results_dir)
    if analyzer.load_experiment_files(args.min_depth, args.max_depth):
        analyzer.plot_scores_by_depth(args.output)
    else:
        print("Analysis failed: No experiment files found matching criteria.")

if __name__ == "__main__":
    main() 