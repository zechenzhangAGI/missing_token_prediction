import json
import os
import argparse
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import networkx as nx
from matplotlib.patches import Patch

class TreeVisualizer:
    """Visualization tool for tree experiment results."""
    
    def __init__(self, results_file: str):
        """Initialize with the path to results JSON file."""
        with open(results_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tree = self.data.get("tree", {})
        self.stats = self.data.get("statistics", {})
        self.config = self.data.get("config", {})
        
        # Create a directory for visualizations
        self.out_dir = os.path.join(os.path.dirname(results_file), "visualizations")
        os.makedirs(self.out_dir, exist_ok=True)
    
    def build_graph(self, node_dict: Dict[str, Any] = None, parent: Optional[str] = None) -> nx.DiGraph:
        """
        Recursively build a NetworkX graph from the tree data.
        
        Args:
            node_dict: Dictionary representation of current node (defaults to root)
            parent: ID of parent node for edge creation
            
        Returns:
            NetworkX DiGraph object
        """
        if node_dict is None:
            node_dict = self.tree
        
        G = nx.DiGraph()
        
        # Process current node
        node_id = node_dict.get("name", "Root")
        avg_score = np.mean(node_dict.get("judgeScores", [0]))
        
        # Add current node with its attributes
        G.add_node(
            node_id,
            name=node_id,
            avg_score=avg_score,
            problem=node_dict.get("problemStatement", ""),
            solution=node_dict.get("problemSolution", ""),
            link_desc=node_dict.get("linkDescription", ""),
            scores=node_dict.get("judgeScores", [])
        )
        
        # Connect to parent if exists
        if parent:
            G.add_edge(
                parent, 
                node_id, 
                description=node_dict.get("linkDescription", "")
            )
        
        # Process children recursively
        for child in node_dict.get("children", []):
            child_graph = self.build_graph(child, node_id)
            G = nx.compose(G, child_graph)
        
        return G
    
    def visualize_tree(self, output_file: str = "tree.png"):
        """
        Create a visualization of the problem tree with node scores.
        
        Args:
            output_file: Filename for the visualization output
        """
        G = self.build_graph()
        
        # Set up the plot
        plt.figure(figsize=(20, 12))
        
        # Layout - hierarchical for trees
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=TB")
        
        # Get scores for coloring nodes
        scores = [data.get('avg_score', 0) for _, data in G.nodes(data=True)]
        
        # Create a colormap based on scores
        vmin, vmax = 0, 10
        cmap = cm.get_cmap('RdYlGn')
        norm = plt.Normalize(vmin, vmax)
        
        # Draw the nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=scores,
            cmap=cmap,
            node_size=1500,
            node_shape='o',
            alpha=0.8,
            vmin=vmin, vmax=vmax
        )
        
        # Draw the edges with arrow
        nx.draw_networkx_edges(
            G, pos,
            arrows=True,
            arrowsize=20,
            width=2,
            alpha=0.7,
            edge_color='gray'
        )
        
        # Add labels - node names only, truncated if too long
        labels = {n: self._truncate(d['name'], 30) for n, d in G.nodes(data=True)}
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=10,
            font_weight='bold',
            font_family='sans-serif'
        )
        
        # Add a colorbar for score reference
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('Average Score', fontsize=12)
        
        # Add title and adjust layout
        plt.title('Problem Tree with Performance Scores', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(self.out_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Tree visualization saved to {output_path}")
        return output_path
    
    def visualize_pass_at_k(self, output_file: str = "pass_at_k.png"):
        """
        Create a bar chart of pass@k scores for different nodes in the tree.
        
        Args:
            output_file: Filename for the visualization output
        """
        # Collect pass@k data from all nodes
        node_stats = self._collect_node_stats(self.stats)
        
        # Extract node names and pass@k values
        node_names = [stat['node_name'] for stat in node_stats]
        pass_at_1 = [stat['pass_at_k'].get('pass@1', 0) for stat in node_stats]
        pass_at_3 = [stat['pass_at_k'].get('pass@3', 0) for stat in node_stats]
        pass_at_5 = [stat['pass_at_k'].get('pass@5', 0) for stat in node_stats]
        
        # Set up the plot
        plt.figure(figsize=(14, 10))
        
        # Set up bar positions
        x = np.arange(len(node_names))
        width = 0.25
        
        # Plot bars
        bar1 = plt.bar(x - width, pass_at_1, width, label='pass@1', color='#5DA5DA')
        bar2 = plt.bar(x, pass_at_3, width, label='pass@3', color='#FAA43A')
        bar3 = plt.bar(x + width, pass_at_5, width, label='pass@5', color='#60BD68')
        
        # Add labels and title
        plt.xlabel('Nodes', fontsize=12)
        plt.ylabel('pass@k Score', fontsize=12)
        plt.title('pass@k Performance Across Tree Nodes', fontsize=16)
        plt.xticks(x, [self._truncate(name, 20) for name in node_names], rotation=45, ha='right')
        plt.yticks(np.arange(0, 1.1, 0.1))
        
        # Add legend and grid
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Tight layout and save
        plt.tight_layout()
        output_path = os.path.join(self.out_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"pass@k visualization saved to {output_path}")
        return output_path
    
    def visualize_score_distribution(self, output_file: str = "score_distribution.png"):
        """
        Create a histogram of scores across all nodes.
        
        Args:
            output_file: Filename for the visualization output
        """
        # Collect all scores from tree
        all_scores = self._collect_all_scores(self.tree)
        
        if not all_scores:
            print("No scores available for visualization")
            return None
        
        # Set up the plot
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        bins = np.arange(0, 11, 1) - 0.5
        plt.hist(all_scores, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        
        # Add a vertical line for the pass threshold
        threshold = self.config.get("pass_threshold", 7.0)
        plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Pass Threshold ({threshold})')
        
        # Add labels and title
        plt.xlabel('Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Score Distribution Across All Problems', fontsize=16)
        plt.xticks(range(0, 11))
        
        # Add legend and grid
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Tight layout and save
        plt.tight_layout()
        output_path = os.path.join(self.out_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Score distribution visualization saved to {output_path}")
        return output_path
    
    def generate_html_report(self, output_file: str = "report.html"):
        """
        Generate an HTML report with all visualizations and detailed node information.
        
        Args:
            output_file: Filename for the HTML report
        """
        # Generate all visualizations
        tree_viz = self.visualize_tree()
        pass_at_k_viz = self.visualize_pass_at_k()
        score_dist_viz = self.visualize_score_distribution()
        
        # Create relative paths for images
        tree_img = os.path.relpath(tree_viz, self.out_dir)
        pass_at_k_img = os.path.relpath(pass_at_k_viz, self.out_dir) if pass_at_k_viz else ""
        score_dist_img = os.path.relpath(score_dist_viz, self.out_dir) if score_dist_viz else ""
        
        # Build HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Tree Experiment Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .section {{ margin-bottom: 30px; }}
                .stats {{ display: flex; flex-wrap: wrap; }}
                .stat-card {{ background: #f5f5f5; border-radius: 5px; padding: 15px; margin: 10px; min-width: 200px; }}
                .viz-container {{ margin: 20px 0; text-align: center; }}
                .viz-container img {{ max-width: 100%; border: 1px solid #ddd; }}
                .node-details {{ margin: 20px 0; }}
                .node {{ background: #f9f9f9; border-left: 4px solid #333; padding: 10px; margin-bottom: 15px; }}
                .problem {{ background: #eef; padding: 10px; border-radius: 3px; }}
                .solution {{ background: #efe; padding: 10px; border-radius: 3px; }}
                .answers {{ margin-top: 10px; }}
                .answer {{ background: #fff; border: 1px solid #ddd; padding: 10px; margin-bottom: 5px; }}
                .score {{ font-weight: bold; }}
                .feedback {{ font-style: italic; color: #555; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>Tree Experiment Results</h1>
            
            <div class="section">
                <h2>Experiment Configuration</h2>
                <table>
                    <tr><th>Setting</th><th>Value</th></tr>
        """
        
        # Add configuration
        for key, value in self.config.items():
            if isinstance(value, dict):
                html_content += f"<tr><td>{key}</td><td>{json.dumps(value)}</td></tr>"
            else:
                html_content += f"<tr><td>{key}</td><td>{value}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
        """
        
        # Add tree visualization
        html_content += f"""
                <div class="viz-container">
                    <h3>Problem Tree Structure with Performance</h3>
                    <img src="{tree_img}" alt="Problem Tree Visualization">
                </div>
        """
        
        # Add pass@k visualization
        if pass_at_k_img:
            html_content += f"""
                <div class="viz-container">
                    <h3>pass@k Performance by Node</h3>
                    <img src="{pass_at_k_img}" alt="pass@k Visualization">
                </div>
            """
        
        # Add score distribution visualization
        if score_dist_img:
            html_content += f"""
                <div class="viz-container">
                    <h3>Score Distribution</h3>
                    <img src="{score_dist_img}" alt="Score Distribution">
                </div>
            """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Overall Statistics</h2>
        """
        
        # Add aggregate statistics
        if "aggregate_average" in self.stats:
            html_content += f"""
                <div class="stats">
                    <div class="stat-card">
                        <h3>Average Score</h3>
                        <p>{self.stats["aggregate_average"]:.2f} / 10.0</p>
                    </div>
            """
            
            if "aggregate_pass_at_k" in self.stats:
                for k, score in self.stats["aggregate_pass_at_k"].items():
                    html_content += f"""
                    <div class="stat-card">
                        <h3>{k}</h3>
                        <p>{score:.2f}</p>
                    </div>
                    """
            
            html_content += "</div>"
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Node Details</h2>
        """
        
        # Add detailed node information
        html_content += self._generate_node_html(self.tree)
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write HTML file
        output_path = os.path.join(self.out_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report saved to {output_path}")
        return output_path
    
    def _collect_node_stats(self, stats_dict: Dict[str, Any], result: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Recursively collect statistics for all nodes."""
        if result is None:
            result = []
        
        # Skip if it's the list of all scores
        if "node_name" not in stats_dict:
            return result
        
        # Add current node stats
        node_stat = {
            "node_name": stats_dict.get("node_name", "Unknown"),
            "average_score": stats_dict.get("average_score", 0),
            "pass_at_k": stats_dict.get("pass_at_k", {})
        }
        result.append(node_stat)
        
        # Process children recursively
        for child_stats in stats_dict.get("children", []):
            self._collect_node_stats(child_stats, result)
        
        return result
    
    def _collect_all_scores(self, node_dict: Dict[str, Any], result: List[float] = None) -> List[float]:
        """Recursively collect all scores from the tree."""
        if result is None:
            result = []
        
        # Add current node scores
        result.extend(node_dict.get("judgeScores", []))
        
        # Process children recursively
        for child in node_dict.get("children", []):
            self._collect_all_scores(child, result)
        
        return result
    
    def _generate_node_html(self, node_dict: Dict[str, Any], depth: int = 0) -> str:
        """Recursively generate HTML for node details."""
        node_html = f"""
        <div class="node" style="margin-left: {depth * 20}px">
            <h3>{node_dict.get("name", "Unknown")}</h3>
        """
        
        if node_dict.get("linkDescription"):
            node_html += f"""
            <p><strong>Connection to parent:</strong> {node_dict.get("linkDescription")}</p>
            """
        
        # Add problem and solution
        problem_statement = node_dict.get("problemStatement", "").replace('\n', '<br>')
        problem_solution = node_dict.get("problemSolution", "").replace('\n', "<br>")
        node_html += f"""
            <div class="problem">
                <h4>Problem Statement:</h4>
                <p>{problem_statement}</p>
            </div>
            
            <div class="solution">
                <h4>Solution:</h4>
                <p>{problem_solution}</p>
            </div>
        """
        
        # Add model answers and evaluations if available
        if node_dict.get("modelAnswers") and node_dict.get("judgeScores"):
            node_html += """
            <div class="answers">
                <h4>Model Answers and Evaluations:</h4>
            """
            
            for i, (answer, score, feedback) in enumerate(zip(
                node_dict.get("modelAnswers", []),
                node_dict.get("judgeScores", []),
                node_dict.get("judgeFeedback", [])
            )):
                answer_html = answer.replace('\n', "<br>")
                node_html += f"""
                <div class="answer">
                    <p class="score">Attempt {i+1} - Score: {score}/10</p>
                    <p class="feedback"><strong>Feedback:</strong> {feedback}</p>
                    <details>
                        <summary>Show Answer</summary>
                        <p>{answer_html}</p>
                    </details>
                </div>
                """
            
            node_html += """
            </div>
            """
        
        # Process children recursively
        for child in node_dict.get("children", []):
            node_html += self._generate_node_html(child, depth + 1)
        
        node_html += """
        </div>
        """
        
        return node_html
    
    def _truncate(self, text: str, max_len: int = 30) -> str:
        """Truncate text to max_len characters with ellipsis if needed."""
        if len(text) <= max_len:
            return text
        return text[:max_len-3] + "..."


def main():
    parser = argparse.ArgumentParser(description="Visualize tree experiment results")
    parser.add_argument("results_file", type=str, help="Path to experiment results JSON file")
    parser.add_argument("--output", type=str, default="report.html", help="Output HTML report filename")
    
    args = parser.parse_args()
    
    visualizer = TreeVisualizer(args.results_file)
    visualizer.generate_html_report(args.output)


if __name__ == "__main__":
    main() 