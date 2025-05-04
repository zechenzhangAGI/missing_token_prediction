import json
import os
import argparse
import time
import random
from typing import Dict, List, Optional, Any, Union, Tuple
import re
import numpy as np
from tqdm import tqdm
import concurrent.futures

class Node:
    """Tree node representing a self-contained problem and solution."""
    
    def __init__(
        self, 
        name: str, 
        problem_statement: str, 
        problem_solution: str,
        link_description: str = "",
        children: List["Node"] = None
    ):
        self.name = name
        self.problem_statement = problem_statement
        self.problem_solution = problem_solution
        self.link_description = link_description
        self.children = children or []
        
        # Fields to track model performance
        self.model_answers = []
        self.judge_scores = []
        self.judge_feedback = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        result = {
            "name": self.name,
            "problemStatement": self.problem_statement,
            "problemSolution": self.problem_solution,
            "linkDescription": self.link_description,
            "children": [child.to_dict() for child in self.children],
        }
        
        # Include model evaluation data if available
        if self.model_answers:
            result["modelAnswers"] = self.model_answers
        if self.judge_scores:
            result["judgeScores"] = self.judge_scores
        if self.judge_feedback:
            result["judgeFeedback"] = self.judge_feedback
            
        return result
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Node":
        """Create a node from a dictionary representation."""
        children = [Node.from_dict(child) for child in data.get("children", [])]
        
        node = Node(
            name=data["name"],
            problem_statement=data["problemStatement"],
            problem_solution=data["problemSolution"],
            link_description=data.get("linkDescription", ""),
            children=children
        )
        
        # Restore any evaluation data
        if "modelAnswers" in data:
            node.model_answers = data["modelAnswers"]
        if "judgeScores" in data:
            node.judge_scores = data["judgeScores"]
        if "judgeFeedback" in data:
            node.judge_feedback = data["judgeFeedback"]
            
        return node


class TreeExperiment:
    """Main class for managing the tree-based experimental pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.root_node = None
        self.llm_client = None
        self.results_dir = config.get("results_dir", "results")
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
    
    def build_tree_from_paper(self, paper_path: str, max_depth: Optional[int] = None) -> Node:
        """
        Build a tree structure from a research paper.
        This should be implemented with an LLM call.
        
        Args:
            paper_path: Path to the research paper
            max_depth: Maximum depth for the generated tree (optional)
        """
        # Read the paper content
        with open(paper_path, 'r', encoding='utf-8') as f:
            paper_content = f.read()
        
        # Prompt the LLM to break down the paper into a tree structure
        prompt = f"""
        Please analyze the following research paper and break it down into a hierarchical tree structure where:
        
        1. STRUCTURE REQUIREMENTS:
           - Each node represents a self-contained problem and its solution
           - The root node captures the overall goal/contribution of the paper
           - Child nodes represent distinct sub-problems that must be solved to achieve the parent's goal
           - A parent node's problem can be solved completely if and only if all its child nodes' problems are solved
        
        2. NODE INDEPENDENCE:
           - Each node should be independent and conceptually self-contained
           - A node's problem statement should be understandable without requiring other nodes' content
           - Solutions should reflect knowledge of child node solutions when appropriate
        
        3. NODE ATTRIBUTES:
           - name: A short descriptive title for the problem
           - problemStatement: Clear definition of what needs to be solved
           - problemSolution: The paper's solution to this problem
           - linkDescription: How this node relates to its parent node
           - children: Array of child nodes (if any)
        
        4. OUTPUT FORMAT:
           Return valid, properly nested JSON that can be parsed directly.
        
        
        Here's a shortened example of the expected output structure, but be more specific than the example:
        ```json
        {{
          "name": "Overall Goal: Establish dS3 / Matrix Model Duality",
          "problemStatement": "Demonstrate the precise duality between pure dS3 quantum gravity and a specific double-scaled matrix integral. ",
          "problemSolution": "The duality holds as both prerequisites are achieved: (1) The observables match with A_n = W_n, and (2) The entropy values match with S_MM = S_GH.",
          "linkDescription": "",
          "children": [
            {{
              "name": "Demonstrate Observables Match (A_n = W_n)",
              "problemStatement": "Prove the equality between dS3 gravity observables (A_n) and matrix model observables (W_n).",
              "problemSolution": "Proven by showing both A_n and W_n equal the complex Liouville amplitude through independent calculations.",
              "linkDescription": "Prerequisite: First pillar of the duality proof.",
              "children": [
                {{
                  "name": "Calculate A_n and Show A_n = A^Liouville",
                  "problemStatement": "Calculate integrated dS3 correlator A_n and prove its equality to the Liouville amplitude.",
                  "problemSolution": "By integrating the boundary correlator over moduli space with the appropriate measure, we obtain the Liouville string amplitude.",
                  "linkDescription": "Prerequisite: Establishes half of the observables equality.",
                  "children": []
                }}
              ]
            }},
            {{
              "name": "Demonstrate Entropy Match (S_MM = S_GH)",
              "problemStatement": "Prove the equality between matrix model entropy (S_MM) and Gibbons-Hawking entropy (S_GH).",
              "problemSolution": "Both entropies are expressed in terms of common parameters and shown to be numerically equal.",
              "linkDescription": "Prerequisite: Second pillar of the duality proof.",
              "children": []
            }}
          ]
        }}
        ```
        
        
        
        Here's the paper content:
        ```
        {paper_content}
        ```
        """
        
        # Add max_depth instruction if specified
        if max_depth is not None:
            prompt += f"""
        IMPORTANT: The tree should have a depth of {max_depth} levels.
        - Level 1: Root node (overall goal)
        - Level 2: Major sub-problems ...
        - Level 3: More specific sub-problems ...
        ...
        - Level {max_depth}: Most specific sub-problems
        In general, the shallow nodes should be a coarse grained problem where the children nodes provide specific research directions to solve the parent node problem. For example, the root node would be really hard to solve without knowing more details about its children nodes.
        As level of depth increases, the sub-problems should become more specific and detailed.
        The leaf nodes can terminate at any level below level {max_depth} and represents the most specific sub-problems.
        
        Do not exceed {max_depth} levels of nesting in the tree structure.
        
        Finally,VERY IMPORTANT: BE AS SPECIFIC AS POSSIBLE FOR THE PROBLEM STATEMENT AND SOLUTION FOR THE LEAF NODES (the most specific sub-problems) SO THAT THE PROBLEM HAS CONTEXT AND DEFINATIONS AS A SELF-CONTAINED PHYSICS PROBLEM, CONTAINING EQUATIONS, DEFINITIONS, AND CONTEXTS.
        """
        prompt += f"""
        Return the result as a valid JSON object with this nested structure.
        
        Here's the paper content:
        ```
        {paper_content}
        ```
        """
        
        # Call LLM API with the prompt
        model_name = self.config.get("tree_builder_model", self.config.get("model_name", "gpt-4"))
        max_tokens = self.config.get("max_tokens", {}).get("tree_building", 8000)
        temperature = self.config.get("temperature", {}).get("tree_building", 0.2)
        
        response = self._call_llm(
            prompt, 
            model_name=model_name,
            max_tokens=max_tokens, 
            temperature=temperature
        )
        
        # Parse JSON from the response (assuming the LLM returns valid JSON)
        # In practice, you might need more robust extraction logic
        tree_data = self._extract_json_from_llm_response(response)
        
        # Convert the dictionary to Node objects
        self.root_node = Node.from_dict(tree_data)
        return self.root_node
    
    def generate_model_answers(self, node: Node, samples: int = 1) -> List[str]:
        """Generate answers for a specific node using the target model."""
        prompt = f"""
        Below is a physics problem adapted from a research paper. 
        PROBLEM:
        {node.problem_statement}
        
        Please solve this problem step by step and be as specific as possible.
        """
        
        model_name = self.config.get("solver_model", self.config.get("model_name", "gpt-3.5-turbo"))
        max_tokens = self.config.get("max_tokens", {}).get("solving", 2000)
        temperature = self.config.get("temperature", {}).get("solving", 0.7)
        
        answers = []
        for _ in range(samples):
            response = self._call_llm(
                prompt, 
                model_name=model_name,
                max_tokens=max_tokens, 
                temperature=temperature
            )
            answers.append(response)
        
        return answers
    
    def judge_model_answers(self, node: Node, answers: List[str]) -> List[Dict[str, Any]]:
        """Evaluate model answers against the ground truth solution."""
        results = []
        
        model_name = self.config.get("judge_model", self.config.get("model_name", "gpt-4"))
        max_tokens = self.config.get("max_tokens", {}).get("judging", 2000)
        temperature = self.config.get("temperature", {}).get("judging", 0.2)
        
        for answer in answers:
            prompt = f"""
            You are a judge evaluating a model's solution to a problem.
            
            PROBLEM:
            {node.problem_statement}
            
            CORRECT SOLUTION:
            {node.problem_solution}
            
            MODEL'S ANSWER:
            {answer}
            
            Please evaluate the model's answer on a scale from 0 to 10, where:
            - 0: Completely incorrect or irrelevant and hallucinated
            - 5: Partially correct but with significant gaps, derivations are not correct
            - 10: Fully correct and complete
            
            Provide your score and a detailed explanation of your evaluation. BE VERY CAREFUL AND STRICT WITH THE SCORE.
            Format your response as a JSON object with 'score' and 'feedback' fields.
            """
            
            response = self._call_llm(
                prompt, 
                model_name=model_name,
                max_tokens=max_tokens, 
                temperature=temperature
            )
            
            # Parse the response to extract score and feedback
            try:
                evaluation = self._extract_json_from_llm_response(response)
                results.append(evaluation)
            except:
                # Fallback in case the LLM doesn't produce valid JSON
                results.append({
                    "score": 0, 
                    "feedback": "Failed to parse evaluation response"
                })
        
        return results
    
    def process_node(self, node: Node) -> None:
        """Process a single node (for parallel execution)."""
        print(f"Processing node: {node.name}")
        
        # Generate answers using the target model
        samples = self.config.get("samples_per_node", 1)
        answers = self.generate_model_answers(node, samples=samples)
        node.model_answers = answers
        
        # Judge the answers
        evaluations = self.judge_model_answers(node, answers)
        node.judge_scores = [eval.get("score", 0) for eval in evaluations]
        node.judge_feedback = [eval.get("feedback", "") for eval in evaluations]
        
        return node
        
    def run_experiment(self):
        """Run the experiment on all nodes in parallel."""
        if self.root_node is None:
            raise ValueError("No tree structure has been loaded or created.")
        
        # Collect all nodes from the tree
        all_nodes = self._collect_all_nodes(self.root_node)
        print(f"Found {len(all_nodes)} nodes to process")
        
        # Process nodes in parallel
        max_workers = self.config.get("max_workers", min(32, os.cpu_count() * 2))
        use_parallel = self.config.get("use_parallel", True) and len(all_nodes) > 1
        
        if use_parallel:
            print(f"Processing nodes in parallel with {max_workers} workers")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_node = {executor.submit(self.process_node, node): node for node in all_nodes}
                
                # Process results as they complete
                for future in tqdm(concurrent.futures.as_completed(future_to_node), total=len(all_nodes)):
                    node = future_to_node[future]
                    try:
                        _ = future.result()
                    except Exception as exc:
                        print(f"Node {node.name} generated an exception: {exc}")
        else:
            print("Processing nodes sequentially")
            for node in tqdm(all_nodes, desc="Processing nodes"):
                self.process_node(node)
    
    def _collect_all_nodes(self, node: Node, nodes: List[Node] = None) -> List[Node]:
        """Recursively collect all nodes in the tree."""
        if nodes is None:
            nodes = []
        
        nodes.append(node)
        
        for child in node.children:
            self._collect_all_nodes(child, nodes)
        
        return nodes
    
    def calculate_stats(self, node: Node = None) -> Dict[str, Any]:
        """Calculate statistics for the experiment results."""
        if node is None:
            node = self.root_node
        
        # Node-level statistics
        stats = {
            "node_name": node.name,
            "average_score": np.mean(node.judge_scores) if node.judge_scores else 0,
            "pass_at_k": {
                f"pass@{k}": self._calculate_pass_at_k(node.judge_scores, k, threshold=7) 
                for k in [1, 3, 5] if k <= len(node.judge_scores)
            }
        }
        
        # Add child stats
        if node.children:
            stats["children"] = [self.calculate_stats(child) for child in node.children]
            
            # Aggregate stats from all descendants
            all_scores = node.judge_scores.copy()
            for child_stat in stats["children"]:
                if "all_descendant_scores" in child_stat:
                    all_scores.extend(child_stat["all_descendant_scores"])
            
            stats["all_descendant_scores"] = all_scores
            stats["aggregate_average"] = np.mean(all_scores) if all_scores else 0
            stats["aggregate_pass_at_k"] = {
                f"pass@{k}": self._calculate_pass_at_k(all_scores, k, threshold=7) 
                for k in [1, 3, 5] if k <= len(all_scores)
            }
        
        return stats
    
    def save_results(self, filename: str = None):
        """Save the experiment results to a file."""
        if not filename:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"experiment_{timestamp}.json"
        
        output_path = os.path.join(self.results_dir, filename)
        
        result = {
            "config": self.config,
            "tree": self.root_node.to_dict() if self.root_node else None,
            "statistics": self.calculate_stats() if self.root_node else None
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        print(f"Results saved to {output_path}")
        return output_path
    
    def load_tree(self, file_path: str):
        """Load a previously saved tree structure from a file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "tree" in data:
            self.root_node = Node.from_dict(data["tree"])
        elif isinstance(data, dict) and "name" in data:
            self.root_node = Node.from_dict(data)
        else:
            raise ValueError("Invalid tree data format in the file")
        
        return self.root_node
    
    def _call_llm(self, prompt: str, model_name: str = None, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """
        Call the LLM API with the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            model_name: The specific model to use (overrides config)
            temperature: Sampling temperature
            max_tokens: Maximum response length
            
        Returns:
            The LLM's response text
        """
        # Use default model from config if not specified
        if model_name is None:
            model_name = self.config.get("model_name", "gpt-4")
            
        # Mock mode for testing without API calls
        if self.config.get("mock_mode", False):
            time.sleep(1)  # Simulate API latency
            return f"Mock response for: {prompt[:50]}..."
        
        # Harvard OpenAI endpoint implementation
        if "harvard" in self.config.get("llm_provider", "").lower() or self.config.get("use_harvard_endpoint", False):
            try:
                import requests
                
                url = self.config.get("harvard_api_url", 
                    "https://go.apis.huit.harvard.edu/ais-openai-direct-limited-schools/v1/chat/completions")
                
                payload = json.dumps({
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],

                    "max_completion_tokens": max_tokens
                })
                
                headers = {
                    'Content-Type': 'application/json',
                    'api-key': '709KLfil8udIORZu9UjjE4jhaGTG6uW0'
                }
                
                max_retries = self.config.get("max_retries", 3)
                for attempt in range(max_retries):
                    try:
                        print(f"Calling Harvard API (attempt {attempt+1}/{max_retries})")
                        print(f"Model: {model_name}, Max tokens: {max_tokens}")
                        
                        response = requests.request("POST", url, headers=headers, data=payload)
                        
                        # Save raw response to debug files
                        debug_dir = os.path.join(self.results_dir, "debug")
                        os.makedirs(debug_dir, exist_ok=True)
                        debug_file = os.path.join(debug_dir, f"harvard_api_{int(time.time())}.txt")
                        with open(debug_file, 'w', encoding='utf-8') as f:
                            f.write(f"API REQUEST:\n")
                            f.write(f"URL: {url}\n")
                            f.write(f"Headers: {headers}\n")
                            f.write(f"Payload: {payload}\n\n")
                            f.write(f"API RESPONSE:\n")
                            f.write(f"Status: {response.status_code}\n")
                            f.write(f"Content: {response.text}\n")
                        
                        # Log response details
                        print(f"API response status: {response.status_code}")
                        print(f"API response length: {len(response.text)} chars")
                        
                        response_dict = json.loads(response.text)
                        
                        # Check if response has expected structure
                        if "choices" not in response_dict:
                            print(f"WARNING: Response missing 'choices' key. Available keys: {list(response_dict.keys())}")
                            if "error" in response_dict:
                                print(f"API ERROR: {response_dict['error']}")
                                return f"API ERROR: {response_dict.get('error', {}).get('message', 'Unknown error')}"
                            
                            # Try to use a different model as fallback
                            if attempt == max_retries - 1 and model_name != "gpt-3.5-turbo-1106":
                                print("Attempting fallback to gpt-3.5-turbo-1106 model...")
                                model_name = "gpt-3.5-turbo-1106"
                                payload = json.dumps({
                                    "model": model_name,
                                    "messages": [{"role": "user", "content": prompt}],
                                    "temperature": temperature,
                                    "max_completion_tokens": max_tokens
                                })
                                continue
                            
                            return "ERROR: Unexpected API response format"
                        
                        content = response_dict["choices"][0]["message"]["content"].strip()
                        print(f"Successfully received content ({len(content)} chars)")
                        return content
                    except Exception as e:
                        print(f"API Call error (attempt {attempt+1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            time.sleep(5)
                        else:
                            print(f"Failed after {max_retries} attempts. Last error: {e}")
                            return "ERROR: Harvard API call failed after multiple attempts."
            except ImportError:
                return "ERROR: Missing requests package for Harvard API calls."
        
        # Standard OpenAI API implementation
        elif "openai" in self.config.get("llm_provider", "").lower():
            try:
                import openai
                client = openai.OpenAI(api_key=self.config.get("api_key"))
                
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful research assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return response.choices[0].message.content
                
            except ImportError:
                raise ImportError("To use OpenAI, you need to install the openai package.")
            
        # Example implementation for Anthropic API
        elif "anthropic" in self.config.get("llm_provider", "").lower():
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=self.config.get("api_key"))
                
                response = client.messages.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system="You are a helpful research assistant.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                return response.content[0].text
                
            except ImportError:
                raise ImportError("To use Anthropic, you need to install the anthropic package.")
        
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.get('llm_provider')}")
    
    def _extract_json_from_llm_response(self, response: str) -> Dict[str, Any]:
        """Extract and parse JSON from an LLM response."""
        # Save raw response to debug file
        debug_dir = os.path.join(self.results_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        debug_file = os.path.join(debug_dir, f"llm_response_{int(time.time())}.txt")
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(f"RAW RESPONSE:\n{response}\n\n")
        
        # Try different strategies to extract JSON
        try:
            # Strategy 1: Direct parsing (if the entire response is valid JSON)
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass
            
            # Strategy 2: Find JSON-like content within the response using regex
            json_pattern = r'(\{.*\}|\[.*\])'
            match = re.search(json_pattern, response, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Remove potential markdown formatting
                    cleaned = json_str.replace('```json', '').replace('```', '')
                    return json.loads(cleaned)
            
            # Strategy 3: Try to find a JSON block within markdown code blocks
            code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
            code_matches = re.findall(code_block_pattern, response)
            for code_match in code_matches:
                try:
                    return json.loads(code_match)
                except json.JSONDecodeError:
                    continue
            
            # Strategy 4: Attempt to repair common JSON errors
            # Try with added outer braces if they might be missing
            if not (response.strip().startswith('{') or response.strip().startswith('[')):
                try:
                    return json.loads('{' + response + '}')
                except json.JSONDecodeError:
                    pass
            
            # Log the failure details for debugging
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write(f"JSON EXTRACTION FAILED - COULD NOT PARSE\n")
            
            # If all extraction methods fail, build a simple structure from the response
            return {
                "name": "Error extracting JSON",
                "problemStatement": "The LLM did not return a valid JSON structure.",
                "problemSolution": response[:500] + ("..." if len(response) > 500 else ""),
                "linkDescription": "",
                "children": []
            }
            
        except Exception as e:
            # Catch any other errors in the JSON extraction process
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write(f"JSON EXTRACTION ERROR: {str(e)}\n")
            
            return {
                "name": f"Error: {str(e)}",
                "problemStatement": "An error occurred while trying to extract JSON from the LLM response.",
                "problemSolution": response[:500] + ("..." if len(response) > 500 else ""),
                "linkDescription": "",
                "children": []
            }
    
    def _calculate_pass_at_k(self, scores: List[float], k: int, threshold: float = 7.0) -> float:
        """
        Calculate the pass@k metric.
        
        Args:
            scores: List of scores
            k: k value for pass@k
            threshold: Minimum score to be considered a "pass"
            
        Returns:
            pass@k score (between 0 and 1)
        """
        if not scores or k > len(scores):
            return 0.0
        
        # Pass@k is the probability of getting at least one correct answer
        # if we sample k times
        n = len(scores)
        c = sum(1 for score in scores if score >= threshold)
        
        if c == 0:
            return 0.0
        
        if k >= n:
            # If k >= n, we're sampling all available answers
            return 1.0 if c > 0 else 0.0
        
        # Calculate 1 - probability of getting no correct answers
        # This is 1 - (n-c choose k) / (n choose k)
        if k == 1:
            return c / n
        
        if c == n:
            return 1.0
            
        # Use the formula: 1 - [(n-c)! / (n-c-k)! / k!] / [n! / (n-k)! / k!]
        # Simplified to: 1 - [(n-c)! * (n-k)!] / [n! * (n-c-k)!]
        # For numerical stability, we compute this differently
        
        # Calculate directly using the hypergeometric probability formula
        prob_no_correct = 1.0
        for i in range(k):
            prob_no_correct *= (n - c - i) / (n - i)
            
        return 1.0 - prob_no_correct


def main():
    parser = argparse.ArgumentParser(description="Run tree-based experiments on research papers")
    parser.add_argument("--paper", type=str, default="papers/ds3_matrix_model/cleaned.md", 
                        help="Path to the research paper")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to configuration file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file name (default: auto-generated)")
    parser.add_argument("--load-tree", type=str, default=None,
                        help="Load pre-existing tree structure from file")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Maximum depth of the tree to generate from the paper")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of samples per node (overrides config)")
    parser.add_argument("--mock", action="store_true",
                        help="Run in mock mode without actual API calls")
    parser.add_argument("--sequential", action="store_true",
                        help="Process nodes sequentially instead of in parallel")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        print(f"Config file {args.config} not found, using defaults")
        config = {}
    
    # Override config with command line args
    if args.samples:
        config["samples_per_node"] = args.samples
    if args.mock:
        config["mock_mode"] = True
    if args.sequential:
        config["use_parallel"] = False
    
    # Initialize experiment
    experiment = TreeExperiment(config)
    
    # Either load existing tree or build new one
    if args.load_tree:
        print(f"Loading tree from {args.load_tree}")
        experiment.load_tree(args.load_tree)
    else:
        print(f"Building tree from {args.paper} with max depth {args.max_depth}")
        experiment.build_tree_from_paper(args.paper, max_depth=args.max_depth)
    
    # Run the experiment
    experiment.run_experiment()
    
    # Save results
    experiment.save_results(filename=args.output)
    
    print("Experiment completed successfully")


if __name__ == "__main__":
    main() 