# Missing Token Prediction Experimental Pipeline

This project implements a hierarchical tree-based experimental pipeline for evaluating large language models (LLMs) on complex academic problems. The pipeline:

1. Breaks down a research paper into a hierarchical tree of self-contained problems and solutions
2. Tests an LLM's ability to solve each sub-problem
3. Evaluates the quality of the model's solutions against ground truth
4. Computes pass@k metrics and visualizes performance

## Installation

```bash
# Clone the repository
git clone [repository-url]
cd missing_token_prediction

# Install required dependencies
pip install -r requirements.txt

# Note: pygraphviz requires graphviz to be installed:
# On Ubuntu/Debian:
# sudo apt-get install graphviz graphviz-dev
# On macOS:
# brew install graphviz
```

## Configuration

Edit `config.json` to configure your experiment:

```json
{
  "llm_provider": "harvard",      // "harvard", "openai" or "anthropic"
  "model_name": "o3-mini-2025-01-31", // Default model
  "tree_builder_model": "o3-mini-2025-01-31", // Model to use for tree building
  "solver_model": "o3-mini-2025-01-31", // Model to test for problem solving
  "judge_model": "o3-mini-2025-01-31",  // Model for evaluating solutions
  "harvard_api_key": "your_harvard_api_key", // Harvard API key
  "harvard_api_url": "https://go.apis.huit.harvard.edu/ais-openai-direct-limited-schools/v1/chat/completions", // Harvard API URL
  "samples_per_node": 3,          // Number of samples per node (for pass@k)
  "pass_threshold": 7.0,          // Score threshold to consider a solution correct
  "use_parallel": true,           // Process nodes in parallel
  "max_workers": 16               // Maximum number of parallel workers
}
```

### API Endpoints

The experiment supports multiple API endpoints:

1. **Harvard API**: For Harvard University users, the system can use the custom Harvard API endpoint for accessing OpenAI models.
   - Set `"llm_provider": "harvard"`
   - Configure `harvard_api_key` and `harvard_api_url` as needed

2. **Standard OpenAI API**: For direct OpenAI API access
   - Set `"llm_provider": "openai"`
   - Configure `api_key` with your OpenAI API key

3. **Anthropic API**: For using Anthropic's Claude models
   - Set `"llm_provider": "anthropic"`
   - Configure `api_key` with your Anthropic API key

## Usage

### Running an Experiment

```bash
# Run with default settings (uses papers/ds3_matrix_model/cleaned.md)
python tree_experiment.py

# Specify a different paper
python tree_experiment.py --paper path/to/your/paper.md

# Control tree depth (maximum levels of nesting in the generated tree)
python tree_experiment.py --max-depth 3

# Specify alternative configuration
python tree_experiment.py --config custom_config.json

# Run in mock mode (for testing without API calls)
python tree_experiment.py --mock

# Control sampling
python tree_experiment.py --samples 5  # 5 samples per node

# Process nodes sequentially (no parallelization)
python tree_experiment.py --sequential
```

### Visualizing Results

The experiment will output a JSON file with results. You can visualize these using:

```bash
python tree_visualizer.py results/experiment_20230101-123456.json
```

This will generate an HTML report with visualizations of the problem tree, performance metrics, and detailed node-by-node results.

## How It Works

### 1. Tree Building

The pipeline uses a powerful LLM to analyze the paper and break it down into a hierarchical tree where:
- The root node represents the paper's overall goal/contribution
- Child nodes represent sub-problems that must be solved
- Each node contains a problem statement and solution
- The `max_depth` parameter controls how deep/detailed the tree becomes

### 2. Model Testing

For each node in the tree:
- The model is presented with just the problem statement
- The model generates N solutions (configurable samples per node)
- Each solution is evaluated against the ground truth
- All nodes are processed in parallel for efficiency (configurable)

### 3. Evaluation

The pipeline uses a judge LLM to:
- Compare model-generated solutions against ground truth
- Assign a score from 0-10 for correctness
- Provide detailed feedback on the solution quality

### 4. Analysis & Visualization

The pipeline calculates:
- Raw scores for each node
- pass@k metrics (probability of getting at least one correct answer in k attempts)
- Aggregate statistics across the tree

Visualizations include:
- Interactive tree with performance heatmap
- pass@k bar charts
- Score distribution histograms

## Example Tree Structure

Here's an example of the tree structure format:

```json
{
  "name": "Overall Goal: Establish dS3 / Matrix Model Duality",
  "problemStatement": "...",
  "problemSolution": "...",
  "linkDescription": "",
  "children": [
    {
      "name": "Sub-Problem 1",
      "problemStatement": "...",
      "problemSolution": "...",
      "linkDescription": "This sub-problem relates to the parent by...",
      "children": [...]
    },
    {
      "name": "Sub-Problem 2",
      "problemStatement": "...",
      "problemSolution": "...",
      "linkDescription": "...",
      "children": [...]
    }
  ]
}
```

## Performance Considerations

- **Parallelization**: By default, all nodes are processed in parallel to improve efficiency. This can be adjusted through the `use_parallel` and `max_workers` config settings or disabled with the `--sequential` flag.

- **API Costs**: When evaluating many nodes with multiple samples, API costs can add up quickly. Consider using `--max-depth` to limit tree complexity or `--samples` to reduce the number of evaluations.

- **API Rate Limits**: The system includes automatic retries for API failures. You can configure `max_retries` in the config file.

## License

[Your License Here] 
