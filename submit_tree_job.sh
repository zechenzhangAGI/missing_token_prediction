#!/bin/bash
#SBATCH --partition=serial_requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=0-5:00
#SBATCH --mem=100G
#SBATCH --job-name=tree_exp
#SBATCH --output=/n/home04/zechenzhang/missing_token_prediction/logs/outs/%x_%j.out
#SBATCH --error=/n/home04/zechenzhang/missing_token_prediction/logs/errs/%x_%j.err

# Load modules and activate environment
module load python
mamba activate ML_2024

# Create log directories if they don't exist
mkdir -p /n/home04/zechenzhang/missing_token_prediction/logs/outs
mkdir -p /n/home04/zechenzhang/missing_token_prediction/logs/errs

# Change to the project directory
cd /n/home04/zechenzhang/missing_token_prediction

# Configuration parameters (modify as needed)
PAPER_PATH="papers/ds3_matrix_model/cleaned.md"  # Path to the research paper
CONFIG_FILE="config.json"                        # Path to configuration file
MAX_DEPTH=7                                      # Maximum tree depth
SAMPLES=3                                        # Samples per node
OUTPUT_NAME="experiment_depth${MAX_DEPTH}_samples${SAMPLES}"  # Auto-generated output filename

# Optional parameters (uncomment to use)
LOAD_TREE=""         # --load-tree path/to/tree.json
USE_MOCK=""          # --mock
USE_SEQUENTIAL=""    # --sequential

# Uncomment to use optional parameters
#LOAD_TREE="--load-tree path/to/existing/tree.json"
#USE_MOCK="--mock"  # Useful for testing without API calls
#USE_SEQUENTIAL="--sequential"  # For debugging parallelization issues

# Print configuration for log
echo "Running tree experiment with configuration:"
echo "Paper: $PAPER_PATH"
echo "Config file: $CONFIG_FILE"
echo "Max depth: $MAX_DEPTH"
echo "Samples per node: $SAMPLES"
echo "Output name: $OUTPUT_NAME"
if [ -n "$LOAD_TREE" ]; then echo "Loading existing tree: $LOAD_TREE"; fi
if [ -n "$USE_MOCK" ]; then echo "Using mock mode"; fi
if [ -n "$USE_SEQUENTIAL" ]; then echo "Using sequential mode"; fi
echo "---------------------------------------"

# Run the experiment
python tree_experiment.py \
  --paper "$PAPER_PATH" \
  --config "$CONFIG_FILE" \
  --max-depth "$MAX_DEPTH" \
  --samples "$SAMPLES" \
  --output "$OUTPUT_NAME.json" \
  $LOAD_TREE $USE_MOCK $USE_SEQUENTIAL

# Check if the experiment completed successfully
if [ $? -eq 0 ]; then
  echo "Experiment completed successfully"
  echo "Results saved to $OUTPUT_NAME.json"
else
  echo "Experiment failed with error code $?"
fi 