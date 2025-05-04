#!/bin/bash
#SBATCH --partition=serial_requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-0:30
#SBATCH --mem=8G
#SBATCH --job-name=depth_plot
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

# Configuration parameters
CONFIG_FILE="config.json"
MIN_DEPTH=3
MAX_DEPTH=7
OUTPUT_FILE="depth_level_scores.png"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --min-depth)
      MIN_DEPTH="$2"
      shift 2
      ;;
    --max-depth)
      MAX_DEPTH="$2"
      shift 2
      ;;
    --output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Extract results_dir from config.json
RESULTS_DIR=$(python -c "import json; print(json.load(open('$CONFIG_FILE'))['results_dir'])")

# Check if RESULTS_DIR exists
if [ ! -d "$RESULTS_DIR" ]; then
  echo "Error: Results directory '$RESULTS_DIR' not found."
  exit 1
fi

# Print configuration for log
echo "Running depth score analysis with configuration:"
echo "Results directory: $RESULTS_DIR"
echo "Min depth: $MIN_DEPTH"
echo "Max depth: $MAX_DEPTH"
echo "Output file: $OUTPUT_FILE"
echo "---------------------------------------"

# Run the depth analysis script
python score_by_depth_plot.py \
  --results-dir "$RESULTS_DIR" \
  --min-depth "$MIN_DEPTH" \
  --max-depth "$MAX_DEPTH" \
  --output "$OUTPUT_FILE"

# Check if the script completed successfully
if [ $? -eq 0 ]; then
  echo "Depth score analysis completed successfully"
  echo "Plot saved to: $OUTPUT_FILE"
else
  echo "Depth score analysis failed with error code $?"
fi 