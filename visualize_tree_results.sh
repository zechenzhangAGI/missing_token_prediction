#!/bin/bash
#SBATCH --partition=serial_requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-1:00
#SBATCH --mem=16G
#SBATCH --job-name=tree_viz
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
RESULTS_FILE=""
MAX_DEPTH=""
SAMPLES="3"
OUTPUT_NAME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --results-file)
      RESULTS_FILE="$2"
      shift 2
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --max-depth)
      MAX_DEPTH="$2"
      shift 2
      ;;
    --samples)
      SAMPLES="$2"
      shift 2
      ;;
    --output)
      OUTPUT_NAME="$2"
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

# If specific result file is provided, use it
if [ -n "$RESULTS_FILE" ]; then
  # Check if the provided file exists
  if [ ! -f "$RESULTS_FILE" ]; then
    echo "Error: Results file '$RESULTS_FILE' not found."
    exit 1
  fi
# If MAX_DEPTH and SAMPLES are provided, find matching result file
elif [ -n "$MAX_DEPTH" ] && [ -n "$SAMPLES" ]; then
  # Define the experiment name pattern with MAX_DEPTH and SAMPLES
  EXPERIMENT_NAME="experiment_depth${MAX_DEPTH}_samples${SAMPLES}"
  
  # Find the most recent matching file
  RESULTS_FILE=$(find "$RESULTS_DIR" -name "${EXPERIMENT_NAME}.json" -type f -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
  
  if [ -z "$RESULTS_FILE" ]; then
    echo "Error: No result files found matching MAX_DEPTH=$MAX_DEPTH and SAMPLES=$SAMPLES in '$RESULTS_DIR'."
    exit 1
  fi
# Default: find the most recent result file
else
  # Find the most recent .json file in the results directory
  RESULTS_FILE=$(find "$RESULTS_DIR" -name "*.json" -type f -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
  
  if [ -z "$RESULTS_FILE" ]; then
    echo "Error: No result files found in '$RESULTS_DIR'."
    exit 1
  fi
fi

# Extract experiment details from results filename for naming the report
if [ -z "$OUTPUT_NAME" ]; then
  BASENAME=$(basename "$RESULTS_FILE" .json)
  OUTPUT_NAME="${BASENAME}_report.html"
fi

# Get the visualization directory that will be created by the Python script
VIZ_DIR=$(dirname "$RESULTS_FILE")/visualizations
mkdir -p "$VIZ_DIR"

echo "Using results file: $RESULTS_FILE"

# Print configuration for log
echo "Running tree visualization with configuration:"
echo "Results file: $RESULTS_FILE"
echo "Config file: $CONFIG_FILE"
echo "Output report: $OUTPUT_NAME"
echo "Expected location: $VIZ_DIR/$OUTPUT_NAME"
echo "---------------------------------------"

# Run the visualization - pass ONLY the filename for output, not the full path
# The TreeVisualizer class creates its own visualizations directory
python tree_visualizer.py "$RESULTS_FILE" --output "$OUTPUT_NAME"

# Check if the visualization completed successfully
if [ $? -eq 0 ]; then
  echo "Visualization completed successfully"
  echo "HTML report saved to: $VIZ_DIR/$OUTPUT_NAME"
  
  # List generated visualizations
  echo "---------------------------------------"
  echo "Generated visualizations:"
  find "$VIZ_DIR" -type f -name "*.png" -printf "- %f\n"
else
  echo "Visualization failed with error code $?"
fi 