#!/bin/bash
#SBATCH --partition=serial_requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0-1:00
#SBATCH --mem=100G
#SBATCH --job-name=gen_flowchart
#SBATCH --output=/n/home04/zechenzhang/missing_token_prediction/logs/outs/%x_%j.out
#SBATCH --error=/n/home04/zechenzhang/missing_token_prediction/logs/errs/%x_%j.err

# Load modules and activate environment (Adjust if needed)
module load python
mamba activate ML_2024 # Or your specific conda/mamba environment

# Create log directories if they don't exist
mkdir -p /n/home04/zechenzhang/missing_token_prediction/logs/outs
mkdir -p /n/home04/zechenzhang/missing_token_prediction/logs/errs

# Change to the project directory (Important for relative paths)
cd /n/home04/zechenzhang/missing_token_prediction || exit 1

# --- Configuration --- #
# Default config file (can be overridden)
CONFIG_FILE="config.json"

# Variables to hold command line args
RESULTS_FILE_ARG=""
MAX_DEPTH_ARG=""
SAMPLES_ARG=""
OUTPUT_NAME_ARG=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --results-file)
      RESULTS_FILE_ARG="$2"
      shift 2
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --max-depth)
      MAX_DEPTH_ARG="$2"
      shift 2
      ;;
    --samples)
      SAMPLES_ARG="$2"
      shift 2
      ;;
    --output)
      OUTPUT_NAME_ARG="$2"
      shift 2
      ;;
    *)
      echo "Usage: sbatch $0 [--results-file <path>] [--config <path>] [--max-depth <int>] [--samples <int>] [--output <name>]"
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# --- Find Results File --- #

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Config file '$CONFIG_FILE' not found."
  exit 1
fi

# Extract results_dir from config.json
RESULTS_DIR=$(python -c "import json; print(json.load(open('$CONFIG_FILE'))['results_dir'])")

# Check if RESULTS_DIR exists
if [ ! -d "$RESULTS_DIR" ]; then
  echo "Error: Results directory '$RESULTS_DIR' (from config) not found."
  exit 1
fi

FINAL_RESULTS_FILE=""

# Priority 1: Use specific results file if provided
if [ -n "$RESULTS_FILE_ARG" ]; then
  if [ ! -f "$RESULTS_FILE_ARG" ]; then
    echo "Error: Specified results file '$RESULTS_FILE_ARG' not found."
    exit 1
  fi
  FINAL_RESULTS_FILE="$RESULTS_FILE_ARG"
# Priority 2: Find file based on depth and samples if provided
elif [ -n "$MAX_DEPTH_ARG" ] && [ -n "$SAMPLES_ARG" ]; then
  EXPERIMENT_NAME="experiment_depth${MAX_DEPTH_ARG}_samples${SAMPLES_ARG}"
  # Find the most recent matching file (adjust pattern if needed)
  FOUND_FILE=$(find "$RESULTS_DIR" -name "${EXPERIMENT_NAME}*.json" -type f -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
  if [ -z "$FOUND_FILE" ]; then
    echo "Error: No result files found matching pattern '${EXPERIMENT_NAME}*.json' in '$RESULTS_DIR'."
    exit 1
  fi
  FINAL_RESULTS_FILE="$FOUND_FILE"
# Priority 3: Default to the most recent result file
else
  FOUND_FILE=$(find "$RESULTS_DIR" -name "*.json" -type f -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
  if [ -z "$FOUND_FILE" ]; then
    echo "Error: No result files found in '$RESULTS_DIR'."
    exit 1
  fi
  FINAL_RESULTS_FILE="$FOUND_FILE"
fi

# --- Determine Output Name --- #

# Use provided output name or generate one from results file
if [ -n "$OUTPUT_NAME_ARG" ]; then
  FINAL_OUTPUT_NAME="$OUTPUT_NAME_ARG"
else
  BASENAME=$(basename "$FINAL_RESULTS_FILE" .json)
  FINAL_OUTPUT_NAME="${BASENAME}_flowchart.html"
fi

# Determine the full output path (place it in the same dir as the results json)
OUTPUT_DIR=$(dirname "$FINAL_RESULTS_FILE")
FULL_OUTPUT_PATH="$OUTPUT_DIR/$FINAL_OUTPUT_NAME"


# --- Execute Visualization --- #

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running flowchart generation with configuration:"
echo "Input Results File: $FINAL_RESULTS_FILE"
echo "Output HTML File: $FULL_OUTPUT_PATH"
echo "Config File: $CONFIG_FILE"
echo "---------------------------------------"

# Construct the python command
PYTHON_CMD="python generate_flowchart.py \"$FINAL_RESULTS_FILE\" -o \"$FULL_OUTPUT_PATH\""

# Execute the command
echo "Running: $PYTHON_CMD"
eval $PYTHON_CMD

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "Flowchart generation completed successfully."
  echo "HTML report saved to: $FULL_OUTPUT_PATH"
else
  echo "Flowchart generation failed with error code $EXIT_CODE."
fi

exit $EXIT_CODE 