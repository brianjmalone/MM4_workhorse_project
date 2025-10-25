#!/bin/bash
# Run a Jupyter notebook remotely on Mac Mini M4
# Builds x86_64 image (Intel) which runs via Rosetta on M4

set -e  # Exit immediately if any command fails

# Parse arguments
NOTEBOOK=$1      # First argument: path to notebook (e.g., notebooks/23_train_model.ipynb)
OUTPUT_NAME=$2   # Second argument: descriptive name for results (e.g., xgboost_baseline)
ADD_TIMESTAMP=$3 # Optional third argument: --timestamp to append timestamp

# Validate input
if [ -z "$NOTEBOOK" ] || [ -z "$OUTPUT_NAME" ]; then
    echo "Usage: ./scripts/run-remote.sh NOTEBOOK OUTPUT_NAME [--timestamp]"
    echo ""
    echo "Examples:"
    echo "  ./scripts/run-remote.sh notebooks/23_train_model.ipynb xgboost_baseline"
    echo "  ./scripts/run-remote.sh notebooks/23_train_model.ipynb xgboost_lr001 --timestamp"
    echo ""
    exit 1
fi

# Create run identifier (with optional timestamp)
if [ "$ADD_TIMESTAMP" == "--timestamp" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_ID="${OUTPUT_NAME}_${TIMESTAMP}"
else
    RUN_ID="$OUTPUT_NAME"
fi

echo "=== Running notebook on Mac Mini ==="
echo "Notebook: $NOTEBOOK"
echo "Results: results/$RUN_ID/"

# Build Docker image on iMac
# - Uses standard `docker build` which creates x86_64 (Intel) images on Intel Macs
# - Tag: analysis:latest (will be replaced on each run)
# - Uses layer caching: only rebuilds changed layers (fast after first build)
echo "Building Docker image..."
docker build -t analysis:latest .

# Transfer image to Mac Mini via SSH
# - `docker save`: Exports image as tar archive to stdout
# - `|`: Pipe (stream) output directly to SSH command
# - `ssh mac-mini-ethernet`: Connect to Mini (uses SSH host alias from ~/.ssh/config)
# - Full path to docker needed because SSH non-interactive shells don't load PATH
echo "Transferring to Mini..."
docker save analysis:latest | ssh mac-mini-ethernet "/Applications/Docker.app/Contents/Resources/bin/docker load"

# Create timestamped results directories on both machines
# - iMac: Where final results will live
# - Mini: Temporary location for container to write output
mkdir -p results/$RUN_ID
ssh mac-mini-ethernet "mkdir -p ~/MM4_workhorse_project/results/$RUN_ID"

# Execute notebook inside container on Mac Mini
# - `docker run`: Start new container from analysis:latest image
# - `--rm`: Automatically delete container when done (cleanup)
# - `-v`: Mount Mini's results folder into container at /output (volume mount)
# - `jupyter nbconvert --execute`: Run notebook, capture outputs
# - `--to notebook`: Save as .ipynb with cell outputs included
# - `--output /output/executed.ipynb`: Write to mounted volume (appears on Mini's filesystem)
echo "Starting execution..."
ssh mac-mini-ethernet "/Applications/Docker.app/Contents/Resources/bin/docker run --rm \
  -v ~/MM4_workhorse_project/results/$RUN_ID:/output \
  analysis:latest \
  jupyter nbconvert --execute --to notebook \
  --output /output/executed.ipynb \
  $NOTEBOOK"

# Copy all results from Mac Mini back to iMac
# - `scp -r`: Secure copy, recursive (gets all files in folder)
# - Source: Mini's results folder for this run
# - Destination: iMac's results folder (same RUN_ID)
echo "Copying results back..."
scp -r mac-mini-ethernet:~/MM4_workhorse_project/results/$RUN_ID/* results/$RUN_ID/

# Show completion message and list result files
echo ""
echo "✓ Done! Results in: results/$RUN_ID/"
ls -lh results/$RUN_ID/
