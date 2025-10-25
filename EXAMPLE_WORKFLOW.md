# Example Workflow: Training a Model Remotely

## Scenario
Development occurs on `23_train_model.ipynb` on the iMac. The notebook trains a machine learning model - a task that will take 30 minutes and saturate the host CPU. The goal is to run it on the Mac Mini M4 while continuing work on other tasks.

## Step 1: Develop Your Notebook on iMac

```python
# File: notebooks/23_train_model.ipynb

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import pickle
import json
from datetime import datetime

# Load data (assuming it's in ../data/)
df = pd.read_csv("../data/training_data.csv")
X = df.drop('target', axis=1)
y = df['target']

# Train model (expensive operation)
print("Training model...")
model = RandomForestClassifier(n_estimators=500, n_jobs=-1)
model.fit(X, y)
print(f"Training complete. Score: {model.score(X, y):.4f}")

# Save results to /output (container mount point)
OUTPUT_DIR = Path("/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Save trained model
with open(OUTPUT_DIR / "model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save metrics
metrics = {
    "timestamp": datetime.now().isoformat(),
    "accuracy": float(model.score(X, y)),
    "n_estimators": 500,
    "feature_importance": {
        col: float(imp)
        for col, imp in zip(X.columns, model.feature_importances_)
    }
}

with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✓ Model and metrics saved to {OUTPUT_DIR}")
```

## Step 2: Trigger Remote Execution

**From your iMac terminal:**

```bash
# Option A: Use standard (Intel) build (runs via Rosetta on M4)
./scripts/run-remote.sh notebooks/23_train_model.ipynb

# Option B: Use ARM64 build (native M4 performance)
./scripts/run-remote-buildx.sh notebooks/23_train_model.ipynb
```

## Step 3: What Happens During Execution

```
=== Running 23_train_model on Mac Mini ===

Building Docker image...
[Docker builds image with your notebook + dependencies]

Transferring to Mini...
[Image sent over ethernet - ~5 seconds]

Starting execution...
[Notebook runs on Mac Mini M4]
Training model...
Training complete. Score: 0.9823
✓ Model and metrics saved to /output

Copying results back...
[Results transferred back to iMac]

✓ Done! Results in: results/23_train_model_20251023_143022/
-rw-r--r--  executed.ipynb   12K
-rw-r--r--  model.pkl        2.3M
-rw-r--r--  metrics.json     1.2K
```

**While this runs (30 minutes), you can:**
- Continue editing `24_evaluate_model.ipynb` on iMac
- Run other quick notebooks locally
- Get coffee, check email, etc.

## Step 4: Load Results in Next Notebook

```python
# File: notebooks/24_evaluate_model.ipynb

from pathlib import Path
import pickle
import json

# Find most recent run
results_dir = Path("../results")
latest_run = sorted(results_dir.glob("23_train_model_*"))[-1]

print(f"Loading results from: {latest_run.name}")

# Load trained model
with open(latest_run / "model.pkl", "rb") as f:
    model = pickle.load(f)

# Load metrics
with open(latest_run / "metrics.json", "r") as f:
    metrics = json.load(f)

print(f"Model accuracy: {metrics['accuracy']:.4f}")
print(f"Trained at: {metrics['timestamp']}")

# Continue with evaluation...
test_data = pd.read_csv("../data/test_data.csv")
predictions = model.predict(test_data)
```

## Key Points

1. **No changes needed to notebook code** - Write normal Python, just save outputs to `/output`

2. **Results are immutable** - Each run gets timestamped folder, never overwritten

3. **Executed notebook included** - `executed.ipynb` shows exactly what ran with all cell outputs

4. **Work continues** - iMac is free while Mini runs expensive computation

5. **Reproducible** - Complete snapshot of code + results for each run

## Common Patterns

**Long-running simulation:**
```bash
./scripts/run-remote.sh notebooks/15_monte_carlo_simulation.ipynb
# Go work on notebooks 16, 17, 18 while it runs
```

**Data processing:**
```bash
./scripts/run-remote.sh notebooks/02_clean_raw_data.ipynb
# Processes large CSV, saves cleaned version to results/
```

**Hyperparameter sweep:**
```bash
# Modify notebook with different params, run multiple times
./scripts/run-remote.sh notebooks/23_train_model.ipynb  # params set A
./scripts/run-remote.sh notebooks/23_train_model.ipynb  # params set B
# Each gets separate timestamped results folder
```

## Best Practices

**DO:**
- Save all outputs to `/output` directory
- Use timestamped filenames for clarity
- Save metadata (params, metrics) as JSON
- Test notebook locally first with small dataset

**DON'T:**
- Store results in cell outputs (won't survive nbconvert)
- Use hardcoded absolute paths (use relative paths like `../data/`)
- Expect interactive input (notebooks run non-interactively)
- Modify files outside the container (they won't persist)
