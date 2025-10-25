# MM4 Workhorse Project Guide

## Project Overview

**Goal:** Use Mac Mini M4 as a computational accelerator for Jupyter notebooks while continuing development work on iMac.

**Setup:**
- **iMac (2019):** Development machine, 1.5TB storage, houses all data
- **Mac Mini M4:** Compute accelerator, connected via gigabit ethernet
- **Connection:** Same desk, ethernet cable, ~125 MB/s transfer speeds

**Key Principle:** iMac is the source of truth for code and data. Mac Mini is ephemeral compute.

## Architecture
```
iMac (development + storage)          Mac Mini M4 (execution)
├─ Develop notebooks                  ├─ Docker daemon
├─ Store data (1.5TB)                 ├─ Pull/run images
├─ Build Docker images                ├─ Mount iMac volumes
└─ Trigger remote execution ────────> └─ Write results back to iMac
         (gigabit ethernet)
```

## Workflow Pattern

### Standard Workflow (Non-MLX)
1. Work on numbered notebooks (e.g., `23_train_model.ipynb`)
2. When ready to run expensive computation, trigger remote execution
3. Continue working on other notebooks (24, 25, etc.) while Mini runs
4. Results appear in timestamped folders on iMac
5. Load results in subsequent notebooks for analysis

### Key Design Decisions

**Notebooks are standalone:**
- Each numbered notebook does one specific thing
- No conflicts - you work on different notebooks while one runs remotely
- Results saved to files, never stored in notebook cell outputs

**Immutable results:**
- Each run gets timestamped folder: `results/23_train_model_20250423_143022/`
- Never overwrite previous runs
- Complete executed notebook saved for reproducibility

**Docker layer caching:**
- Heavy dependencies (pandas, numpy, sklearn) built once, cached
- Only code changes rebuild (fast - seconds not minutes)
- First build: ~10 minutes
- Subsequent builds: ~30 seconds

## Project Structure
```
MM4_workhorse_project/
├── notebooks/
│   ├── 00_test_remote.ipynb
│   ├── 01_data_cleaning.ipynb
│   ├── 23_train_model.ipynb
│   └── ...
├── results/
│   └── 23_train_model_20250423_143022/
│       ├── executed.ipynb          # Snapshot of what ran
│       ├── model.pkl               # Computed outputs
│       ├── metrics.json
│       └── figures/
├── scripts/
│   └── run-remote.sh               # Main execution script
├── data/                           # Input data (optional)
├── Dockerfile                      # Container definition
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

## Usage

### Basic Execution
```bash
# Run a notebook on Mac Mini
./scripts/run-remote.sh notebooks/23_train_model.ipynb

# Results appear in: results/23_train_model_TIMESTAMP/
```

### Notebook Pattern

**Every notebook that produces results should:**
```python
from pathlib import Path
import pickle
import json

# Use /output when running in Docker
OUTPUT_DIR = Path("/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Do expensive computation
results = expensive_function()

# Save results to files (NOT notebook cell outputs)
with open(OUTPUT_DIR / "results.pkl", "wb") as f:
    pickle.dump(results, f)

# Save metadata
metadata = {
    "timestamp": datetime.now().isoformat(),
    "parameters": {"learning_rate": 0.001},
    "metrics": {"accuracy": 0.95}
}
with open(OUTPUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Results saved to {OUTPUT_DIR}")
```

**Loading results in subsequent notebooks:**
```python
from pathlib import Path
import pickle

# Load from most recent run (or specify exact run)
results_dir = Path("../results")
latest_run = sorted(results_dir.glob("23_train_model_*"))[-1]

with open(latest_run / "results.pkl", "rb") as f:
    results = pickle.load(f)

# Continue analysis with loaded results
plot_results(results)
```

## Docker Details

### Dockerfile Structure
```dockerfile
FROM python:3.11-slim

WORKDIR /workspace

# Layer 1: Heavy dependencies (built once, cached forever)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Layer 2: Your code (rebuilt when code changes - fast)
COPY notebooks/ ./notebooks/

CMD ["bash"]
```

**Why this matters:**
- Layer 1 is ~2-3GB, takes minutes to build, but only happens once
- Layer 2 is ~MBs, rebuilds in seconds when you change code
- Docker only transfers changed layers over network

### How run-remote.sh Works

1. **Build image** on iMac (uses layer caching)
2. **Transfer** to Mac Mini (only changed layers sent)
3. **Execute** notebook in container on Mini
4. **Mount** iMac's results folder so output writes directly back
5. **Done** - results appear on iMac automatically

## ML/Compute Options (Without Dedicated GPU)

### Recommended: PyTorch with MPS Backend

**MPS (Metal Performance Shaders)** uses Apple's GPU + Neural Engine without needing MLX:
```python
import torch

# Automatically uses Apple Silicon acceleration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = MyModel().to(device)
data = data.to(device)
output = model(data)
```

**Why this is best:**
- Standard PyTorch (ecosystem you know)
- Uses Apple's GPU/Neural Engine automatically
- No code rewriting needed
- Works in Docker on M4

### Other Good Options

**Traditional ML (very fast on CPU):**
- scikit-learn: `RandomForestClassifier(n_jobs=-1)`
- XGBoost/LightGBM: Often faster on CPU than GPU for tabular data
- Already optimized, no special setup

**Fast data processing:**
- Polars: Much faster than pandas for large datasets
- DuckDB: Fast SQL analytics on CSVs/Parquet

**CPU-parallel tasks:**
- NumPy/SciPy: Heavily optimized
- Numba: JIT compiler for numerical Python
- Perfect for simulations, Monte Carlo, etc.

### About MLX

**MLX is Apple's ML framework** (like PyTorch but Apple-only):
- Separate framework, not a translation layer
- Would require rewriting existing PyTorch/TensorFlow code
- Optimized for Apple Silicon, sometimes faster than PyTorch
- Smaller ecosystem, fewer pre-trained models

**When to use MLX:**
- Starting new projects and want maximum M4 performance
- Running LLMs locally (via mlx-lm)
- Specific Apple Silicon optimization needed

**When NOT to use MLX:**
- You have existing PyTorch/TensorFlow code (stick with it)
- Need broad model zoo / pre-trained models
- Collaborating with others (smaller ecosystem)

**For this project:** Start with PyTorch + MPS. Consider MLX later for specific use cases.

## Phased Approach

### Phase 1: Basic Workflow (Current)
- ✓ Docker setup with standard Python stack
- ✓ Remote execution of notebooks
- ✓ Results saved to timestamped folders
- ✓ No MLX, just pandas/numpy/sklearn

### Phase 2: Add ML Acceleration (When Needed)
- Ad