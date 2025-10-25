# MM4 Workhorse Project Guide

A practical guide to using a Mac Mini M4 as a computational accelerator for Jupyter notebooks while continuing development work on an iMac.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Notebook Conventions](#notebook-conventions)
- [ML/Compute Options](#mlcompute-options-without-dedicated-gpu)
- [Docker Details](#docker-details)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)
- [Getting Help](#getting-help)
- [Design Patterns](#design-patterns-used)

## Overview

**Goal:** Use Mac Mini M4 as a computational accelerator for Jupyter notebooks while continuing development work on iMac.

**Setup:**
- **iMac (2019):** Development machine, 1.5TB storage, houses all data
- **Mac Mini M4:** Compute accelerator, connected via gigabit ethernet
- **Connection:** Same desk, ethernet cable, ~125 MB/s transfer speeds

**Key Principle:** iMac is the source of truth for code and data. Mac Mini is ephemeral compute.

## Prerequisites

Before starting, you'll need:

### Hardware
- Two Macs on the same local network (or one Mac + remote server)
- Recommended: Ethernet connection for faster transfers (WiFi works but slower)

### Software
- **macOS** on both machines
- **Docker Desktop** installed on Mac Mini ([download here](https://www.docker.com/products/docker-desktop))
- **SSH access** between machines (we'll verify during setup)
- **Git** (optional, but recommended for version control)

### Skills
- Basic terminal/command line familiarity
- Python and Jupyter notebook experience
- Understanding of what Docker containers are (don't need to be an expert)
- Ability to edit text files

### What You Don't Need
- Deep Docker expertise (you'll learn as you go)
- DevOps background
- Cloud infrastructure knowledge
- Advanced networking skills

**The focus here is on getting it working, not becoming a Docker expert.** With AI assistance (like Claude, ChatGPT, or Claude Code), you can troubleshoot issues as they arise.

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

### Standard Workflow
1. Work on numbered notebooks (e.g., `23_train_xgboost_model.ipynb`)
2. When ready to run expensive computation, trigger remote execution
3. Continue working on other notebooks (24, 25, etc.) while Mini runs
4. Results appear in named folders on iMac
5. Load results in subsequent notebooks for analysis

### Key Design Decisions

**Numbered notebooks with descriptive titles:**
- Format: `##_descriptive_title.ipynb` (e.g., `23_train_xgboost_model.ipynb`)
- **Why numbers:** Enables easy TAB completion in terminal/CLI tools
- **Why descriptive:** Makes purpose clear at a glance
- **CLI benefit:** Can reference with just the number (Claude Code, scripts, etc.)
  - `./run-remote.sh 23` auto-completes to full notebook name
  - Easy to remember and reference in conversation with AI tools
  - Chronological organization shows project evolution

**Notebooks are standalone:**
- Each numbered notebook does one specific thing
- No conflicts - you work on different notebooks while one runs remotely
- Results saved to files, never stored in notebook cell outputs

**Named results folders:**
- Each run creates a descriptive folder based on notebook name and parameters
- Can run same notebook multiple times with different configurations
- Easy to find specific results later

**Docker layer caching:**
- Heavy dependencies (pandas, numpy, sklearn) built once, cached
- Only code changes rebuild (fast - seconds not minutes)
- First build: ~10 minutes
- Subsequent builds: ~30 seconds

## Project Structure
```
MM4_workhorse_project/
├── notebooks/                    # Jupyter notebooks
│   ├── 00_test_remote.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 10_baseline_model.ipynb
│   ├── 23_train_xgboost_model.ipynb
│   ├── 24_hyperparameter_tuning.ipynb
│   └── ...
├── src/                          # Standalone Python scripts
│   └── test_remote.py
├── results/
│   ├── test_remote/              # Results from notebook 00
│   │   ├── executed.ipynb
│   │   └── results.csv
│   ├── test_python/              # Results from src/test_remote.py
│   │   ├── execution.log
│   │   └── results.csv
│   ├── xgboost_baseline/         # Named run from notebook 23
│   │   ├── executed.ipynb
│   │   ├── model.pkl
│   │   └── metadata.json
│   ├── xgboost_lr001_n100/       # Different parameters, same notebook
│   │   ├── executed.ipynb
│   │   ├── model.pkl
│   │   └── metadata.json
│   └── hyperparameter_search/    # Results from notebook 24
│       └── ...
├── scripts/
│   ├── run-remote.sh             # Run Jupyter notebooks remotely
│   └── run-remote-python.sh      # Run Python scripts remotely
├── data/                         # Input data (optional)
├── Dockerfile                    # Container definition
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md
```

### Notebook Numbering Convention

**Recommended numbering scheme:**
- `00-09`: Setup, testing, exploration
- `10-19`: Data cleaning and preprocessing
- `20-29`: Feature engineering
- `30-39`: Baseline models
- `40-49`: Model development and training
- `50-59`: Model evaluation and comparison
- `60-69`: Hyperparameter tuning
- `70-79`: Final models and production prep
- `80-89`: Analysis and visualization
- `90-99`: Reports and documentation

**Benefits:**
- Leave gaps (0, 5, 10, 15...) for inserting related notebooks later
- Grouping by tens shows workflow phases
- Easy to find notebooks: "Where's the feature engineering?" → "20s"
- TAB completion: `./run-remote.sh 23<TAB>` → `23_train_xgboost_model.ipynb`

## Getting Started

### 1. Verify SSH Access
```bash
# From iMac, test connection to Mac Mini
ssh macmini hostname

# If it asks for password every time, set up SSH keys:
ssh-keygen -t ed25519  # Press enter for defaults
ssh-copy-id macmini    # Enter password one last time
```

### 2. Install Docker on Mac Mini
```bash
# Check if Docker is already installed
ssh macmini 'docker --version'

# If not installed:
# 1. Download Docker Desktop from https://www.docker.com/products/docker-desktop
# 2. Install on Mac Mini
# 3. Start Docker Desktop
# 4. Verify: ssh macmini 'docker --version'
```

### 3. Set Up File Sharing (if needed)

If Mac Mini needs to write to iMac's folders:

**On iMac:**
1. System Settings → Sharing → File Sharing
2. Enable file sharing
3. Add your project folder to shared folders
4. Note your iMac's hostname (e.g., `imac.local`)

**Test from Mini:**
```bash
ssh macmini 'ls ~/MM4_workhorse_project/results'
```

### 4. Create Your Environment
```bash
# Export your current Python environment
conda list --export > requirements.txt
# or
pip freeze > requirements.txt
```

### 5. Test the Workflow
```bash
# Run the test notebook
./scripts/run-remote.sh notebooks/00_test_remote.ipynb test_remote

# Check results
ls -lh results/test_remote/
cat results/test_remote/results.csv
```

## Usage

> **Note on Examples:** This guide uses **toy computations** (simple loops, basic calculations) to demonstrate the workflow. The focus is on learning the remote execution pattern, not on complex ML models. Once you understand the workflow, you can apply it to your own compute-intensive tasks (training models, data processing, simulations, etc.).

### Tested Working Examples

The following workflows have been tested and verified working:

#### 1. Jupyter Notebooks

**Run a notebook:**
```bash
./scripts/run-remote.sh notebooks/00_test_remote.ipynb test_remote
```

**What happens:**
- Builds Docker image (or uses cached layers)
- Transfers to Mac Mini
- Executes notebook remotely
- Copies results back to `results/test_remote/`

**Results folder contains:**
```
results/test_remote/
├── executed.ipynb    # Notebook with all outputs
└── results.csv       # Data saved by notebook
```

#### 2. Standalone Python Scripts

**Run a Python script:**
```bash
./scripts/run-remote-python.sh src/test_remote.py test_python
```

**What happens:**
- Same Docker build/transfer process
- Executes Python script remotely
- Captures all output to execution log
- Copies results back to `results/test_python/`

**Results folder contains:**
```
results/test_python/
├── execution.log     # All stdout/stderr from script
└── results.csv       # Data saved by script
```

**When to use scripts vs notebooks:**
- **Notebooks**: Exploratory work, prototyping, results you want to see inline
- **Scripts**: Production pipelines, scheduled jobs, cleaner for version control

### Basic Execution Syntax

```bash
# Notebooks - full path required
./scripts/run-remote.sh notebooks/23_train_xgboost_model.ipynb xgboost_baseline

# Python scripts - full path required
./scripts/run-remote-python.sh src/train_model.py xgboost_baseline

# With timestamp (for multiple runs of same config)
./scripts/run-remote.sh notebooks/23_train_xgboost_model.ipynb xgboost_baseline --timestamp
# Results in: results/xgboost_baseline_20251024_143022/

# CLI tools can use TAB completion
./scripts/run-remote.sh notebooks/23<TAB>  # auto-completes to full name
```

### Notebook Pattern

**Every notebook that produces results should:**
```python
from pathlib import Path
import pickle
import json
from datetime import datetime

# Use /output when running in Docker
OUTPUT_DIR = Path("/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Do expensive computation
results = expensive_function()

# Save results to files (NOT notebook cell outputs)
with open(OUTPUT_DIR / "results.pkl", "wb") as f:
    pickle.dump(results, f)

# Save metadata with timestamp for audit trail
metadata = {
    "run_name": "xgboost_baseline",  # From command line arg
    "timestamp": datetime.now().isoformat(),
    "notebook": "23_train_xgboost_model.ipynb",
    "parameters": {"learning_rate": 0.001, "n_estimators": 100},
    "metrics": {"accuracy": 0.95, "f1": 0.93}
}
with open(OUTPUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Results saved to {OUTPUT_DIR}")
```

**Loading results in subsequent notebooks:**
```python
from pathlib import Path
import pickle

# Load from specific named run
results_dir = Path("../results/xgboost_baseline")

with open(results_dir / "results.pkl", "rb") as f:
    results = pickle.load(f)

# Or compare multiple runs
baseline = pickle.load(open("../results/xgboost_baseline/results.pkl", "rb"))
tuned = pickle.load(open("../results/xgboost_lr001_n100/results.pkl", "rb"))

# Continue analysis
compare_results(baseline, tuned)
```

### Python Script Pattern

**Every Python script that runs remotely should:**
```python
#!/usr/bin/env python3
"""
Description of what this script does.
"""
from pathlib import Path
import pandas as pd
import pickle
import json
from datetime import datetime

# Use /output when running in Docker
OUTPUT_DIR = Path("/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Do expensive computation
print("Starting computation...")
results = expensive_function()
print("Computation complete")

# Save results to files
with open(OUTPUT_DIR / "results.pkl", "wb") as f:
    pickle.dump(results, f)

# Save metadata
metadata = {
    "script": "train_model.py",
    "timestamp": datetime.now().isoformat(),
    "parameters": {"learning_rate": 0.001},
    "metrics": {"accuracy": 0.95}
}
with open(OUTPUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Results saved to {OUTPUT_DIR}")
```

**Key differences from notebooks:**
- No separate executed artifact (execution.log captures all output)
- All print statements go to execution.log for debugging
- More suitable for production pipelines and scheduled runs
- Cleaner version control (no cell outputs, no metadata)

## Notebook Conventions

### Naming Strategy

**For notebooks:**
- Format: `##_descriptive_name.ipynb`
- Number first (for TAB completion and ordering)
- Descriptive second (for clarity)
- Use underscores, not spaces
- Examples: `23_train_xgboost_model.ipynb`, `45_deep_learning_baseline.ipynb`

**For result folders:**
- Descriptive: `xgboost_baseline`, `random_forest_tuned`
- Include key parameters: `xgboost_lr001_depth5`, `rf_n500_depth10`
- Indicate iterations: `feature_eng_v2`, `model_retrain_march`

**Avoid:**
- Just timestamps: `20250423_143022` (what was this?)
- Generic names: `run1`, `test`, `final` (not descriptive)
- Too long: `xgboost_with_learning_rate_0.001_and_100_estimators` (unwieldy)

**Tip:** If you need to run the same configuration multiple times, append timestamp to the name:
```bash
./scripts/run-remote.sh notebooks/23_train_xgboost_model.ipynb xgboost_baseline_$(date +%Y%m%d)
```

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

## Security Considerations

**This setup is designed for trusted local networks.** If you're using this at home or in a lab with machines you control, the default configuration is fine.

### What's Safe in This Setup

✓ **Local network only** - No internet exposure
✓ **SSH with keys** - More secure than passwords
✓ **Docker isolation** - Notebooks run in containers
✓ **Read-only data mounts** - Prevents accidental data modification

### What to Consider

⚠️ **HTTP Docker registry** - Fine for local network, but unencrypted
⚠️ **Network file sharing** - Anyone on your network could potentially access shared folders
⚠️ **No authentication on Docker** - Mini trusts commands from iMac

### If You Need More Security

For untrusted networks or shared lab environments:
```bash
# 1. Use HTTPS for Docker registry
# Follow: https://docs.docker.com/registry/deploying/

# 2. Use VPN between machines
# Or SSH tunneling for all communication

# 3. Enable Docker TLS authentication
# Follow: https://docs.docker.com/engine/security/protect-access/

# 4. Encrypt file sharing
# Use encrypted volumes or SSH-based file transfers
```

**For most solo data science work on your own network, the default setup is appropriate.** Add security layers only if your threat model requires it.

## Phased Approach

### Phase 1: Basic Workflow (Current)
- ✓ Docker setup with standard Python stack
- ✓ Remote execution of notebooks
- ✓ Results saved to named folders
- ✓ No MLX, just pandas/numpy/sklearn

### Phase 2: Add ML Acceleration (When Needed)
- Add PyTorch with MPS support to Dockerfile
- Test GPU acceleration on M4
- Benchmark vs CPU-only approaches

### Phase 3: Production-Grade Additions (If Needed)
- Git commit tracking in metadata
- Dependency pinning (requirements.lock)
- Comprehensive logging
- Experiment tracking (MLflow)
- Data versioning (DVC)

**Current focus:** Get Phase 1 working solidly before adding complexity.

## Troubleshooting

### Common Issues

**Docker not found on Mini:**
```bash
ssh macmini 'docker --version'
# If fails, install Docker Desktop on Mini
# Download from: https://www.docker.com/products/docker-desktop
```

**SSH requires password every time:**
```bash
# Set up SSH keys (one-time setup)
ssh-keygen -t ed25519
ssh-copy-id macmini
# Test: ssh macmini hostname (should not ask for password)
```

**Results folder not accessible:**
```bash
# Verify Mini can access iMac's filesystem
ssh macmini 'ls ~/MM4_workhorse_project/results'

# If fails, check:
# 1. File sharing enabled on iMac (System Settings → Sharing)
# 2. Project folder is shared
# 3. Permissions allow read/write
```

**Image transfer very slow:**
```bash
# Check connection type
# Ethernet: ~30 seconds for 3GB image
# WiFi: ~3-5 minutes for 3GB image

# Verify ethernet connection:
# System Settings → Network → Ethernet (should show connected)

# First transfer is always slow (full image)
# Subsequent transfers should be fast (only changed layers)
```

**Notebook execution fails:**
```bash
# Check Docker logs on Mini
ssh macmini 'docker ps -a'  # Get container ID
ssh macmini 'docker logs CONTAINER_ID'

# Common issues:
# - Missing dependencies in requirements.txt
# - Path issues (use /output not relative paths)
# - Memory limits (Docker Desktop settings)
```

**Permission denied errors:**
```bash
# Make sure run-remote.sh is executable
chmod +x scripts/run-remote.sh

# Check volume mount permissions
ssh macmini 'ls -ld ~/MM4_workhorse_project/results'
```

## Getting Help

**The most important thing to understand is the goal, not every technical detail.**

This setup was created with help from AI assistants (Claude, specifically), and you can use AI assistance to troubleshoot any issues you encounter.

### How to Get Effective Help from AI Assistants

When you run into problems, describe:

1. **What you're trying to do** - "I want to run a notebook on my Mac Mini"
2. **What happened** - Copy/paste the error message
3. **What you expected** - "Results should appear in results/ folder"
4. **Your setup** - "iMac to Mac Mini, gigabit ethernet, Docker installed"

**Example prompt for Claude Code or ChatGPT:**
```
I'm following the MM4 Workhorse guide to run Jupyter notebooks on my Mac Mini.
When I run ./scripts/run-remote.sh notebooks/00_test.ipynb test, I get:

[paste error here]

My setup: iMac (2019) connected to Mac Mini M4 via ethernet. 
Docker is installed on both machines.
Goal: Execute the notebook on Mini and get results back on iMac.

What's wrong and how do I fix it?
```

### Why This Works

- You don't need to be a Docker expert
- You don't need to memorize all the commands
- You need to understand the **architecture** and **goal**
- AI can help with the implementation details

### Recommended Tools for Getting Help

- **Claude Code** - Can directly interact with your files and terminal
- **ChatGPT** - Good for explaining concepts and debugging
- **Claude.ai** - Good for understanding trade-offs and architecture
- **Stack Overflow** - Still useful for specific error messages

### What to Learn Deeply vs. What to Look Up

**Understand deeply:**
- The workflow (iMac builds → Mini executes → results return)
- Why Docker (reproducibility, isolation)
- Notebook conventions (numbered, descriptive, results to files)

**Look up as needed:**
- Specific Docker commands
- SSH troubleshooting
- Python packaging details
- File permission fixes

**This guide gives you enough context to ask good questions and understand the answers.** That's the goal.

## Design Patterns Used

- **Separation of Concerns:** Compute vs storage vs development
- **Immutability:** Results in named folders, never overwritten
- **DRY:** Docker layers cached, not rebuilding everything
- **Single Responsibility:** Each notebook does one thing
- **Explicit over Implicit:** Named folders, clear metadata
- **Convention over Configuration:** Numbered notebooks enable tooling

## Notes

This is a **solo data science development workflow**, not a production ML deployment system. It optimizes for:
- Fast iteration
- Hardware constraints (iMac storage, Mini compute)
- Reproducibility
- Simplicity
- CLI-friendly (numbered notebooks for TAB completion)

Add production features (CI/CD, monitoring, orchestration) only when actually needed.

---

## Contributing

Found an issue or have a suggestion? Please open an issue or pull request.

## License

MIT License - feel free to use and adapt for your own projects.