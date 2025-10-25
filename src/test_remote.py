#!/usr/bin/env python3
"""
Test script for remote execution.
Performs the same trivial computation as 00_test_remote.ipynb
"""
import pandas as pd
import time
from pathlib import Path

print("Starting computation on remote...")

# Trivial 'expensive' computation
results = []
for i in range(10):
    time.sleep(1)  # Simulate work
    results.append({"iteration": i, "value": i ** 2})
    print(f"Iteration {i} complete")

df = pd.DataFrame(results)

# Save to results
OUTPUT_DIR = Path("/output")
OUTPUT_DIR.mkdir(exist_ok=True)

df.to_csv(OUTPUT_DIR / "results.csv", index=False)
print(f"✓ Saved {len(df)} rows to results.csv")
