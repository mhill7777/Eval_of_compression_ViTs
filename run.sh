#!/bin/bash

# Exit immediately if any command fails
set -e

echo "STARTING BENCHMARKING PIPELINE"

# Synchronize disks and drop OS memory caches (requires sudo)
# This clears out any leftover file system indexing from the training dataset
echo "Performing hard system cache flush"
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches

echo "Pausing for 5 seconds to let CPU/GPU power draw settle..."
sleep 5


echo "Running benchmark.py ..."
python benchmark.py

echo ""
echo ""
echo "EXPERIMENT COMPLETE! Check results/benchmark_comparison.csv"
