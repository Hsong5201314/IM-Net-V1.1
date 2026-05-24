#!/bin/bash

# ==============================================================================
# Script for RQ2: Ablation Study
# This script reproduces the ablation study results (e.g., Table 2).
# It will automatically evaluate various degraded versions of IM-Net.
# ==============================================================================

set -e

echo "================================================================="
echo "🔬 Starting RQ2: Ablation Study for IM-Net"
echo "================================================================="
echo "[INFO] Running ablation configurations via run_table2_ablation.py"
echo "================================================================="

python run_table2_ablation.py

echo ""
echo "================================================================="
echo "🎉 RQ2 Ablation Study completed successfully! Check the output logs."
echo "================================================================="
