#!/bin/bash

# ==============================================================================
# Script for RQ3: Training Efficiency Analysis
# This script runs the efficiency comparison (e.g., time cost, convergence speed)
# between IM-Net and baseline methods.
# ==============================================================================

set -e

echo "================================================================="
echo "⏱️  Starting RQ3: Efficiency Analysis"
echo "================================================================="
echo "[INFO] Running efficiency tests via run_rq3_efficiency.py"
echo "================================================================="

python run_rq3_efficiency.py

echo ""
echo "================================================================="
echo "🎉 RQ3 Efficiency Analysis completed successfully!"
echo "[INFO] Please check the terminal output or the generated figures/logs."
echo "================================================================="
