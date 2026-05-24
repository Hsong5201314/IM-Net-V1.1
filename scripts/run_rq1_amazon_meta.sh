#!/bin/bash

# ==============================================================================
# Script for RQ1: Main Performance Evaluation of IM-Net
# Backbone: LightGCN
# Datasets: Amazon Books & Yelp
# ==============================================================================

set -e

echo "================================================================="
echo "🚀 Starting RQ1 Experiment 1/2: Amazon Books (LightGCN + IM-Net)"
echo "================================================================="

python main_gpu.py \
    --dataset amazon \
    --model_name LightGCN \
    --mode meta \
    --data_path ./data/amazon_books_processedDataV3 \
    --epochs 400 \
    --meta_loss_weight 0.001 \
    --meta_update_freq 20 \
    --cl_lambda 0.2 \
    --neg_sample_ratio 8 \

echo ""
echo "================================================================="
echo "🎉 Amazon Books (LightGCN + IM-Net) Experiment completed successfully!"
echo "================================================================="
