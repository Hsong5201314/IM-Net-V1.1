#!/bin/bash

echo ""
echo "================================================================="
echo "🚀 Starting RQ1 Experiment 2/2: Yelp (LightGCN + IM-Net)"
echo "================================================================="

python main_gpu.py \
    --dataset yelp \
    --model_name LightGCN \
    --mode meta \
    --data_path ./data/yelp_processed_for_meta \
    --epochs 400

echo ""
echo "================================================================="
echo "🎉 Yelp (LightGCN + IM-Net) Experiment completed successfully!"
echo "================================================================="