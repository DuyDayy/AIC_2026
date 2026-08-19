#!/bin/bash
set -e
source .venv/bin/activate
modal profile activate overfiters
modal volume create aic-frame-vol || true
modal volume create aic-data-vol || true
echo "Uploading checkpoints..."
modal volume put -f aic-frame-vol export_for_fusion/l21-l25-prod-v1 /runs/
echo "Uploading pilot boundaries..."
modal volume put -f aic-data-vol pilot_boundaries /
echo "Uploading shot boundaries..."
modal volume put -f aic-data-vol shot_boundaries /
echo "Uploading videos (this will take a while)..."
modal volume put -f aic-data-vol data/video /
echo "Deploying app..."
cd scripts/frame_extracting
modal deploy modal_app.py
cd ../..
echo "All done!"
