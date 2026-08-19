#!/bin/bash
set -e
source .venv/bin/activate
export MISSING=$(cat << 'INNER_EOF'
L21_V013
L21_V014
L21_V016
L21_V019
L21_V021
L21_V024
L21_V025
L21_V028
L21_V030
L22_V002
L22_V003
L22_V005
L22_V007
L22_V011
L22_V019
L22_V022
L22_V023
L22_V024
L22_V025
L22_V026
L22_V027
L22_V028
L22_V029
L22_V031
L25_V004
L25_V012
L25_V015
L25_V017
L25_V020
L25_V029
L25_V035
L25_V038
L25_V042
L25_V044
L25_V047
L25_V059
L25_V062
L25_V064
L25_V073
L25_V082
INNER_EOF
)

echo "Cleaning up corrupted local files..."
for v in $MISSING; do
    rm -rf export_for_fusion/l21-l25-prod-v1/candidates/$v
    rm -rf export_for_fusion/l21-l25-prod-v1/vectors/$v
done

echo "Switching to degeabeo to download properly..."
modal profile activate degeabeo
for v in $MISSING; do
    echo "Downloading $v"
    modal volume get --force aic-frame-vol /runs/l21-l25-prod-v1/candidates/$v/ export_for_fusion/l21-l25-prod-v1/candidates/
    modal volume get --force aic-frame-vol /runs/l21-l25-prod-v1/vectors/$v/ export_for_fusion/l21-l25-prod-v1/vectors/
done

echo "Switching to overfiters to upload..."
modal profile activate overfiters
for v in $MISSING; do
    echo "Uploading $v"
    modal volume put --force aic-frame-vol export_for_fusion/l21-l25-prod-v1/candidates/$v /runs/l21-l25-prod-v1/candidates/
    modal volume put --force aic-frame-vol export_for_fusion/l21-l25-prod-v1/vectors/$v /runs/l21-l25-prod-v1/vectors/
done

echo "Starting app..."
python scripts/frame_extracting/resume_production.py --run-id l21-l25-prod-v1

echo "Starting billing monitor..."
python billing_monitor.py
