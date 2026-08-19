#!/bin/bash
echo "Waiting for sync_from_modal.py to finish..."
while pgrep -f "sync_from_modal.py" > /dev/null; do
    sleep 5
done
echo "Sync finished! Starting upload to overfiters..."
./upload_overfiters.sh

echo "Starting pipeline resume..."
source .venv/bin/activate
modal profile activate overfiters
python scripts/frame_extracting/resume_production.py --run-id l21-l25-prod-v1

echo "Starting billing monitor..."
python billing_monitor.py
