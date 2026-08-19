import os
import subprocess

with open("remote_vectors.txt") as f:
    remote_paths = [line.strip() for line in f if line.strip()]
remote_videos = [p.split("/")[-1] for p in remote_paths]

with open("local_vectors.txt") as f:
    local_videos = [line.strip() for line in f if line.strip()]

missing = [v for v in remote_videos if v not in local_videos]

print(f"Found {len(missing)} missing videos to download.")

for i, v in enumerate(missing, 1):
    print(f"Downloading {i}/{len(missing)}: {v}")
    
    # Download candidates
    c_remote = f"/runs/l21-l25-prod-v1/candidates/{v}"
    c_local = f"export_for_fusion/l21-l25-prod-v1/candidates/{v}"
    subprocess.run(["modal", "volume", "get", "aic-frame-vol", c_remote, c_local], check=False)
    
    # Download vectors
    v_remote = f"/runs/l21-l25-prod-v1/vectors/{v}"
    v_local = f"export_for_fusion/l21-l25-prod-v1/vectors/{v}"
    subprocess.run(["modal", "volume", "get", "aic-frame-vol", v_remote, v_local], check=False)

# Download updated manifests
print("Downloading manifests...")
subprocess.run(["modal", "volume", "get", "aic-frame-vol", "/runs/l21-l25-prod-v1/RUN_MANIFEST.json", "export_for_fusion/l21-l25-prod-v1/RUN_MANIFEST.json"], check=False)
subprocess.run(["modal", "volume", "get", "aic-frame-vol", "/runs/l21-l25-prod-v1/BUDGET_PLAN.json", "export_for_fusion/l21-l25-prod-v1/BUDGET_PLAN.json"], check=False)

print("Done sync!")
