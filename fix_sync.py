import os
import subprocess

with open("remote_vectors.txt") as f:
    remote_paths = [line.strip() for line in f if line.strip()]
remote_videos = [p.split("/")[-1] for p in remote_paths]

with open("local_vectors.txt") as f:
    local_videos = [line.strip() for line in f if line.strip()]

missing = [v for v in remote_videos if v not in local_videos]

print("Deleting corrupted files locally...")
for v in missing:
    c_local = f"export_for_fusion/l21-l25-prod-v1/candidates/{v}"
    v_local = f"export_for_fusion/l21-l25-prod-v1/vectors/{v}"
    if os.path.exists(c_local): os.remove(c_local) if os.path.isfile(c_local) else None
    if os.path.exists(v_local): os.remove(v_local) if os.path.isfile(v_local) else None

print("Deleting corrupted files remotely on overfiters...")
subprocess.run(["modal", "profile", "activate", "overfiters"], check=True)
for v in missing:
    c_remote = f"/runs/l21-l25-prod-v1/candidates/{v}"
    v_remote = f"/runs/l21-l25-prod-v1/vectors/{v}"
    subprocess.run(["modal", "volume", "rm", "aic-frame-vol", c_remote], check=False)
    subprocess.run(["modal", "volume", "rm", "aic-frame-vol", v_remote], check=False)

print("Redownloading properly from degeabeo...")
subprocess.run(["modal", "profile", "activate", "degeabeo"], check=True)
for i, v in enumerate(missing, 1):
    print(f"Downloading {i}/{len(missing)}: {v}")
    c_remote = f"/runs/l21-l25-prod-v1/candidates/{v}"
    c_local = f"export_for_fusion/l21-l25-prod-v1/candidates/{v}"
    v_remote = f"/runs/l21-l25-prod-v1/vectors/{v}"
    v_local = f"export_for_fusion/l21-l25-prod-v1/vectors/{v}"
    
    os.makedirs(c_local, exist_ok=True)
    os.makedirs(v_local, exist_ok=True)
    
    subprocess.run(["modal", "volume", "get", "aic-frame-vol", f"{c_remote}/", f"{c_local}/"], check=False)
    subprocess.run(["modal", "volume", "get", "aic-frame-vol", f"{v_remote}/", f"{v_local}/"], check=False)

print("Done fixing!")
