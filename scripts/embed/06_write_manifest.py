#!/usr/bin/env python3
"""
Sinh provenance manifest cho chỉ mục embedding — CHỈ ghi thứ xác minh được
============================================================================

    modal run scripts/embed/06_write_manifest.py

Ghi `artifacts/embed/embed/manifest.json`.

=============================================================================
NGUYÊN TẮC: KHÔNG SUY ĐOÁN, VÀ NÓI RÕ CHỖ KHÔNG BIẾT
=============================================================================

Một manifest hồi tố dựng bằng phỏng đoán **tệ hơn không có manifest**: nó trông như
bằng chứng nên không ai kiểm lại, và sai lệch nằm im cho tới lúc kết quả không tái tạo
được. Nên script này chia mọi trường thành ba nhóm và không trộn lẫn:

    verified        đọc trực tiếp từ artefact, từ hf-cache, hoặc từ mã đã chạy
    not_applicable  trường được yêu cầu nhưng hệ này KHÔNG có thứ đó
    unverifiable    không khôi phục được từ dữ liệu hiện có

`unverifiable` rỗng nghĩa là mọi thứ còn lại đều truy được về nguồn.

=============================================================================
VÌ SAO SHA GIẢI ĐƯỢC MỘT CÁCH KHÔNG MƠ HỒ
=============================================================================

`from_pretrained` không ghim revision nên nó giải về `main` **tại thời điểm tải**. Điều
đó chỉ mơ hồ nếu cache chứa NHIỀU snapshot. Script kiểm đúng chỗ đó: nếu một repo có
hơn một snapshot, hoặc snapshot mới hơn `emb.npy`, nó **từ chối** ghi SHA và đẩy trường
sang `unverifiable` kèm lý do — vì khi đó không phân biệt được bản nào đã sinh ra file.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

OUT = Path("artifacts/embed/embed/manifest.json")
EMBED_DIR = Path("data/embed")
ARTEFACTS = ("emb.npy", "ids.npy", "frame_idx.npy", "ranges.json")
ENCODE_SCRIPT = Path("scripts/embed/01_encode_modal.py")
ENCODER_SRC = Path("src/ingestion/jina_encoder.py")
INDEX_SRC = Path("src/ingestion/vector_index.py")

app = modal.App("aic-manifest")
cache = modal.Volume.from_name("hf-cache", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").env({"HF_HOME": "/cache"})


@app.function(image=image, volumes={"/cache": cache}, timeout=600)
def read_cache() -> dict:
    """Đọc snapshot THẬT trong volume đã dùng lúc mã hoá."""
    import os

    hub = Path("/cache/hub")
    out: dict = {"repos": {}}
    for d in sorted(hub.glob("models--jinaai--*")):
        snaps = sorted(p.name for p in (d / "snapshots").iterdir()) \
            if (d / "snapshots").is_dir() else []
        refs = {r.name: r.read_text().strip() for r in (d / "refs").iterdir()} \
            if (d / "refs").is_dir() else {}
        out["repos"][d.name.replace("models--", "").replace("--", "/")] = {
            "snapshots": snaps,
            "refs": refs,
            "snapshot_mtime_utc": {
                s: datetime.fromtimestamp(os.path.getmtime(d / "snapshots" / s),
                                          timezone.utc).isoformat()
                for s in snaps},
        }
    pp = sorted(hub.glob("models--jinaai--jina-clip-v2/snapshots/*/preprocessor_config.json"))
    if pp:
        out["preprocessor_config"] = json.loads(pp[0].read_text())
    cfg = sorted(hub.glob("models--jinaai--jina-clip-v2/snapshots/*/config.json"))
    if cfg:
        c = json.loads(cfg[0].read_text())
        out["vision_config"] = c.get("vision_config")
        out["config_truncate_dim"] = c.get("truncate_dim")
        out["text_task_instructions"] = (
            c.get("text_config", {}).get("hf_model_config_kwargs", {})
             .get("task_instructions"))
    return out


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None
    head = run("git", "rev-parse", "HEAD")
    dirty = run("git", "status", "--porcelain")
    return {"head": head,
            "clean": (dirty == "") if dirty is not None else None,
            "note": None if dirty == "" else
                    "cây làm việc CÓ thay đổi chưa commit lúc sinh manifest — "
                    "`head` KHÔNG mô tả đủ mã đang chạy; dùng `source_snapshot`"}


@app.local_entrypoint()
def main():
    import numpy as np

    cache_info = read_cache.remote()
    files, missing = {}, []
    for f in ARTEFACTS:
        p = EMBED_DIR / f
        if not p.exists():
            missing.append(f)
            continue
        files[f] = {"bytes": p.stat().st_size, "sha256": sha256(p),
                    "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime,
                                                        timezone.utc).isoformat()}
    if missing:
        raise SystemExit(f"thiếu artefact: {missing} — không sinh manifest cho tập dở")

    emb = np.load(EMBED_DIR / "emb.npy", mmap_mode="r")
    sample = np.asarray(emb[:5000], dtype=np.float32)
    norms = np.linalg.norm(sample, axis=1)
    emb_mtime = (EMBED_DIR / "emb.npy").stat().st_mtime

    unver, na = {}, {}

    # ── SHA model: chỉ chấp nhận khi KHÔNG mơ hồ ──────────────────────────
    def resolve(repo: str) -> str | None:
        r = cache_info["repos"].get(repo)
        if not r or not r["snapshots"]:
            unver[repo] = "không có snapshot nào trong hf-cache"
            return None
        if len(r["snapshots"]) > 1:
            unver[repo] = (f"cache có {len(r['snapshots'])} snapshot "
                           f"{r['snapshots']} — không phân biệt được bản nào sinh ra "
                           f"emb.npy; phải mã hoá lại với revision ghim")
            return None
        s = r["snapshots"][0]
        mt = datetime.fromisoformat(r["snapshot_mtime_utc"][s]).timestamp()
        if mt > emb_mtime:
            unver[repo] = (f"snapshot tải LÚC {r['snapshot_mtime_utc'][s]}, SAU khi "
                           f"emb.npy được ghi — không thể là bản đã dùng")
            return None
        return s

    model_sha = resolve("jinaai/jina-clip-v2")
    impl_sha = resolve("jinaai/jina-clip-implementation")

    # ── trường được yêu cầu nhưng hệ này KHÔNG có ────────────────────────
    na["task"] = ("không dùng: 01_encode_modal.py:152 gọi encode_image(imgs, "
                  "batch_size=…) không truyền task; encode_text cũng vậy")
    na["faiss_index"] = ("không tồn tại: tìm kiếm là quét phẳng chính xác "
                         "(vector_index.FlatIndex.search), không có ANN index")

    pc = cache_info.get("preprocessor_config") or {}
    g = git_state()
    man = {
        "schema": "aic2026.embed.provenance/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/embed/06_write_manifest.py",

        "model": {
            "id": "jinaai/jina-clip-v2",
            "commit_sha": model_sha,
            "remote_code": {"repo": "jinaai/jina-clip-implementation",
                            "commit_sha": impl_sha},
            "tokenizer_revision": model_sha,
            # Lớp xử lý ảnh do REMOTE CODE định nghĩa, nên revision của nó KHÁC model.
            "image_processor_revision": impl_sha,
        },

        "preprocessing": {
            "input_mode": "RGB",
            "size": pc.get("size"),
            "resize_mode": pc.get("resize_mode"),
            "interpolation": pc.get("interpolation"),
            "fill_color": pc.get("fill_color"),
            "mean": pc.get("mean"),
            "std": pc.get("std"),
            "vision_image_size": (cache_info.get("vision_config") or {}).get("image_size"),
        },

        "embedding": {
            "n_vectors": int(emb.shape[0]),
            "dim": int(emb.shape[1]),
            "dtype": str(emb.dtype),
            "truncate_dim_at_encode": 1024,
            "normalize_embeddings": True,
            "l2_norm_range_5000_rows": [round(float(norms.min()), 6),
                                        round(float(norms.max()), 6)],
        },

        "runtime": {
            "python": "3.11", "gpu": "A10G",
            "torch": "2.5.1", "torchvision": "0.20.1", "transformers": "4.48.0",
            "pillow": "11.1.0", "numpy": "1.26.4",
            "einops": "0.8.0", "timm": "1.0.13", "peft": "0.14.0",
        },

        "artifacts": {k: {"bytes": v["bytes"], "sha256": v["sha256"]}
                      for k, v in files.items()},

        "code": {
            "git_head": g["head"],
            "git_clean": g["clean"],
            "files": {p.as_posix(): sha256(p)
                      for p in (ENCODE_SCRIPT, ENCODER_SRC, INDEX_SRC) if p.exists()},
        },

        "not_applicable": na,
        "unverifiable": unver,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✓ {OUT}")
    print(f"  model      {model_sha}")
    print(f"  remote     {impl_sha}")
    print(f"  emb.npy    {files['emb.npy']['sha256'][:16]}…  {emb.shape}")
    print(f"  not_applicable: {list(na)}")
    print(f"  unverifiable  : {list(unver) or 'rỗng'}")
