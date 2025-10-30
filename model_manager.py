"""Model manager utility

Centralizes publishing and retrieval of the "latest" trained model.

Functions:
  - publish_latest(model_run_dir): copy weights and artifacts from a
    training run into seatbelt_model/latest/ and archive any previous latest.
  - get_latest_model_path(): return path to latest/weights/best.pt if present.
  - copy_latest_model_to_flutter(): best-effort copy of latest best.pt into
    the Flutter app assets folder.

This module avoids printing non-ASCII characters to prevent Windows
console encoding errors.
"""

from pathlib import Path
from datetime import datetime
import shutil
import json


BASE = Path(__file__).parent


def publish_latest(model_run_dir):
    """Publish the weights and common artifacts from a training run.

    Args:
        model_run_dir (str | Path): path to the training run folder which
            contains a `weights/` subfolder (containing best.pt/last.pt).

    Returns:
        dict: {"latest_dir": str, "copied": [<paths>]}
    """
    model_run_dir = Path(model_run_dir)
    weights_src = model_run_dir / "weights"
    if not weights_src.exists():
        raise FileNotFoundError(f"Weights folder not found: {weights_src}")

    seatbelt_dir = BASE / "seatbelt_model"
    latest_dir = seatbelt_dir / "latest"
    archive_dir = seatbelt_dir / "archive"

    archive_dir.mkdir(parents=True, exist_ok=True)

    # Archive existing latest if present
    if latest_dir.exists() and any(latest_dir.iterdir()):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = archive_dir / f"latest_{ts}"
        shutil.move(str(latest_dir), str(dst))

    # Recreate latest structure
    latest_weights_dst = latest_dir / "weights"
    latest_weights_dst.mkdir(parents=True, exist_ok=True)

    copied = []
    # Copy primary weight files
    for name in ("best.pt", "last.pt"):
        src = weights_src / name
        if src.exists():
            dst = latest_weights_dst / name
            shutil.copy2(str(src), str(dst))
            copied.append(str(dst))

    # Copy optional artifacts
    for fname in ("args.yaml", "results.csv"):
        src = model_run_dir / fname
        if src.exists():
            dst = latest_dir / fname
            shutil.copy2(str(src), str(dst))
            copied.append(str(dst))

    # Copy jpg/png plots from run root
    for ext in ("*.jpg", "*.png"):
        for p in model_run_dir.glob(ext):
            dst = latest_dir / p.name
            shutil.copy2(str(p), str(dst))
            copied.append(str(dst))

    # Write small metadata file
    meta = {
        "source": str(model_run_dir),
        "created": datetime.now().isoformat(),
        "weights": "weights/best.pt"
    }
    with open(latest_dir / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Attempt flutter copy (best-effort, non-fatal)
    try:
        copy_latest_model_to_flutter()
    except Exception:
        pass

    return {"latest_dir": str(latest_dir), "copied": copied}


def get_latest_model_path():
    """Return the path to seatbelt_model/latest/weights/best.pt if available.
    Falls back to any .pt file in latest/weights if best.pt not present.
    Returns None if no weights found.
    """
    p = BASE / "seatbelt_model" / "latest" / "weights"
    best = p / "best.pt"
    if best.exists():
        return str(best)

    # fallback to any pt
    if p.exists():
        pts = list(p.glob("*.pt"))
        if pts:
            # return the most recently modified
            pts_sorted = sorted(pts, key=lambda x: x.stat().st_mtime)
            return str(pts_sorted[-1])

    return None


def copy_latest_model_to_flutter():
    """Copy latest best.pt into the Flutter app assets folder (best-effort).
    Returns True if copy happened, False otherwise.
    """
    latest_path = get_latest_model_path()
    if not latest_path:
        return False
    latest = Path(latest_path)
    if not latest.exists():
        return False

    flutter_models = BASE / "Flutter APp" / "seatbelt_detector_app" / "assets" / "models"
    flutter_models.mkdir(parents=True, exist_ok=True)
    
    # Copy with multiple naming conventions for compatibility
    dst_best = flutter_models / "best.pt"
    dst_latest = flutter_models / "latest.pt"
    dst_seatbelt = flutter_models / "seatbelt_model.pt"  # Original name used by Flutter app
    
    shutil.copy2(str(latest), str(dst_best))
    shutil.copy2(str(latest), str(dst_latest))
    shutil.copy2(str(latest), str(dst_seatbelt))
    
    print(f"Copied latest model to Flutter app:")
    print(f"  - {dst_best}")
    print(f"  - {dst_latest}")
    print(f"  - {dst_seatbelt}")
    
    return True


if __name__ == "__main__":
    lp = get_latest_model_path()
    if lp:
        print(f"Latest model: {lp}")
    else:
        print("No latest model available")