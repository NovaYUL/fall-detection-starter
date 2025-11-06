import argparse
import shutil
import zipfile
from pathlib import Path
import requests

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg")

HELP = """
URFD download helper

This script tries to:
1) Download a URFD archive if a direct public URL is provided, OR
2) Use a local --zip you already downloaded, then
3) Extract and structure files into:
   <out>/falls/*.mp4 and <out>/non_falls/*.mp4

Usage examples:
  python scripts/download_urfd.py --out data/raw/urfd --url https://<public>/URFD.zip
  python scripts/download_urfd.py --out data/raw/urfd --zip /path/to/URFD.zip

If no URL is public (license required), manually download from the official page,
then pass --zip to structure them.
"""

def download(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def extract(zip_path: Path, tmp_dir: Path):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(tmp_dir)

def classify_and_copy(src_root: Path, out_root: Path):
    falls_dir = out_root / "falls"
    non_dir = out_root / "non_falls"
    falls_dir.mkdir(parents=True, exist_ok=True)
    non_dir.mkdir(parents=True, exist_ok=True)

    def is_fall(p: Path):
        s = p.stem.lower()
        sp = str(p.parent).lower()
        return ("fall" in s or "fall" in sp) and not ("non" in s or "adl" in s or "no_fall" in s or "nonfall" in s)

    cnt_f, cnt_n = 0, 0
    for p in src_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            target = falls_dir / p.name if is_fall(p) else non_dir / p.name
            if not target.exists():
                shutil.copy2(p, target)
            if is_fall(p):
                cnt_f += 1
            else:
                cnt_n += 1
    return cnt_f, cnt_n

def main():
    ap = argparse.ArgumentParser(description="URFD downloader/structurer")
    ap.add_argument("--out", required=True, help="Output directory (will create falls/ and non_falls/)")
    ap.add_argument("--url", default="", help="Direct public URL to URFD zip (if available)")
    ap.add_argument("--zip", default="", help="Path to a locally downloaded zip")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.url and not args.zip:
        print(HELP)
        return

    tmp = out_root / "_tmp"
    zip_path = Path(args.zip) if args.zip else out_root / "_tmp" / "urfd.zip"
    try:
        if args.url:
            print(f"Downloading: {args.url}")
            download(args.url, zip_path)
        print(f"Extracting: {zip_path}")
        extract(zip_path, tmp)
        f, n = classify_and_copy(tmp, out_root)
        print(f"Structured URFD into {out_root}")
        print(f"Falls: {f}, Non-falls: {n}")
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()
