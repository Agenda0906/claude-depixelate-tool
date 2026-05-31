"""
Empirical mosaic parameter analysis.

Compares three block-size detection methods:
  (A) Intra-block variance  - minimise mean within-block pixel variance
                               (most robust; immune to compression artifacts)
  (B) FFT autocorrelation  - peak of the column-diff autocorrelation signal
  (C) Edge detection        - original approach: count column/row transitions

Also scans candidate fonts and reports the per-font MSE score so you can
confirm whether the image was produced with a known or unknown font.

Run from repo root:
    python depixhmm_tools/analyze.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Allow sibling-module imports
sys.path.insert(0, str(Path(__file__).parent))
from detect_params import (
    detect_block_size,
    detect_block_size_fft,
    detect_font,
    _DEFAULT_FONT_CANDIDATES,
)

HERE = Path(__file__).parent.parent.parent

IMAGES = [
    "user1_username_pixelated.png", "user1_password_pixelated.png",
    "user2_username_pixelated.png", "user2_password_pixelated.png",
    "user3_username_pixelated.png", "user3_password_pixelated.png",
    "user4_username_pixelated.png", "user4_password_pixelated.png",
]


# ---------------------------------------------------------------------------
# Original edge-detection approach (kept for comparison)
# ---------------------------------------------------------------------------

def detect_edges(path: Path):
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.int32)
    h, w = arr.shape
    col_diff = np.abs(np.diff(arr, axis=1)).sum(axis=0)
    row_diff = np.abs(np.diff(arr, axis=0)).sum(axis=1)
    col_edges = [i + 1 for i, d in enumerate(col_diff) if d > 0]
    row_edges = [i + 1 for i, d in enumerate(row_diff) if d > 0]
    cds = np.diff([0] + col_edges + [w]).tolist()
    rds = np.diff([0] + row_edges + [h]).tolist()
    return h, w, col_edges[:12], sorted(set(cds)), sorted(set(rds))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze(name: str) -> None:
    p = HERE / name
    if not p.exists():
        print(f"[skip] {name} not found\n")
        return

    h_px, w_px, ce, cds, rds = detect_edges(p)
    b_var,  scores_var  = detect_block_size(p)
    b_fft, _ac          = detect_block_size_fft(p)

    sep = "-" * 60
    print(sep)
    print(f"  {name}  ({w_px}x{h_px} px)")
    print(sep)

    # Method A: variance
    top6_var = sorted(scores_var.items(), key=lambda kv: kv[1])[:6]
    print(f"  (A) Variance scores: "
          f"{['b=%d->%.2f' % (b, s) for b, s in top6_var]}")
    print(f"      best block_size = {b_var}")

    # Method B: FFT
    print(f"  (B) FFT autocorr   -> best block_size = {b_fft}")

    # Method C: edges
    print(f"  (C) Edge detection -> col widths seen: {cds}")
    print(f"                        row heights seen: {rds}")
    print(f"      first col edges: {ce}")

    if b_var == b_fft:
        print(f"  [OK] All methods agree: block_size = {b_var}")
    else:
        print(f"  [!] Methods disagree -- variance={b_var}  fft={b_fft}"
              f"  (recommend variance result)")

    # Font detection (compact view)
    print()
    print("  Font detection (compatible sizes only):")
    existing_fonts = [(n, fp) for n, fp in _DEFAULT_FONT_CANDIDATES if Path(fp).exists()]
    font_info = detect_font(p, b_var, existing_fonts, verbose=True)
    if font_info:
        fn, fp, fs, adv = font_info
        print(f"  -> Best font: {fn}  size={fs}  advance={adv} px")
    else:
        print("  -> No compatible font found in candidate list")

    print()


if __name__ == "__main__":
    import sys as _sys
    targets = _sys.argv[1:] if len(_sys.argv) > 1 else IMAGES
    for n in targets:
        analyze(n)
