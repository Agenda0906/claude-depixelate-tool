"""Beam-search depixelizer for mosaic text screenshots.

Default calibration (byte-exact for the current images):
  Font: Courier New 36,  advance=22 px,  monospaced
  Text origin: x=8, y=0
  Mosaic: block=8, grid origin (0,0), partial edge blocks averaged
  Alphabet: A-Z a-z 0-9

Method: left-to-right beam search. At each position we append a candidate
character, render the FULL prefix (left-neighbour bleed into shared mosaic
blocks is reproduced exactly), re-pixelate, and score greyscale block MSE.

Usage
-----
    python depixhmm_tools/solve.py                      # hardcoded defaults
    python depixhmm_tools/solve.py --auto               # auto-detect all params
    python depixhmm_tools/solve.py --block 4 --size 24 # override specific params
    python depixhmm_tools/solve.py --auto img.png       # single image, auto mode
"""
from __future__ import annotations

import argparse
from pathlib import Path
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Default calibration constants (override via CLI or --auto)
# ---------------------------------------------------------------------------
import sys as _sys, platform as _platform
from pathlib import Path as _Path

def _resolve_font_path() -> str:
    """Return Courier New path for Windows/macOS/Linux."""
    _sys.path.insert(0, str(_Path(__file__).parent.parent / "analysis"))
    from detect_params import get_default_font_path
    return get_default_font_path()

FONT_PATH = _resolve_font_path()

BLOCK     = 8
SIZE      = 36
X0        = 8       # left margin in pixels
ADV       = 22      # character advance in pixels
ALPHABET  = string.ascii_uppercase + string.ascii_lowercase + string.digits
HERE      = Path(__file__).parent.parent.parent

IMAGES = [
    "user1_username_pixelated.png", "user1_password_pixelated.png",
    "user2_username_pixelated.png", "user2_password_pixelated.png",
    "user3_username_pixelated.png", "user3_password_pixelated.png",
    "user4_username_pixelated.png", "user4_password_pixelated.png",
]


# ---------------------------------------------------------------------------
# Core rendering / pixelation
# ---------------------------------------------------------------------------

def pixelate(arr: np.ndarray, block: int) -> np.ndarray:
    """Block-average over FULL blocks only (partial edge row/col dropped)."""
    h, w = arr.shape
    nr, nc = h // block, w // block
    arr = arr[:nr * block, :nc * block]
    return arr.reshape(nr, block, nc, block).mean(axis=(1, 3))


def render_blocks(
    text: str,
    W: int,
    H: int,
    y: int,
    font: ImageFont.FreeTypeFont,
    block: int,
    x0: int,
) -> np.ndarray:
    img = Image.new("L", (W, H), 255)
    ImageDraw.Draw(img).text((x0, y), text, font=font, fill=0)
    return pixelate(np.asarray(img, dtype=np.float32), block)


def col_limit(i: int, x0: int, adv: int, block: int) -> int:
    """Number of block-columns fully determined after placing character index i."""
    right_px = x0 + (i + 1) * adv
    return (right_px + block - 1) // block


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def decode_with_y(
    tb: np.ndarray,
    W: int,
    H: int,
    y: int,
    beam_width: int,
    font: ImageFont.FreeTypeFont,
    block: int,
    x0: int,
    adv: int,
) -> list:
    n_chars = (W - x0) // adv
    beams = [(0.0, "")]

    for i in range(n_chars):
        climit = min(col_limit(i, x0, adv, block), tb.shape[1])
        cand = []
        for _, prefix in beams:
            for ch in ALPHABET:
                rb = render_blocks(prefix + ch, W, H, y, font, block, x0)
                cl = min(climit, rb.shape[1])
                score = ((rb[:, :cl] - tb[:, :cl]) ** 2).mean()
                cand.append((score, prefix + ch))
        cand.sort(key=lambda t: t[0])
        beams = cand[:beam_width]

    final = []
    for _, s in beams:
        rb = render_blocks(s, W, H, y, font, block, x0)
        cl = min(rb.shape[1], tb.shape[1])
        mse = ((rb[:, :cl] - tb[:, :cl]) ** 2).mean()
        final.append((mse, s))
    final.sort(key=lambda t: t[0])
    return final


def decode(
    name: str,
    beam_width: int = 40,
    y_offsets: range = range(-1, 4),
    font: ImageFont.FreeTypeFont | None = None,
    block: int = BLOCK,
    x0: int = X0,
    adv: int = ADV,
    img_root: Path = HERE,
) -> tuple:
    """
    Decode one image.

    Returns (top_list, best_y) where top_list is sorted [(mse, string), ...].
    """
    if font is None:
        font = ImageFont.truetype(FONT_PATH, SIZE)

    target = np.asarray(
        Image.open(img_root / name).convert("L"), dtype=np.float32
    )
    H, W = target.shape
    tb = pixelate(target, block)

    best: tuple | None = None
    for y in y_offsets:
        res = decode_with_y(tb, W, H, y, beam_width, font, block, x0, adv)
        if best is None or res[0][0] < best[0][0][0]:
            best = (res, y)

    return best  # (top_list, chosen_y)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, font_size)


def _char_advance(font: ImageFont.FreeTypeFont) -> int:
    return font.getbbox("MM")[2] - font.getbbox("M")[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Beam-search depixelizer for mosaic text images"
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Auto-detect block size and font from the image (slower first pass)",
    )
    parser.add_argument(
        "--block", type=int, default=None,
        help=f"Block size override (default: {BLOCK})",
    )
    parser.add_argument(
        "--size", type=int, default=None,
        help=f"Font size override (default: {SIZE})",
    )
    parser.add_argument(
        "--font", default=None,
        help=f"Font path override (default: {FONT_PATH})",
    )
    parser.add_argument(
        "--x0", type=int, default=None,
        help=f"Left text margin in px (default: {X0})",
    )
    parser.add_argument(
        "--beam", type=int, default=40,
        help="Beam width (default: 40)",
    )
    parser.add_argument(
        "images", nargs="*",
        help="Image filenames relative to repo root (default: all user images)",
    )
    args = parser.parse_args()

    images = args.images if args.images else IMAGES

    for name in images:
        img_path = HERE / name
        if not img_path.exists():
            print(f"[skip] {name} not found")
            continue

        # ----- parameter resolution -----
        if args.auto:
            _ANALYSIS = Path(__file__).resolve().parent.parent / "analysis"
            if str(_ANALYSIS) not in _sys.path:
                _sys.path.insert(0, str(_ANALYSIS))
            from detect_params import detect_all   # lazy import (only when needed)
            print(f"[auto] {name} — detecting params ...", flush=True)
            params = detect_all(img_path, verbose=False)
            block     = params["block_size"]
            font_path = params["font_path"]
            font_size = params["font_size"]
            adv       = params["advance"]
            x0        = params["left_pad"]
            font      = _build_font(font_path, font_size)
            print(
                f"         block={block}  font={params['font_name']}"
                f"  size={font_size}  adv={adv}",
                flush=True,
            )
        else:
            block     = args.block if args.block is not None else BLOCK
            font_path = args.font  if args.font  is not None else FONT_PATH
            font_size = args.size  if args.size  is not None else SIZE
            x0        = args.x0   if args.x0    is not None else X0
            font      = _build_font(font_path, font_size)
            # Recompute advance whenever any font param was overridden
            adv = _char_advance(font) if (args.block or args.size or args.font) else ADV

        # ----- decode -----
        top, y = decode(
            name,
            beam_width=args.beam,
            font=font,
            block=block,
            x0=x0,
            adv=adv,
        )
        best_mse, best_s = top[0]
        alts = "   ".join(f"{s}({m:.2f})" for m, s in top[1:3])
        print(
            f"{name:<32}  {best_s:<20}  MSE={best_mse:.3f}  y={y}"
            f"   alt: {alts}"
        )


if __name__ == "__main__":
    main()
