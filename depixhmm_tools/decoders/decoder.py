"""
DepixHMM-based decoder for pixelated screenshots.

Approach
--------
1. Crop each input PNG to an integer multiple of block_size (8) in both
   dimensions, top-left aligned so the visible pixelization grid is preserved.
2. For each image, sweep (font_size, offset_y) combinations and pick the
   reconstruction whose training-set accuracy is highest.
3. Use a permissive alphanumeric regex for the regex text generator.

The image heights observed are 38/39 (-> 4 block rows = 32 px) and
45/46 (-> 5 block rows = 40 px). The mapping from "4 block rows" / "5
block rows" to a Courier New font size depends on offset_y, so we sweep.
"""
from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageFont

DEPIX_HMM_ROOT = Path(__file__).parent.parent.parent / "DepixHMM"
sys.path.insert(0, str(DEPIX_HMM_ROOT))

from text_depixelizer.HMM.depix_hmm import DepixHMM  # noqa: E402
from text_depixelizer.parameters import (  # noqa: E402
    PictureParameters,
    TrainingParameters,
)

logging.basicConfig(level=logging.WARNING)

import sys as _sys, platform as _platform
from pathlib import Path as _Path

def _resolve_font_path() -> str:
    """Return Courier New path for Windows/macOS/Linux."""
    _sys.path.insert(0, str(_Path(__file__).parent.parent / "analysis"))
    from detect_params import get_default_font_path
    return get_default_font_path()

FONT_PATH = _resolve_font_path()

BLOCK_SIZE = 8
WINDOW_SIZE = 3  # 3 blocks wide window

IMAGES = [
    "user1_username_pixelated.png",
    "user1_password_pixelated.png",
    "user2_username_pixelated.png",
    "user2_password_pixelated.png",
    "user3_username_pixelated.png",
    "user3_password_pixelated.png",
    "user4_username_pixelated.png",
    "user4_password_pixelated.png",
]


def crop_to_block_grid(img: Image.Image, block_size: int) -> Image.Image:
    """Crop bottom/right so width and height are exact multiples of block_size."""
    w, h = img.size
    return img.crop((0, 0, w - (w % block_size), h - (h % block_size)))


def font_tile_height(size: int, offset_y: int, block_size: int = 8) -> int:
    font = ImageFont.truetype(FONT_PATH, size)
    ascent, descent = font.getmetrics()
    off_y = offset_y % block_size
    return (
        math.ceil((ascent - off_y) / block_size)
        + math.ceil((descent + off_y) / block_size)
    )


def candidate_params_for(target_tile_y: int) -> List[Tuple[int, int]]:
    """Enumerate (font_size, offset_y) that produce the requested tile-row count."""
    candidates: List[Tuple[int, int]] = []
    for size in range(18, 40):
        for off in range(0, BLOCK_SIZE):
            if font_tile_height(size, off, BLOCK_SIZE) == target_tile_y:
                candidates.append((size, off))
    return candidates


def decode_image(
    image_path: Path,
    pattern: str,
    target_tile_y: int,
    n_train: int = 2000,
    n_clusters: int = 200,
    n_test: int = 200,
) -> List[Tuple[float, str, int, int]]:
    """Decode one image. Returns list of (accuracy, decoded, font_size, offset_y)."""
    img = Image.open(image_path).convert("RGB")
    img = crop_to_block_grid(img, BLOCK_SIZE)

    results: List[Tuple[float, str, int, int]] = []

    for size, off in candidate_params_for(target_tile_y):
        font = ImageFont.truetype(FONT_PATH, size)
        pp = PictureParameters(
            pattern=pattern,
            font=font,
            block_size=BLOCK_SIZE,
            window_size=WINDOW_SIZE,
            randomize_pixelization_origin_x=True,
            offset_y=off,
        )
        tp = TrainingParameters(
            n_img_train=n_train, n_img_test=n_test, n_clusters=n_clusters
        )
        try:
            hmm = DepixHMM(pp, tp)
            hmm.train()
            acc, _ = hmm.evaluate()
            with Image.open(image_path) as raw:
                cropped = crop_to_block_grid(raw.convert("RGB"), BLOCK_SIZE)
                decoded = hmm.test_image(cropped)
            results.append((acc, decoded, size, off))
            print(
                f"  size={size:2d} offset_y={off} acc={acc:.2%}  -> {decoded!r}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  size={size} offset_y={off} ERROR {exc}", flush=True)

    results.sort(key=lambda r: -r[0])
    return results


def main() -> None:
    here = Path(__file__).parent.parent.parent
    for name in IMAGES:
        path = here / name
        img = Image.open(path)
        w, h = img.size
        tile_y = h // BLOCK_SIZE
        # use a wide alphanumeric pattern, length proportional to image width
        n_chars_min = max(4, w // 25)
        n_chars_max = max(n_chars_min + 4, w // 12)
        pattern = rf"[A-Za-z0-9 ]{{{n_chars_min},{n_chars_max}}}"
        print(
            f"\n=== {name} ({w}x{h}, tile_y={tile_y}, pattern={pattern}) ===",
            flush=True,
        )
        results = decode_image(path, pattern, tile_y)
        if results:
            best = results[0]
            print(
                f">>> best for {name}: size={best[2]} offset_y={best[3]} "
                f"acc={best[0]:.2%} -> {best[1]!r}",
                flush=True,
            )


if __name__ == "__main__":
    main()
