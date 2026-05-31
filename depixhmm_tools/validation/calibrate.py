"""Joint calibration: given a guess string + target image, search font size / x / y
that minimises block MSE when re-pixelated with the exact target grid (block=8,
origin (0,0), partial edge blocks averaged)."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys as _sys, platform as _platform
from pathlib import Path as _Path

def _resolve_font_path() -> str:
    """Return Courier New path for Windows/macOS/Linux."""
    _sys.path.insert(0, str(_Path(__file__).parent.parent / "analysis"))
    from detect_params import get_default_font_path
    return get_default_font_path()

FONT_PATH = _resolve_font_path()

BLOCK = 8
HERE = Path(__file__).parent.parent.parent


def pixelate_like_target(arr, block=BLOCK):
    """Average every block cell including partial edge cells -> per-block grid (grayscale)."""
    h, w = arr.shape[:2]
    nr = (h + block - 1) // block
    nc = (w + block - 1) // block
    out = np.zeros((nr, nc), dtype=np.float32)
    for r in range(nr):
        for c in range(nc):
            cell = arr[r*block:min((r+1)*block, h), c*block:min((c+1)*block, w)]
            out[r, c] = cell.mean()
    return out


def target_blocks(path):
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return pixelate_like_target(arr), arr.shape


def render(text, size, x, y, W, H):
    font = ImageFont.truetype(FONT_PATH, size)
    img = Image.new("L", (W, H), 255)
    ImageDraw.Draw(img).text((x, y), text, font=font, fill=0)
    return np.asarray(img, dtype=np.float32)


def best_fit(guess, target_name, sizes=range(30, 46), verbose=True):
    tb, (H, W) = target_blocks(HERE / target_name)
    best = None
    for size in sizes:
        for x in range(-4, 12):
            for y in range(-12, 6):
                rarr = render(guess, size, x, y, W, H)
                rb = pixelate_like_target(rarr)
                mse = ((rb - tb) ** 2).mean()
                if best is None or mse < best[0]:
                    best = (mse, size, x, y)
    if verbose:
        print(f"{target_name}: guess={guess!r} -> MSE={best[0]:.1f} "
              f"size={best[1]} x={best[2]} y={best[3]} (img {W}x{H})")
    return best


if __name__ == "__main__":
    best_fit("DrSunYatSen1866", "user1_username_pixelated.png")
    best_fit("HarryPotter89", "user3_username_pixelated.png")
    best_fit("ElonMusk2030", "user4_username_pixelated.png")
    best_fit("UncleWilliam", "user2_username_pixelated.png")
