"""Verify calibration: render a guess, pixelate it the same way, compare to target."""
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


def render_tight(text, font, left_pad=0):
    ascent, descent = font.getmetrics()
    bbox = font.getbbox(text)
    width = max(bbox[2], 1) + left_pad
    img = Image.new("RGB", (width, ascent + descent), (255, 255, 255))
    ImageDraw.Draw(img).text((left_pad, 0), text, font=font, fill=(0, 0, 0))
    return img


def block_avg(arr, block=BLOCK):
    h, w = arr.shape[:2]
    nr, nc = h // block, w // block
    arr = arr[:nr * block, :nc * block]
    return arr.reshape(nr, block, nc, block, 3).mean(axis=(1, 3))


def blocks_of_image(path):
    return block_avg(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32))


def compare(guess, target_name, size, left_pad=0):
    font = ImageFont.truetype(FONT_PATH, size)
    asc, desc = font.getmetrics()
    rendered = render_tight(guess, font, left_pad)
    rb = block_avg(np.asarray(rendered, dtype=np.float32))
    tb = blocks_of_image(HERE / target_name)
    nc = min(rb.shape[1], tb.shape[1])
    nr = min(rb.shape[0], tb.shape[0])
    rb2, tb2 = rb[:nr, :nc], tb[:nr, :nc]
    mse = ((rb2 - tb2) ** 2).mean()
    th = Image.open(HERE / target_name).size[1]
    print(f"{target_name}: size={size} asc+desc={asc+desc} img_h={th} "
          f"render_cols={rb.shape[1]} tgt_cols={tb.shape[1]} pad={left_pad} MSE={mse:.1f}")
    return mse


if __name__ == "__main__":
    # confident guess: Sun Yat-sen born 1866
    for size in range(30, 42):
        font = ImageFont.truetype(FONT_PATH, size)
        print(size, "asc+desc=", sum(font.getmetrics()), "advance=",
              font.getbbox("MM")[2] - font.getbbox("M")[2])
    print("---")
    for pad in range(0, 9):
        compare("DrSunYatSen1866", "user1_username_pixelated.png", 33, pad)
