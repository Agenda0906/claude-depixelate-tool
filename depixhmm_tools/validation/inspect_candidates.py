"""Per-position top-K diagnostic for a given image."""
from __future__ import annotations

import string
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont

# Ensure decoders/ is on the path so template_decoder can be found
sys.path.insert(0, str(Path(__file__).parent.parent / "decoders"))
from template_decoder import (
    BLOCK_SIZE,
    FONT_PATH,
    block_columns,
    char_advance,
    render_glyph_columns,
)


def show(image_path: Path, font: ImageFont.FreeTypeFont, alphabet: list, k: int = 5) -> None:
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    blocks = block_columns(arr, BLOCK_SIZE)
    n_rows, n_cols, _ = blocks.shape
    advance = char_advance(font)

    best_align: int = 0
    best_total = None
    for alignment in range(BLOCK_SIZE):
        tpl = render_glyph_columns(font, alphabet, BLOCK_SIZE, alignment)
        n_per = tpl.shape[1]
        max_chars = (n_cols * BLOCK_SIZE - alignment) // advance
        total = 0.0
        for ci in range(max_chars):
            ce_left = alignment + ci * advance
            fbc = ce_left // BLOCK_SIZE
            if fbc + n_per > n_cols:
                break
            test = blocks[:, fbc : fbc + n_per, :].transpose(1, 0, 2)
            diff = tpl - test[None, :, :, :]
            sq = (diff ** 2).sum(axis=-1).sum(axis=-1)
            weights = np.ones(n_per, dtype=np.float32); weights[0] = 0.3; weights[-1] = 0.3
            costs = (sq * weights[None, :]).sum(axis=-1)
            total += costs.min()
        if best_total is None or total < best_total:
            best_total = total
            best_align = alignment
    print(f"{image_path.name}  font={font.size}  advance={advance}  best_alignment={best_align}")

    tpl = render_glyph_columns(font, alphabet, BLOCK_SIZE, best_align)
    n_per = tpl.shape[1]
    max_chars = (n_cols * BLOCK_SIZE - best_align) // advance
    for ci in range(max_chars):
        ce_left = best_align + ci * advance
        fbc = ce_left // BLOCK_SIZE
        if fbc + n_per > n_cols:
            break
        test = blocks[:, fbc : fbc + n_per, :].transpose(1, 0, 2)
        diff = tpl - test[None, :, :, :]
        sq = (diff ** 2).sum(axis=-1).sum(axis=-1)
        weights = np.ones(n_per, dtype=np.float32); weights[0] = 0.3; weights[-1] = 0.3
        costs = (sq * weights[None, :]).sum(axis=-1)
        order = np.argsort(costs)[:k]
        top = [(alphabet[i], float(costs[i])) for i in order]
        print(f"  pos {ci:2d}:  " + "  ".join(f"{c}:{s:.0f}" for c, s in top))


def main() -> None:
    here = Path(__file__).parent.parent.parent
    alphabet = list(string.ascii_letters + string.digits + " ")
    for name, size in [
        ("user3_username_pixelated.png", 35),
        ("user3_username_pixelated.png", 36),
        ("user3_username_pixelated.png", 39),
        ("user4_password_pixelated.png", 35),
        ("user4_password_pixelated.png", 36),
        ("user1_username_pixelated.png", 32),
        ("user1_username_pixelated.png", 33),
        ("user2_password_pixelated.png", 33),
    ]:
        font = ImageFont.truetype(FONT_PATH, size)
        show(here / name, font, alphabet, k=5)
        print()


if __name__ == "__main__":
    main()
