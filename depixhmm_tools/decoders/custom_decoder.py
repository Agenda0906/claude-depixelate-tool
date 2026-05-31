"""
Custom DepixHMM-based decoder for screenshots pixelated as:
- Image cropped tightly to height = ascent + descent (no padding)
- Block size 8, grid origin at (0, 0) of the image
- Floor(height / 8) full block rows used, partial row at bottom ignored

This mirrors what a typical "pixelate this image" filter does on a tightly
cropped line of text. Training synthesises images of the same format,
clusters their windows, and trains an HMM directly on them, bypassing the
default DepixHMM pixelization pipeline (which assumes padded images).
"""
from __future__ import annotations

import logging
import random
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEPIX_HMM_ROOT = Path(__file__).parent.parent.parent / "DepixHMM"
sys.path.insert(0, str(DEPIX_HMM_ROOT))

from text_depixelizer.HMM.clusterer import KmeansClusterer  # noqa: E402
from text_depixelizer.HMM.hmm import HMM  # noqa: E402
from text_depixelizer.HMM.hmm_result_reconstructor import (  # noqa: E402
    reconstruct_string_from_window_characters,
)
from text_depixelizer.training_pipeline.original_image import (  # noqa: E402
    CharacterBoundingBox,
)
from text_depixelizer.training_pipeline.windows import Window  # noqa: E402

import sys as _sys, platform as _platform
from pathlib import Path as _Path

def _resolve_font_path() -> str:
    """Return Courier New path for Windows/macOS/Linux."""
    _sys.path.insert(0, str(_Path(__file__).parent.parent / "analysis"))
    from detect_params import get_default_font_path
    return get_default_font_path()

FONT_PATH = _resolve_font_path()

BLOCK_SIZE = 8


def render_tight(text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    """Render text with NO padding. Image height = ascent + descent."""
    ascent, descent = font.getmetrics()
    bbox = font.getbbox(text)
    width = max(bbox[2], 1)
    img = Image.new("RGB", (width, ascent + descent), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font, fill=(0, 0, 0))
    return img


def pixelate(image: Image.Image, block_size: int, origin_x: int = 0) -> Image.Image:
    """Pixelate from (origin_x, 0); keep partial blocks at the right/bottom."""
    w, h = image.size
    arr = np.asarray(image).copy()
    # left partial (before origin_x)
    if origin_x > 0:
        block = arr[0:h, 0:origin_x]
        if block.size:
            arr[0:h, 0:origin_x] = np.rint(block.mean(axis=(0, 1))).astype(np.uint8)
    # full / partial blocks
    x = origin_x
    while x < w:
        x2 = min(x + block_size, w)
        for y in range(0, h, block_size):
            y2 = min(y + block_size, h)
            cell = arr[y:y2, x:x2]
            if cell.size:
                arr[y:y2, x:x2] = np.rint(cell.mean(axis=(0, 1))).astype(np.uint8)
        x = x2
    return Image.fromarray(arr)


def char_bboxes(text: str, font: ImageFont.FreeTypeFont) -> List[CharacterBoundingBox]:
    """Compute approximate per-character horizontal bounding boxes (no padding)."""
    bbs: List[CharacterBoundingBox] = []
    cursor = 0
    for ch in text:
        # for monospaced Courier New this is constant, but use getbbox for safety
        bb = font.getbbox(ch)
        w = bb[2] - bb[0]
        # In Pillow, draw.text(x, y) advances by the font's character advance
        # which for monospaced fonts equals the bbox width. We measure
        # cumulative advance via getbbox of substrings:
        bbs.append(
            CharacterBoundingBox(
                char=ch,
                top=0,
                bottom=font.getmetrics()[0] + font.getmetrics()[1],
                left=cursor,
                right=cursor + w,
            )
        )
        cursor += w
    return bbs


def char_advance(font: ImageFont.FreeTypeFont) -> int:
    """For monospaced Courier New, advance width per character."""
    a = font.getbbox("M")
    b = font.getbbox("MM")
    return (b[2] - b[0]) - (a[2] - a[0])


def extract_windows(
    image: Image.Image,
    text: str,
    font: ImageFont.FreeTypeFont,
    origin_x: int,
    block_size: int,
    window_size: int,
) -> List[Window]:
    """Extract windows from a (custom-)pixelated training image."""
    arr = np.asarray(image)
    h, w, _ = arr.shape
    n_tiles_x = (w - origin_x) // block_size  # only full blocks
    n_tiles_y = h // block_size  # only full block rows
    advance = char_advance(font)

    windows: List[Window] = []
    for window_index in range(n_tiles_x - window_size + 1):
        wl = origin_x + window_index * block_size
        wr = wl + window_size * block_size - 1
        # which characters fall in [wl, wr]?
        chars: List[str] = []
        for i, ch in enumerate(text):
            left = i * advance
            right = (i + 1) * advance
            if min(right, wr) > max(left, wl):
                chars.append(ch)
        values = arr[0 : n_tiles_y * block_size : block_size,
                     wl : wl + window_size * block_size : block_size,
                     :].flatten()
        windows.append(Window(tuple(chars), values, window_index))
    return windows


def gen_sample_text(alphabet: str, length: int) -> str:
    return "".join(random.choice(alphabet) for _ in range(length))


def train_hmm(
    font: ImageFont.FreeTypeFont,
    alphabet: str,
    length_range: Tuple[int, int],
    n_images: int,
    n_clusters: int,
    window_size: int,
    randomize_origin: bool = True,
):
    block = BLOCK_SIZE
    print(f"  training: {n_images} imgs, |alphabet|={len(alphabet)}, win={window_size}, "
          f"clusters={n_clusters}, randomize_x={randomize_origin}", flush=True)
    t = time.perf_counter()
    all_windows: List[Window] = []
    for _ in range(n_images):
        L = random.randint(*length_range)
        text = gen_sample_text(alphabet, L)
        img = render_tight(text, font)
        origin_x = random.randint(0, block - 1) if randomize_origin else 0
        pix = pixelate(img, block, origin_x)
        # ensure pix wider than window
        n_x = (pix.size[0] - origin_x) // block
        if n_x < window_size:
            continue
        all_windows.extend(extract_windows(pix, text, font, origin_x, block, window_size))
    print(f"  generated {len(all_windows)} windows in {time.perf_counter() - t:.1f}s",
          flush=True)

    t = time.perf_counter()
    clusterer = KmeansClusterer(all_windows, n_clusters)
    all_windows = clusterer.map_windows_to_cluster(all_windows)
    print(f"  clustering done in {time.perf_counter() - t:.1f}s", flush=True)

    states = list({w.characters for w in all_windows})
    observations = list({w.k for w in all_windows})
    state_idx = {s: i for i, s in enumerate(states)}
    obs_idx = {o: i for i, o in enumerate(observations)}

    n_s, n_o = len(states), len(observations)
    sp = np.zeros(n_s)
    tp = np.zeros((n_s, n_s))
    ep = np.zeros((n_s, n_o))

    for w in all_windows:
        ep[state_idx[w.characters], obs_idx[w.k]] += 1

    for i, w in enumerate(all_windows):
        if w.window_index == 0:
            sp[state_idx[w.characters]] += 1
        elif i > 0 and all_windows[i - 1].window_index == w.window_index - 1:
            tp[state_idx[all_windows[i - 1].characters], state_idx[w.characters]] += 1

    # smoothing to avoid log(0)
    eps_sp, eps_tp, eps_ep = 1.0, 1.0, 1.0
    sp = (sp + eps_sp) / (sp.sum() + eps_sp * n_s)
    tp_row_sums = tp.sum(axis=1, keepdims=True)
    tp = (tp + eps_tp) / (tp_row_sums + eps_tp * n_s)
    ep_row_sums = ep.sum(axis=1, keepdims=True)
    ep = (ep + eps_ep) / (ep_row_sums + eps_ep * n_o)

    print(f"  HMM ready: |states|={n_s} |obs|={n_o}", flush=True)
    return states, observations, sp, tp, ep, clusterer


def log_viterbi(observations: List[int], sp, tp, ep) -> np.ndarray:
    """Vectorised log-space Viterbi. Returns indices of most likely state sequence."""
    T = len(observations)
    n_s = sp.shape[0]
    log_sp = np.log(sp)
    log_tp = np.log(tp)
    log_ep = np.log(ep)
    v = np.empty((n_s, T))
    ptrs = np.empty((n_s, T), dtype=np.int32)
    v[:, 0] = log_sp + log_ep[:, observations[0]]
    ptrs[:, 0] = 0
    for t in range(1, T):
        scores = v[:, t - 1][:, None] + log_tp  # shape (n_s_from, n_s_to)
        ptrs[:, t] = np.argmax(scores, axis=0)
        v[:, t] = scores[ptrs[:, t], np.arange(n_s)] + log_ep[:, observations[t]]
    x = np.empty(T, dtype=np.int32)
    x[-1] = int(np.argmax(v[:, -1]))
    for t in range(T - 1, 0, -1):
        x[t - 1] = ptrs[x[t], t]
    return x


def decode(image_path: Path, font, states, observations, sp, tp, ep, clusterer,
           window_size: int) -> str:
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img)
    h, w, _ = arr.shape
    n_tiles_x = w // BLOCK_SIZE
    n_tiles_y = h // BLOCK_SIZE
    seqs: List[List[int]] = []
    # try every possible x-origin offset (0..block_size-1) and pick the one
    # whose viterbi gives the highest path probability
    best = None
    for ox in range(BLOCK_SIZE):
        n_x = (w - ox) // BLOCK_SIZE
        if n_x < window_size + 1:
            continue
        win_values: List[np.ndarray] = []
        for wi in range(n_x - window_size + 1):
            wl = ox + wi * BLOCK_SIZE
            v = arr[0 : n_tiles_y * BLOCK_SIZE : BLOCK_SIZE,
                    wl : wl + window_size * BLOCK_SIZE : BLOCK_SIZE,
                    :].flatten()
            win_values.append(v)
        ks = clusterer.map_values_to_cluster(win_values)
        obs_map = {o: i for i, o in enumerate(observations)}
        seq = [obs_map.get(k, 0) for k in ks]
        path = log_viterbi(seq, sp, tp, ep)
        # final path log-likelihood
        log_ep = np.log(ep)
        log_tp = np.log(tp)
        log_sp = np.log(sp)
        ll = log_sp[path[0]] + log_ep[path[0], seq[0]]
        for t in range(1, len(seq)):
            ll += log_tp[path[t - 1], path[t]] + log_ep[path[t], seq[t]]
        decoded = reconstruct_string_from_window_characters(
            [states[i] for i in path], BLOCK_SIZE, font
        )
        if best is None or ll > best[0]:
            best = (ll, decoded, ox)
    return best[1] if best else ""


def main() -> None:
    logging.basicConfig(level=logging.ERROR)
    np.seterr(divide="ignore")
    random.seed(0)

    here = Path(__file__).parent.parent.parent
    images = [
        "user1_username_pixelated.png",
        "user1_password_pixelated.png",
        "user2_username_pixelated.png",
        "user2_password_pixelated.png",
        "user3_username_pixelated.png",
        "user3_password_pixelated.png",
        "user4_username_pixelated.png",
        "user4_password_pixelated.png",
    ]

    # Group images by height (-> font size)
    short_imgs, tall_imgs = [], []
    for name in images:
        with Image.open(here / name) as img:
            (short_imgs if img.size[1] <= 40 else tall_imgs).append(name)

    alphabet = string.ascii_letters + string.digits + " "
    window_size = 3
    n_images = 1500
    n_clusters = 350

    # ---- short group: image height 38-39 -> font size 33 (ascent+descent=38)
    print("=== SHORT group (image_h in 38..39) ===", flush=True)
    font_small = ImageFont.truetype(FONT_PATH, 33)
    states, obs, sp, tp, ep, cl = train_hmm(
        font_small,
        alphabet,
        length_range=(10, 25),
        n_images=n_images,
        n_clusters=n_clusters,
        window_size=window_size,
    )
    for name in short_imgs:
        t = time.perf_counter()
        decoded = decode(here / name, font_small, states, obs, sp, tp, ep, cl, window_size)
        print(f"  {name}: {decoded!r}  ({time.perf_counter() - t:.1f}s)", flush=True)

    # ---- tall group: image height 45-46 -> font size 39 (ascent+descent=45)
    print("=== TALL group (image_h in 45..46) ===", flush=True)
    font_big = ImageFont.truetype(FONT_PATH, 39)
    states, obs, sp, tp, ep, cl = train_hmm(
        font_big,
        alphabet,
        length_range=(10, 20),
        n_images=n_images,
        n_clusters=n_clusters,
        window_size=window_size,
    )
    for name in tall_imgs:
        t = time.perf_counter()
        decoded = decode(here / name, font_big, states, obs, sp, tp, ep, cl, window_size)
        print(f"  {name}: {decoded!r}  ({time.perf_counter() - t:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
