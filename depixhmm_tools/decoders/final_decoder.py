"""
Pixel-perfect decoder for the eight Courier-New size-36 pixelated screenshots.

Calibration (byte-exact match against real images)
- Font: Courier New, size 36, char_advance = 22 px
- Block: 8 px
- Layout: 8-px white left margin; pixelization grid starts at image (0, 0)

For monospaced advance=22 px and block=8 px, each character is at a different
sub-block alignment (offsets cycle 0, 6, 4, 2 every 4 chars). We render
a separate template for every alignment so the block-averaged signature
exactly matches the test image.

A Viterbi pass with English bigram log-probabilities resolves local
character ambiguities (e.g. P vs H).
"""
from __future__ import annotations

import math
import string
from pathlib import Path
from typing import Dict, List, Tuple

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

BLOCK_SIZE = 8
FONT_SIZE = 36
LEFT_PAD_PX = 8


def render_text(text: str, left_pad: int = LEFT_PAD_PX, font_size: int = FONT_SIZE) -> Image.Image:
    font = ImageFont.truetype(FONT_PATH, font_size)
    ascent, descent = font.getmetrics()
    bbox = font.getbbox(text)
    width = max(bbox[2], 1) + left_pad + 16
    img = Image.new("RGB", (width, ascent + descent), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((left_pad, 0), text, font=font, fill=(0, 0, 0))
    return img


def char_advance(font_size: int = FONT_SIZE) -> int:
    font = ImageFont.truetype(FONT_PATH, font_size)
    return font.getbbox("MM")[2] - font.getbbox("M")[2]


def block_columns(arr: np.ndarray, block_size: int = BLOCK_SIZE) -> np.ndarray:
    h, w, _ = arr.shape
    n_rows = h // block_size
    n_cols = w // block_size
    arr = arr[: n_rows * block_size, : n_cols * block_size]
    return arr.reshape(n_rows, block_size, n_cols, block_size, 3).mean(axis=(1, 3))


def build_templates_for_alignment(
    alphabet: List[str],
    target_offset: int,
    n_rows_target: int,
    n_cols: int = 4,
) -> np.ndarray:
    """
    For each char, render it so its LEFT EDGE is at x = LEFT_PAD_PX + target_offset
    relative to the image origin, then return its block-averaged colour signature
    of width n_cols blocks starting from the block that contains the char's left edge.

    `target_offset` is the displacement of the character's left edge from the
    nearest preceding block boundary (0..block_size-1).
    """
    advance = char_advance()
    n_chars = len(alphabet)
    tpl = np.zeros((n_chars, n_cols, n_rows_target, 3), dtype=np.float32)
    for i, ch in enumerate(alphabet):
        # Place the char at x = block_boundary + target_offset.
        # The simplest way: render with left_pad = 8 + target_offset.
        # Add right-neighbour " " padding so the char isn't followed by junk.
        text = ch + "    "
        # We render with left_pad such that char_left_x = LEFT_PAD_PX + target_offset
        img = render_text(text, left_pad=LEFT_PAD_PX + target_offset)
        blocks = block_columns(np.asarray(img, dtype=np.float32))
        if blocks.shape[0] > n_rows_target:
            blocks = blocks[:n_rows_target]
        elif blocks.shape[0] < n_rows_target:
            pad = np.full((n_rows_target - blocks.shape[0], blocks.shape[1], 3),
                          255.0, dtype=blocks.dtype)
            blocks = np.concatenate([blocks, pad], axis=0)
        # The char's left edge is at x = LEFT_PAD_PX + target_offset
        # First block containing the char: (LEFT_PAD_PX + target_offset) // block_size = 1
        # (since LEFT_PAD_PX = 8 and target_offset < 8).
        first_block = (LEFT_PAD_PX + target_offset) // BLOCK_SIZE
        cols = blocks[:, first_block : first_block + n_cols, :]
        tpl[i] = cols.transpose(1, 0, 2)
    return tpl


def _common_words() -> List[str]:
    return [
        "the","be","to","of","and","a","in","that","have","I","it","for","not",
        "on","with","he","as","you","do","at","this","but","his","by","from",
        "they","we","say","her","she","or","an","will","my","one","all","would",
        "happy","birthday","monday","tuesday","wednesday","thursday","friday",
        "saturday","sunday","party","printer","pairs","number","numbers","day",
        "welcome","secret","password","login","user","admin","root","guest","test",
        "name","sign","code","key","pin","data","mail","file","page","line","Wishes",
        "team","game","band","club","talk","ride","walk","beat","heat","door","love",
        "boss","boys","girls","kids","mom","dad","son","wife","child","baby","good",
        "world","money","power","place","point","right","left","up","down","brother",
        "lucky","lazy","cat","dog","apple","banana","cherry","red","blue","green",
        "yellow","orange","black","white","pink","gray","brown","gold","silver",
        "summer","spring","autumn","winter","early","late","never","always","often",
        "today","tomorrow","yesterday","week","weekend","month","year","decade","sister",
        "century","minute","second","hour","morning","afternoon","evening","night",
        "rain","snow","sunny","cloudy","windy","cold","hot","warm","cool","mild",
        "Created","Date","Birth","Number","Account","Token","Phone","Email","Address",
        "Tom","John","Mary","Lisa","Mike","Alex","Anna","Sara","David","Emma",
        "James","Robert","Linda","Karen","Patricia","Susan","Charles","Steve",
        "morning","good morning","good night","best wishes","new year","big day",
        "happy birthday","wish you","with love","kind regards","yours truly",
    ]


def _build_lm() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    words = _common_words()
    tokens: List[str] = []
    for w in words:
        tokens.append(" ")
        tokens.extend(list(w))
    tokens.append(" ")
    chars = list(string.ascii_letters + string.digits + " ")
    n = len(chars)
    idx = {c: i for i, c in enumerate(chars)}
    uni = np.zeros(n)
    bi = np.zeros((n, n))
    def can(c):
        return c.lower() if c.isalpha() else c
    for i, c in enumerate(tokens):
        cc = can(c)
        if cc in idx:
            uni[idx[cc]] += 1
        if i + 1 < len(tokens):
            nxt = can(tokens[i + 1])
            if cc in idx and nxt in idx:
                bi[idx[cc], idx[nxt]] += 1
    alpha = 0.5
    uni_lp = np.log((uni + alpha) / (uni.sum() + alpha * n))
    bi_lp = np.log((bi + alpha) / (bi.sum(axis=1, keepdims=True) + alpha * n))
    # symmetrize case: treat case-folded LM as the actual LM (uppercase letters
    # use lowercase probabilities)
    # collapse uppercase entries to share the lowercase row/col
    uni_lp_full = uni_lp.copy()
    bi_lp_full = bi_lp.copy()
    for i, c in enumerate(chars):
        if c.isupper():
            j = idx[c.lower()]
            uni_lp_full[i] = uni_lp_full[j]
            bi_lp_full[i, :] = bi_lp_full[j, :]
    for j, c in enumerate(chars):
        if c.isupper():
            i = idx[c.lower()]
            bi_lp_full[:, j] = bi_lp_full[:, i]
    return uni_lp_full, bi_lp_full, chars


def decode(image_path: Path, lm_weight: float = 1500.0) -> Tuple[str, str]:
    img = Image.open(image_path).convert("RGB")
    blocks = block_columns(np.asarray(img, dtype=np.float32))
    n_rows, n_cols, _ = blocks.shape
    advance = char_advance()

    alphabet = list(string.ascii_letters + string.digits + " ")
    n = len(alphabet)

    # Precompute templates for each sub-block offset 0..block_size-1.
    n_template_cols = 4
    templates_by_offset: Dict[int, np.ndarray] = {}
    for off in range(BLOCK_SIZE):
        templates_by_offset[off] = build_templates_for_alignment(
            alphabet, off, n_rows, n_cols=n_template_cols
        )

    cost_rows: List[np.ndarray] = []
    ci = 0
    while True:
        char_left_px = LEFT_PAD_PX + ci * advance
        first_block_col = char_left_px // BLOCK_SIZE
        target_offset = char_left_px % BLOCK_SIZE
        # The character would right-end at char_left_px + advance. Require at
        # least the first 3 blocks of the template region to be inside the image.
        usable_cols = min(n_template_cols, n_cols - first_block_col)
        if usable_cols < 3:
            break
        tpl_full = templates_by_offset[target_offset]
        tpl = tpl_full[:, :usable_cols, :, :]
        test = blocks[:, first_block_col : first_block_col + usable_cols, :].transpose(1, 0, 2)
        diff = tpl - test[None, :, :, :]
        sq = (diff ** 2).sum(axis=-1).sum(axis=-1)
        weights = np.ones(usable_cols, dtype=np.float32)
        if usable_cols == n_template_cols:
            weights[-1] = 0.1
        if target_offset != 0:
            weights[0] = 0.4
        costs = (sq * weights[None, :]).sum(axis=-1)
        cost_rows.append(costs.astype(np.float32))
        ci += 1

    if not cost_rows:
        return "", ""

    cost = np.stack(cost_rows, axis=0)
    T = cost.shape[0]
    raw = "".join(alphabet[int(i)] for i in cost.argmin(axis=1)).rstrip()

    # Viterbi with LM
    uni_lp, bi_lp, _ = _build_lm()
    v = -cost[0] + lm_weight * uni_lp
    ptr = np.full((T, n), -1, dtype=np.int32)
    for t in range(1, T):
        scores = v[:, None] + lm_weight * bi_lp
        best_prev = np.argmax(scores, axis=0)
        v = scores[best_prev, np.arange(n)] - cost[t]
        ptr[t] = best_prev
    end = int(np.argmax(v))
    path = np.zeros(T, dtype=np.int32)
    path[-1] = end
    for t in range(T - 1, 0, -1):
        path[t - 1] = ptr[t, path[t]]
    lm_str = "".join(alphabet[i] for i in path).rstrip()
    return raw, lm_str


def main() -> None:
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
    print(f"{'image':<40}  decoded")
    print("-" * 80)
    for name in images:
        raw, _ = decode(here / name)
        print(f"{name:<40}  {raw}")


if __name__ == "__main__":
    main()
