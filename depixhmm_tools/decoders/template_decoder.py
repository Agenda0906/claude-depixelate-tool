"""
Template-matching decoder for pixelated monospaced text, augmented with
English bigram statistics as a soft language prior.

Pipeline
--------
1. Per font size (`ad`-rows must match image rows) and per pixelization-grid
   alignment 0..block_size-1, render every alphabet character in a centred
   neighbour context and capture its block-averaged signature.
2. Slide the templates over the test image one character advance at a time;
   per position compute squared-RGB distance to every candidate character.
3. Run a Viterbi pass with English unigram + bigram log-probabilities over
   the candidate cost matrix to bias decoding toward plausible English.
4. Pick the (size, alignment) with the best combined Viterbi score.
"""
from __future__ import annotations

import math
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# ---------- rendering helpers ----------

def render_text(text: str, font: ImageFont.FreeTypeFont, left_pad: int = 0) -> Image.Image:
    ascent, descent = font.getmetrics()
    bbox = font.getbbox(text)
    width = max(bbox[2], 1) + left_pad
    img = Image.new("RGB", (width, ascent + descent), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((left_pad, 0), text, font=font, fill=(0, 0, 0))
    return img


def char_advance(font: ImageFont.FreeTypeFont) -> int:
    return font.getbbox("MM")[2] - font.getbbox("M")[2]


def block_columns(arr: np.ndarray, block_size: int) -> np.ndarray:
    h, w, _ = arr.shape
    n_rows = h // block_size
    n_cols = w // block_size
    arr = arr[: n_rows * block_size, : n_cols * block_size]
    return arr.reshape(n_rows, block_size, n_cols, block_size, 3).mean(axis=(1, 3))


def render_glyph_columns(
    font: ImageFont.FreeTypeFont,
    alphabet: List[str],
    block_size: int,
    alignment: int,
) -> np.ndarray:
    """(n_chars, n_per, n_rows, 3) — block-averaged colour signature per char."""
    advance = char_advance(font)
    h_px = sum(font.getmetrics())
    n_rows = h_px // block_size
    n_per = (advance + block_size - 1) // block_size + 1
    tpl = np.zeros((len(alphabet), n_per, n_rows, 3), dtype=np.float32)

    for ci, ch in enumerate(alphabet):
        text = "   " + ch + "   "
        img = render_text(text, font, left_pad=alignment)
        arr = np.asarray(img, dtype=np.float32)
        blocks = block_columns(arr, block_size)
        ce_left = alignment + 3 * advance
        first_block = ce_left // block_size
        cols = blocks[:, first_block : first_block + n_per, :]
        tpl[ci] = cols.transpose(1, 0, 2)
    return tpl


# ---------- bigram language model ----------

def _common_words() -> List[str]:
    # very small built-in list - good enough for soft bias toward English
    return [
        "the","be","to","of","and","a","in","that","have","I","it","for","not",
        "on","with","he","as","you","do","at","this","but","his","by","from",
        "they","we","say","her","she","or","an","will","my","one","all","would",
        "there","their","what","so","up","out","if","about","who","get","which",
        "go","me","when","make","can","like","time","no","just","him","know",
        "take","people","into","year","your","good","some","could","them","see",
        "other","than","then","now","look","only","come","its","over","think",
        "also","back","after","use","two","how","our","work","first","well","way",
        "even","new","want","because","any","these","give","day","most","us",
        "happy","birthday","monday","tuesday","wednesday","thursday","friday",
        "saturday","sunday","january","february","march","april","may","june",
        "july","august","september","october","november","december","number",
        "party","printer","pairs","pair","numbers","mister","missus","welcome",
        "secret","password","login","user","admin","root","guest","test",
        "name","sign","code","key","pin","data","mail","file","page","line",
        "team","game","band","club","talk","ride","walk","beat","heat","door",
        "make","fast","slow","high","love","care","fire","mind","mark","jump",
        "play","take","help","hope","wish","best","luck","brave","bold","calm",
        "love","life","live","loud","loose","lose","laser","lever","later","letter",
        "boss","boys","girls","kids","mom","dad","son","wife","child","baby",
        "world","money","power","time","place","point","right","left","up","down",
        "happy birthday","new year","good morning","good night","thank you",
    ]


def _build_lm(extra_alphabet: str = "") -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    """Unigram + bigram log probabilities. Includes a leading/trailing space token."""
    words = _common_words()
    corpus_tokens: List[str] = []
    for w in words:
        corpus_tokens.append(" ")
        corpus_tokens.extend(list(w))
    corpus_tokens.append(" ")

    uni: Dict[str, int] = {}
    bi: Dict[Tuple[str, str], int] = {}
    for i, c in enumerate(corpus_tokens):
        uni[c] = uni.get(c, 0) + 1
        if i + 1 < len(corpus_tokens):
            nxt = corpus_tokens[i + 1]
            key = (c, nxt)
            bi[key] = bi.get(key, 0) + 1

    # build log probs with Laplace smoothing across alphabet
    chars = sorted(set(list(string.ascii_letters + string.digits + " ") + list(extra_alphabet)))
    alpha = 0.5
    uni_total = sum(uni.values()) + alpha * len(chars)
    uni_lp = {c: math.log((uni.get(c, 0) + alpha) / uni_total) for c in chars}
    # case-insensitive bigram (uppercase folded into lowercase)
    bi_lp: Dict[Tuple[str, str], float] = {}
    for a in chars:
        # marginal P(. | a)
        denom = sum(bi.get((a, b), 0) for b in chars) + alpha * len(chars)
        for b in chars:
            num = bi.get((a, b), 0) + alpha
            bi_lp[(a, b)] = math.log(num / denom)
    return uni_lp, bi_lp


# ---------- decoding ----------

def decode_image(
    image_path: Path,
    font: ImageFont.FreeTypeFont,
    alphabet: List[str],
    uni_lp: Dict[str, float],
    bi_lp: Dict[Tuple[str, str], float],
    block_size: int = BLOCK_SIZE,
    lm_weight: float = 800.0,
) -> Optional[Tuple[str, float, int]]:
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    blocks = block_columns(arr, block_size)
    n_rows, n_cols, _ = blocks.shape
    advance = char_advance(font)
    n_chars_alpha = len(alphabet)

    char_to_idx = {c: i for i, c in enumerate(alphabet)}
    bi_matrix = np.full((n_chars_alpha, n_chars_alpha), -1e9, dtype=np.float32)
    for a, ai in char_to_idx.items():
        for b, bi_i in char_to_idx.items():
            bi_matrix[ai, bi_i] = bi_lp.get((a.lower() if a.isalpha() else a,
                                             b.lower() if b.isalpha() else b),
                                             -10.0)
    uni_vec = np.array(
        [uni_lp.get(c.lower() if c.isalpha() else c, -10.0) for c in alphabet],
        dtype=np.float32,
    )

    best = None
    for alignment in range(block_size):
        tpl = render_glyph_columns(font, alphabet, block_size, alignment)
        n_per = tpl.shape[1]
        max_chars = (n_cols * block_size - alignment) // advance
        if max_chars <= 2:
            continue

        # collect cost matrix per position: shape (T, n_chars)
        rows: List[np.ndarray] = []
        for ci in range(max_chars):
            ce_left_px = alignment + ci * advance
            first_block_col = ce_left_px // block_size
            if first_block_col + n_per > n_cols:
                break
            test = blocks[:, first_block_col : first_block_col + n_per, :].transpose(1, 0, 2)
            diff = tpl - test[None, :, :, :]
            sq = (diff ** 2).sum(axis=-1).sum(axis=-1)  # (n_chars, n_per)
            weights = np.ones(n_per, dtype=np.float32)
            weights[0] = 0.3
            weights[-1] = 0.3
            costs = (sq * weights[None, :]).sum(axis=-1)
            rows.append(costs.astype(np.float32))
        if not rows:
            continue
        cost = np.stack(rows, axis=0)  # (T, n_chars)
        T = cost.shape[0]

        # Viterbi: minimise (img_cost) - lm_weight * log_prob
        v = -cost[0] / 1.0 + lm_weight * uni_vec
        ptr = np.full((T, n_chars_alpha), -1, dtype=np.int32)
        for t in range(1, T):
            # transition from each prev state to each next state
            scores = v[:, None] + lm_weight * bi_matrix  # (n_prev, n_next)
            best_prev = np.argmax(scores, axis=0)
            v = scores[best_prev, np.arange(n_chars_alpha)] - cost[t]
            ptr[t] = best_prev

        end_state = int(np.argmax(v))
        path = np.zeros(T, dtype=np.int32)
        path[-1] = end_state
        for t in range(T - 1, 0, -1):
            path[t - 1] = ptr[t, path[t]]
        decoded = "".join(alphabet[i] for i in path).rstrip(" ")
        # raw cost (image-only) as secondary objective
        img_cost = float(sum(cost[t, path[t]] for t in range(T)))
        viterbi_score = float(v[end_state])
        if best is None or viterbi_score > best[1]:
            best = (decoded, viterbi_score, alignment, img_cost)
    return best


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

    alphabet = list(string.ascii_letters + string.digits + " ")
    uni_lp, bi_lp = _build_lm()

    for name in images:
        path = here / name
        with Image.open(path) as img:
            h = img.size[1]
        h_rows = h // BLOCK_SIZE
        best_overall = None
        for size in range(18, 50):
            font = ImageFont.truetype(FONT_PATH, size)
            ad = sum(font.getmetrics())
            if ad // BLOCK_SIZE != h_rows:
                continue
            result = decode_image(path, font, alphabet, uni_lp, bi_lp)
            if result is None:
                continue
            decoded, score, alignment, img_cost = result
            if best_overall is None or score > best_overall[1]:
                best_overall = (decoded, score, alignment, size, img_cost)
        if best_overall is None:
            print(f"{name}  NO FONT MATCH")
            continue
        decoded, score, alignment, size, img_cost = best_overall
        print(f"{name}  size={size:2d}  align={alignment}  img={img_cost:.0f}  "
              f"viterbi={score:.0f}  -> {decoded!r}")


if __name__ == "__main__":
    main()
