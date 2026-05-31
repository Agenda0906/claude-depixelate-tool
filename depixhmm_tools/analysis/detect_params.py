"""
Auto-detection of mosaic parameters: block size and font.

Block size — intra-block variance minimisation
    For the correct block size b, the pixelated image tiles perfectly into b×b
    uniform blocks (every pixel within a block is identical), giving near-zero
    within-block variance. Every wrong candidate crosses block boundaries and
    accumulates variance. We also provide an FFT-autocorrelation alternative.

Font detection — minimum-over-alphabet MSE scoring
    Build greyscale block-average fingerprints for each character rendered in
    a neutral (space) context. For each character position in the target image,
    compute the MSE against every fingerprint and take the minimum. Sum the
    per-position minimums across the whole image — the correct (font, size)
    minimises this total, because its fingerprints match what is actually there.

Usage
-----
    python depixhmm_tools/detect_params.py                 # all user images
    python depixhmm_tools/detect_params.py path/to/img.png
"""
from __future__ import annotations

import platform
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Cross-platform font resolution
# ---------------------------------------------------------------------------

def _platform_font_candidates() -> List[Tuple[str, str]]:
    """Return (name, path) candidates that exist on the current OS."""
    system = platform.system()
    candidates: List[Tuple[str, str]] = []

    if system == "Windows":
        win = Path("C:/Windows/Fonts")
        candidates = [
            ("Courier New",    str(win / "cour.ttf")),
            ("Consolas",       str(win / "consola.ttf")),
            ("Lucida Console", str(win / "lucon.ttf")),
            ("Cascadia Mono",  str(win / "CascadiaMono.ttf")),
            ("Courier",        str(win / "courier.ttf")),
        ]
    elif system == "Darwin":  # macOS
        mac_dirs = [
            Path.home() / "Library/Fonts",
            Path("/Library/Fonts"),
            Path("/System/Library/Fonts"),
            Path("/System/Library/Fonts/Supplemental"),
        ]
        mac_map = [
            ("Courier New",    "Courier New.ttf"),
            ("Courier New",    "CourierNew.ttf"),
            ("Courier",        "Courier.dfont"),
            ("Andale Mono",    "Andale Mono.ttf"),
            ("Menlo",          "Menlo.ttc"),
        ]
        for name, fname in mac_map:
            for d in mac_dirs:
                p = d / fname
                if p.exists():
                    candidates.append((name, str(p)))
                    break
    else:  # Linux
        linux_dirs = [
            Path.home() / ".fonts",
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
        ]
        linux_map = [
            ("Courier New",  "cour.ttf"),
            ("Courier New",  "CourierNew.ttf"),
            ("FreeMono",     "FreeMono.ttf"),
            ("DejaVu Mono",  "DejaVuSansMono.ttf"),
            ("Liberation Mono", "LiberationMono-Regular.ttf"),
        ]
        for name, fname in linux_map:
            for d in linux_dirs:
                matches = list(d.rglob(fname))
                if matches:
                    candidates.append((name, str(matches[0])))
                    break

    # Keep only paths that actually exist
    return [(n, p) for n, p in candidates if Path(p).exists()]


def get_default_font_path() -> str:
    """Return the path to Courier New (or best available monospaced font) on this OS."""
    cands = _platform_font_candidates()
    if cands:
        return cands[0][1]
    raise FileNotFoundError(
        "No suitable monospaced font found. "
        "Please install Courier New or specify --font /path/to/font.ttf"
    )


_DEFAULT_FONT_CANDIDATES: List[Tuple[str, str]] = _platform_font_candidates()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pixelate(arr: np.ndarray, block: int) -> np.ndarray:
    """Block-average (full blocks only). 2-D (grey) or 3-D (RGB) input."""
    h, w = arr.shape[:2]
    nr, nc = h // block, w // block
    if nr == 0 or nc == 0:
        return np.zeros((0, 0), dtype=np.float32)
    a = arr[:nr * block, :nc * block]
    if arr.ndim == 2:
        return a.reshape(nr, block, nc, block).mean(axis=(1, 3)).astype(np.float32)
    return a.reshape(nr, block, nc, block, arr.shape[2]).mean(axis=(1, 3)).astype(np.float32)


def _char_advance(font: ImageFont.FreeTypeFont) -> int:
    """Pixel advance for one character in a monospaced font."""
    return font.getbbox("MM")[2] - font.getbbox("M")[2]


# ---------------------------------------------------------------------------
# 1. Block size detection
# ---------------------------------------------------------------------------

def detect_block_size(
    img_path: Path,
    candidates: Optional[List[int]] = None,
) -> Tuple[int, Dict[int, float]]:
    """
    Detect mosaic block size by minimising mean intra-block pixel variance.

    For the correct block size every tile is uniform → variance ≈ 0.
    Any wrong size straddles real block boundaries and accumulates variance.

    Parameters
    ----------
    img_path   : path to the pixelated PNG
    candidates : list of block sizes to evaluate;
                 defaults to [2,3,4,5,6,7,8,9,10,12,16,20,24,32]

    Returns
    -------
    (best_block_size, {block_size: variance_score})
    """
    if candidates is None:
        candidates = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 32]

    arr = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
    h, w = arr.shape

    scores: Dict[int, float] = {}

    for b in candidates:
        nr, nc = h // b, w // b
        if nr < 2 or nc < 2:
            continue
        tiles = arr[:nr * b, :nc * b].reshape(nr, b, nc, b)
        block_means = tiles.mean(axis=(1, 3), keepdims=True)
        var = float(((tiles - block_means) ** 2).mean())
        scores[b] = var

    if not scores:
        return candidates[0], scores

    # Any true divisor of the real block size will also have near-zero variance
    # (sub-blocks are subsets of uniform tiles). The TRUE block size is the
    # LARGEST candidate with near-zero variance, because the next larger size
    # will straddle two different real blocks and accumulate variance.
    min_var = min(scores.values())
    zero_threshold = max(min_var * 4 + 0.5, 1.0)   # robust to minor JPEG noise
    near_zero = [b for b, v in scores.items() if v <= zero_threshold]

    if near_zero:
        best_b = max(near_zero)                     # largest near-zero = true block
    else:
        best_b = min(scores, key=scores.get)        # fallback: plain minimum

    return best_b, scores


def detect_block_size_fft(img_path: Path) -> Tuple[int, np.ndarray]:
    """
    Alternative block-size detector: FFT autocorrelation of the column-diff signal.

    Block boundaries produce spikes in the absolute column-mean derivative.
    The autocorrelation of those spikes peaks at multiples of the block size.

    Returns
    -------
    (block_size, normalised_autocorrelation_array)
    """
    arr = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
    col_mean = arr.mean(axis=0)
    diff = np.abs(np.diff(col_mean))          # spike at every block boundary
    n = len(diff)
    f = np.fft.rfft(diff, n=2 * n)
    ac = np.fft.irfft(f * f.conj())[:n]
    ac_norm = ac / max(float(ac[0]), 1e-9)

    best_lag, best_val = 2, -1.0
    for lag in range(2, min(33, n // 2)):
        if ac_norm[lag] > best_val:
            best_val = ac_norm[lag]
            best_lag = lag

    return best_lag, ac_norm


# ---------------------------------------------------------------------------
# 2. Font discovery
# ---------------------------------------------------------------------------

def discover_monospaced_fonts(
    font_dir: str = "C:/Windows/Fonts",
    extra_paths: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """
    Scan a font directory and return (stem_name, path) for every monospaced font.

    A font is classified as monospaced when advance("M") == advance("i") ±1 px.
    """
    roots = [Path(font_dir)]
    if extra_paths:
        for p in extra_paths:
            pp = Path(p)
            roots.append(pp if pp.is_dir() else pp.parent)

    found: List[Tuple[str, str]] = []
    seen: set = set()
    for root in roots:
        for fp in sorted(root.glob("*.ttf")) + sorted(root.glob("*.otf")):  # type: ignore[operator]
            if str(fp) in seen:
                continue
            seen.add(str(fp))
            try:
                font = ImageFont.truetype(str(fp), 20)
                adv_M = font.getbbox("MM")[2] - font.getbbox("M")[2]
                adv_i = font.getbbox("ii")[2] - font.getbbox("i")[2]
                if abs(adv_M - adv_i) <= 1 and adv_M > 0:
                    found.append((fp.stem, str(fp)))
            except Exception:
                continue

    return found


# ---------------------------------------------------------------------------
# 3. Character fingerprints
# ---------------------------------------------------------------------------

def build_fingerprints(
    font_path: str,
    font_size: int,
    block_size: int,
    n_rows_target: int,
    alphabet: str = string.ascii_letters + string.digits,
    left_pad: int = 8,
    n_context: int = 3,
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Build a greyscale block-average fingerprint for each character.

    Each character is rendered flanked by spaces so the block averages are
    dominated by the character itself and not by distant neighbours.

    Returns
    -------
    fingerprints : dict  char → ndarray shape (n_rows, n_char_blocks)
    advance      : int   pixel advance per character
    """
    font = ImageFont.truetype(font_path, font_size)
    ascent, descent = font.getmetrics()
    H = ascent + descent
    advance = _char_advance(font)
    if advance <= 0:
        return {}, 0

    context = " " * n_context
    # +2 to capture partial bleed from left/right neighbours
    n_char_blocks = (advance + block_size - 1) // block_size + 2

    fingerprints: Dict[str, np.ndarray] = {}
    for ch in alphabet:
        text = context + ch + context
        W_render = left_pad + len(text) * advance + left_pad
        img = Image.new("L", (W_render, H), 255)
        ImageDraw.Draw(img).text((left_pad, 0), text, font=font, fill=0)
        arr = np.asarray(img, dtype=np.float32)
        tb = _pixelate(arr, block_size)
        if tb.size == 0:
            continue

        # First block column that belongs to the target character
        char_left_px = left_pad + n_context * advance
        char_first_block = char_left_px // block_size
        n_take = min(n_char_blocks, tb.shape[1] - char_first_block)
        if n_take < 1:
            continue

        sig = tb[: min(tb.shape[0], n_rows_target), char_first_block : char_first_block + n_take]
        fingerprints[ch] = sig.astype(np.float32)

    return fingerprints, advance


# ---------------------------------------------------------------------------
# 4. Font scoring
# ---------------------------------------------------------------------------

def score_font(
    img_path: Path,
    block_size: int,
    font_path: str,
    font_size: int,
    left_pad: int = 8,
    alphabet: str = string.ascii_letters + string.digits,
) -> float:
    """
    Score how well (font_path, font_size) explains the pixelated image.

    Algorithm
    ---------
    For each character position in the image, find the fingerprint
    (across the whole alphabet) that minimises the MSE against the target
    blocks at that position. Average the per-position minimums.
    Lower score  ←→  better font match.

    Returns float("inf") when the font/size is geometrically incompatible
    (height mismatch) or the font file is missing.
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        return float("inf")

    img = Image.open(img_path).convert("L")
    target = np.asarray(img, dtype=np.float32)
    H, W = target.shape

    ascent, descent = font.getmetrics()
    font_h = ascent + descent
    # Height must match within one block (strictly less than block_size margin).
    # Using >= excludes fonts whose height overshoots the image by a full block.
    if abs(font_h - H) >= block_size:
        return float("inf")

    n_rows_target = H // block_size
    tb = _pixelate(target, block_size)
    if tb.size == 0:
        return float("inf")
    n_rows, n_cols = tb.shape

    fingerprints, advance = build_fingerprints(
        font_path, font_size, block_size, n_rows_target,
        alphabet=alphabet, left_pad=left_pad,
    )
    if not fingerprints or advance <= 0:
        return float("inf")

    chars = list(fingerprints.keys())
    fps = list(fingerprints.values())
    n_fp_rows = fps[0].shape[0]
    n_fp_cols = max(fp.shape[1] for fp in fps)

    # Stack into a single array for fast vectorised comparison
    fp_array = np.full((len(chars), n_fp_rows, n_fp_cols), 255.0, dtype=np.float32)
    for i, fp in enumerate(fps):
        r, c = fp.shape
        fp_array[i, :r, :c] = fp

    n_chars = max(1, (W - left_pad) // advance)
    total_mse = 0.0
    n_valid = 0

    for pos in range(n_chars):
        char_left = left_pad + pos * advance
        first_block = char_left // block_size
        n_take = min(n_fp_cols, n_cols - first_block)
        if n_take < 1:
            continue

        test_seg = tb[:n_fp_rows, first_block : first_block + n_take]   # (n_rows, n_take)
        fp_seg   = fp_array[:, :, :n_take]                               # (n_chars, n_rows, n_take)
        diff = fp_seg - test_seg[None, :, :]                             # broadcast
        mse_per_char = (diff ** 2).mean(axis=(1, 2))                     # (n_chars,)
        total_mse += float(mse_per_char.min())
        n_valid += 1

    return total_mse / n_valid if n_valid > 0 else float("inf")


# ---------------------------------------------------------------------------
# 5. Font detection
# ---------------------------------------------------------------------------

def detect_font(
    img_path: Path,
    block_size: int,
    candidate_fonts: Optional[List[Tuple[str, str]]] = None,
    size_range: range = range(18, 55),
    left_pad: int = 8,
    alphabet: str = string.ascii_letters + string.digits,
    verbose: bool = False,
) -> Optional[Tuple[str, str, int, int]]:
    """
    Find the best (font_name, font_path, font_size, advance) for the image.

    Sweeps every (font, size) pair, skipping combinations whose height is
    incompatible with the image. Returns the pair with the lowest MSE score.

    Returns None if no compatible font is found.
    """
    if candidate_fonts is None:
        candidate_fonts = [(n, p) for n, p in _DEFAULT_FONT_CANDIDATES if Path(p).exists()]

    best: Optional[Tuple[str, str, int, int]] = None
    best_score = float("inf")

    for font_name, font_path in candidate_fonts:
        for font_size in size_range:
            sc = score_font(img_path, block_size, font_path, font_size,
                            left_pad=left_pad, alphabet=alphabet)
            if sc == float("inf"):
                continue
            if verbose:
                try:
                    fnt = ImageFont.truetype(font_path, font_size)
                    adv = _char_advance(fnt)
                except Exception:
                    adv = -1
                print(f"    {font_name:20s}  size={font_size:2d}  adv={adv:2d}  score={sc:.3f}")
            if sc < best_score:
                best_score = sc
                try:
                    fnt = ImageFont.truetype(font_path, font_size)
                    adv = _char_advance(fnt)
                except Exception:
                    adv = 0
                best = (font_name, font_path, font_size, adv)

    return best


# ---------------------------------------------------------------------------
# 6. Combined auto-detection entry point
# ---------------------------------------------------------------------------

def detect_all(
    img_path: Path,
    candidate_fonts: Optional[List[Tuple[str, str]]] = None,
    block_candidates: Optional[List[int]] = None,
    left_pad: int = 8,
    verbose: bool = True,
) -> dict:
    """
    Auto-detect block_size, font, font_size, and advance for a pixelated image.

    Returns
    -------
    dict with keys:
        block_size, font_name, font_path, font_size, advance, left_pad
    """
    img_path = Path(img_path)

    # --- Step 1: block size ---
    best_b, scores = detect_block_size(img_path, block_candidates)
    b_fft, _ = detect_block_size_fft(img_path)

    if verbose:
        top5 = sorted(scores.items(), key=lambda kv: kv[1])[:5]
        print(f"  Block variance scores (top-5): "
              f"{[(b, round(s, 3)) for b, s in top5]}")
        print(f"  Block size  variance-method={best_b}  fft-method={b_fft}")
        if best_b != b_fft:
            print(f"  (methods disagree -- using variance result {best_b})")

    # --- Step 2: font ---
    if verbose:
        print(f"  Scanning fonts for block_size={best_b} ...")
    font_info = detect_font(
        img_path, best_b,
        candidate_fonts=candidate_fonts,
        left_pad=left_pad,
        verbose=verbose,
    )

    if font_info is None:
        if verbose:
            print("  Font detection found no match -- using Courier New 36 defaults")
        return {
            "block_size": best_b,
            "font_name":  "Courier New",
            "font_path":  "C:/Windows/Fonts/cour.ttf",
            "font_size":  36,
            "advance":    22,
            "left_pad":   left_pad,
        }

    font_name, font_path, font_size, advance = font_info
    if verbose:
        print(f"  -> font={font_name}  size={font_size}  advance={advance} px")

    return {
        "block_size": best_b,
        "font_name":  font_name,
        "font_path":  font_path,
        "font_size":  font_size,
        "advance":    advance,
        "left_pad":   left_pad,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    HERE = Path(__file__).parent.parent.parent
    _default_images = [
        "user1_username_pixelated.png", "user1_password_pixelated.png",
        "user2_username_pixelated.png", "user2_password_pixelated.png",
        "user3_username_pixelated.png", "user3_password_pixelated.png",
        "user4_username_pixelated.png", "user4_password_pixelated.png",
    ]

    targets = sys.argv[1:] if len(sys.argv) > 1 else _default_images

    for name in targets:
        p = HERE / name if not Path(name).is_absolute() else Path(name)
        if not p.exists():
            print(f"[skip] {name} not found")
            continue
        print(f"\n=== {name} ===")
        params = detect_all(p, verbose=True)
        print(f"  Result -> {params}")
