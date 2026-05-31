# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Recover hidden text (usernames and passwords) from eight pixelated PNG images in the repository root:
`user1_username_pixelated.png`, `user1_password_pixelated.png`, `user2_*`, `user3_*`, `user4_*`.

## Dependencies

Install from `DepixHMM/requirements.txt` (numpy, Pillow, rstr, scikit-learn). Python 3.x. Courier New font must be present at `C:/Windows/Fonts/cour.ttf`.

```
pip install -r DepixHMM/requirements.txt
```

## Running the Decoders

All scripts must be run from the **repository root** (`Pixelated/`).

### 🔍 Analysis / Detection

```bash
# Auto-detect block size + font, then run full detection report
python depixhmm_tools/analysis/detect_params.py

# Block/grid geometry analysis — three-method comparison (variance, FFT, edge)
python depixhmm_tools/analysis/analyze.py
```

### 🔓 Decoders

```bash
# ★ Beam-search decoder (recommended) — default uses calibrated constants
python depixhmm_tools/decoders/solve.py
python depixhmm_tools/decoders/solve.py --auto              # auto-detect all params
python depixhmm_tools/decoders/solve.py --block 4 --size 24 # override individual params
python depixhmm_tools/decoders/solve.py --auto img.png       # single image

# Template-matching + Viterbi + bigram LM decoder
python depixhmm_tools/decoders/template_decoder.py

# Per-alignment templates + Viterbi + LM (colour/RGB)
python depixhmm_tools/decoders/final_decoder.py

# DepixHMM full pipeline decoder (sweeps font size + offset_y)
python depixhmm_tools/decoders/decoder.py

# Custom DepixHMM decoder (tight-crop images, HMM-based)
python depixhmm_tools/decoders/custom_decoder.py
```

### ✅ Validation / Diagnostics

```bash
# Calibrate against a known guess string (finds best font size + x/y offsets)
python depixhmm_tools/validation/calibrate.py

# Verify a rendered guess vs. target image (compute MSE)
python depixhmm_tools/validation/verify.py

# Per-position top-K candidates diagnostic
python depixhmm_tools/validation/inspect_candidates.py
```

> **Backward compatibility:** The old flat paths (e.g. `depixhmm_tools/solve.py`) still work — they forward to the new locations automatically.

## Architecture

### Two-Layer Design

**`DepixHMM/`** — upstream HMM library (do not modify):
- `text_depixelizer/depix_hmm.py`: Entry point `depix_hmm(picture_parameters, training_parameters, img_path)` — trains HMM, evaluates, then decodes a target image.
- `text_depixelizer/HMM/`: `DepixHMM` class wraps `KmeansClusterer`, `HMM`, and `hmm_result_reconstructor`. Training generates synthetic text images, pixelates them, extracts windows, clusters them, then fits start/transition/emission probabilities.
- `text_depixelizer/parameters.py`: `PictureParameters` (font, block_size, pattern, window_size, offset_y, randomize_pixelization_origin_x) and `TrainingParameters` (n_img_train, n_img_test, n_clusters).

**`depixhmm_tools/`** — project-specific decoder scripts (organised into three groups):

| Group | Script | Approach | Notes |
|---|---|---|---|
| analysis | `analysis/detect_params.py` | Block size + font auto-detection | Intra-block variance + min-alphabet MSE |
| analysis | `analysis/analyze.py` | Block geometry diagnostics | Three-method comparison: variance, FFT, edge |
| decoders | `decoders/solve.py` ★ | Beam search, pixel MSE | Best accuracy; `--auto/--block/--size/--font/--x0` |
| decoders | `decoders/template_decoder.py` | Template match + Viterbi + bigram LM | Sweeps font size and alignment |
| decoders | `decoders/final_decoder.py` | Per-alignment templates + Viterbi + LM | Color (RGB), per-alignment offset 0..7 |
| decoders | `decoders/custom_decoder.py` | DepixHMM with tight-crop images | Trains HMM with custom pixelation |
| decoders | `decoders/decoder.py` | Standard DepixHMM pipeline | Sweeps (font_size, offset_y) combos |
| validation | `validation/calibrate.py` | Joint (size, x, y) MSE minimisation | Use when you have a guess string |
| validation | `validation/verify.py` | Render + compare MSE | Validates a candidate answer |
| validation | `validation/inspect_candidates.py` | Per-position top-K | Diagnostic; imports from `template_decoder` |

### Parameter Auto-Detection (`analysis/detect_params.py`)

Two independent block-size methods; font detection via fingerprint scoring:

**Block size — intra-block variance minimisation (`detect_block_size`)**
For each candidate block size `b`, tile the image into `b×b` blocks and compute mean within-block pixel variance. The correct `b` yields variance ≈ 0 (uniform tiles). All proper divisors of the true block size also score 0, so the algorithm takes the **largest near-zero candidate**. FFT autocorrelation (`detect_block_size_fft`) is also available as a cross-check.

**Font — minimum-over-alphabet MSE scoring (`score_font` / `detect_font`)**
For each (font, size) candidate that passes the height constraint (`abs(font_height - image_height) < block_size`), build greyscale block-average fingerprints for every character. At each character position in the image, compute MSE against all fingerprints and take the minimum. Average across positions — the correct font minimises this total. Courier New scores ~650; wrong fonts score ~1200+.

`detect_all(img_path)` runs both detections and returns a dict:
```python
{block_size, font_name, font_path, font_size, advance, left_pad}
```

Candidate fonts in `_DEFAULT_FONT_CANDIDATES` (add more for non-Windows systems). `discover_monospaced_fonts(font_dir)` scans a directory for monospaced fonts automatically.

### Image Properties (empirically confirmed)

| Property | Value |
|---|---|
| Block size | 8 px |
| Grid x-origin | 0 (block edges at 8, 16, 24, …) |
| Left text margin | 8 px |
| Font | Courier New (monospaced), advance = 22 px at size 36 |
| Alphabet | A–Z, a–z, 0–9 (and space for some decoders) |
| Height 38–39 px | 4 full block rows; `custom_decoder.py` uses font size 33 |
| Height 45–46 px | 5 full block rows; `custom_decoder.py` uses font size 39 |

`decoders/solve.py` uses size 36 universally (calibrated via MSE ≈ 0.1 against real images).

### Path Conventions (after reorganisation)

All scripts use `Path(__file__).parent.parent.parent` to reach the repository root (`Pixelated/`) for image files. The DepixHMM library root is resolved as `Path(__file__).parent.parent.parent / "DepixHMM"`.

Cross-group imports use explicit `sys.path` insertion:
- `analysis/analyze.py` → inserts `analysis/` dir, imports `detect_params`
- `decoders/solve.py` → lazy-inserts `analysis/` dir before importing `detect_params`
- `validation/inspect_candidates.py` → inserts `decoders/` dir, imports `template_decoder`

### Decoding Workflow

1. Run `analysis/detect_params.py` (or `analysis/analyze.py`) to confirm block size and font.
2. Run `decoders/solve.py` (beam search) for a fast first-pass result; use `--auto` if parameters are unknown.
3. Use `decoders/template_decoder.py` or `decoders/final_decoder.py` for LM-assisted disambiguation.
4. Use `validation/calibrate.py` with a candidate answer to confirm via MSE ≈ 0.
5. Use `validation/verify.py` to validate the final answer.
