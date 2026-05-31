# DepixHMM Tools — 馬賽克文字還原工具集

> **⚠️ 資安教育用途聲明**  
> 本工具集為學術研究與資安教育目的而開發，旨在揭示馬賽克遮蔽的根本侷限，促進更安全的資料遮蔽實踐。請勿用於未授權的資料存取。

本專案是在 [DepixHMM](https://github.com/beurtschipper/DepixHMM) 開源庫的基礎上，透過 Claude Code (AI 輔助開發) 建構的 Python 工具鏈，用於還原被馬賽克（Pixelation）遮蔽的等寬字型文字截圖。核心解碼器採用 **Beam Search 純像素 MSE 比對**，對等寬字型密碼字串的還原正確率可達 MSE < 0.22。

---

## 為什麼馬賽克可以被破解？

馬賽克並非「刪除」資訊，而是將每個區塊內的像素取**平均值**。只要知道原始字型與方塊大小，就能反向推算出最接近的字元組合。

```
原始像素 → 取平均 → 馬賽克方塊
                 ↑
         這個方向可以被逆推
```

---

## 工具一覽（`depixhmm_tools/`）

| 腳本 | 用途 | 使用時機 |
|------|------|----------|
| `decoders/solve.py` | **主要解碼器**：Beam Search 純像素 MSE | 首選，效果最佳 |
| `analysis/detect_params.py` | 自動偵測方塊大小與字型 | 不知道圖片參數時 |
| `analysis/analyze.py` | 方塊幾何分析（三種方法比較） | 診斷/除錯用 |
| `validation/calibrate.py` | 已知猜測字串時，搜尋最佳 font size / x / y | 驗證參數 |
| `validation/verify.py` | 渲染猜測字串並計算 MSE | 驗證解碼結果 |
| `decoders/template_decoder.py` | 樣板比對 + Viterbi + Bigram 語言模型 | 有語言上下文時 |
| `decoders/final_decoder.py` | 每個對齊偏移的樣板 + Viterbi + LM（彩色） | 彩色截圖 |
| `decoders/decoder.py` | 標準 DepixHMM 流程（掃描 font size + offset_y） | 原始 HMM 方法 |
| `decoders/custom_decoder.py` | 緊密裁切圖片的自訂 HMM | 無白邊的截圖 |
| `validation/inspect_candidates.py` | 每個位置的 Top-K 候選字元 | 除錯用 |

### 解碼方法比較

| 方法 | 優點 | 缺點 |
|------|------|------|
| **Beam Search (solve.py)** | 不假設字元統計關係，對密碼最準確 | 較慢（beam width 越大越慢） |
| HMM + 語言模型 | 對英文單字效果好 | 對密碼有「語言偏置」，會猜成英文字 |
| Viterbi 純像素 | 無語言偏置 | 仍受 HMM 馬可夫性假設制約，不穩定 |

---

## 快速開始

### Clone 專案

本專案使用 Git Submodule 引入 DepixHMM，請使用以下指令一次 clone 完整內容：

```bash
git clone --recurse-submodules https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

若已 clone 但 `DepixHMM/` 是空的，補執行：

```bash
git submodule update --init
```

### 環境需求

| 項目 | 需求 |
|------|------|
| Python | 3.8 以上 |
| 作業系統 | Windows / macOS / Linux（字型路徑自動偵測） |
| 字型 | Courier New 或系統內建等寬字型 |

### 安裝

```bash
pip install -r requirements.txt
```

依賴套件：`numpy`, `Pillow`, `rstr`, `scikit-learn`

### 字型說明

工具會自動偵測作業系統並尋找可用的等寬字型：

| 作業系統 | 尋找字型 |
|----------|---------|
| Windows | `C:/Windows/Fonts/cour.ttf` |
| macOS | `/Library/Fonts/Courier New.ttf` 等 |
| Linux | `/usr/share/fonts/` 下的 FreeMono / DejaVu Mono 等 |

若自動偵測失敗，可用 `--font` 參數手動指定：

```bash
python depixhmm_tools/decoders/solve.py --font /path/to/your/font.ttf
```

### 基本使用流程

所有腳本皆需從**專案根目錄**執行（`Pixelated/`），而非從 `depixhmm_tools/` 內部執行。

**Step 1：分析圖片參數（不確定字型/方塊大小時）**

```bash
python depixhmm_tools/analysis/detect_params.py
```

會輸出偵測到的 `block_size`、字型名稱、字型大小、字元寬度等資訊。

**Step 2：執行 Beam Search 解碼（主要工具）**

```bash
# 使用預設參數（自動偵測字型, block=8）
python depixhmm_tools/decoders/solve.py

# 自動偵測所有參數
python depixhmm_tools/decoders/solve.py --auto

# 手動指定參數
python depixhmm_tools/decoders/solve.py --block 8 --size 36

# 解碼單張圖片
python depixhmm_tools/decoders/solve.py --auto my_screenshot.png
```

**Step 3：驗證結果**

```bash
python depixhmm_tools/validation/verify.py
```

### 進階：逐步診斷

```bash
# 三種方法比對方塊大小與字型
python depixhmm_tools/analysis/analyze.py

# 已知部分答案時，用 calibrate 確認參數
python depixhmm_tools/validation/calibrate.py

# 查看每個字元位置的前 5 個候選
python depixhmm_tools/validation/inspect_candidates.py
```

---

## 工作原理

```
目標截圖（馬賽克圖）
        ↓
  偵測方塊大小（detect_params.py）
        ↓
  偵測字型（detect_params.py）
        ↓
  Beam Search 解碼（solve.py）
  ┌─────────────────────────────────────┐
  │ 逐字元位置：                         │
  │ 渲染字元 → 重新馬賽克化 → 計算 MSE  │
  │ 保留 MSE 最低的 K 條路徑（K=beam）   │
  └─────────────────────────────────────┘
        ↓
  輸出還原文字 + MSE 值
```

### Beam Search 為何優於 HMM？

密碼字元之間沒有統計依存關係（不像英文單字），因此：
- HMM + 語言模型：會把 `TaiwanNumber1` 猜成 `TaiwanNumbex1`（英文偏置）
- Beam Search：只看像素 MSE，完全不假設字元間的統計關係

實驗結果：6 張截圖的 MSE 全部低於 **0.22**，備選答案的 MSE 是最佳解的 4–16 倍。

---

## 使用限制

| 限制 | 說明 |
|------|------|
| **僅支援等寬字型** | Courier New、Consolas、Lucida Console 等；比例字型（Arial、Times New Roman）無法準確還原 |
| **需知道（或能偵測）字型** | 完全未知的自訂字型需手動加入 `_DEFAULT_FONT_CANDIDATES` |
| **字元集限制** | 預設為 A–Z、a–z、0–9；不支援中文、日文、特殊符號（可修改 `ALPHABET` 變數） |
| **Windows 字型路徑** | 字型預設路徑為 `C:/Windows/Fonts/`；Linux/macOS 需修改 `FONT_PATH` |
| **方塊大小需可偵測** | 若截圖經過二次壓縮（JPEG）或縮放，方塊邊界可能模糊，偵測精度下降 |
| **文字需水平排列** | 不支援旋轉、彎曲的文字 |
| **背景需接近純色** | 複雜背景會降低 MSE 比對精度 |

---

## 資安啟示：正確的遮蔽方式

| 遮蔽方式 | 安全性 | 說明 |
|----------|--------|------|
| 馬賽克 | ❌ 不安全 | 保留均值色彩特徵，可被反推 |
| 模糊（Blur） | ❌ 不安全 | 與馬賽克類似，資訊仍被保留 |
| **純色矩形覆蓋** | ✅ 安全 | 覆蓋區域資訊歸零，數學上不可逆 |

**建議做法**：使用截圖工具的「黑色矩形」填充功能覆蓋敏感資料，而非馬賽克或模糊效果。

---

## 專案結構

```
Pixelated/
├── depixhmm_tools/
│   ├── analysis/            # 🔍 分析/偵測
│   │   ├── detect_params.py # 參數自動偵測
│   │   └── analyze.py       # 幾何分析診斷
│   ├── decoders/            # 🔓 解碼器
│   │   ├── solve.py         # ★ 主要解碼器（Beam Search）
│   │   ├── template_decoder.py
│   │   ├── final_decoder.py
│   │   ├── decoder.py
│   │   └── custom_decoder.py
│   └── validation/          # ✅ 驗證/診斷
│       ├── calibrate.py
│       ├── verify.py
│       └── inspect_candidates.py
├── DepixHMM/                # 上游 HMM 函式庫（Git Submodule）
├── requirements.txt
├── LICENSE
└── *.png                    # 示範用馬賽克截圖（虛構帳密）
```

---

## 致謝

- [DepixHMM](https://github.com/beurtschipper/DepixHMM)：上游 HMM 函式庫
- [Depix](https://github.com/spipm/Depix)：最初的開源馬賽克逆推工具（spipm, 2021）
- 本工具鏈透過 [Claude Code](https://github.com/anthropics/claude-code) 輔助開發

---
