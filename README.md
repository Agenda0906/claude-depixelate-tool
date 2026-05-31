# AI 輔助還原馬賽克：以 Claude Code 重構 DepixHMM 結合 Beam Search 演算法的實踐與資安啟示

> **⚠️ 資安教育用途聲明**  
> 本工具集為學術研究與資安目的而開發，旨在揭示馬賽克遮蔽的侷限性，促進更安全的資料遮蔽實踐。請勿用於未授權的資料存取。

本專案以 [DepixHMM](https://github.com/JonasSchatz/DepixHMM) 為基礎，透過 [Claude Code](https://github.com/anthropics/claude-code)（AI 驅動的命令列開發工具）輔助，建構出一套改良版 Python 工具鏈，用於還原被馬賽克遮蔽的等寬字型文字截圖。核心解碼器採用 **Beam Search 純像素 MSE 比對**，實驗中 MSE 全部低於 0.22。

---

## 為什麼馬賽克可以被破解？

馬賽克並非「刪除」資訊，而是將每個區塊內的像素取**平均值**。只要知道原始字型與方塊大小，就能反向推算出最接近的字元組合。

```
原始像素 → 取平均 → 馬賽克方塊
                 ↑
         這個方向可以被逆推
```

---

## 這個工具是怎麼做出來的

### 第一步：讓 Claude Code 學習 DepixHMM

首先在 `Pixelated/` 資料夾內啟動 Claude Code，指示它閱讀 DepixHMM 的原始碼並生成中文說明，逐步理解其演算架構：

- 馬賽克的數學本質（均值降採樣）
- de Bruijn 序列參考圖比對機制
- 隱藏馬可夫模型（HMM）與 Viterbi 解碼

### 第二步：Claude Code 識別並修正原始碼缺陷

Claude Code 在執行過程中自動發現並修正了 DepixHMM 中的 6 項問題，涵蓋：

| 類別 | 問題 |
|------|------|
| 演算法錯誤 | Viterbi off-by-one |
| 數值穩定性 | 缺少 Log-Space 計算，直接相乘導致下溢 |
| 環境相依性 | 背景色偵測失敗、EXIF 方向未校正 |
| 多語言支援 | 中文字元不支援 |

### 第三步：嘗試三種解碼架構，最終選定 Beam Search

Claude Code 先後嘗試了三種方法，最終確立以 **Beam Search 純像素 MSE** 為最佳解：

| 解碼方案 | 問題 | 採用 |
|----------|------|------|
| HMM + 語言模型 | 對密碼字串有「語言偏置」，把密碼當英文單字猜 | ❌ |
| Viterbi 純像素 | 仍受 HMM 馬可夫性假設制約，密碼字元間無統計依存關係，結果不穩定 | ❌ |
| **Beam Search 純像素 MSE** | 完全脫離 HMM 框架，以純視覺 MSE 為唯一標準 | ✅ |

Beam Search 不假設任何字元間的統計關係，在每個位置渲染所有候選字元並重新馬賽克化，保留 MSE 最低的 K 條路徑，走完全程取最佳解。

### 第四步：建構工具鏈

Claude Code 根據以上研究，自行撰寫了 `depixhmm_tools/` 內的所有腳本，並整合 Claude API 進行語意分析，形成完整的「馬賽克逆向攻擊鏈」展示。

---

## 工具一覽（`depixhmm_tools/`）

| 腳本 | 用途 |
|------|------|
| `decoders/solve.py` ★ | **主要解碼器**：Beam Search 純像素 MSE |
| `analysis/detect_params.py` | 自動偵測方塊大小與字型 |
| `analysis/analyze.py` | 方塊幾何分析（三種方法比較） |
| `validation/calibrate.py` | 已知猜測字串時，搜尋最佳 font size / x / y |
| `validation/verify.py` | 渲染猜測字串並計算 MSE |
| `decoders/template_decoder.py` | 樣板比對 + Viterbi + Bigram 語言模型 |
| `decoders/final_decoder.py` | 每個對齊偏移的樣板 + Viterbi + LM（彩色） |
| `decoders/decoder.py` | 標準 DepixHMM 流程（掃描 font size + offset_y） |
| `decoders/custom_decoder.py` | 緊密裁切圖片的自訂 HMM |
| `validation/inspect_candidates.py` | 每個位置的 Top-K 候選字元（除錯用） |

---

## 使用限制

| 限制 | 說明 |
|------|------|
| **僅支援等寬字型** | Courier New、Consolas、Lucida Console 等；比例字型（Arial、Times New Roman）無法準確還原 |
| **需知道（或能偵測）字型** | 完全未知的自訂字型需手動加入 `_DEFAULT_FONT_CANDIDATES` |
| **字元集限制** | 預設為 A–Z、a–z、0–9；不支援中文、日文、特殊符號（可修改 `ALPHABET` 變數） |
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
│   │   ├── detect_params.py
│   │   └── analyze.py
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
├── DepixHMM/                # 上游 HMM 函式庫
├── requirements.txt
├── LICENSE
└── *.png                    # 示範用馬賽克截圖（虛構帳密）
```

---

## 致謝

- [DepixHMM](https://github.com/JonasSchatz/DepixHMM)：上游 HMM 函式庫（Jonas Schatz）
- [Depix](https://github.com/spipm/Depix)：最初的開源馬賽克逆推工具（spipm, 2021）
- 本工具鏈透過 [Claude Code](https://github.com/anthropics/claude-code) 輔助開發

---
