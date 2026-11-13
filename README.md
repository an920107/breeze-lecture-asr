# Breeze-ASR-25 課程錄音轉錄工具

用 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) + CTranslate2 載入
MediaTek **Breeze-ASR-25** 模型，把音訊檔或網路影片（YouTube 等）轉成
**SRT 字幕**與**純文字稿**。

針對台灣資工系課程最佳化：中英夾雜的專有名詞 prompt / hotwords、VAD 靜音切分、
GPU 批次推論，以及 VRAM 不足時自動降低 batch size。

---

## 需求

- **Linux + NVIDIA GPU**：程式固定以 `device="cuda"` 載入模型，不支援純 CPU。
  8GB VRAM 即可執行（`float16` 約佔 3.8GB）。
- **Python 3.12**（見 `mise.toml`；`pyproject.toml` 允許 3.10–3.12）
- **[uv](https://docs.astral.sh/uv/)**：建立虛擬環境與安裝依賴
- **[mise](https://mise.jdx.dev/)**（選用）：自動安裝正確的 Python 版本
- 網路連線：首次需下載模型；處理 YouTube 連結時也需要
- 選用：`deno` / `node` / `bun`（YouTube JS challenge）、已登入的瀏覽器（抓 cookies）

確認 GPU 可用：

```bash
nvidia-smi
```

---

## 初始化

### 1. 安裝 Python 與依賴

在專案根目錄執行：

```bash
# 有 mise 的話，依 mise.toml 安裝 Python 3.12
mise install

# 建立 .venv 並依 uv.lock 安裝所有依賴
uv sync
```

沒有 mise 時，改用：

```bash
uv sync --python 3.12
```

`uv sync` 會一併安裝 `nvidia-cublas-cu12`，這是 CTranslate2 在 Linux 上
執行 CUDA 推論所需的 `libcublas.so.12`（程式啟動時會自動預先載入）。

### 2. 準備模型（必要）

模型資料夾 `breeze-asr-25-ct2/` 約 3GB，已被 `.gitignore` 排除，**不會隨專案提供**。
請從官方 Hugging Face 模型轉換成 CTranslate2 格式：

```bash
uv run ct2-transformers-converter \
  --model MediaTek-Research/Breeze-ASR-25 \
  --output_dir breeze-asr-25-ct2 \
  --quantization float16 \
  --copy_files tokenizer.json preprocessor_config.json
```

- 需要額外約 6GB 的磁碟與記憶體空間做轉換，第一次會下載約 3GB 的權重。
- 轉換完成後，`breeze-asr-25-ct2/` 應包含：
  `config.json`、`model.bin`、`tokenizer.json`、`vocabulary.json`、`preprocessor_config.json`。
- 若你已經有轉好的模型，直接放到專案根目錄的 `./breeze-asr-25-ct2/` 即可。

### 3. 驗證

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

應印出 `True`。

---

## 執行

> 請在**專案根目錄**執行，因為模型路徑是相對於目前工作目錄的 `./breeze-asr-25-ct2`。

最簡單的用法：

```bash
uv run python main.py -i path/to/lecture.mp3
```

會在目前目錄產生 `lecture.srt` 與 `lecture.txt`。

### 常見範例

```bash
# 指定輸出目錄
uv run python main.py -i lecture.m4a -o output/

# 一次轉錄多個檔案（批次；此時不能用 -s/-t，請用 -o）
uv run python main.py -i a.mp3 b.mp3 -o output/

# 直接轉錄 YouTube（需要 yt-dlp，已包含在依賴中）
uv run python main.py -i "https://www.youtube.com/watch?v=XXXXXXXXXXX" -o output/

# 需要登入才看得到的影片：從瀏覽器讀 cookies
uv run python main.py -i "https://www.youtube.com/watch?v=XXXXXXXXXXX" \
  --cookies-from-browser brave

# 先看看有哪些格式可用（不下載、不轉錄）
uv run python main.py -i "https://www.youtube.com/watch?v=XXXXXXXXXXX" --list-formats

# 英文課程（預設語言是 zh）
uv run python main.py -i lecture.mp3 --language en

# VRAM 吃緊時：改用 int8 精度並縮小 batch
uv run python main.py -i lecture.mp3 --compute-type int8_float16 --batch-size 1
```

### 常用參數

| 參數 | 說明 |
| --- | --- |
| `-i, --input` | 音訊檔路徑或網路影片 URL，可一次給多個（必填） |
| `-o, --output-dir` | SRT/TXT 輸出目錄（預設：目前目錄） |
| `-s, --srt` | 指定 SRT 檔名（僅限單一輸入） |
| `-t, --txt` | 指定純文字檔名（僅限單一輸入） |
| `--language` | 辨識語言，`zh`（預設）、`en`，或 `auto` 自動偵測 |
| `--batch-size` | 批次推論大小（預設 4）；VRAM 不足會自動減半重試 |
| `--compute-type` | 模型精度：`float16`（預設）、`int8_float16`、`int8`、`float32` |
| `--cookies-from-browser` | 例如 `brave`、`chrome`、`brave+GNOMEKEYRING:Default` |
| `--cookies` | Netscape 格式 cookies 檔 |
| `--format` | 自訂 yt-dlp 格式選擇，例如 `bestaudio/best` |
| `--list-formats` | 列出可用格式後結束 |
| `--extractor-args` | 透傳給 yt-dlp，例如 `'youtube:player_client=tv'` |
| `--js-runtime` | 指定 yt-dlp 用的 JS runtime，例如 `node:/usr/bin/node` |

完整說明：

```bash
uv run python main.py -h
```

### 輸出

- `*.srt`：UTF-8 字幕檔，可直接匯入播放器或剪輯軟體。
- `*.txt`：逐行純文字稿，方便再餵給 LLM 整理筆記。
- 執行過程會即時印出時間軸與辨識文字，結束時顯示總耗時與 RTFx（即時倍率）。

---

## 常見問題

**`libcublas.so.12` 找不到 / CUDA 相關錯誤**
請確認是用 `uv run` 執行（使用專案的 `.venv`），並先跑過 `uv sync`。
`main.py` 已內建預先載入 pip 版 CUDA 12 函式庫的邏輯。

**`CUDA out of memory`**
程式會自動把 batch size 減半重試。也可以主動降低：
`--compute-type int8_float16 --batch-size 1`。

**YouTube 要求登入（`Sign in to confirm...`）**
加上 `--cookies-from-browser brave`（或 `chrome`），或改用 `--cookies cookies.txt`。

**只拿到 storyboard / 找不到可用格式**
通常是 YouTube 要求 PO token。可依序嘗試：

```bash
uv run python main.py -i "URL" --extractor-args 'youtube:formats=missing_pot'
uv run python main.py -i "URL" --extractor-args 'youtube:player_client=tv'
# 或安裝 PO token provider：
uv pip install bgutil-ytdlp-pot-provider
```

**`failed to load cookies`**
確認瀏覽器已登入 YouTube；GNOME 環境需 `secretstorage`（`uv sync` 已安裝），
必要時明確指定 keyring 與 profile：
`--cookies-from-browser brave+GNOMEKEYRING:Default`。

**模型載入失敗 / 找不到模型**
確認 `./breeze-asr-25-ct2/` 存在且包含 `model.bin`，並從專案根目錄執行。
若模型放在別處，請修改 `main.py` 中的 `MODEL_PATH`。

**YouTube 下載很慢或失敗**
確認網路可連外，並安裝 `deno`、`node` 或 `bun` 其中之一；
程式會自動偵測，也可用 `--js-runtime` 指定。
