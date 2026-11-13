import argparse
import ctypes
import gc
import glob
import importlib.util
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from faster_whisper import BatchedInferencePipeline, WhisperModel

MODEL_PATH = "./breeze-asr-25-ct2"

# 針對資工系課程的語境引導與專有名詞權重提升
CS_PROMPT = "這是一堂台灣大學資訊工程學系的課程錄音，包含中英夾雜的專有名詞與程式碼講解。"
CS_HOTWORDS = (
    "compiler, LLVM, IR, RISC-V, pointer, mutex, deadlock, thread, "
    "process, kernel, Linux, Rust, C++, algorithm, dynamic programming, "
    "time complexity, cache miss, page fault, context switch, register"
)

# 拿不到音軌時的排查建議 (YouTube 要求 PO token 的典型症狀)
CLIENT_HINT = (
    "若所有 client 都只拿到 storyboard，通常是 YouTube 要求 PO token：\n"
    "  1) 先試 --extractor-args 'youtube:formats=missing_pot'\n"
    "  2) 裝 PO token provider 外掛：uv pip install bgutil-ytdlp-pot-provider\n"
    "  3) 或手動帶入：--extractor-args 'youtube:po_token=web.gvs+<token>'"
)

# 只拿到 storyboard、沒有媒體格式時，依序換這些 client 重試
FALLBACK_CLIENTS = ("tv", "tv_simply", "web_safari", "mweb", "android")


def preload_cuda12_libs() -> None:
    """預先載入 pip 版 nvidia-* 提供的 CUDA 12 函式庫。

    CTranslate2 是用 dlopen("libcublas.so.12") 在執行期找函式庫，而
    .venv/.../site-packages/nvidia/cublas/lib 不在動態連結器的預設搜尋路徑，
    因此在建立模型前先以 RTLD_GLOBAL 載入，讓後續 dlopen 能命中已載入的 soname。
    """
    if sys.platform != "linux":
        return

    try:
        spec = importlib.util.find_spec("nvidia.cublas.lib")
    except ModuleNotFoundError:
        return
    if spec is None or not spec.submodule_search_locations:
        return

    for lib_dir in spec.submodule_search_locations:
        for lib_path in sorted(glob.glob(os.path.join(lib_dir, "libcublas*.so.12"))):
            try:
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                print(f"警告：無法預先載入 {lib_path}: {exc}")


def format_timestamp(seconds: float) -> str:
    """將秒數轉換為 SRT 字幕格式 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def is_url(value: str) -> bool:
    """判斷傳入的是否為 http/https 連結 (例如 YouTube)。"""
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def free_gpu_memory() -> None:
    """盡量釋放上一次推論失敗時仍被佔用的 GPU 記憶體。"""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    # is_initialized() 可避免為了 empty_cache 而額外建立一個 CUDA context
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "使用 Breeze-ASR-25 將音訊檔或網路影片轉錄為 SRT 字幕與純文字稿。"
            "可一次傳入多個輸入做批次轉錄。"
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        action="extend",
        nargs="+",
        required=True,
        metavar="SOURCE",
        help=(
            "輸入音訊檔路徑 (支援 wav, mp3, m4a, flac, mp4 等格式) "
            "或 YouTube 等網路影片連結 (需安裝 yt-dlp)；"
            "可一次給多個，例如 -i a.mp3 b.mp3，或反覆指定 -i a.mp3 -i b.mp3"
        ),
    )
    parser.add_argument(
        "-s",
        "--srt",
        type=Path,
        default=None,
        help="輸出 SRT 字幕檔路徑，僅限單一輸入時使用 (預設: <輸出目錄>/<輸入檔名>.srt)",
    )
    parser.add_argument(
        "-t",
        "--txt",
        type=Path,
        default=None,
        help="輸出純文字稿路徑，僅限單一輸入時使用 (預設: <輸出目錄>/<輸入檔名>.txt)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        help="批次轉錄時輸出 SRT/TXT 的目錄 (預設: 目前目錄)",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        metavar="BROWSER[:PROFILE]",
        help=(
            "向瀏覽器讀取 cookies，用於需要登入的 YouTube 影片，"
            "例如 --cookies-from-browser brave，"
            "或 brave+GNOMEKEYRING:Default (指定 keyring 與 profile)"
        ),
    )
    parser.add_argument(
        "--cookies",
        type=Path,
        default=None,
        help="使用 Netscape 格式的 cookies 檔案 (等同 yt-dlp 的 --cookies)",
    )
    parser.add_argument(
        "--format",
        default=None,
        metavar="SELECTOR",
        help=(
            "自訂 yt-dlp 格式選擇字串；預設自動挑選含音軌且不需額外合併的格式。"
            "例如 --format bestaudio/best 或 --format 251"
        ),
    )
    parser.add_argument(
        "--list-formats",
        action="store_true",
        help="只列出每個網址可用的格式後結束（不下載、不轉錄）",
    )
    parser.add_argument(
        "--extractor-args",
        action="append",
        default=None,
        metavar="IE_KEY:KEY=VALUE",
        help=(
            "透傳給 yt-dlp 的 extractor 參數，可重複指定。"
            "例如 --extractor-args 'youtube:player_client=tv' "
            "或 --extractor-args 'youtube:player_client=web,android'"
        ),
    )
    parser.add_argument(
        "--js-runtime",
        action="append",
        default=None,
        metavar="RUNTIME[:PATH]",
        help=(
            "指定 yt-dlp 解 YouTube JS challenge 用的 JS runtime，"
            "例如 node、deno 或 node:/usr/bin/node。"
            "未指定時自動從 PATH 偵測 deno/node/bun (預設會自動偵測)"
        ),
    )
    parser.add_argument(
        "--remote-components",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "允許 yt-dlp 下載的遠端元件，預設 ejs:github (YouTube challenge solver；"
            "會從 GitHub 下載腳本)。可改用 ejs:npm"
        ),
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=4,
        help="批次推論大小；GPU 記憶體不足時會自動減半重試 (預設: 4)",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        choices=["float16", "int8_float16", "int8", "float32"],
        help=(
            "模型精度；8GB VRAM 建議 float16 (約 3.8GB)，"
            "記憶體吃緊時改用 int8_float16 (約 2.3GB) (預設: float16)"
        ),
    )
    parser.add_argument(
        "--language",
        default="zh",
        help="辨識語言代碼，例如 zh / en；設為 auto 則自動偵測 (預設: zh)",
    )
    args = parser.parse_args(argv)

    for source in args.input:
        if not is_url(source) and not Path(source).is_file():
            parser.error(f"找不到輸入音訊檔: {source}")

    if len(args.input) > 1 and (args.srt or args.txt):
        parser.error(
            "批次轉錄多個輸入時不能指定 -s/--srt 或 -t/--txt，"
            "請改用 -o/--output-dir 指定輸出目錄"
        )

    if args.cookies is not None and not args.cookies.is_file():
        parser.error(f"找不到 cookies 檔案: {args.cookies}")

    if args.extractor_args:
        try:
            args.extractor_args = parse_extractor_args(args.extractor_args)
        except ValueError as exc:
            parser.error(str(exc))

    if args.js_runtime:
        try:
            args.js_runtimes = parse_js_runtimes(args.js_runtime)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        args.js_runtimes = detect_js_runtime()

    # YouTube 的 JS challenge 需要 solver 腳本，預設允許從 GitHub 取得
    args.remote_components = args.remote_components or ["ejs:github"]

    return args


def parse_extractor_args(values: list[str]) -> dict[str, dict[str, list[str]]]:
    """將 'youtube:player_client=tv;key=v' 形式轉成 yt-dlp 的 extractor_args 結構。"""
    result: dict[str, dict[str, list[str]]] = {}
    for value in values:
        ie_key, sep, rest = value.partition(":")
        if not sep or not ie_key or not rest:
            raise ValueError(
                f"--extractor-args 格式應為 IE_KEY:KEY=VALUE，收到: {value}"
            )
        for pair in rest.split(";"):
            key, sep, raw = pair.partition("=")
            if not sep or not key:
                raise ValueError(
                    f"--extractor-args 格式應為 IE_KEY:KEY=VALUE，收到: {value}"
                )
            result.setdefault(ie_key, {}).setdefault(key, []).extend(raw.split(","))
    return result


def parse_browser_spec(spec: str) -> tuple[str, str | None, str | None, str | None]:
    """將 BROWSER[+KEYRING][:PROFILE][::CONTAINER] 拆成 yt-dlp 需要的 4-tuple。"""
    container = None
    if "::" in spec:
        spec, _, container = spec.partition("::")

    profile = None
    if ":" in spec:
        spec, _, profile = spec.partition(":")

    keyring = None
    if "+" in spec:
        spec, _, keyring = spec.partition("+")

    return (spec, profile or None, keyring or None, container or None)


def build_ytdlp_opts(args: argparse.Namespace) -> dict[str, Any]:
    """把 CLI 參數轉成 yt-dlp 的通用 params (cookies、JS runtime、remote components)。"""
    opts: dict[str, Any] = {}

    if args.cookies is not None:
        opts["cookiefile"] = str(args.cookies)
    if args.cookies_from_browser:
        opts["cookiesfrombrowser"] = parse_browser_spec(args.cookies_from_browser)

    if args.js_runtimes:
        opts["js_runtimes"] = args.js_runtimes
        runtimes = ", ".join(
            f"{name} ({cfg.get('path')})" for name, cfg in args.js_runtimes.items()
        )
        print(f"JS runtime: {runtimes}")
    else:
        print(
            "警告：找不到 deno/node/bun，YouTube 可能只回傳 storyboard；"
            "請安裝 deno 或 node，或用 --js-runtime 指定"
        )

    if args.remote_components:
        opts["remote_components"] = set(args.remote_components)

    return opts


def detect_js_runtime() -> dict[str, dict[str, str]]:
    """從 PATH 找一個可用的 JS runtime (YouTube 解 challenge 必需)。"""
    for name, executable in (
        ("deno", "deno"),
        ("node", "node"),
        ("bun", "bun"),
        ("quickjs", "qjs"),
    ):
        path = shutil.which(executable)
        if path:
            return {name: {"path": path}}
    return {}


def parse_js_runtimes(values: list[str]) -> dict[str, dict[str, str]]:
    """將 --js-runtime 的 RUNTIME[:PATH] 轉成 yt-dlp 的 js_runtimes 結構。"""
    runtimes: dict[str, dict[str, str]] = {}
    for value in values:
        name, sep, path = value.partition(":")
        name = name.strip()
        path = path.strip()
        if not sep:
            resolved = shutil.which(name)
            if resolved is None:
                raise ValueError(f"找不到 JS runtime: {name}")
            path = resolved
        if not name or not path:
            raise ValueError(f"--js-runtime 格式應為 RUNTIME[:PATH]，收到: {value}")
        runtimes[name] = {"path": path}
    return runtimes


def pick_audio_format(info: Mapping[str, Any]) -> str | None:
    """從 yt-dlp 的格式清單挑一個有音軌、且不需要 ffmpeg 額外合併的格式。

    YouTube 直播常常只有 HLS 的 muxed 格式 (影+音同一個流)，固定的
    "bestaudio/best" 有時反而挑不到東西，所以先看實際清單再決定。
    """
    formats = [
        f
        for f in info.get("formats") or []
        if not f.get("has_drm")
        # 排除 storyboard (只含縮圖影格，沒有真正的音軌)
        and f.get("format_note") != "storyboard"
        and f.get("vcodec") != "images"
    ]

    def has_audio(fmt: Mapping[str, Any]) -> bool:
        return fmt.get("acodec") not in (None, "none")

    audio_only = [
        f for f in formats if has_audio(f) and f.get("vcodec") in (None, "none")
    ]
    muxed = [
        f for f in formats if has_audio(f) and f.get("vcodec") not in (None, "none")
    ]

    # 優先選純音訊 (檔案較小)，沒有就退而求其次用影音合一的流
    candidates = audio_only or muxed
    if not candidates:
        return None

    best = max(candidates, key=lambda f: (f.get("abr") or 0, f.get("tbr") or 0))
    return best.get("format_id")


def describe_formats(info: Mapping[str, Any], limit: int = 100) -> str:
    """列出可用格式，方便使用者自行用 --format 指定。"""
    lines = [
        f"  {fmt.get('format_id')}: ext={fmt.get('ext')} "
        f"acodec={fmt.get('acodec')} vcodec={fmt.get('vcodec')} "
        f"protocol={fmt.get('protocol')} tbr={fmt.get('tbr')} "
        f"note={fmt.get('format_note')} "
        f"filesize={fmt.get('filesize') or fmt.get('filesize_approx')} "
        f"drm={bool(fmt.get('has_drm'))}"
        for fmt in (info.get("formats") or [])[:limit]
    ]
    return "\n".join(lines) if lines else "  (此影片沒有提供任何格式)"


def _import_yt_dlp() -> Any:
    try:
        import yt_dlp
    except ImportError:
        raise SystemExit(
            "錯誤：處理網路影片連結需要 yt-dlp，請先執行 'uv sync' 或 'pip install yt-dlp'。"
        )
    return yt_dlp


def _base_download_opts(
    dest_dir: Path,
    ytdlp_opts: dict[str, Any],
    extractor_args: dict[str, dict[str, list[str]]] | None = None,
) -> Any:
    # 值型別混雜 (str/bool/tuple)，且 yt-dlp 各版本的 params 型別不一致，故標註為 Any
    opts: Any = {
        "outtmpl": str(dest_dir / "%(title).120s [%(id)s].%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        **ytdlp_opts,
    }
    if extractor_args:
        opts["extractor_args"] = extractor_args
    return opts


def probe_media(
    url: str, base_opts: Any, *, process: bool = False
) -> Mapping[str, Any]:
    """解析中繼資料 (不下載)，取得包含 formats 的 info dict。

    process=False 很關鍵：它會跳過 yt-dlp 的預設格式選擇，因此即使預設選擇器
    挑不到東西 (例如直播只有 HLS 格式)，我們仍能拿到原始格式清單自行決定。
    """
    youtube_dl = _import_yt_dlp()
    with youtube_dl.YoutubeDL(params=base_opts) as ydl:
        info = ydl.extract_info(url, download=False, process=process)
    if info is None:
        raise RuntimeError(f"無法解析影片連結: {url}")
    return info


def with_player_client(
    extractor_args: dict[str, dict[str, list[str]]] | None, client: str
) -> dict[str, dict[str, list[str]]]:
    """複製一份 extractor_args 並把 youtube 的 player_client 換成指定值。"""
    merged = _copy_extractor_args(extractor_args)
    merged.setdefault("youtube", {})["player_client"] = [client]
    return merged


def _copy_extractor_args(
    extractor_args: dict[str, dict[str, list[str]]] | None,
) -> dict[str, dict[str, list[str]]]:
    return {
        ie_key: {key: list(values) for key, values in args.items()}
        for ie_key, args in (extractor_args or {}).items()
    }


def with_missing_pot(
    extractor_args: dict[str, dict[str, list[str]]] | None,
) -> dict[str, dict[str, list[str]]]:
    """允許 yt-dlp 保留「缺少 PO token」的格式，否則它們會被直接丟掉。"""
    merged = _copy_extractor_args(extractor_args)
    youtube_args = merged.setdefault("youtube", {})
    formats = youtube_args.setdefault("formats", [])
    if "missing_pot" not in formats:
        formats.append("missing_pot")
    return merged


def find_working_probe(
    url: str,
    dest_dir: Path,
    ytdlp_opts: dict[str, Any],
    extractor_args: dict[str, dict[str, list[str]]] | None = None,
) -> tuple[Mapping[str, Any], Any]:
    """探測影片資訊，若拿不到含音軌的格式就換 YouTube client 重試。

    回傳 (info, base_opts)，base_opts 會帶著真正成功的那組 extractor_args，
    下載時必須用同一組，否則可能又拿不到格式。
    """
    attempts: list[tuple[str, dict[str, dict[str, list[str]]] | None]] = [
        ("預設", extractor_args)
    ]
    user_clients = (extractor_args or {}).get("youtube", {}).get("player_client")
    if not user_clients:  # 使用者沒指定才自動嘗試其他 client
        attempts += [
            (client, with_player_client(extractor_args, client))
            for client in FALLBACK_CLIENTS
        ]
        # 最後手段：保留因缺少 PO token 而被過濾的格式
        attempts.append(("missing_pot", with_missing_pot(extractor_args)))

    last_probe: Mapping[str, Any] | None = None
    last_opts: Any = None
    last_error: Exception | None = None

    for label, attempt_args in attempts:
        opts = _base_download_opts(dest_dir, ytdlp_opts, attempt_args)
        try:
            probe = probe_media(url, opts, process=False)
        except Exception as exc:  # noqa: BLE001 - 換下一個 client 再試
            last_error = exc
            print(f"client={label} 探測失敗，換下一個...")
            continue

        if pick_audio_format(probe):
            if label == "missing_pot":
                print(
                    "已保留缺少 PO token 的格式；若下載時回 403，代表仍需 PO token provider",
                    flush=True,
                )
            elif label != "預設":
                print(f"改用 client={label} 取得可用格式", flush=True)
            return probe, opts
        last_probe, last_opts = probe, opts
        print(f"client={label} 只拿到不含音軌的格式，換下一個...")

    if last_probe is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"無法解析影片連結: {url}")
    # 全部 client 都拿不到音軌：回傳最後一次結果，讓呼叫端把清單印出來
    return last_probe, last_opts


def download_with_format(url: str, base_opts: Any, format_id: str) -> Path:
    """用指定的 format 下載，回傳音訊檔路徑。"""
    youtube_dl = _import_yt_dlp()
    # faster-whisper (PyAV) 可直接解 webm/m4a/mp4，不需 ffmpeg 轉檔
    with youtube_dl.YoutubeDL(params={**base_opts, "format": format_id}) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            raise RuntimeError(f"無法解析影片連結: {url}")

        downloads = info.get("requested_downloads") or []
        if downloads and downloads[0].get("filepath"):
            return Path(downloads[0]["filepath"])
        return Path(ydl.prepare_filename(info))


def print_formats(
    url: str,
    ytdlp_opts: dict[str, Any],
    extractor_args: dict[str, dict[str, list[str]]] | None = None,
) -> None:
    """探測並印出某個網址可用的格式清單。"""
    probe, _ = find_working_probe(url, Path("."), ytdlp_opts, extractor_args)
    formats = probe.get("formats") or []
    print(f"\n=== {probe.get('title') or url} ===")
    print(
        f"共 {len(formats)} 個格式 "
        f"(live_status={probe.get('live_status')}, "
        f"is_live={probe.get('is_live')}):"
    )
    print(describe_formats(probe))
    if not formats or pick_audio_format(probe) is None:
        print(CLIENT_HINT)


def download_audio(
    url: str,
    dest_dir: Path,
    ytdlp_opts: dict[str, Any],
    format_selector: str | None = None,
    extractor_args: dict[str, dict[str, list[str]]] | None = None,
) -> Path:
    """用 yt-dlp 下載網路影片的音軌，回傳下載後的檔案路徑。"""
    print(f"正在透過 yt-dlp 下載音訊: {url}")

    # 先找到「真的含音軌」的格式清單 (必要時自動換 client)，並沿用同一組設定下載
    probe, base_opts = find_working_probe(url, dest_dir, ytdlp_opts, extractor_args)
    auto_format = pick_audio_format(probe)

    if probe.get("is_live"):
        print("偵測到直播，將從現在開始下載直到直播結束...")

    chosen = format_selector or auto_format
    if chosen is None:
        raise RuntimeError(
            "找不到含音軌且不需額外合併的格式，可用格式如下：\n"
            f"{describe_formats(probe)}\n"
            f"{CLIENT_HINT}\n"
            "或用 --format 自行指定格式"
        )

    print(f"選用格式: {chosen}")
    try:
        return download_with_format(url, base_opts, chosen)
    except Exception as exc:
        # 使用者指定的格式根本不存在 (例如直播沒有純音訊格式) 時，改用手動挑選的結果
        if "Requested format is not available" not in str(exc):
            raise
        fallback = auto_format or "best"
        if fallback == chosen:
            raise RuntimeError(
                f"格式 {chosen} 無法使用，可用格式如下：\n{describe_formats(probe)}"
            ) from exc
        print(f"格式 {chosen} 無法使用，改用 {fallback} 重試...")
        return download_with_format(url, base_opts, fallback)


def build_model(compute_type: str) -> BatchedInferencePipeline:
    """載入模型並包成批次推論管道 (模型只載入一次，供所有輸入重複使用)。"""
    print("正在載入 Breeze-ASR-25 模型至 GPU...")
    preload_cuda12_libs()
    # 針對 3060 Ti 8GB：
    # compute_type="float16" 約佔用 3.8GB VRAM
    # 若同時有其他程式吃 VRAM，可改為 "int8_float16" (僅佔約 2.3GB VRAM)
    model = WhisperModel(MODEL_PATH, device="cuda", compute_type=compute_type)
    # BatchedInferencePipeline 自帶 VAD 靜音過濾與切塊
    return BatchedInferencePipeline(model=model)


def resolve_output_paths(audio_file: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    """決定這筆輸入的 SRT/TXT 輸出路徑，並建立缺少的父目錄。"""
    # -s/-t 只在單一輸入時有效 (parse_args 已檢查)，適合指定完整檔名
    srt_path = args.srt or args.output_dir / f"{audio_file.stem}.srt"
    txt_path = args.txt or args.output_dir / f"{audio_file.stem}.txt"

    for out_path in (srt_path, txt_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    return srt_path, txt_path


def transcribe_file(
    batched_model: BatchedInferencePipeline,
    audio_file: Path,
    srt_path: Path,
    txt_path: Path,
    *,
    batch_size: int,
    language: str | None,
) -> None:
    """轉錄單一音訊檔並寫出 SRT/TXT；VRAM 不足時自動降低 batch size 重試。"""
    # 中文語境引導只對中文有用；其他語言若帶入中文 prompt 反而會把內容帶偏
    initial_prompt = CS_PROMPT if language == "zh" else None
    effective_batch_size = max(1, batch_size)

    print(
        f"開始轉錄 {audio_file} "
        f"(language={language or 'auto'}, batch_size={effective_batch_size})..."
    )
    start_time = time.time()

    while True:
        try:
            segments, info = batched_model.transcribe(
                str(audio_file),
                batch_size=effective_batch_size,
                language=language,
                beam_size=5,
                initial_prompt=initial_prompt,
                hotwords=CS_HOTWORDS,
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 500,  # 超過 0.5 秒的靜音即切斷，避免教授停頓時產生幻覺
                    "speech_pad_ms": 200,  # 語音前後保留 0.2 秒緩衝，避免切掉字首字尾
                },
            )

            # 注意：segments 是 Generator，在迴圈迭代時才會真正觸發 GPU 推論
            with (
                open(srt_path, "w", encoding="utf-8") as srt_file,
                open(txt_path, "w", encoding="utf-8") as txt_file,
            ):
                for idx, segment in enumerate(segments, start=1):
                    start_ts = format_timestamp(segment.start)
                    end_ts = format_timestamp(segment.end)
                    text = segment.text.strip()

                    # 即時印出進度
                    print(f"[{start_ts} -> {end_ts}] {text}")

                    # 寫入 SRT 字幕檔
                    srt_file.write(f"{idx}\n{start_ts} --> {end_ts}\n{text}\n\n")
                    # 寫入純文字稿 (方便後續餵給 LLM 整理筆記)
                    txt_file.write(f"{text}\n")
            break
        except RuntimeError as exc:
            is_oom = "out of memory" in str(exc).lower()
            if not is_oom or effective_batch_size <= 1:
                raise
            effective_batch_size = max(1, effective_batch_size // 2)
            print(
                f"GPU 記憶體不足，降低 batch_size 至 {effective_batch_size} 後重試..."
            )
            free_gpu_memory()

    elapsed = time.time() - start_time
    print(f"\n轉錄完成！總耗時: {elapsed:.2f} 秒 (音訊總長: {info.duration:.2f} 秒)")
    print(f"即時因子 (RTFx): {info.duration / elapsed:.1f}x")
    print(f"SRT 字幕: {srt_path}")
    print(f"純文字稿: {txt_path}")


def main() -> None:
    args = parse_args()
    language = None if args.language.lower() == "auto" else args.language
    total = len(args.input)
    ytdlp_opts = build_ytdlp_opts(args)

    if args.list_formats:
        for source in args.input:
            if not is_url(source):
                print(f"跳過非網址輸入: {source}")
                continue
            try:
                print_formats(source, ytdlp_opts, args.extractor_args)
            except Exception as exc:  # noqa: BLE001 - 單一網址失敗不應中斷其他網址
                print(f"錯誤：{source} 取得格式失敗 ({type(exc).__name__}: {exc})")
        return

    batched_model: BatchedInferencePipeline | None = None
    failures: list[tuple[str, str]] = []
    used_outputs: set[Path] = set()

    # 所有網路影片都下載到同一個暫存目錄，程式結束後統一清除
    with tempfile.TemporaryDirectory(prefix="breeze-asr-") as tmp_root:
        for index, source in enumerate(args.input, start=1):
            print(f"\n=== [{index}/{total}] {source} ===")
            try:
                if is_url(source):
                    audio_file = download_audio(
                        source,
                        Path(tmp_root),
                        ytdlp_opts,
                        args.format,
                        args.extractor_args,
                    )
                    print(f"已下載音訊: {audio_file}")
                else:
                    audio_file = Path(source)

                srt_path, txt_path = resolve_output_paths(audio_file, args)
                for out_path in (srt_path, txt_path):
                    if out_path in used_outputs:
                        print(f"警告：輸出路徑重複，將覆蓋先前的結果: {out_path}")
                    used_outputs.add(out_path)

                # 模型只載入一次，批次轉錄時重複使用以節省時間與 VRAM
                if batched_model is None:
                    batched_model = build_model(args.compute_type)

                transcribe_file(
                    batched_model,
                    audio_file,
                    srt_path,
                    txt_path,
                    batch_size=args.batch_size,
                    language=language,
                )
            except Exception as exc:  # noqa: BLE001 - 單一輸入失敗不應中斷整批
                print(f"錯誤：{source} 處理失敗 ({type(exc).__name__}: {exc})")
                if "Sign in to confirm" in str(exc):
                    print(
                        "提示：YouTube 要求登入驗證，請加上 "
                        "--cookies-from-browser brave (或改用 --cookies cookies.txt)"
                    )
                elif "failed to load cookies" in str(exc):
                    print(
                        "提示：無法讀取瀏覽器 cookies。請確認該瀏覽器已登入 YouTube、"
                        "GNOME 環境需安裝 secretstorage (執行 uv sync)，"
                        "必要時明確指定 keyring 與 profile，例如 "
                        "--cookies-from-browser brave+GNOMEKEYRING:Default"
                    )
                elif "Requested format is not available" in str(exc):
                    print(
                        "提示：yt-dlp 挑不到任何可用格式（常發生在直播或需要 PO token 的情況）。"
                        "請執行 --list-formats 看清單，或改用"
                        "--extractor-args 'youtube:player_client=tv' 換一個 client"
                    )
                failures.append((source, str(exc)))
                free_gpu_memory()

    if failures:
        print(f"\n完成，但有 {len(failures)}/{total} 個輸入失敗：")
        for source, message in failures:
            print(f"  - {source}: {message}")
        raise SystemExit(1)

    print(f"\n全部完成：{total} 個輸入皆已轉錄。")


if __name__ == "__main__":
    main()
