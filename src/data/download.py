"""Download NASA C-MAPSS turbofan degradation datasets.

Official source (NASA Prognostics Data Repository / Open Data Portal):
  https://data.nasa.gov/download/w58c-4req/application%2Fzip
Alternative mirror commonly used in research:
  https://ti.arc.nasa.gov/c/6/  (legacy; may redirect)

If download fails (network, 403, etc.), callers should fall back to the
synthetic generator in ``src.data.synthetic`` and clearly label all results.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

# Primary public URL (NASA data portal package for C-MAPSS)
CMAPSS_URLS = [
    "https://data.nasa.gov/download/w58c-4req/application%2Fzip",
    "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
]

# Per-file mirror (research GitHub mirrors of the public C-MAPSS txt files)
CMAPSS_FILE_MIRRORS = [
    "https://github.com/LahiruJayasinghe/RUL-Net/raw/master/CMAPSSData",
    "https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData",
]

EXPECTED_FILES = [
    "train_FD001.txt",
    "test_FD001.txt",
    "RUL_FD001.txt",
    "train_FD002.txt",
    "test_FD002.txt",
    "RUL_FD002.txt",
    "train_FD003.txt",
    "test_FD003.txt",
    "RUL_FD003.txt",
    "train_FD004.txt",
    "test_FD004.txt",
    "RUL_FD004.txt",
]


def _has_cmapss(data_dir: Path) -> bool:
    return all((data_dir / f).exists() for f in EXPECTED_FILES[:3])  # at least FD001


def download_cmapss(data_dir: str | Path = "data/raw", timeout: int = 120) -> bool:
    """Download and extract C-MAPSS into ``data_dir``.

    Returns True on success, False if all URLs fail.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if _has_cmapss(data_dir):
        print(f"[download] C-MAPSS already present in {data_dir}")
        return True

    last_err: Exception | None = None
    for url in CMAPSS_URLS:
        try:
            print(f"[download] Trying {url} ...")
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 (research-download)"})
            with urlopen(req, timeout=timeout) as resp:
                content = resp.read()
            if len(content) < 1000:
                raise RuntimeError(f"Response too small ({len(content)} bytes)")
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                # Extract txt files, flattening nested dirs
                for name in zf.namelist():
                    if name.endswith(".txt") and not name.endswith("/"):
                        target = data_dir / Path(name).name
                        with zf.open(name) as src, target.open("wb") as dst:
                            dst.write(src.read())
            if _has_cmapss(data_dir):
                print(f"[download] Extracted C-MAPSS to {data_dir}")
                return True
            print("[download] Zip extracted but expected files missing; trying next URL")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[download] Failed: {exc}")

    # Try per-file mirrors
    for base in CMAPSS_FILE_MIRRORS:
        try:
            print(f"[download] Trying file mirror {base} ...")
            ok_files = 0
            for fname in EXPECTED_FILES:
                url = f"{base.rstrip('/')}/{fname}"
                req = Request(url, headers={"User-Agent": "Mozilla/5.0 (research-download)"})
                with urlopen(req, timeout=timeout) as resp:
                    content = resp.read()
                if len(content) < 50:
                    raise RuntimeError(f"{fname} too small")
                (data_dir / fname).write_bytes(content)
                ok_files += 1
            if _has_cmapss(data_dir):
                print(f"[download] Downloaded {ok_files} files from mirror to {data_dir}")
                return True
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[download] Mirror failed: {exc}")

    print(
        "[download] All C-MAPSS URLs failed. "
        "Use synthetic fallback (src.data.synthetic) and label results. "
        f"Last error: {last_err}"
    )
    return False


if __name__ == "__main__":
    ok = download_cmapss()
    raise SystemExit(0 if ok else 1)
