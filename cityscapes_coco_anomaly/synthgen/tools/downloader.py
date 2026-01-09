import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm


def require_exists(p: Path, what: str = "path") -> None:
    if not p.exists():
        raise FileNotFoundError(f"Missing {what}: {p}")


def download_file(
        url: str,
        dst: Path,
        *,
        session: Optional[requests.Session] = None,
        chunk_bytes: int = 10_000_000,
        timeout: int = 60,
        force: bool = False) -> Path:
    """
    Stream-download `url` to `dst`.
    Idempotent unless force=True.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not force and dst.exists() and dst.stat().st_size > 0:
        return dst

    s = session or requests.Session()
    with s.get(url, stream=True, allow_redirects=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))

        tmp = dst.with_suffix(dst.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=dst.name) as pbar:

            for chunk in r.iter_content(chunk_size=chunk_bytes):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))

        tmp.replace(dst)

    return dst


def safe_extract_zip(zip_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        base = extract_to.resolve()
        for member in zf.infolist():
            member_path = (extract_to / member.filename).resolve()
            if not str(member_path).startswith(str(base)):
                raise RuntimeError(f"Unsafe zip member path: {member.filename}")
        zf.extractall(extract_to)
