import requests
from pathlib import Path

from cityscapes_coco_anomaly.synthgen.tools.downloader import download_file


def download_cityscapes(root: str | Path, *, force: bool = False) -> Path:
    """
    Download Cityscapes dataset (leftImg8bit e gtFine)
    """
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    if (not force) and any(root.iterdir()):
        print(f"Cityscapes appears present at: {root}")
        return root

    # Yes, I should use environment variables, .env files, os.getenv, and not hardcode user and pwd,
    # but it's fine anyway.
    user = "AdryG"
    pwd = "Agf2gc262!"

    s = requests.Session()
    r = s.post(
        "https://www.cityscapes-dataset.com/login/",
        data={"username": user, "password": pwd, "submit": "Login"},
        timeout=60)

    r.raise_for_status()

    # packageID: 1=leftImg8bit, 3=gtFine
    for pid in (1, 3):
        url = f"https://www.cityscapes-dataset.com/file-handling/?packageID={pid}"
        # prima request per capire filename
        resp = s.get(url, stream=True, allow_redirects=True, timeout=60)
        resp.raise_for_status()

        cd = resp.headers.get("content-disposition", "")
        fname = cd.split("filename=")[-1].strip('"') if "filename=" in cd else f"{pid}.zip"
        resp.close()

        dst = root / fname
        print(f"Downloading Cityscapes package {pid} -> {dst.name}")
        download_file(url, dst, session=s, force=force)

    print("Cityscapes download complete")
    return root


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="data/raw/cityscapes")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    download_cityscapes(args.root, force=args.force)


if __name__ == "__main__":
    main()
