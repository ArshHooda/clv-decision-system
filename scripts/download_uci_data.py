from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clv.config import RAW_DIR, UCI_DATASET_PAGE, UCI_ZIP_URL  # noqa: E402


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "online_retail_ii.zip"

    print("Downloading UCI Online Retail II...")
    print(UCI_ZIP_URL)
    try:
        urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)
    except Exception as exc:
        print(f"Automatic download failed: {exc}")
        print("Manual download page:")
        print(UCI_DATASET_PAGE)
        print(f"Place online_retail_II.xlsx in: {RAW_DIR}")
        raise SystemExit(1)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(RAW_DIR)

    print(f"Downloaded and extracted files to: {RAW_DIR}")


if __name__ == "__main__":
    main()
