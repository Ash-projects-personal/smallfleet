"""Download the demo models (GGUF files from Hugging Face)."""

from __future__ import annotations

from .backends import download_file
from .config import (FRONTIER_MODEL_PATH, FRONTIER_MODEL_URL,
                     LOCAL_MODEL_PATH, LOCAL_MODEL_URL)


def main() -> None:
    download_file(LOCAL_MODEL_URL, LOCAL_MODEL_PATH)
    download_file(FRONTIER_MODEL_URL, FRONTIER_MODEL_PATH)


if __name__ == "__main__":
    main()
