"""
live_monitor.py — Watch a folder for new .c64 captures and POST to /api/predict.

Usage:
    python live_monitor.py
    python live_monitor.py --watch /path/to/captures
    python live_monitor.py --watch /path/to/captures --url http://myhost:5000/api/predict
"""

import argparse
import os
import time
from datetime import datetime

import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

_MAX_FILE_BYTES = 100 * 1_048_576   # 100 MB
_STABLE_WAIT    = 2.0               # seconds to wait before checking file size stability
_RETRY_DELAY    = 10                # seconds before connection retry (Render cold start)


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _wait_until_stable(path: str) -> bool:
    """Return True if the file size hasn't changed after _STABLE_WAIT seconds."""
    try:
        size_before = os.path.getsize(path)
        time.sleep(_STABLE_WAIT)
        return os.path.getsize(path) == size_before
    except OSError:
        return False


class _C64Handler(FileSystemEventHandler):
    def __init__(self, url: str):
        self._url = url

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith(".c64"):
            return
        self._handle(event.src_path)

    def _handle(self, path: str):
        filename = os.path.basename(path)

        try:
            size = os.path.getsize(path)
        except OSError:
            return

        if size > _MAX_FILE_BYTES:
            print(f"[{_ts()}] SKIP  {filename} — {size / 1_048_576:.1f} MB exceeds 100 MB limit")
            return

        if not _wait_until_stable(path):
            print(f"[{_ts()}] WARN  {filename} — could not verify file is fully written")
            return

        self._post(path, filename, retry=True)

    def _post(self, path: str, filename: str, retry: bool = False):
        try:
            with open(path, "rb") as fh:
                resp = requests.post(
                    self._url,
                    files={"file": (filename, fh, "application/octet-stream")},
                    timeout=30,
                )
            resp.raise_for_status()
            data = resp.json()
            print(
                f"[{_ts()}]  {filename}"
                f"  result={data.get('result', '?')}"
                f"  confidence={data.get('confidence', 0):.2f}"
                f"  bursts={data.get('bursts_processed', '?')}"
            )
        except requests.exceptions.ConnectionError:
            if retry:
                print(f"[{_ts()}] CONN  {filename} — connection failed, retrying in {_RETRY_DELAY}s")
                time.sleep(_RETRY_DELAY)
                self._post(path, filename, retry=False)
            else:
                print(f"[{_ts()}] ERR   {filename} — connection failed after retry")
        except Exception as e:
            print(f"[{_ts()}] ERR   {filename} — {e}")


def main():
    parser = argparse.ArgumentParser(description="Watch for .c64 files and POST to /api/predict")
    parser.add_argument(
        "--watch", default=".",
        help="Directory to watch (default: current directory)",
    )
    parser.add_argument(
        "--url", default="http://localhost:5000/api/predict",
        help="API endpoint URL (default: http://localhost:5000/api/predict)",
    )
    args = parser.parse_args()

    watch_dir = os.path.abspath(args.watch)
    print(f"[{_ts()}] Watching {watch_dir}")
    print(f"[{_ts()}] Posting to {args.url}")

    handler  = _C64Handler(url=args.url)
    observer = Observer()
    observer.schedule(handler, watch_dir, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
