#!/usr/bin/env python
"""
download_laz.py  --  Sol HPC version
====================================
Downloads all LAZ tiles from USGS rockyweb listed in LINKS_FILE.

- Resume-friendly: existing files are skipped.
- Failed downloads can be retried by re-running the script.
- Parallel workers tuned for Sol's network (16 by default).
"""

import os
import time
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------#
# CONFIG                                                                     #
# ---------------------------------------------------------------------------#
LINKS_FILE  = "/path/to/Phoenix/LiDAR/all_laz_links.txt"
OUTPUT_DIR  = "/path/to/Phoenix/LiDAR/raw_data"
MAX_WORKERS = 16          # USGS will throttle if too aggressive; 16 is safe
TIMEOUT     = 300         # per-file connect+read timeout (sec)
MAX_RETRIES = 3           # retry transient failures
RETRY_WAIT  = 10          # seconds between retries
# ---------------------------------------------------------------------------#


def download_file(url):
    """Download a single LAZ with resume + retry."""
    url = url.strip()
    if not url:
        return ("skip", None)

    filename = url.split("/")[-1]
    filepath = os.path.join(OUTPUT_DIR, filename)

    # Skip if already present and non-empty
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return ("exists", filename)

    for attempt in range(MAX_RETRIES):
        try:
            with requests.get(url, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()
                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MB
                        if chunk:
                            f.write(chunk)
            # sanity check size
            if os.path.getsize(filepath) > 0:
                return ("ok", filename)
            else:
                os.remove(filepath)
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT)
                continue
            return ("fail", f"{filename}: {e}")
    return ("fail", filename)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(LINKS_FILE) as f:
        urls = [l.strip() for l in f if l.strip()]

    print(f"Links file : {LINKS_FILE}",     flush=True)
    print(f"Output     : {OUTPUT_DIR}",     flush=True)
    print(f"Total URLs : {len(urls)}",      flush=True)
    print(f"Workers    : {MAX_WORKERS}",    flush=True)
    print(flush=True)

    ok = exists = fail = 0
    failures = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(download_file, u): u for u in urls}
        with tqdm(total=len(futures), desc="LAZ", smoothing=0.05) as pbar:
            for fut in as_completed(futures):
                status, info = fut.result()
                if status == "ok":
                    ok += 1
                elif status == "exists":
                    exists += 1
                elif status == "fail":
                    fail += 1
                    failures.append(info)
                pbar.update(1)
                pbar.set_postfix(ok=ok, exists=exists, fail=fail)

    print(flush=True)
    print(f"DONE  downloaded={ok}  already_present={exists}  failed={fail}",
          flush=True)

    if failures:
        fail_log = os.path.join(OUTPUT_DIR, "_failed.txt")
        with open(fail_log, "w") as f:
            f.write("\n".join(failures))
        print(f"Failed list -> {fail_log}", flush=True)
        print("Re-run this script to retry failed downloads.", flush=True)


if __name__ == "__main__":
    main()
