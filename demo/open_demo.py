#!/usr/bin/env python3
"""Opens the interactive Redline tour in your browser. Standard library only, no GPU needed.

    python3 demo/open_demo.py            # open demo/index.html directly
    python3 demo/open_demo.py --serve    # serve it on http://localhost:8000 instead
    python3 demo/open_demo.py --rebuild  # regenerate index.html from the template first

The page is a single self-contained HTML file (code excerpts, measured results and
traces are baked in), so it also works offline and can simply be double-clicked.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import pathlib
import subprocess
import sys
import webbrowser

DEMO = pathlib.Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", action="store_true", help="serve over http://localhost instead of file://")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--rebuild", action="store_true", help="run demo/build_demo.py first")
    ap.add_argument("--no-browser", action="store_true", help="print the URL without opening a browser")
    args = ap.parse_args()

    if args.rebuild:
        subprocess.run([sys.executable, str(DEMO / "build_demo.py")], check=True)
    page = DEMO / "index.html"
    if not page.exists():
        sys.exit(f"{page} not found; run python3 demo/build_demo.py")

    if not args.serve:
        url = page.as_uri()
        print(f"Redline tour: {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DEMO))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as httpd:
        url = f"http://localhost:{args.port}/index.html"
        print(f"Redline tour: {url}  (Ctrl+C to stop)")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    os.chdir(DEMO)
    main()
