"""Review and correct the labels of recorded walks before training.

    python3 training/label_walks.py          # copy new walks off the board, open the page
    python3 training/label_walks.py --no-pull

Copies data/walks/ off the board (USB/adb) into walks/, then serves a page at
http://localhost:8800 that shows each walk as a list of segments (straight,
left, right, stairs up/down, arrive) next to a chart of the heading, with the
marks you tapped. The segments start as walklib.auto_labels()' guess; fix
whatever's wrong and press Save. That writes walks/<id>/labels.json and
labels.csv, which training reads. Only walks you've saved are trained on.

Needs only the Python standard library and numpy.
"""

import json
import re
import shutil
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import walklib

PORT = 8800
BOARD_WALKS = "/home/arduino/ArduinoApps/campus-nav/data/walks"
PAGE = Path(__file__).resolve().parent / "label_ui.html"
ID_RE = re.compile(r"^[\w-]+$")


def pull():
    """Copy every walk off the board (adb). Local labels.* files are never touched."""
    if not shutil.which("adb"):
        return {"ok": False, "message": "adb isn't installed"}
    r = subprocess.run(["adb", "devices"], capture_output=True, text=True, timeout=10)
    if not any(l.strip().endswith("device") for l in r.stdout.splitlines()[1:]):
        return {"ok": False, "message": "Board not connected over USB"}
    walklib.WALKS_DIR.mkdir(exist_ok=True)
    before = set(walklib.list_walks())
    r = subprocess.run(["adb", "pull", f"{BOARD_WALKS}/.", str(walklib.WALKS_DIR)],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        return {"ok": False, "message": (r.stderr or r.stdout).strip()[-300:]}
    new = sorted(set(walklib.list_walks()) - before)
    return {"ok": True, "message": f"Copied walks from the board ({len(new)} new)", "new": new}


def walk_summaries():
    out = []
    for wid in walklib.list_walks():
        d = walklib.WALKS_DIR / wid
        try:
            meta = json.loads((d / "meta.json").read_text())
        except (OSError, ValueError):
            meta = {}
        try:
            saved = json.loads((d / "labels.json").read_text())
        except (OSError, ValueError):
            saved = None
        out.append({"id": wid, "route": (saved or meta).get("route", ""),
                    "kind": (saved or meta).get("kind", "demo"), "person": (saved or meta).get("person", ""),
                    "started": meta.get("started", ""), "steps": meta.get("steps"),
                    "dist_m": meta.get("dist_m"), "reviewed": saved is not None,
                    "use": (saved or {}).get("use", False)})
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.path = self.path.split("?", 1)[0]
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/walks":
            return self._send(200, walk_summaries())
        m = re.match(r"^/api/walk/([\w-]+)$", self.path)
        if m and (walklib.WALKS_DIR / m[1]).is_dir():
            return self._send(200, walklib.load_walk(m[1]))
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/api/pull":
            return self._send(200, pull())
        m = re.match(r"^/api/walk/([\w-]+)$", self.path)
        if m and (walklib.WALKS_DIR / m[1]).is_dir():
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            return self._send(200, walklib.save_labels(m[1], body))
        self._send(404, {"error": "not found"})


def main():
    if "--no-pull" not in sys.argv:
        print(pull()["message"])
    print(f"{len(walklib.list_walks())} walks in {walklib.WALKS_DIR}")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"Labeling page: {url}   (Ctrl+C to quit)")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
