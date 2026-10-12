"""
Bridge from the Python game to the Three.js view in the browser.

Python does all the sensing and thinking; the browser only draws.
  http://localhost:8765   serves the page in web/ (index.html, game.js, three.js)
  ws://localhost:8766     Python -> page: game state (JSON) + webcam inset (JPEG)
                          page -> Python: key presses
Everything stays on this laptop -- no internet needed.
"""

import asyncio
import functools
import json
import os
import queue
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import websockets

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")  # always load the newest game.js
        super().end_headers()


class WebView:
    def __init__(self, http_port=8765, ws_port=8766, open_browser=True, low=False):
        self.keys = queue.Queue()
        self._lock = threading.Lock()
        self._state, self._seq = None, 0
        self._jpeg, self._jseq = None, 0

        handler = functools.partial(_QuietHandler, directory=WEB_DIR)
        self._http = ThreadingHTTPServer(("127.0.0.1", http_port), handler)
        threading.Thread(target=self._http.serve_forever, daemon=True).start()
        threading.Thread(target=self._run_ws, args=(ws_port,), daemon=True).start()

        self.url = f"http://localhost:{http_port}/?ws={ws_port}" + ("&low" if low else "")
        print(f"3D view: {self.url}")
        if open_browser:
            webbrowser.open(self.url)

    # --- called from the game loop -----------------------------------------
    def send_state(self, state):
        s = json.dumps(state)
        with self._lock:
            self._state, self._seq = s, self._seq + 1

    def send_camera(self, jpeg_bytes):
        with self._lock:
            self._jpeg, self._jseq = jpeg_bytes, self._jseq + 1

    def poll_keys(self):
        out = []
        while not self.keys.empty():
            out.append(self.keys.get_nowait())
        return out

    def close(self):
        self._http.shutdown()

    # --- websocket server (its own thread + asyncio loop) -------------------
    def _run_ws(self, port):
        async def main():
            async with websockets.serve(self._client, "127.0.0.1", port, max_size=2 ** 20):
                await asyncio.Future()  # run forever
        asyncio.run(main())

    async def _client(self, ws):
        async def read_keys():
            async for msg in ws:
                try:
                    key = json.loads(msg).get("key")
                except (ValueError, AttributeError):
                    continue
                if isinstance(key, str) and key:
                    self.keys.put(key)

        reader = asyncio.ensure_future(read_keys())
        seq = jseq = -1
        try:
            while not reader.done():
                with self._lock:
                    state, s, jpeg, js = self._state, self._seq, self._jpeg, self._jseq
                # Only ever send the NEWEST frame, so a slow page never lags behind.
                if state is not None and s != seq:
                    await ws.send(state)
                    seq = s
                if jpeg is not None and js != jseq:
                    await ws.send(jpeg)
                    jseq = js
                await asyncio.sleep(1 / 120)
        except websockets.ConnectionClosed:
            pass
        finally:
            reader.cancel()
