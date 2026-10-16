"""Preview server: the camera feed and a live copy of the log, on localhost.

The log side mirrors the terminal: every INFO line any module logs shows up
here through `PreviewHandler`, so nothing has to report to the preview
separately. Disabled unless `preview.start()` is called; until then every
call is a no-op.
"""
import json
import logging
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

logger = logging.getLogger(__name__)

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>pal</title>
<style>
  body { margin: 0; padding: 8px; background: #000; color: #ccc; font: 13px/1.4 monospace; }
  img { display: block; max-width: 640px; max-height: 45vh; margin-bottom: 8px; }
  a { color: #ccc; }
  #log { margin: 0; white-space: pre-wrap; }
</style>
</head>
<body>
<div>pal preview -- <span id="status">disconnected</span> -- <a href="#" id="clear">clear</a></div>
<img id="frame" alt="">
<pre id="log"></pre>
<script>
  const log = document.getElementById('log');
  const status = document.getElementById('status');
  const frame = document.getElementById('frame');

  setInterval(() => {
    const img = new Image();
    img.onload = () => { frame.src = img.src; };
    img.src = '/frame.jpg?t=' + Date.now();
  }, 100);

  function append(line) {
    const atBottom = window.innerHeight + window.scrollY >= document.body.scrollHeight - 4;
    log.textContent += line + '\\n';
    const lines = log.textContent.split('\\n');
    if (lines.length > 2000) log.textContent = lines.slice(-2000).join('\\n');
    if (atBottom) window.scrollTo(0, document.body.scrollHeight);
  }

  document.getElementById('clear').onclick = (e) => {
    e.preventDefault();
    fetch('/reset', { method: 'POST' });
    log.textContent = '';
  };

  function connect() {
    const es = new EventSource('/events');
    es.onopen = () => { status.textContent = 'connected'; };
    es.onerror = () => { status.textContent = 'disconnected'; es.close(); setTimeout(connect, 1000); };
    es.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.reset) log.textContent = ''; else append(msg.line);
    };
  }
  connect();
</script>
</body>
</html>
"""


class PreviewServer:
    """Singleton preview server. Safe to call methods even when disabled."""

    def __init__(self):
        self._enabled = False
        self._frame_jpeg = None
        self._frame_lock = threading.Lock()
        self._clients = []
        self._clients_lock = threading.Lock()
        self._history = deque(maxlen=500)
        self._httpd = None

    def start(self, port: int = 8765):
        if self._enabled:
            return
        self._enabled = True
        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), self._make_handler())
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        logger.info(f"preview at http://127.0.0.1:{port}")

    def stop(self):
        if not self._enabled:
            return
        self._enabled = False
        try:
            self._httpd.shutdown()
        except Exception:
            pass

    # ----- public push API -----

    def set_frame(self, frame_np):
        if not self._enabled or frame_np is None:
            return
        try:
            ok, buf = cv2.imencode(".jpg", frame_np, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self._frame_lock:
                    self._frame_jpeg = buf.tobytes()
        except Exception as e:
            logger.warning(f"frame encode failed: {e}")

    def add_line(self, line: str):
        if not self._enabled:
            return
        message = self._encode({"line": line})
        self._history.append(message)
        self._broadcast(message)

    def reset(self):
        """Clear server history and tell connected clients to clear."""
        if not self._enabled:
            return
        self._history.clear()
        self._broadcast(self._encode({"reset": True}))

    # ----- internals -----

    @staticmethod
    def _encode(payload: dict) -> bytes:
        return f"data: {json.dumps(payload)}\n\n".encode()

    def _broadcast(self, message: bytes):
        with self._clients_lock:
            for client in list(self._clients):
                try:
                    client.wfile.write(message)
                    client.wfile.flush()
                except OSError:
                    self._clients.remove(client)

    def _register_client(self, handler):
        with self._clients_lock:
            self._clients.append(handler)
        for message in list(self._history):
            try:
                handler.wfile.write(message)
                handler.wfile.flush()
            except Exception:
                break

    def _unregister_client(self, handler):
        with self._clients_lock:
            if handler in self._clients:
                self._clients.remove(handler)

    def _get_frame(self):
        with self._frame_lock:
            return self._frame_jpeg

    def _make_handler(server_self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    self._send(200, "text/html; charset=utf-8", HTML_PAGE.encode())
                elif path == "/frame.jpg":
                    buf = server_self._get_frame()
                    if buf is None:
                        self.send_response(204)
                        self.end_headers()
                    else:
                        self._send(200, "image/jpeg", buf, {"Cache-Control": "no-store"})
                elif path == "/events":
                    self._serve_events()
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path.split("?", 1)[0] == "/reset":
                    server_self.reset()
                    self.send_response(204)
                    self.end_headers()
                else:
                    self.send_error(404)

            def _send(self, code, content_type, body, headers=None):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _serve_events(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                try:
                    self.wfile.write(b": connected\n\n")
                    self.wfile.flush()
                except Exception:
                    return
                server_self._register_client(self)
                try:
                    while server_self._enabled:
                        time.sleep(15)
                        try:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                        except Exception:
                            break
                finally:
                    server_self._unregister_client(self)

        return Handler


# Module-level singleton.
preview = PreviewServer()


class PreviewHandler(logging.Handler):
    """Sends formatted log lines to the preview page."""

    def emit(self, record):
        try:
            preview.add_line(self.format(record))
        except Exception:
            self.handleError(record)
