"""
Preview server — debug dashboard for Scout.

Serves a localhost HTML page with:
  - Live camera frame (latest annotated, refreshed via <img> tag timer)
  - Event stream (SSE) for transcripts, tool calls, image attachments, etc.

Disabled unless PREVIEW_MODE env var is set (or .start() is called explicitly).

Usage from Scout code:
    from Testing.preview.server import preview
    preview.set_frame(annotated_np_array)
    preview.add_event("user_speech", {"text": "what is this"})
    preview.add_event("tool_call", {"name": "save_note", "args": {...}})

All calls are no-ops when disabled, so hooks are cheap to leave in place.
"""

import os
import json
import time
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Scout Preview</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #1a1a1a;
    color: #e0e0e0;
    height: 100vh;
    display: flex;
    overflow: hidden;
  }
  #camera {
    flex: 0 0 60%;
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
  }
  #camera img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
  }
  #camera .label {
    position: absolute;
    top: 12px;
    left: 12px;
    background: rgba(0,0,0,0.6);
    padding: 4px 10px;
    font-size: 12px;
    border-radius: 4px;
    letter-spacing: 0.5px;
  }
  #events {
    flex: 1;
    border-left: 1px solid #333;
    display: flex;
    flex-direction: column;
    background: #222;
  }
  #events-header {
    padding: 12px 16px;
    border-bottom: 1px solid #333;
    font-size: 13px;
    letter-spacing: 0.5px;
    color: #888;
    display: flex;
    justify-content: space-between;
  }
  #status { font-size: 11px; }
  .connected { color: #4caf50; }
  .disconnected { color: #f44336; }
  #log {
    flex: 1;
    overflow-y: auto;
    padding: 8px 0;
  }
  .event {
    padding: 8px 16px;
    border-bottom: 1px solid #2a2a2a;
    font-size: 13px;
    line-height: 1.5;
    display: flex;
    gap: 10px;
  }
  .event .icon { flex: 0 0 22px; font-size: 16px; }
  .event .body { flex: 1; min-width: 0; word-wrap: break-word; }
  .event .ts {
    flex: 0 0 60px;
    font-size: 10px;
    color: #666;
    text-align: right;
    margin-top: 3px;
  }
  .event.user_speech .body { color: #82b1ff; }
  .event.scout_speech .body { color: #ffe082; }
  .event.tool_call .body { color: #b9f6ca; font-family: ui-monospace, monospace; font-size: 12px; }
  .event.tool_result .body { color: #c5e1a5; font-family: ui-monospace, monospace; font-size: 12px; }
  .event.image_attached .body { color: #ce93d8; font-style: italic; }
  .event.system .body { color: #888; font-style: italic; }
  pre { margin: 0; white-space: pre-wrap; }
</style>
</head>
<body>
  <div id="camera">
    <img id="frame" src="/frame.jpg" alt="camera">
    <div class="label">CAMERA</div>
  </div>
  <div id="events">
    <div id="events-header">
      <span>EVENTS</span>
      <span id="status" class="disconnected">disconnected</span>
    </div>
    <div id="log"></div>
  </div>
<script>
  const log = document.getElementById('log');
  const status = document.getElementById('status');
  const frame = document.getElementById('frame');

  // Refresh camera frame every 100ms
  setInterval(() => {
    frame.src = '/frame.jpg?t=' + Date.now();
  }, 100);

  const ICONS = {
    user_speech: '🎤',
    scout_speech: '🔊',
    tool_call: '🔧',
    tool_result: '✅',
    image_attached: '📷',
    system: 'ℹ️',
  };

  function addEvent(evt) {
    const div = document.createElement('div');
    div.className = 'event ' + evt.type;
    const icon = ICONS[evt.type] || '•';
    const ts = new Date(evt.ts * 1000).toLocaleTimeString('en-US', {hour12: false});
    let body;
    if (typeof evt.data === 'string') {
      body = evt.data;
    } else if (evt.data.text) {
      body = evt.data.text;
    } else {
      body = JSON.stringify(evt.data);
    }
    div.innerHTML = `<div class="icon">${icon}</div><div class="body"><pre>${escapeHtml(body)}</pre></div><div class="ts">${ts}</div>`;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
    // Cap DOM at 500 events
    while (log.children.length > 500) log.removeChild(log.firstChild);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  function connect() {
    const es = new EventSource('/events');
    es.onopen = () => { status.textContent = 'connected'; status.className = 'connected'; };
    es.onerror = () => {
      status.textContent = 'disconnected';
      status.className = 'disconnected';
      es.close();
      setTimeout(connect, 1000);
    };
    es.onmessage = (e) => {
      try { addEvent(JSON.parse(e.data)); } catch (err) { console.error(err); }
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
        self._port = 4200
        self._frame_jpeg = None
        self._frame_lock = threading.Lock()
        self._clients = []
        self._clients_lock = threading.Lock()
        self._history = deque(maxlen=50)  # replayed to new clients
        self._httpd = None
        self._thread = None

    def start(self, port: int = 4200):
        """Start the server. Idempotent."""
        if self._enabled:
            return
        self._port = port
        self._enabled = True

        handler = self._make_handler()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        print(f"[Preview] http://127.0.0.1:{port}")

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
            print(f"[Preview] frame encode failed: {e}")

    def add_event(self, event_type: str, data):
        if not self._enabled:
            return
        evt = {"type": event_type, "data": data, "ts": time.time()}
        line = f"data: {json.dumps(evt)}\n\n".encode()
        self._history.append(line)
        with self._clients_lock:
            dead = []
            for client in self._clients:
                try:
                    client.wfile.write(line)
                    client.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    dead.append(client)
            for d in dead:
                try:
                    self._clients.remove(d)
                except ValueError:
                    pass

    # ----- internals -----

    def _register_client(self, handler):
        with self._clients_lock:
            self._clients.append(handler)
        # Replay recent history so the page isn't empty on connect
        for line in list(self._history):
            try:
                handler.wfile.write(line)
                handler.wfile.flush()
            except Exception:
                break

    def _unregister_client(self, handler):
        with self._clients_lock:
            try:
                self._clients.remove(handler)
            except ValueError:
                pass

    def _get_frame(self):
        with self._frame_lock:
            return self._frame_jpeg

    def _make_handler(server_self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass  # silence default per-request stderr logging

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/" or path == "/index.html":
                    self._serve_html()
                elif path == "/frame.jpg":
                    self._serve_frame()
                elif path == "/events":
                    self._serve_events()
                else:
                    self.send_error(404)

            def _serve_html(self):
                body = HTML_PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve_frame(self):
                buf = server_self._get_frame()
                if buf is None:
                    # Tiny placeholder JPEG (1x1 black)
                    self.send_response(204)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(buf)))
                self.end_headers()
                self.wfile.write(buf)

            def _serve_events(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                # Initial comment to flush headers in some clients
                try:
                    self.wfile.write(b": connected\n\n")
                    self.wfile.flush()
                except Exception:
                    return
                server_self._register_client(self)
                try:
                    # Block this handler thread until the client disconnects.
                    # We detect disconnect via heartbeat writes.
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


# Module-level singleton. Auto-starts if PREVIEW_MODE=1.
preview = PreviewServer()
if os.environ.get("PREVIEW_MODE") == "1":
    port = int(os.environ.get("PREVIEW_MODE_PORT", "4200"))
    preview.start(port=port)