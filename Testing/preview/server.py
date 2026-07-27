"""
Preview server — debug dashboard for Scout.

Serves a localhost HTML page with:
  - Live camera frame (latest annotated, refreshed via <img> tag timer)
  - Event stream (SSE) for transcripts, tool calls, image attachments, etc.

Disabled unless preview.start() is called.

Usage from Scout code:
    from Testing.preview.server import preview
    preview.set_frame(annotated_np_array)
    preview.add_event("user_speech", {"text": "what is this"})
    preview.add_event("tool_call", {"name": "save_note", "args": {...}})

All calls are no-ops when disabled, so hooks are cheap to leave in place.
"""

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
<title>Scout</title>
<style>
  :root {
    --bg: #0e0f11;
    --panel: #131518;
    --border: rgba(255,255,255,0.06);
    --text: #e6e7ea;
    --text-dim: #8a8d93;
    --text-faint: #565a61;
    --accent: #7dd3c0;
    --accent-dim: rgba(125, 211, 192, 0.4);
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', sans-serif;
    font-size: 14px;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    overflow: hidden;
    -webkit-font-smoothing: antialiased;
  }

  /* ---- Camera panel ---- */
  #camera {
    flex: 0 0 58%;
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
  #camera-empty {
    color: var(--text-faint);
    font-size: 12px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 500;
  }
  .camera-label {
    position: absolute;
    top: 20px;
    left: 24px;
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-faint);
    font-weight: 600;
  }

  /* ---- Events panel ---- */
  #events {
    flex: 1;
    border-left: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    background: var(--panel);
    min-width: 0;
  }
  #events-header {
    padding: 18px 24px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
  }
  #events-header h1 {
    margin: 0;
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-dim);
    font-weight: 600;
  }
  .header-actions {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  #status {
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-faint);
    display: flex;
    align-items: center;
    gap: 6px;
  }
  #status::before {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text-faint);
    transition: background 0.2s;
  }
  #status.connected { color: var(--text-dim); }
  #status.connected::before {
    background: var(--accent);
    box-shadow: 0 0 6px var(--accent-dim);
  }
  #reset {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-dim);
    font: inherit;
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 5px 12px;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.15s;
  }
  #reset:hover {
    color: var(--text);
    border-color: rgba(255,255,255,0.15);
    background: rgba(255,255,255,0.02);
  }
  #reset:active { transform: translateY(1px); }

  #log {
    flex: 1;
    overflow-y: auto;
    padding: 8px 0 24px;
  }
  #log::-webkit-scrollbar { width: 8px; }
  #log::-webkit-scrollbar-track { background: transparent; }
  #log::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
  }
  #log::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.12); }

  .event {
    padding: 10px 24px;
    display: grid;
    grid-template-columns: 80px 1fr auto;
    gap: 16px;
    align-items: baseline;
    animation: fadeIn 0.18s ease-out;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(2px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .event .label {
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 600;
    color: var(--text-faint);
    white-space: nowrap;
  }
  .event .body {
    color: var(--text);
    line-height: 1.5;
    word-wrap: break-word;
    min-width: 0;
  }
  .event .ts {
    font-size: 10px;
    color: var(--text-faint);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }

  /* per-type accents (label only — keep body neutral) */
  .event.user_speech .label { color: var(--accent); }
  .event.scout_speech .label { color: #c9a96e; }
  .event.tool_call .label { color: #b794d4; }
  .event.tool_result .label { color: #7fa67d; }
  .event.image_attached .label { color: #d4a574; }
  .event.system .label { color: var(--text-faint); }

  .event.tool_call .body, .event.tool_result .body {
    font-family: ui-monospace, 'SF Mono', Menlo, monospace;
    font-size: 12.5px;
    color: var(--text-dim);
  }
  .event.system .body {
    color: var(--text-dim);
    font-style: italic;
  }
  .event.image_attached .body { color: var(--text-dim); }

  pre { margin: 0; white-space: pre-wrap; font: inherit; }

  .empty-state {
    padding: 60px 24px;
    text-align: center;
    color: var(--text-faint);
    font-size: 12px;
    letter-spacing: 0.05em;
  }
</style>
</head>
<body>
  <div id="camera">
    <span class="camera-label">Camera</span>
    <img id="frame" alt="" style="display:none">
    <div id="camera-empty">No signal</div>
  </div>
  <div id="events">
    <div id="events-header">
      <h1>Activity</h1>
      <div class="header-actions">
        <span id="status">disconnected</span>
        <button id="reset" title="Clear log and start fresh">Clear</button>
      </div>
    </div>
    <div id="log">
      <div class="empty-state">Waiting for events…</div>
    </div>
  </div>
<script>
  const log = document.getElementById('log');
  const status = document.getElementById('status');
  const frame = document.getElementById('frame');
  const cameraEmpty = document.getElementById('camera-empty');
  const resetBtn = document.getElementById('reset');

  let frameLoaded = false;
  setInterval(() => {
    const img = new Image();
    img.onload = () => {
      frame.src = img.src;
      if (!frameLoaded) {
        frame.style.display = 'block';
        cameraEmpty.style.display = 'none';
        frameLoaded = true;
      }
    };
    img.onerror = () => {};
    img.src = '/frame.jpg?t=' + Date.now();
  }, 100);

  const LABELS = {
    user_speech: 'Heard',
    scout_speech: 'Said',
    tool_call: 'Tool',
    tool_result: 'Result',
    image_attached: 'Image',
    system: 'System',
  };

  function clearLog() {
    log.innerHTML = '<div class="empty-state">Waiting for events…</div>';
  }

  function addEvent(evt) {
    const empty = log.querySelector('.empty-state');
    if (empty) empty.remove();

    if (evt.type === '_reset') { clearLog(); return; }

    const div = document.createElement('div');
    div.className = 'event ' + evt.type;
    const label = LABELS[evt.type] || evt.type;
    const ts = new Date(evt.ts * 1000).toLocaleTimeString('en-US', {hour12: false});

    let body;
    if (typeof evt.data === 'string') {
      body = evt.data;
    } else if (evt.data.text) {
      body = evt.data.text;
    } else if (evt.type === 'tool_call') {
      const args = evt.data.args ? JSON.stringify(evt.data.args) : '';
      body = `${evt.data.name}(${args})`;
    } else if (evt.type === 'tool_result') {
      body = `${evt.data.name || ''} → ${evt.data.result || ''}`.trim();
    } else if (evt.type === 'image_attached') {
      const conf = evt.data.conf !== undefined ? ` (${(evt.data.conf * 100).toFixed(0)}%)` : '';
      body = `${evt.data.label || 'frame'}${conf}`;
    } else {
      body = JSON.stringify(evt.data);
    }

    div.innerHTML = `<div class="label">${label}</div><div class="body"><pre>${escapeHtml(body)}</pre></div><div class="ts">${ts}</div>`;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
    while (log.children.length > 500) log.removeChild(log.firstChild);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  resetBtn.addEventListener('click', async () => {
    try {
      await fetch('/reset', { method: 'POST' });
      clearLog();
    } catch (err) {
      console.error('reset failed', err);
    }
  });

  function connect() {
    const es = new EventSource('/events');
    es.onopen = () => { status.textContent = 'live'; status.className = 'connected'; };
    es.onerror = () => {
      status.textContent = 'disconnected';
      status.className = '';
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
        self._port = 8765
        self._frame_jpeg = None
        self._frame_lock = threading.Lock()
        self._clients = []
        self._clients_lock = threading.Lock()
        self._history = deque(maxlen=200)
        self._httpd = None
        self._thread = None

    def start(self, port: int = 8765):
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
        self._broadcast(line)

    def reset(self):
        """Clear server history and tell connected clients to clear their UI."""
        if not self._enabled:
            return
        self._history.clear()
        evt = {"type": "_reset", "data": {}, "ts": time.time()}
        line = f"data: {json.dumps(evt)}\n\n".encode()
        self._broadcast(line)

    # ----- internals -----

    def _broadcast(self, line: bytes):
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

    def _register_client(self, handler):
        with self._clients_lock:
            self._clients.append(handler)
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
                pass

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

            def do_POST(self):
                path = self.path.split("?", 1)[0]
                if path == "/reset":
                    server_self.reset()
                    self.send_response(204)
                    self.end_headers()
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