"""Preview server: the camera feed and a live copy of the log, on localhost.

The camera image fills the page with what was said overlaid on it, next to a
panel with the speaker score per utterance and the system log. The log mirrors
the terminal: every INFO line any module logs shows up here through
`PreviewHandler`, so nothing has to report to the preview separately; speech
lines are tagged so they land on the image instead. Disabled unless `preview.start()` is called; until then every
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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pal</title>
<style>
  :root {
    --bg: #000; --panel: #1a1a19; --line: #383835;
    --text: #f0efec; --muted: #c3c2b7; --faint: #8a8984;
    --you: #3987e5; --other: #d95926;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body { margin: 0; background: var(--bg); color: var(--muted); font: 13px/1.4 monospace;
         display: grid; grid-template-columns: 1fr 400px; }
  a { color: var(--muted); }

  #stage { position: relative; min-height: 0; overflow: hidden; }
  #frame { width: 100%; height: 100%; object-fit: contain; display: block; }
  #subs { position: absolute; left: 0; right: 0; bottom: 0; padding: 48px 24px 20px;
          background: linear-gradient(transparent, rgba(0,0,0,.85) 40%);
          font: 18px/1.45 system-ui, sans-serif; color: var(--text); }
  #subs div { margin-top: 4px; text-shadow: 0 1px 3px #000; }
  #subs .who { display: inline-block; min-width: 3.5em; font: 12px monospace; color: var(--muted);
               border-left: 4px solid var(--faint); padding-left: 6px; margin-right: 8px; }
  #subs .user .who { border-color: var(--you); }
  #subs .other .who { border-color: var(--other); }
  #subs .pal .who { border-color: var(--text); }

  #side { display: flex; flex-direction: column; min-height: 0; background: var(--panel);
          border-left: 1px solid var(--line); }
  #side header { padding: 8px 12px; border-bottom: 1px solid var(--line); }
  #score { padding: 12px; border-bottom: 1px solid var(--line); }
  #score h2 { margin: 0 0 2px; font-size: 13px; font-weight: normal; color: var(--text); }
  #latest { font-size: 28px; color: var(--text); }
  #latest small { font-size: 13px; color: var(--muted); }
  .legend { color: var(--muted); margin: 4px 0 6px; }
  .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 10px;
              vertical-align: -1px; }
  .legend i:first-child { margin-left: 0; }
  #chart { width: 100%; height: 150px; display: block; }
  #chart text { fill: var(--faint); font: 10px monospace; }
  #tip { position: fixed; pointer-events: none; display: none; max-width: 280px; padding: 6px 8px;
         background: #000; border: 1px solid var(--line); border-radius: 4px; color: var(--text); }
  #log { flex: 1; min-height: 0; overflow: auto; margin: 0; padding: 8px 12px; white-space: pre-wrap; }

  @media (max-width: 800px) {
    body { grid-template-columns: 1fr; grid-template-rows: 55vh auto; height: auto; }
    #subs { font-size: 15px; padding: 32px 16px 12px; }
    #side { border-left: 0; }
    #log { max-height: 50vh; }
  }
</style>
</head>
<body>
<div id="stage">
  <img id="frame" alt="">
  <div id="subs"></div>
</div>
<aside id="side">
  <header>pal -- <span id="status">disconnected</span> -- <a href="#" id="clear">clear</a></header>
  <section id="score">
    <h2>Speaker score per utterance</h2>
    <div id="latest">--</div>
    <div class="legend"><i style="background:var(--you)"></i>you <i style="background:var(--other)"></i>other
      <span style="margin-left:10px">--- threshold</span></div>
    <svg id="chart" role="img" aria-label="Speaker score per utterance"></svg>
  </section>
  <pre id="log"></pre>
</aside>
<div id="tip"></div>
<script>
  const $ = (id) => document.getElementById(id);
  const log = $('log'), subs = $('subs'), chart = $('chart'), tip = $('tip');
  const MAX_SUBS = 6, MAX_BARS = 40;
  let utterances = [];

  setInterval(() => {
    const img = new Image();
    img.onload = () => { $('frame').src = img.src; };
    img.src = '/frame.jpg?t=' + Date.now();
  }, 100);

  function addLine(line) {
    const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
    log.textContent += line + '\\n';
    const lines = log.textContent.split('\\n');
    if (lines.length > 2000) log.textContent = lines.slice(-2000).join('\\n');
    if (atBottom) log.scrollTop = log.scrollHeight;
  }

  function addSpeech({ who, text }) {
    const row = document.createElement('div');
    row.className = who;
    const label = document.createElement('span');
    label.className = 'who';
    label.textContent = who === 'user' ? 'you' : who;
    row.append(label, text);
    subs.append(row);
    while (subs.children.length > MAX_SUBS) subs.firstChild.remove();
    [...subs.children].forEach((el, i, all) => { el.style.opacity = 0.35 + 0.65 * (i + 1) / all.length; });
  }

  function addUtterance(u) {
    utterances.push(u);
    if (utterances.length > MAX_BARS) utterances.shift();
    $('latest').innerHTML = u.score.toFixed(2) + ' <small>' + (u.is_user ? 'you' : 'other') +
      ', ' + u.seconds.toFixed(1) + 's</small>';
    drawChart();
  }

  function drawChart() {
    const w = chart.clientWidth, h = chart.clientHeight, left = 26, top = 6, bottom = 4;
    const plotH = h - top - bottom, y = (v) => top + plotH * (1 - Math.max(0, Math.min(1, v)));
    const slot = (w - left) / MAX_BARS, bar = Math.max(2, slot - 2);
    const ns = 'http://www.w3.org/2000/svg';
    const el = (name, attrs) => { const n = document.createElementNS(ns, name);
      for (const k in attrs) n.setAttribute(k, attrs[k]); return n; };
    chart.replaceChildren();
    for (const v of [0, 0.5, 1]) {
      chart.append(el('line', { x1: left, x2: w, y1: y(v), y2: y(v), stroke: '#383835' }));
      const label = el('text', { x: left - 4, y: y(v) + 3, 'text-anchor': 'end' });
      label.textContent = v.toFixed(1);
      chart.append(label);
    }
    const threshold = utterances.length ? utterances[utterances.length - 1].threshold : 0.5;
    utterances.forEach((u, i) => {
      const x = left + i * slot + 1, top_ = y(u.score);
      const height = Math.max(1, y(0) - top_);
      chart.append(el('rect', { x, y: top_, width: bar, height, rx: Math.min(2, bar / 2),
        fill: u.is_user ? 'var(--you)' : 'var(--other)' }));
      const hit = el('rect', { x: x - 1, y: top, width: slot, height: plotH, fill: 'transparent' });
      hit.onmousemove = (e) => {
        tip.style.display = 'block';
        tip.style.left = Math.min(e.clientX + 12, innerWidth - 290) + 'px';
        tip.style.top = (e.clientY + 12) + 'px';
        tip.textContent = u.score.toFixed(2) + ' -> ' + (u.is_user ? 'you' : 'other') + ' (' +
          u.seconds.toFixed(1) + 's)\\n' + u.text;
        tip.style.whiteSpace = 'pre-wrap';
      };
      hit.onmouseleave = () => { tip.style.display = 'none'; };
      chart.append(hit);
    });
    chart.append(el('line', { x1: left, x2: w, y1: y(threshold), y2: y(threshold),
      stroke: '#c3c2b7', 'stroke-dasharray': '4 3', 'pointer-events': 'none' }));
  }
  addEventListener('resize', drawChart);
  drawChart();

  function clearAll() {
    log.textContent = ''; subs.replaceChildren(); utterances = [];
    $('latest').textContent = '--'; drawChart();
  }

  $('clear').onclick = (e) => {
    e.preventDefault();
    fetch('/reset', { method: 'POST' });
    clearAll();
  };

  function connect() {
    const es = new EventSource('/events');
    es.onopen = () => { $('status').textContent = 'connected'; };
    es.onerror = () => { $('status').textContent = 'disconnected'; es.close(); setTimeout(connect, 1000); };
    es.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.reset) clearAll();
      else if (msg.speech) addSpeech(msg.speech);
      else if (msg.utterance) addUtterance(msg.utterance);
      else addLine(msg.line);
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
        """A system line for the log panel."""
        self._push({"line": line})

    def add_speech(self, who: str, text: str):
        """Something said, by "user", "other" or "pal": shown over the camera image."""
        self._push({"speech": {"who": who, "text": text}})

    def add_utterance(self, score: float, threshold: float, is_user: bool, seconds: float, text: str):
        """One bar in the speaker score chart."""
        self._push({"utterance": {"score": score, "threshold": threshold, "is_user": is_user,
                                  "seconds": seconds, "text": text}})

    def reset(self):
        """Clear server history and tell connected clients to clear."""
        if not self._enabled:
            return
        self._history.clear()
        self._broadcast(self._encode({"reset": True}))

    # ----- internals -----

    def _push(self, payload: dict):
        if not self._enabled:
            return
        message = self._encode(payload)
        self._history.append(message)
        self._broadcast(message)

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
    """Sends formatted log lines to the preview page. A record logged with
    `extra={"speech": (who, text)}` goes over the camera image instead of the log panel."""

    def emit(self, record):
        try:
            speech = getattr(record, "speech", None)
            if speech:
                preview.add_speech(*speech)
            else:
                preview.add_line(self.format(record))
        except Exception:
            self.handleError(record)
