"""Preview server: the camera feed and a live copy of the log, on localhost.

The camera image fills the page with what was said overlaid on it, next to a
panel with the speaker score, where the time went for each reply, what recall
found and how well it matched, and the system log. The log mirrors
the terminal: every INFO line any module logs shows up here through
`PreviewHandler`, so nothing has to report to the preview separately; speech
lines are tagged so they land on the image instead. Disabled unless `preview.start()` is called; until then every
call is a no-op.
"""
import base64
import json
import logging
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

logger = logging.getLogger(__name__)

SENT_THUMB_HEIGHT = 192  # shown at half size, sharp on high-DPI screens

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
         display: grid; grid-template-columns: 1fr 440px; }
  a { color: var(--muted); }

  #stage { position: relative; min-height: 0; overflow: hidden; }
  #frame { width: 100%; height: 100%; object-fit: contain; display: block; }
  #sent { position: absolute; top: 12px; left: 12px; right: 12px; display: none; transition: opacity 1s; }
  #sent.on { display: block; }
  #sent .label { display: inline-block; margin-bottom: 6px; padding: 2px 8px; border-radius: 4px;
                 background: var(--you); color: #fff; }
  #sent figure { display: inline-block; margin: 0 8px 0 0; }
  #sent img { display: block; height: 96px; border: 2px solid var(--you); border-radius: 4px; }
  #sent figcaption { color: var(--text); text-shadow: 0 1px 3px #000; }
  #stage.sending { box-shadow: inset 0 0 0 4px var(--you); }
  #subs { position: absolute; left: 0; right: 0; bottom: 0; padding: 48px 24px 20px;
          background: linear-gradient(transparent, rgba(0,0,0,.85) 40%);
          font: 18px/1.45 system-ui, sans-serif; color: var(--text); }
  #subs div { margin-top: 4px; text-shadow: 0 1px 3px #000; }
  #subs .who { display: inline-block; min-width: 3.5em; font: 12px monospace; color: var(--muted);
               border-left: 4px solid var(--faint); padding-left: 6px; margin-right: 8px; }
  #subs .user .who { border-color: var(--you); }
  #subs .other .who { border-color: var(--other); }
  #subs .pal .who { border-color: var(--text); }

  #side { display: flex; flex-direction: column; min-height: 0; overflow-y: auto; background: var(--panel);
          border-left: 1px solid var(--line); }
  #side header { padding: 8px 12px; border-bottom: 1px solid var(--line); }
  #side details { padding: 8px 12px 10px; border-bottom: 1px solid var(--line); flex: none; }
  #side summary { cursor: pointer; color: var(--text); margin-bottom: 4px; }
  #side summary .note { color: var(--faint); }
  #latest { font-size: 28px; color: var(--text); }
  #latest small { font-size: 13px; color: var(--muted); }
  .legend { color: var(--muted); margin: 4px 0 6px; }
  .legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 10px;
              vertical-align: -1px; }
  .legend i:first-child { margin-left: 0; }
  #live { width: 100%; height: 90px; display: block; margin-bottom: 8px; }
  #chart { width: 100%; height: 80px; display: block; }
  #score .sub { color: var(--faint); margin: 2px 0 4px; }
  #score .sub b { color: var(--muted); font-weight: normal; }
  #chart text { fill: var(--faint); font: 10px monospace; }
  #tip { position: fixed; pointer-events: none; display: none; max-width: 280px; padding: 6px 8px;
         background: #000; border: 1px solid var(--line); border-radius: 4px; color: var(--text); }

  /* latency: one stacked bar per reply, newest on top */
  .stages i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 8px; vertical-align: -1px; }
  .stages i:first-child { margin-left: 0; }
  .lat { display: grid; grid-template-columns: 1fr 52px; gap: 6px; align-items: center; margin-top: 5px; }
  .lat .bar { display: flex; height: 12px; gap: 2px; }
  .lat .bar span { height: 100%; border-radius: 2px; min-width: 2px; }
  .lat .bar span:last-child { border-radius: 2px 4px 4px 2px; }
  .lat .total { color: var(--text); text-align: right; }
  .lat .what { grid-column: 1 / -1; color: var(--faint); margin-top: -3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .axis { display: grid; grid-template-columns: 1fr 52px; gap: 6px; color: var(--faint); font-size: 10px; }
  .axis div { display: flex; justify-content: space-between; }

  /* recall: what was found for the last question, and how well it matched */
  #recall .query { color: var(--faint); margin-bottom: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .hit { display: grid; grid-template-columns: 54px 110px 34px 1fr; gap: 6px; align-items: center; margin-top: 3px; }
  .hit .kind { color: var(--faint); }
  .hit .sim { position: relative; height: 10px; background: #262624; border-radius: 2px; }
  .hit .sim span { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 2px 4px 4px 2px; }
  .hit .sim b { position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--text); }
  .hit .val { color: var(--text); text-align: right; }
  .hit .text { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .hit .kw { color: var(--text); border: 1px solid var(--line); border-radius: 3px; padding: 0 3px; margin-right: 4px; font-size: 10px; }

  /* log: one row per line, tagged by where it came from */
  #logbox { flex: 1 0 220px; min-height: 220px; display: flex; flex-direction: column; padding: 8px 12px; }
  #filters { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 6px; }
  #filters button { font: 11px monospace; color: var(--muted); background: none; border: 1px solid var(--line);
                    border-left: 3px solid var(--c, var(--faint)); border-radius: 3px; padding: 1px 6px; cursor: pointer; }
  #filters button.off { opacity: 0.35; }
  #log { flex: 1; min-height: 0; overflow: auto; }
  .row { display: grid; grid-template-columns: 58px 58px 1fr; gap: 6px; padding: 1px 0; border-bottom: 1px solid #222220; }
  .row .t { color: var(--faint); }
  .row .tag { color: var(--muted); border-left: 3px solid var(--c, var(--faint)); padding-left: 5px; overflow: hidden; }
  .row .msg { color: var(--text); white-space: pre-wrap; overflow-wrap: anywhere; }
  .row details { grid-column: 3; padding: 0; border: 0; }
  .row details summary { color: var(--text); margin: 0; }
  .row details pre { margin: 2px 0 4px; color: var(--muted); white-space: pre-wrap; }
  .row.WARNING .msg { color: #e6b450; }
  .row.ERROR .msg, .row.CRITICAL .msg { color: #e66767; }

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
  <div id="sent"></div>
  <div id="subs"></div>
</div>
<aside id="side">
  <header>pal -- <span id="status">disconnected</span> -- <a href="#" id="clear">clear</a></header>
  <details id="score" open>
    <summary>Speaker score</summary>
    <div id="latest">--</div>
    <div class="legend"><i style="background:var(--you)"></i>you <i style="background:var(--other)"></i>other
      <span style="margin-left:10px">--- threshold</span></div>
    <div class="sub">live 20 s: <b>∿</b> last 1 s of speech, <b>▬</b> utterance</div>
    <canvas id="live" role="img" aria-label="Live speaker score, last 20 seconds"></canvas>
    <div class="sub">per utterance, last 40</div>
    <svg id="chart" role="img" aria-label="Speaker score per utterance"></svg>
  </details>
  <details id="latency" open>
    <summary>Latency <span class="note">speech end → first audio</span></summary>
    <div class="stages" id="stages"></div>
    <div id="latbars"><div class="sub" style="color:var(--faint)">no replies yet</div></div>
  </details>
  <details id="recall" open>
    <summary>Recall <span class="note">last question, | = cutoff</span></summary>
    <div id="hits"><div class="query">nothing searched yet</div></div>
  </details>
  <section id="logbox">
    <div id="filters"></div>
    <div id="log"></div>
  </section>
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

  // --- log ---
  // Where a line came from (the logger's last name part) -> a short tag and its colour
  const TAGS = {
    router: ['route', '#c98500'], gate: ['route', '#c98500'], listen: ['hear', '#9085e9'],
    see: ['look', '#199e70'], describe: ['look', '#199e70'], agent: ['pal', '#c3c2b7'],
    responder: ['reply', '#d55181'], remote: ['net', '#3987e5'], store: ['memory', '#008300'],
    recall: ['memory', '#008300'], knowledge: ['docs', '#008300'], voice: ['voice', '#d95926'],
  };
  const MAX_ROWS = 1500;
  let hidden = new Set();
  try { hidden = new Set(JSON.parse(localStorage.getItem('pal-hidden-tags') || '[]')); } catch (e) {}
  const seenTags = new Map();  // tag -> colour, for the filter buttons
  function tagFor(name) { return TAGS[name] || [name || 'log', '#8a8984']; }
  function renderFilters() {
    $('filters').replaceChildren(...[...seenTags].map(([tag, c]) => {
      const b = document.createElement('button');
      b.textContent = tag; b.style.setProperty('--c', c);
      b.className = hidden.has(tag) ? 'off' : '';
      b.onclick = () => {
        hidden.has(tag) ? hidden.delete(tag) : hidden.add(tag);
        try { localStorage.setItem('pal-hidden-tags', JSON.stringify([...hidden])); } catch (e) {}
        renderFilters();
        for (const row of log.children) row.style.display = hidden.has(row.dataset.tag) ? 'none' : '';
      };
      return b;
    }));
  }
  function addLog({ t, name, level, msg }) {
    const [tag, c] = tagFor(name);
    if (!seenTags.has(tag)) { seenTags.set(tag, c); renderFilters(); }
    const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
    const row = document.createElement('div');
    row.className = 'row ' + level; row.dataset.tag = tag;
    if (hidden.has(tag)) row.style.display = 'none';
    const time = document.createElement('span');
    time.className = 't'; time.textContent = new Date(t * 1000).toTimeString().slice(0, 8);
    const label = document.createElement('span');
    label.className = 'tag'; label.textContent = tag; label.style.setProperty('--c', c);
    const [first, ...rest] = msg.split('\\n');
    let body;
    if (rest.length) {  // e.g. a memory update: one line, the details folded away
      body = document.createElement('details');
      const sum = document.createElement('summary'); sum.textContent = first;
      const pre = document.createElement('pre'); pre.textContent = rest.join('\\n');
      body.append(sum, pre);
    } else {
      body = document.createElement('span'); body.className = 'msg'; body.textContent = first;
    }
    row.append(time, label, body);
    log.append(row);
    while (log.children.length > MAX_ROWS) log.firstChild.remove();
    if (atBottom) log.scrollTop = log.scrollHeight;
  }
  function addLine(line) { addLog({ t: Date.now() / 1000, name: '', level: 'INFO', msg: line }); }

  // --- latency ---
  const STAGES = [['hearing', '#9085e9'], ['gate', '#c98500'], ['look + recall', '#199e70'],
                  ['model', '#d55181'], ['voice', '#008300']];
  const STAGE_COLOR = Object.fromEntries(STAGES);
  $('stages').innerHTML = STAGES.map(([n, c]) => `<i style="background:${c}"></i>${n}`).join(' ');
  let replies = [];
  function addLatency(r) {
    replies.unshift(r);
    replies = replies.slice(0, 6);
    const scale = Math.max(5, ...replies.map((x) => x.total));  // seconds across; same for all rows
    const rows = replies.map((x) => {
      const bar = x.segments.map(([name, s]) =>
        `<span style="width:${(100 * s / scale).toFixed(2)}%;background:${STAGE_COLOR[name] || '#8a8984'}" ` +
        `data-tip="${name} ${s.toFixed(2)} s"></span>`).join('');
      const what = x.text.replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]));
      return `<div class="lat"><div class="bar">${bar}</div><div class="total">${x.total.toFixed(2)} s</div>` +
             `<div class="what">${what}${x.parallel ? ' · gate ran alongside' : ''}</div></div>`;
    }).join('');
    const ticks = Array.from({ length: Math.floor(scale) + 1 }, (_, i) => `<span>${i}</span>`).join('');
    $('latbars').innerHTML = rows + `<div class="axis"><div>${ticks}</div><span>s</span></div>`;
  }

  // --- recall ---
  const SIM_LOW = 0.4;  // left end of the similarity bars
  function addRecall({ query, hits }) {
    const box = $('hits');
    const q = document.createElement('div');
    q.className = 'query'; q.textContent = 'for: ' + query;
    if (!hits.length) {
      const none = document.createElement('div');
      none.className = 'query'; none.textContent = 'nothing matched well enough';
      box.replaceChildren(q, none);
      return;
    }
    const pos = (v) => (100 * Math.max(0, Math.min(1, (v - SIM_LOW) / (1 - SIM_LOW)))).toFixed(1) + '%';
    box.replaceChildren(q, ...hits.map((h) => {
      const row = document.createElement('div');
      row.className = 'hit';
      row.dataset.tip = `${h.similarity.toFixed(2)} similarity, cutoff ${h.cutoff.toFixed(2)}` +
        (h.keyword ? ', keyword match' : '') + '\\n' + (h.at ? h.at + ' ' : '') + h.text;
      const above = h.similarity >= h.cutoff;
      row.innerHTML = `<span class="kind">${h.kind}</span>` +
        `<div class="sim"><span style="width:${pos(h.similarity)};background:${above ? '#199e70' : '#5a5a56'}"></span>` +
        `<b style="left:${pos(h.cutoff)}"></b></div><span class="val"></span><span class="text"></span>`;
      row.querySelector('.val').textContent = h.similarity.toFixed(2);
      const text = row.querySelector('.text');
      if (h.keyword) { const kw = document.createElement('span'); kw.className = 'kw'; kw.textContent = 'kw'; text.append(kw); }
      text.append(h.text);
      return row;
    }));
  }

  // tooltips for anything with data-tip (latency segments, recall rows)
  document.addEventListener('mousemove', (e) => {
    const el = e.target.closest && e.target.closest('[data-tip]');
    if (!el) { if (tip.dataset.owner === 'dom') { tip.style.display = 'none'; tip.dataset.owner = ''; } return; }
    tip.dataset.owner = 'dom';
    tip.style.display = 'block'; tip.style.whiteSpace = 'pre-wrap';
    tip.textContent = el.dataset.tip;
    tip.style.left = Math.min(e.clientX + 12, innerWidth - 290) + 'px';
    tip.style.top = (e.clientY + 12) + 'px';
  });

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

  // Live chart: drawn every animation frame so it scrolls smoothly; points arrive ~8/s while speaking.
  const LIVE_SECONDS = 20, GAP_SECONDS = 0.4;
  let livePoints = [];  // {t, score}
  const live = $('live'), css = getComputedStyle(document.documentElement);
  const color = (name) => css.getPropertyValue(name).trim();
  function drawLive() {
    const dpr = devicePixelRatio || 1, w = live.clientWidth, h = live.clientHeight;
    if (live.width !== w * dpr || live.height !== h * dpr) { live.width = w * dpr; live.height = h * dpr; }
    const g = live.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const left = 26, top = 6, plotH = h - top - 4, now = Date.now() / 1000;
    const x = (t) => left + (w - left) * (1 - (now - t) / LIVE_SECONDS);
    const y = (v) => top + plotH * (1 - Math.max(0, Math.min(1, v)));
    g.font = '10px monospace'; g.fillStyle = color('--faint'); g.textAlign = 'right';
    for (const v of [0, 0.5, 1]) {
      g.strokeStyle = '#383835'; g.lineWidth = 1; g.beginPath(); g.moveTo(left, y(v)); g.lineTo(w, y(v)); g.stroke();
      g.fillText(v.toFixed(1), left - 4, y(v) + 3);
    }
    const threshold = utterances.length ? utterances[utterances.length - 1].threshold : 0.5;
    g.setLineDash([4, 3]); g.strokeStyle = color('--muted');
    g.beginPath(); g.moveTo(left, y(threshold)); g.lineTo(w, y(threshold)); g.stroke(); g.setLineDash([]);
    // whole utterances: a span at their final score, over the time they were spoken
    g.lineWidth = 3; g.lineCap = 'round';
    for (const u of utterances) {
      if (!u.ended_at || u.ended_at < now - LIVE_SECONDS) continue;
      g.strokeStyle = u.is_user ? color('--you') : color('--other');
      g.globalAlpha = 0.55;
      g.beginPath(); g.moveTo(Math.max(left, x(u.ended_at - u.seconds)), y(u.score)); g.lineTo(x(u.ended_at), y(u.score)); g.stroke();
      g.globalAlpha = 1;
    }
    // last second of speech: a line, coloured per segment by which side of the threshold it ends on
    livePoints = livePoints.filter((p) => p.t >= now - LIVE_SECONDS - 1);
    g.lineWidth = 2;
    for (let i = 1; i < livePoints.length; i++) {
      const a = livePoints[i - 1], b = livePoints[i];
      if (b.t - a.t > GAP_SECONDS) continue;  // silence between utterances: leave a gap
      g.strokeStyle = b.score >= threshold ? color('--you') : color('--other');
      g.beginPath(); g.moveTo(x(a.t), y(a.score)); g.lineTo(x(b.t), y(b.score)); g.stroke();
    }
    const last = livePoints[livePoints.length - 1];
    if (last && now - last.t < GAP_SECONDS) {
      g.fillStyle = last.score >= threshold ? color('--you') : color('--other');
      g.beginPath(); g.arc(x(last.t), y(last.score), 4, 0, 2 * Math.PI); g.fill();
    }
    requestAnimationFrame(drawLive);
  }
  requestAnimationFrame(drawLive);

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

  const SENT_SHOW_MS = 8000;
  let sentTimer = null;
  function showSent({ t, frames }) {
    const age = Date.now() - t * 1000;
    if (age > SENT_SHOW_MS) return;  // replayed history from before this page opened
    const sent = $('sent'), stage = $('stage');
    const label = document.createElement('div');
    label.className = 'label';
    label.textContent = 'sent ' + frames.length + ' frame' + (frames.length === 1 ? '' : 's') + ' to model';
    const figures = frames.map((f) => {
      const fig = document.createElement('figure');
      const img = document.createElement('img');
      img.src = f.src;
      const cap = document.createElement('figcaption');
      cap.textContent = f.ago < 0.05 ? 'now' : '-' + f.ago.toFixed(1) + 's';
      fig.append(img, cap);
      return fig;
    });
    sent.replaceChildren(label, document.createElement('br'), ...figures);
    sent.style.opacity = 1;
    sent.classList.add('on');
    stage.classList.add('sending');
    setTimeout(() => stage.classList.remove('sending'), 1200);
    clearTimeout(sentTimer);
    sentTimer = setTimeout(() => {
      sent.style.opacity = 0;
      sentTimer = setTimeout(() => sent.classList.remove('on'), 1000);
    }, SENT_SHOW_MS - age - 1000);
  }

  function clearAll() {
    log.replaceChildren(); subs.replaceChildren(); utterances = []; livePoints = []; replies = [];
    $('latbars').replaceChildren(); $('hits').replaceChildren();
    $('sent').classList.remove('on');
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
      else if (msg.sent) showSent(msg.sent);
      else if (msg.live) livePoints.push(msg.live);
      else if (msg.log) addLog(msg.log);
      else if (msg.latency) addLatency(msg.latency);
      else if (msg.recall) addRecall(msg.recall);
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

    def add_log(self, record: logging.LogRecord, message: str):
        """A log record for the log panel, tagged by the module it came from."""
        self._push({"log": {"t": record.created, "name": record.name.rsplit(".", 1)[-1],
                            "level": record.levelname, "msg": message}})

    def add_latency(self, text: str, marks: dict[str, float]):
        """Where the time went between the end of what the user said and the first audio of the
        reply, from the stage marks (pal.debug.trace) plus the responder's own."""
        # The reply may have started before the gate finished (pal.routing.router, parallel_gate);
        # then the gate ran alongside the model and isn't a step of its own.
        parallel = "start" in marks and marks["start"] < marks.get("gate_end", 0)
        if parallel:
            stages = [("hearing", "speech_end", "transcribed"), ("look + recall", "transcribed", "start")]
        else:
            stages = [("hearing", "speech_end", "transcribed"), ("gate", "transcribed", "gate_end"),
                      ("look + recall", "gate_end", "start")]
        stages += [("model", "start", "first_sentence"), ("voice", "first_sentence", "first_audio")]
        segments = [(name, marks[b] - marks[a]) for name, a, b in stages if a in marks and b in marks]
        if not segments or "first_audio" not in marks:
            return  # nothing was said aloud
        total = marks["first_audio"] - min(marks.values())
        self._push({"latency": {"text": text, "segments": segments, "total": total, "parallel": parallel}})

    def add_recall(self, query: str, hits: list, knowledge: list):
        """What automatic recall found for a question: memories and reference material, each with
        its similarity to the question and the cutoff a meaning match needed."""
        rows = [{"kind": "doc" if h in knowledge else h.entry.kind,
                 "at": f"{h.entry.at.astimezone():%H:%M}" if h.entry.at else "",
                 "text": h.entry.text, "similarity": h.similarity, "keyword": h.keyword > 0, "cutoff": h.cutoff}
                for h in [*hits, *knowledge]]
        self._push({"recall": {"query": query, "hits": rows}})

    def add_speech(self, who: str, text: str):
        """Something said, by "user", "other" or "pal": shown over the camera image."""
        self._push({"speech": {"who": who, "text": text}})

    @property
    def enabled(self) -> bool:
        """For callers that would compute something only the preview needs."""
        return self._enabled

    def add_utterance(self, score: float, threshold: float, is_user: bool, seconds: float, text: str,
                      ended_at: float):
        """A finished utterance: a bar in the history chart and a span in the live one."""
        self._push({"utterance": {"score": score, "threshold": threshold, "is_user": is_user,
                                  "seconds": seconds, "text": text, "ended_at": ended_at}})

    def add_live_score(self, score: float):
        """The speaker score of the last moment of speech, for the live line. Not kept for
        clients that connect later: there are many, and they're only worth seeing as they come."""
        self._push({"live": {"t": time.time(), "score": score}}, keep=False)

    def reset(self):
        """Clear server history and tell connected clients to clear."""
        if not self._enabled:
            return
        self._history.clear()
        self._broadcast(self._encode({"reset": True}))

    def show_sent(self, frames):
        """Frames just sent to the model, as (BGR image, seconds ago) pairs: shown over the image for a while."""
        if not self._enabled:
            return
        thumbs = []
        for image, ago in frames:
            h, w = image.shape[:2]
            small = cv2.resize(image, (int(w * SENT_THUMB_HEIGHT / h), SENT_THUMB_HEIGHT), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                thumbs.append({"src": "data:image/jpeg;base64," + base64.b64encode(buf).decode(), "ago": ago})
        self._push({"sent": {"t": time.time(), "frames": thumbs}})

    # ----- internals -----

    def _push(self, payload: dict, keep: bool = True):
        if not self._enabled:
            return
        message = self._encode(payload)
        if keep:
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
                message = record.getMessage()
                if record.exc_info:
                    message += "\n" + self.formatException(record.exc_info)
                preview.add_log(record, message)
        except Exception:
            self.handleError(record)
