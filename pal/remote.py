"""PAL's link to a remote device (the phone app), over LiveKit.

The device joins a LiveKit room and publishes its mic (the Bluetooth earpiece)
and camera; PAL joins the same room and publishes its voice. Events go
both ways on the data channel, topic "events", as JSON with a "type".

    device mic ──► 512-sample chunks ──► listen   (instead of the laptop mic)
    device camera ──► latest frame ──► see        (instead of the webcam)
    voice ──► 20 ms frames ──► device earpiece    (instead of the laptop speaker)

The device gets its room token from a small HTTP endpoint served here, so the
only other thing to run is `livekit-server`. LiveKit runs on its own asyncio
loop in a background thread; everything else talks to it through this class.
"""
import asyncio
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Generator
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from livekit import api, rtc

from pal.perception.hearing import listen
from pal.voice.sounds import capture_tone
from pal.voice.voice import SAMPLE_RATE as VOICE_RATE, Voice

logger = logging.getLogger(__name__)

# Defaults match `livekit-server --dev`, running on this machine.
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")
API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")
ROOM = "pal"
IDENTITY = "pal"
TOKEN_PORT = 8000

MIC_RATE = 16000
MIC_CHUNK_MS = 32  # 512 samples at 16 kHz: what the VAD takes
VOICE_FRAME = VOICE_RATE // 50  # 20 ms
VOICE_BUFFER_MS = 100  # audio queued ahead in LiveKit; this much can still play after a barge-in stop
RECONNECT_AFTER = 5.0


def make_token(identity: str) -> str:
    return (
        api.AccessToken(API_KEY, API_SECRET)
        .with_identity(identity)
        .with_grants(api.VideoGrants(room_join=True, room=ROOM))
        .to_jwt()
    )


class Remote:
    def __init__(self, voice: Voice, token_port: int = TOKEN_PORT):
        self.voice = voice
        self.token_port = token_port
        self._loop = asyncio.new_event_loop()
        self._room: rtc.Room | None = None
        self._source: rtc.AudioSource | None = None
        self._tasks: set[asyncio.Task] = set()

        self._frame: np.ndarray | None = None  # latest camera frame, BGR
        self._frame_at = 0.0
        self._frame_interval = 0.0
        self._new_frame = threading.Condition()

    def start(self):
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        asyncio.run_coroutine_threadsafe(self._run(), self._loop)
        threading.Thread(target=self._serve_tokens, daemon=True).start()

    # --- Public ------------------------------------------------------------

    def frames(self, fps: float) -> Generator[np.ndarray, None, None]:
        """The device camera's frames, at most `fps` per second; waits while the device is away."""
        self._frame_interval = 1 / fps
        last = None
        while True:
            with self._new_frame:
                self._new_frame.wait_for(lambda: self._frame is not last)
                last = self._frame
            yield last

    def send(self, event: dict):
        """Send an event to the device. Dropped when no device is connected."""
        if self._room is not None:
            asyncio.run_coroutine_threadsafe(self._publish(event), self._loop)

    # --- LiveKit -------------------------------------------------------------

    async def _run(self):
        self._source = rtc.AudioSource(VOICE_RATE, 1, queue_size_ms=VOICE_BUFFER_MS)
        self.voice.on_stop = lambda: self._loop.call_soon_threadsafe(self._source.clear_queue)
        self._spawn(self._speak())
        while True:
            room = rtc.Room(loop=self._loop)
            disconnected = asyncio.Event()
            room.on("track_subscribed", self._on_track)
            room.on("data_received", self._on_data)
            room.on("participant_connected", lambda p: logger.info(f"device joined ({p.identity})"))
            room.on("participant_disconnected", lambda p: logger.info(f"device left ({p.identity})"))
            room.on("disconnected", lambda *_: disconnected.set())
            try:
                await room.connect(LIVEKIT_URL, make_token(IDENTITY))
                track = rtc.LocalAudioTrack.create_audio_track("pal-voice", self._source)
                await room.local_participant.publish_track(
                    track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
            except Exception as e:
                logger.warning(f"can't reach LiveKit at {LIVEKIT_URL} ({e}); retrying in {RECONNECT_AFTER:.0f}s")
                await asyncio.sleep(RECONNECT_AFTER)
                continue
            self._room = room
            logger.info(f"in LiveKit room '{ROOM}', waiting for a device")
            await disconnected.wait()
            self._room = None
            logger.warning("lost the LiveKit room, reconnecting")

    def _spawn(self, coro):
        task = self._loop.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _on_track(self, track: rtc.Track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"hearing {participant.identity}'s mic")
            self.voice.play(capture_tone())  # so the user hears the link is up
            self._spawn(self._hear(track))
        elif track.kind == rtc.TrackKind.KIND_VIDEO:
            logger.info(f"seeing {participant.identity}'s camera")
            self._spawn(self._see(track))

    async def _hear(self, track: rtc.Track):
        stream = rtc.AudioStream(track, sample_rate=MIC_RATE, num_channels=1, frame_size_ms=MIC_CHUNK_MS)
        async for event in stream:
            pcm = np.frombuffer(event.frame.data, dtype=np.int16)
            listen.feed(pcm.astype(np.float32) / 32768)

    async def _see(self, track: rtc.Track):
        stream = rtc.VideoStream(track)
        async for event in stream:
            now = time.monotonic()
            if now - self._frame_at < self._frame_interval:
                continue  # converting every frame costs more than see ever looks at
            self._frame_at = now
            frame = event.frame.convert(rtc.VideoBufferType.RGB24)
            rgb = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
            with self._new_frame:
                self._frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                self._new_frame.notify_all()

    async def _speak(self):
        # Silence included: pushing continuously keeps the pace, since capture_frame
        # waits while VOICE_BUFFER_MS is already queued.
        while True:
            samples = self.voice.read(VOICE_FRAME)
            frame = rtc.AudioFrame.create(VOICE_RATE, 1, VOICE_FRAME)
            np.frombuffer(frame.data, dtype=np.int16)[:] = np.clip(samples, -1, 1) * 32767
            await self._source.capture_frame(frame)

    def _on_data(self, packet: rtc.DataPacket):
        if packet.topic != "events":
            return
        try:
            event = json.loads(packet.data)
        except ValueError:
            logger.warning(f"unreadable event from the device: {packet.data[:100]!r}")
            return
        logger.info(f"device event: {event}")
        if event.get("type") == "ping":
            self.send({"type": "pong", "t": event.get("t")})

    async def _publish(self, event: dict):
        room = self._room
        if room is not None:
            await room.local_participant.publish_data(json.dumps(event), reliable=True, topic="events")

    # --- Token endpoint ------------------------------------------------------

    def _serve_tokens(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                url = urlparse(self.path)
                if url.path != "/token":
                    self.send_error(404)
                    return
                identity = parse_qs(url.query).get("identity", ["device"])[0]
                body = json.dumps({"token": make_token(identity)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                logger.info(f"handed out a room token to {identity}")

        server = ThreadingHTTPServer(("0.0.0.0", self.token_port), Handler)
        logger.info(f"token endpoint on port {self.token_port}")
        server.serve_forever()
