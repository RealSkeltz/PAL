# PAL

PAL is a wearable voice assistant ("Buddy" is the name it answers to). The wearer has a
camera (eye level), a microphone and an earpiece (Shokz bone-conduction
headphones); PAL listens, looks when it helps, remembers what happened, and answers in
the earpiece. The longer-term aim is an assistant that learns the wearer's world: the
people, places and things around them. A demo for security personnel (a guard on patrol)
is one target; the software is meant to stay hardware-agnostic.

## Running it

```bash
poetry run python main.py --preview            # laptop mic, webcam and speaker; preview at http://127.0.0.1:4200
./start-remote.sh                              # phone app as mic/camera/earpiece over LiveKit + Tailscale, detection on
poetry run python main.py --text               # type instead of speaking; "> text" / "other: text" are overheard
poetry run python -m pal.perception.vision.see # watch object detection and recognition on the webcam
```

Flags: `--yolo` (object detection and recognition), `--remote` (phone app), `--knowledge PATH`
(extra reference documents), `--verbose`. The response model comes from `PAL_MODEL` in `.env`
(e.g. `foundry:claude-haiku-5-5`, or `ollama:...` for a local one). Each run logs to
`data/logs/<timestamp>/` (`pal.log`, `session.jsonl`); the preview shows the camera, speech,
speaker score, latency per reply, recall hits and the log.

`tests/manual_test_script.md` is the manual test walkthrough. `research/` holds notebooks
and experiments.

## How it fits together

```
perception ──► events ──► router ──► memory ──► responder ──► voice
  hearing: VAD, Whisper STT, speaker ID (is this the user?)
  vision: frame buffer for looks; optional YOLOE + tracking + DINOv2 recognition
  router: the gate (small local model, tev1:4b) decides "is this for PAL?" and "should it look?"
```

- `pal/agent.py` wires the threads together (`Pal.run`).
- `pal/perception/hearing/listen.py`: speech in, transcribed, scored against the user's voiceprint.
- `pal/perception/vision/`: `see.py` is the capture loop (looks are whole frames from a rolling buffer, `camera.py`); object recognition is `detect.py` (YOLOE + tracking), `embed.py` (DINOv2), `stored.py` (the stored objects, with keys), `objects.py` (`Objects`: following, recognizing, what's in view) and `draw.py`.
- `pal/routing/`: `router.py` sorts events into turns and context; `gate.py` + `system_one.py` are the gate.
- `pal/response/`: `responder.py` runs a reply and speaks it; `run_turn.py` is the model/tool loop; `client_turns.py` turns history into what the model sees.
- `pal/memory/`: everything PAL knows (context, history, situation, user facts, goals, episodes, reference documents) plus hybrid recall (BM25 + vectors). `store.py` holds it, with the seen marker and the edits the agent makes.
- `pal/tools/`: the agent's tools; `memory_tools.py` is how it keeps its own memory. `pal/llm/`: model clients (Anthropic/Foundry, ollama, gate, embeddings).
- `pal/remote.py` + `application/client/`: the Android app and its LiveKit link.
- `pal/debug/preview.py`: the preview page (one HTML string, server-sent events).

## The agent in control

The agent owns its picture of the world; perception and the gate only deliver facts.

- **One agent loop** (`Pal._agent_loop`) with two modes, on one memory:
  - **respond** when there's a question for PAL: answer, then update memory with tools where
    the exchange calls for it.
  - **reflect** when nothing is waiting but there are unseen events (speech not for PAL, looks
    with their description): after 5 s without new input, at most 60 s after the first one,
    update memory with tools. Nothing is said aloud. A question for PAL cuts it off.
- **Memory is kept through tools** (`pal/tools/memory_tools.py`): `set_situation` (as a whole),
  `remember` / `revise` / `forget` (user facts, people, preferences), `add_goal` / `update_goal`,
  `log_episode`. Items have stable keys ("u3", "g2") the agent sees in `Memory.view()`; code keeps
  the timestamps.
- **Seen marker:** every turn gets the events after `Memory.seen`, and moves it on when done.
- **Every turn ends with a note** (`pal/response/prompt.py`: `respond_note`, `reflect_note`) with
  what PAL knows and what it hasn't seen. It's at the end of the conversation, not in the system
  prompt, so memory edits don't break the prompt cache.
- **Replies stay quiet about memory:** memory tools make no sound, and after an answer, a step that
  only updated memory isn't followed by speech (`responder.py`); prompting alone held about half the time.
- **The dividing line:** perception and the gate produce facts without judgment (transcribing,
  detecting, embedding, retrieving, "is this for PAL"); everything that needs interpretation goes
  to the agent. New capabilities are new tools, not new loops or gate categories.

Next on this track: object recognition events (stored/recognized) as events the agent sees, a
tool to name recognized people and objects (people's names confirmed by the user first), and
letting a reflect turn ask the user something. Known gap: if a reply misses something worth
remembering, reflect doesn't catch it, since it only looks at events outside the conversation.

Also known: with two people talking, speech recognition (large-v3-turbo on CPU, ~2 s per
utterance, one at a time) falls behind. Ideas: transcribe queued utterances together, a
small model for other people's speech, or Whisper on a GPU.

## Working on this project

- **Agree on the approach before building a new capability.** Lay out the design decisions
  and a recommendation first; keep the first slice small.
- **Dependencies:** never add or update one without asking first and checking it on OSV.dev.
  Ultralytics' auto-install is off (`YOLO_AUTOINSTALL=false` in `see.py`) for this reason.
- **Never read or edit `.env`.**
- **Models live in `models/`** (gitignored): e.g. `models/vision/yoloe-26m-seg.pt`,
  `models/vision/dinov2-base`. YOLOE's text model `mobileclip2_b.ts` sits in the repo root
  (gitignored) because Ultralytics looks for it in the working directory.
- **Privacy:** recognizing people is biometric data. Face recognition is out of scope for the
  security demo; people's names should be confirmed by the user before they're stored.
- **Style:** match the surrounding code. Modules start with a docstring that explains the
  design; comments say why, not what. American English, no em dashes.
- **Commits:** `feat(...)` / `task(...)` / `fix(...)` with a short description in brackets;
  work on a branch, merge into `development`.
