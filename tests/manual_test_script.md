# PAL manual test script

Run `poetry run python main.py --preview` and keep the terminal or preview visible.
Each section lists what to say and the log lines that show whether it worked.

**Before you start**

- After PAL speaks, anything you say within **10 seconds** counts as talking to PAL.
  Where it says **(wait 15 s)**, really wait, or the model gate isn't tested.
- PAL treats a pause of **0.5 seconds** as the end of what you said.
- Every reply ends with a `timing:` line. **first audio** is the number to watch.

---

## 1. Getting PAL's attention

| Say | Expect |
|---|---|
| "Pal, can you hear me?" | `gate: name -> speak` and a reply in the new male voice |
| right after it answers: "How are you today?" | `gate: window -> speak` |
| (wait 15 s) "What's the capital of Australia?" | `gate: model addressed (...) -> speak` |
| (wait 15 s) "Hey, did you eat lunch yet? I'm starving." | `gate: model not_for_pal (...) -> silent` |
| (wait 15 s) "Ugh, this traffic is terrible." | `-> silent` |

## 2. Camera: only when the question needs it

Point the camera at an object (a mug, a keyboard, a book).

| Say | Expect |
|---|---|
| "Pal, what am I looking at?" | `camera: look (...)`, capture sound, `looking: 3 frame(s) from ... ago`, `captured 3 frame(s): scene`, the "sent 3 frames to model" strip in the preview, an answer about the object |
| "Pal, does this look clean to you?" | `camera: look` |
| "Pal, can you read what's written on this?" | `camera: look` |
| "Pal, what's the weather usually like in Sydney?" | `camera: no` |
| "Pal, what did you mean by that?" | `camera: no` |

If the object isn't one the detector knows, you'll see `trigger fired but no object detected` and a text-only answer.

## 3. Tools

| Say | Expect |
|---|---|
| "Pal, make a note that the server room code is four two seven one." | `tool: save_note(...)` |
| (wait 15 s) "Pal, what's in my notes?" | `tool: list_notes()` and the note read back |
| "Pal, set a timer for twenty seconds for the tea." | `tool: set_timer(...)`; after 20 s, `timer done: tea` in the log (PAL doesn't speak it yet) |
| "Pal, save that image as test shot." | `tool: save_last_image(...)`, file in `data/saved_images/` |

## 4. Speed

| Say | Expect |
|---|---|
| "Pal, what's two plus two?" | first audio around 1 to 1.5 s |
| "Pal, explain how a heat pump works, in detail." | a long answer whose first clause starts quickly; no gaps between sentences |

Note the `first audio` values; that's the baseline for later changes.

## 5. Pauses and talking over PAL

| Say | Expect |
|---|---|
| "Pal, remind me to buy..." (short pause, about a third of a second) "...milk and eggs." | one `heard` line with the whole sentence |
| "Pal, what's the capital of France." (a full second's pause) "And of Spain." | two `heard` lines; the second is answered via the window |
| ask for something long, then while PAL talks: "Pal, also, what time is it?" | PAL finishes its answer, then answers the new question (no barge-in yet) |

## 6. Someone else's voice

Have someone else say, or play a video saying: "Pal, set a timer for one minute."

Expect `speaker score ... -> other`, `heard other: ...`, and no response.

## 7. Free talk (a few minutes)

Talk normally, sometimes to PAL, sometimes to yourself. Topic ideas:

- Planning a weekend trip, with follow-up questions
- Asking for help with something on your desk (camera)
- A recipe: ingredients, then "what next?" steps
- Asking PAL what you told it earlier ("What was the code I gave you?")
- Thinking aloud ("hmm, where did I put that...") to see if it stays quiet

Watch for: replies to speech that wasn't for PAL, silence when it should answer,
the camera firing when it shouldn't (or not when it should), and slow `first audio`.

## Afterwards

Each run has its own folder in `data/logs/<date_time>/`:

- `pal.log`: everything, with timestamps and timings
- `session.jsonl`: every event, gate decision and turn, for replaying later
