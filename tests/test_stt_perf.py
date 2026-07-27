import numpy as np
import time
import sys
from faster_whisper import WhisperModel

print("starting", flush=True)
audio = np.random.randn(32000).astype(np.float32) * 0.01
print(f"audio: {len(audio)} samples", flush=True)

configs = [
    {"compute_type": "int8", "cpu_threads": 1},
    {"compute_type": "int8", "cpu_threads": 2},
    {"compute_type": "int8", "cpu_threads": 4},
    {"compute_type": "int8", "cpu_threads": 8},
    {"compute_type": "int8_float32", "cpu_threads": 4},
    {"compute_type": "float32", "cpu_threads": 4},
]

for cfg in configs:
    print(f"\nloading model with {cfg}...", flush=True)
    model = WhisperModel("distil-small.en", device="cpu", **cfg)
    print("loaded, warming up...", flush=True)
    
    list(model.transcribe(audio, language="en", vad_filter=False)[0])
    print("warmed up, timing...", flush=True)
    
    times = []
    for i in range(3):
        t0 = time.time()
        segments, _ = model.transcribe(audio, language="en", vad_filter=False, condition_on_previous_text=False)
        list(segments)
        elapsed = (time.time() - t0) * 1000
        times.append(elapsed)
        print(f"  run {i+1}: {elapsed:.0f}ms", flush=True)
    
    avg = sum(times) / len(times)
    print(f"-> {cfg}: avg {avg:.0f}ms", flush=True)