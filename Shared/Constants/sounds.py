import numpy as np
import random

SAMPLE_RATE = 24000


def _make_chime(freqs, duration_s, sample_rate=SAMPLE_RATE, volume=0.25,
                decay_power=2.5, attack_s=0.02):
    """Layered sines with polynomial decay — sounds like a soft chime."""
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    samples = np.zeros_like(t)
    for i, f in enumerate(freqs):
        samples += np.sin(2 * np.pi * f * t) / (i + 1)
    samples /= np.max(np.abs(samples))

    # Polynomial decay: smooth, naturally tapers to zero
    envelope = np.power(1 - t / duration_s, decay_power)
    samples *= envelope * volume

    # Soft attack
    fade_in = int(attack_s * sample_rate)
    samples[:fade_in] *= np.linspace(0, 1, fade_in)
    return samples.astype(np.float32)


def capture_tone():
    base = [523, 659]
    freqs = [f * random.uniform(0.98, 1.02) for f in base]
    duration = random.uniform(0.25, 0.35)
    decay = random.uniform(2.0, 3.0)
    return _make_chime(freqs, duration_s=duration, decay_power=decay)


def processing_tone():
    """Lower, slower chime with subtle variation each time. Occasionally picks a different key."""
    if random.random() < 0.2:
        # Occasional surprise: pick a different key
        pairs = [
            (220, 330),   # A3 + E4
            (294, 440),   # D4 + A4
            (247, 370),   # B3 + F#4
            (277, 415),   # C#4 + G#4
        ]
        base = list(random.choice(pairs))
    else:
        # Most of the time: same notes with subtle wobble
        base = [261, 392]  # C4 + G4

    freqs = [f * random.uniform(0.98, 1.02) for f in base]
    duration = random.uniform(1.3, 1.7)
    decay = random.uniform(2.0, 3.0)
    return _make_chime(freqs, duration_s=duration, decay_power=decay)

def tool_tone():
    """Quick subtle tick for tool execution. Different feel from chimes."""
    base = [880, 1320]  # higher fifth — feels mechanical, not musical
    freqs = [f * random.uniform(0.99, 1.01) for f in base]
    duration = random.uniform(0.12, 0.18)
    decay = random.uniform(3.5, 4.5)  # faster decay = snappier
    return _make_chime(freqs, duration_s=duration, decay_power=decay, volume=0.18)