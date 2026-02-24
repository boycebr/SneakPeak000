"""
Audio analysis utilities.
MVP uses simulated values; production will integrate Librosa or cloud APIs.
"""

import numpy as np


def analyze_audio_simulated(video_path: str) -> dict:
    """Return simulated audio metrics (MVP placeholder).

    In production, replace with real audio extraction via Librosa,
    Azure Audio, or similar.
    """
    np.random.seed(hash(video_path) % 2**32)

    return {
        "bpm": int(np.random.uniform(90, 140)),
        "volume_level": round(np.random.uniform(65, 95), 1),
        "genre": np.random.choice(
            ["Electronic", "Hip-Hop", "Pop", "Latin", "Rock", "House"]
        ),
        "energy_level": np.random.choice(["low", "medium", "high"]),
        "tempo_consistency": round(np.random.uniform(0.7, 0.95), 2),
    }
