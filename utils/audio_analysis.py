"""
Audio analysis — real extraction via Librosa with simulated fallback.
"""

import numpy as np
import tempfile
import os

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False


def _extract_audio_from_video(video_path: str) -> str | None:
    """Extract audio track from video to a temporary WAV file."""
    if not MOVIEPY_AVAILABLE:
        return None
    try:
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            clip.close()
            return None
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        clip.audio.write_audiofile(tmp.name, logger=None)
        clip.close()
        return tmp.name
    except Exception:
        return None


def analyze_audio_real(video_path: str) -> dict:
    """Extract real BPM, volume, and energy from a video's audio track.

    Uses Librosa for tempo detection and RMS volume analysis.
    Falls back to simulated values if extraction fails.
    """
    if not LIBROSA_AVAILABLE:
        return analyze_audio_simulated(video_path)

    audio_path = _extract_audio_from_video(video_path)
    if audio_path is None:
        return analyze_audio_simulated(video_path)

    try:
        # Load audio (mono, 22050 Hz sample rate)
        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=60)

        # Tempo (BPM)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(float(np.atleast_1d(tempo)[0]))

        # RMS volume (convert to approximate dB-like scale 0-100)
        rms = librosa.feature.rms(y=y)[0]
        avg_rms = float(np.mean(rms))
        # Map RMS to a 0-100 scale (RMS typically 0.0-0.5 for normalized audio)
        volume_level = round(min(100, max(0, avg_rms * 200)), 1)

        # Spectral centroid (brightness indicator — higher = brighter/more energetic)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        avg_centroid = float(np.mean(centroid))

        # Energy level based on RMS + tempo
        if avg_rms > 0.15 and bpm > 120:
            energy_level = "high"
        elif avg_rms > 0.08 or bpm > 100:
            energy_level = "medium"
        else:
            energy_level = "low"

        # Tempo consistency (std of beat intervals)
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        if len(beat_frames) > 2:
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            intervals = np.diff(beat_times)
            consistency = round(1.0 - min(1.0, float(np.std(intervals) / (np.mean(intervals) + 1e-6))), 2)
        else:
            consistency = 0.5

        # Genre estimation based on spectral features (simplified heuristic)
        if avg_centroid > 3000 and bpm > 120:
            genre = "Electronic"
        elif avg_centroid > 2500 and bpm > 110:
            genre = "House"
        elif bpm > 130:
            genre = "Hip-Hop"
        elif avg_centroid < 1500:
            genre = "Jazz"
        elif bpm < 100:
            genre = "Lounge"
        else:
            genre = "Pop"

        return {
            "bpm": bpm,
            "volume_level": volume_level,
            "genre": genre,
            "energy_level": energy_level,
            "tempo_consistency": consistency,
            "is_real": True,
        }

    except Exception:
        return analyze_audio_simulated(video_path)
    finally:
        try:
            os.unlink(audio_path)
        except Exception:
            pass


def analyze_audio_simulated(video_path: str) -> dict:
    """Return simulated audio metrics (fallback when Librosa unavailable)."""
    np.random.seed(hash(video_path) % 2**32)

    return {
        "bpm": int(np.random.uniform(90, 140)),
        "volume_level": round(np.random.uniform(65, 95), 1),
        "genre": np.random.choice(
            ["Electronic", "Hip-Hop", "Pop", "Latin", "Rock", "House"]
        ),
        "energy_level": np.random.choice(["low", "medium", "high"]),
        "tempo_consistency": round(np.random.uniform(0.7, 0.95), 2),
        "is_real": False,
    }
