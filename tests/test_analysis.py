"""Tests for analysis functions: energy score, motion, mood, audio."""

import numpy as np
import pytest


# ── Energy Score ──────────────────────────────────────────────────────────

def _energy(audio, visual, crowd, motion=None, mood=None):
    """Inline copy of calculate_energy_score for testing without Streamlit."""
    bpm_score = min(max((audio.get("bpm", 100) - 60) / 140 * 100, 0), 100)
    volume_score = min(max((audio.get("volume_level", 70) - 40) / 60 * 100, 0), 100)
    audio_energy = (bpm_score + volume_score) / 2
    brightness = visual.get("brightness_level", 50)
    vmap = {"low": 30, "medium": 50, "high": 80}
    visual_activity = vmap.get(visual.get("visual_energy", "medium"), 50)
    visual_score = brightness * 0.4 + visual_activity * 0.6
    motion_score = motion["motion_score"] if motion and "motion_score" in motion else 50.0
    crowd_score = (crowd.get("density_score", 5) / 20) * 100
    mood_score = mood["mood_score"] if mood and "mood_score" in mood else 50.0
    raw = (audio_energy * 0.25 + visual_score * 0.15 +
           motion_score * 0.20 + crowd_score * 0.20 + mood_score * 0.20)
    return round(max(0, min(100, raw)), 1)


def test_energy_high():
    score = _energy(
        {"bpm": 140, "volume_level": 90},
        {"brightness_level": 80, "visual_energy": "high"},
        {"density_score": 15},
        {"motion_score": 85},
        {"mood_score": 90},
    )
    assert score > 65


def test_energy_low():
    score = _energy(
        {"bpm": 70, "volume_level": 45},
        {"brightness_level": 20, "visual_energy": "low"},
        {"density_score": 2},
        {"motion_score": 10},
        {"mood_score": 30},
    )
    assert score < 35


def test_energy_clamped():
    score = _energy(
        {"bpm": 300, "volume_level": 200},
        {"brightness_level": 200, "visual_energy": "high"},
        {"density_score": 100},
        {"motion_score": 200},
        {"mood_score": 200},
    )
    assert score == 100.0


def test_energy_defaults():
    """No motion/mood → defaults to 50."""
    score = _energy(
        {"bpm": 120, "volume_level": 75},
        {"brightness_level": 50, "visual_energy": "medium"},
        {"density_score": 8},
    )
    assert 30 < score < 70


# ── Motion Detection ──────────────────────────────────────────────────────

def test_motion_static_frames():
    from utils.video_processing import compute_motion_score
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frames = [frame.copy() for _ in range(5)]
    result = compute_motion_score(frames)
    assert result["motion_score"] == 0.0
    assert result["motion_level"] == "low"


def test_motion_random_frames():
    from utils.video_processing import compute_motion_score
    frames = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(5)]
    result = compute_motion_score(frames)
    assert 0 <= result["motion_score"] <= 100
    assert result["motion_level"] in ("low", "medium", "high")


def test_motion_single_frame():
    from utils.video_processing import compute_motion_score
    result = compute_motion_score([np.zeros((10, 10, 3), dtype=np.uint8)])
    assert result["motion_score"] == 50.0  # fallback


# ── Mood Extraction ───────────────────────────────────────────────────────

def test_mood_happy():
    from utils.video_processing import extract_mood_from_faces
    faces = [{"joy": "VERY_LIKELY", "sorrow": "VERY_UNLIKELY"}]
    mood = extract_mood_from_faces(faces)
    assert mood["dominant_mood"] == "happy"
    assert mood["mood_score"] > 50


def test_mood_empty():
    from utils.video_processing import extract_mood_from_faces
    mood = extract_mood_from_faces([])
    assert mood["dominant_mood"] == "neutral"
    assert mood["mood_score"] == 50.0


def test_mood_no_sentiment_keys():
    from utils.video_processing import extract_mood_from_faces
    faces = [{"x1": 0, "y1": 0, "x2": 100, "y2": 100}]
    mood = extract_mood_from_faces(faces)
    assert mood["dominant_mood"] == "neutral"


# ── Audio Analysis ────────────────────────────────────────────────────────

def test_audio_simulated():
    from utils.audio_analysis import analyze_audio_simulated
    result = analyze_audio_simulated("test.mp4")
    assert result["is_real"] is False
    assert 60 <= result["bpm"] <= 200
    assert result["genre"] in ["Electronic", "Hip-Hop", "Pop", "Latin", "Rock", "House"]


def test_audio_simulated_deterministic():
    from utils.audio_analysis import analyze_audio_simulated
    r1 = analyze_audio_simulated("same_path.mp4")
    r2 = analyze_audio_simulated("same_path.mp4")
    assert r1["bpm"] == r2["bpm"]


# ── Validation ────────────────────────────────────────────────────────────

def test_validate_video_ok():
    from utils.video_processing import validate_video
    errors = validate_video(10 * 1024 * 1024, 30, 5, 60, 100)
    assert errors == []


def test_validate_video_too_large():
    from utils.video_processing import validate_video
    errors = validate_video(200 * 1024 * 1024, 30, 5, 60, 100)
    assert len(errors) == 1
    assert "too large" in errors[0].lower()


def test_validate_video_too_short():
    from utils.video_processing import validate_video
    errors = validate_video(5 * 1024 * 1024, 2, 5, 60, 100)
    assert len(errors) == 1
    assert "too short" in errors[0].lower()


def test_validate_video_too_long():
    from utils.video_processing import validate_video
    errors = validate_video(5 * 1024 * 1024, 120, 5, 60, 100)
    assert len(errors) == 1
    assert "too long" in errors[0].lower()
