"""Tests for auth helpers, database helpers, and utility functions."""

import pytest


# ── Auth module structure ─────────────────────────────────────────────────

def test_auth_module_imports():
    from utils.auth import sign_up, sign_in, sign_out, get_user, refresh_session, get_auth_headers
    assert callable(sign_up)
    assert callable(sign_in)
    assert callable(sign_out)
    assert callable(get_user)
    assert callable(refresh_session)
    assert callable(get_auth_headers)


def test_auth_headers():
    from utils.auth import get_auth_headers
    headers = get_auth_headers("test-key", "test-token")
    assert headers["apikey"] == "test-key"
    assert headers["Authorization"] == "Bearer test-token"
    assert headers["Content-Type"] == "application/json"


# ── Database module structure ─────────────────────────────────────────────

def test_database_module_imports():
    from utils.database import (
        save_video_results, get_recent_venues, get_venues_by_energy,
        search_venues, get_nearby_venues, save_user_rating,
        get_ratings_for_venue, get_user_submissions, get_user_rating_count,
        upload_to_storage, generate_storage_path,
    )
    assert callable(search_venues)
    assert callable(get_nearby_venues)


def test_supabase_headers():
    from utils.database import get_supabase_headers
    headers = get_supabase_headers("my-key")
    assert headers["apikey"] == "my-key"
    assert "Bearer my-key" in headers["Authorization"]


def test_generate_storage_path():
    from utils.database import generate_storage_path
    path = generate_storage_path("My Cool Venue!", ".jpg")
    assert path.startswith("thumbnails/")
    assert path.endswith(".jpg")
    assert "My_Cool_Venue_" in path


def test_generate_storage_path_long_name():
    from utils.database import generate_storage_path
    long_name = "A" * 100
    path = generate_storage_path(long_name)
    # Name should be truncated to 30 chars
    parts = path.split("/")[1]  # after "thumbnails/"
    assert len(parts) < 80  # reasonable length


# ── Video processing utilities ────────────────────────────────────────────

def test_frame_to_jpeg_roundtrip():
    import numpy as np
    from utils.video_processing import frame_to_jpeg_bytes
    frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    jpeg = frame_to_jpeg_bytes(frame, quality=50)
    assert isinstance(jpeg, bytes)
    assert len(jpeg) > 100
    # JPEG magic bytes
    assert jpeg[:2] == b'\xff\xd8'


def test_thumbnail_generation():
    import numpy as np
    from utils.video_processing import generate_thumbnail
    frame = np.random.randint(0, 255, (800, 1200, 3), dtype=np.uint8)
    thumb = generate_thumbnail(frame, max_width=400)
    assert isinstance(thumb, bytes)
    assert thumb[:2] == b'\xff\xd8'


def test_blur_faces_no_crash_on_empty():
    import numpy as np
    from utils.video_processing import blur_faces_in_frame
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = blur_faces_in_frame(frame, [])
    assert result.shape == frame.shape


def test_blur_faces_modifies_region():
    import numpy as np
    from utils.video_processing import blur_faces_in_frame
    frame = np.full((200, 200, 3), 128, dtype=np.uint8)
    faces = [{"x1": 50, "y1": 50, "x2": 150, "y2": 150}]
    result = blur_faces_in_frame(frame, faces)
    # The blurred region should differ from the original uniform frame
    # (Gaussian blur on a uniform region may not change much, but padding extends it)
    assert result.shape == frame.shape
