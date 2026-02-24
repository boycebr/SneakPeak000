"""
Video processing utilities — frame extraction, metadata, face blurring.
"""

import numpy as np
import tempfile
import os

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def extract_video_metadata(video_path: str) -> dict:
    """Extract basic metadata from a video file."""
    metadata = {
        "duration": 0,
        "fps": 0,
        "width": 0,
        "height": 0,
        "frame_count": 0,
    }

    if MOVIEPY_AVAILABLE:
        try:
            with VideoFileClip(video_path) as clip:
                metadata["duration"] = clip.duration
                metadata["fps"] = clip.fps
                metadata["width"] = clip.w
                metadata["height"] = clip.h
                metadata["frame_count"] = int(clip.duration * clip.fps)
        except Exception:
            pass

    if CV2_AVAILABLE and metadata["duration"] == 0:
        try:
            cap = cv2.VideoCapture(video_path)
            metadata["fps"] = cap.get(cv2.CAP_PROP_FPS)
            metadata["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            metadata["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            metadata["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if metadata["fps"] > 0:
                metadata["duration"] = metadata["frame_count"] / metadata["fps"]
            cap.release()
        except Exception:
            pass

    return metadata


def extract_frames(video_path: str, num_frames: int = 5) -> list:
    """Extract evenly-spaced frames from a video."""
    frames = []
    if not CV2_AVAILABLE:
        return frames

    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return frames

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        cap.release()
    except Exception:
        pass

    return frames


def blur_faces_in_frame(frame, face_locations: list):
    """Apply Gaussian blur to detected face regions in a frame."""
    if not CV2_AVAILABLE:
        return frame

    blurred = frame.copy()
    for face in face_locations:
        try:
            vertices = face.get("boundingPoly", {}).get("vertices", [])
            if len(vertices) >= 4:
                x1 = max(0, vertices[0].get("x", 0) - 20)
                y1 = max(0, vertices[0].get("y", 0) - 20)
                x2 = min(frame.shape[1], vertices[2].get("x", 0) + 20)
                y2 = min(frame.shape[0], vertices[2].get("y", 0) + 20)

                region = blurred[y1:y2, x1:x2]
                if region.size > 0:
                    blurred[y1:y2, x1:x2] = cv2.GaussianBlur(region, (99, 99), 30)
        except Exception:
            continue
    return blurred
