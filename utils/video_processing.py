"""
Video processing utilities — frame extraction, metadata, face blurring.
"""

import io
import numpy as np

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
    """Extract evenly-spaced frames from a video as RGB numpy arrays."""
    frames = []
    if not CV2_AVAILABLE:
        return frames

    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
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


def frame_to_jpeg_bytes(frame: np.ndarray, quality: int = 85) -> bytes:
    """Convert an RGB numpy frame to JPEG bytes."""
    if PIL_AVAILABLE:
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    if CV2_AVAILABLE:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return encoded.tobytes()

    raise RuntimeError("Neither Pillow nor OpenCV available for JPEG encoding")


def blur_faces_in_frame(frame: np.ndarray, face_boxes: list, padding: int = 20) -> np.ndarray:
    """Apply Gaussian blur to detected face regions.

    Args:
        frame: RGB numpy array (H, W, 3)
        face_boxes: List of dicts with keys x1, y1, x2, y2 (normalized format
                    from api_clients.detect_faces)
        padding: Extra pixels around each face box for more complete coverage

    Returns:
        Frame with faces blurred.
    """
    if not CV2_AVAILABLE:
        return frame

    blurred = frame.copy()
    h, w = frame.shape[:2]

    for face in face_boxes:
        try:
            x1 = max(0, int(face["x1"]) - padding)
            y1 = max(0, int(face["y1"]) - padding)
            x2 = min(w, int(face["x2"]) + padding)
            y2 = min(h, int(face["y2"]) + padding)

            if x2 <= x1 or y2 <= y1:
                continue

            region = blurred[y1:y2, x1:x2]
            if region.size > 0:
                # Heavy blur — kernel must be odd and large enough to obscure
                ksize = max(99, ((x2 - x1) // 2) | 1)
                blurred[y1:y2, x1:x2] = cv2.GaussianBlur(
                    region, (ksize, ksize), 30
                )
        except (KeyError, ValueError):
            continue

    return blurred


def process_frame_privacy(
    frame: np.ndarray,
    google_api_key: str = None,
    azure_api_key: str = None,
    azure_endpoint: str = None,
) -> dict:
    """Full privacy pipeline for a single frame.

    1. Convert frame to JPEG bytes
    2. Detect faces using the fallback chain
    3. Blur all detected faces
    4. Return results dict

    Returns:
        {
            "blurred_frame": np.ndarray (RGB),
            "face_count": int,
            "faces": list of normalized face dicts,
            "source": str (which detector found faces),
        }
    """
    from utils.api_clients import detect_faces

    # Convert frame to JPEG for API calls
    jpeg_bytes = frame_to_jpeg_bytes(frame)

    # Detect faces
    faces = detect_faces(
        jpeg_bytes,
        google_api_key=google_api_key,
        azure_api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
    )

    # Blur faces
    blurred = blur_faces_in_frame(frame, faces)

    source = faces[0]["source"] if faces else "none"

    return {
        "blurred_frame": blurred,
        "face_count": len(faces),
        "faces": faces,
        "source": source,
    }


def generate_thumbnail(frame: np.ndarray, max_width: int = 400) -> bytes:
    """Resize a frame to thumbnail size and return JPEG bytes."""
    if not CV2_AVAILABLE:
        return frame_to_jpeg_bytes(frame)

    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        resized = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return frame_to_jpeg_bytes(resized_rgb, quality=80)

    return frame_to_jpeg_bytes(frame, quality=80)


def validate_video(file_size_bytes: int, duration: float, min_dur: int, max_dur: int, max_size_mb: int) -> list:
    """Validate video constraints. Returns list of error strings (empty = valid)."""
    errors = []
    max_bytes = max_size_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        errors.append(f"File too large ({file_size_bytes / 1024 / 1024:.1f}MB). Max is {max_size_mb}MB.")
    if duration > 0 and duration < min_dur:
        errors.append(f"Video too short ({duration:.1f}s). Minimum is {min_dur}s.")
    if duration > max_dur:
        errors.append(f"Video too long ({duration:.1f}s). Maximum is {max_dur}s.")
    return errors


# ── Motion Detection ─────────────────────────────────────────────────────

def compute_motion_score(frames: list) -> dict:
    """Compute motion/activity score from frame differences.

    Compares consecutive frames to measure how much movement exists.
    Returns a 0-100 score and a descriptive level.
    """
    if not CV2_AVAILABLE or len(frames) < 2:
        return {"motion_score": 50.0, "motion_level": "medium"}

    diffs = []
    for i in range(len(frames) - 1):
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        mean_diff = float(np.mean(diff))
        diffs.append(mean_diff)

    avg_motion = np.mean(diffs)
    # Scale: 0-5 = still, 5-15 = moderate, 15+ = active
    # Map to 0-100
    motion_score = round(min(100, (avg_motion / 25.0) * 100), 1)

    if motion_score >= 65:
        level = "high"
    elif motion_score >= 30:
        level = "medium"
    else:
        level = "low"

    return {
        "motion_score": motion_score,
        "motion_level": level,
        "raw_diff": round(float(avg_motion), 2),
    }


# ── Mood Extraction ──────────────────────────────────────────────────────

LIKELIHOOD_SCORES = {
    "VERY_LIKELY": 95,
    "LIKELY": 75,
    "POSSIBLE": 50,
    "UNLIKELY": 20,
    "VERY_UNLIKELY": 5,
    "UNKNOWN": 0,
}


def extract_mood_from_faces(all_faces: list) -> dict:
    """Aggregate mood data from face detection results.

    Uses joy/sorrow likelihood from Google Vision (already in normalized face dicts).
    Falls back to neutral if no mood data available.
    """
    if not all_faces:
        return {
            "dominant_mood": "neutral",
            "joy_percentage": 0.0,
            "mood_confidence": 0.0,
            "mood_score": 50.0,
        }

    joy_scores = []
    sorrow_scores = []

    for face in all_faces:
        joy = face.get("joy", "UNKNOWN")
        sorrow = face.get("sorrow", "UNKNOWN")
        joy_scores.append(LIKELIHOOD_SCORES.get(joy, 0))
        sorrow_scores.append(LIKELIHOOD_SCORES.get(sorrow, 0))

    avg_joy = np.mean(joy_scores) if joy_scores else 0
    avg_sorrow = np.mean(sorrow_scores) if sorrow_scores else 0

    # Determine dominant mood
    if avg_joy >= 60:
        dominant = "happy"
    elif avg_joy >= 40:
        dominant = "positive"
    elif avg_sorrow >= 40:
        dominant = "somber"
    else:
        dominant = "neutral"

    # Mood score: 0 (very sad) to 100 (very happy)
    mood_score = round(min(100, max(0, 50 + (avg_joy - avg_sorrow) / 2)), 1)

    return {
        "dominant_mood": dominant,
        "joy_percentage": round(avg_joy, 1),
        "mood_confidence": round(max(avg_joy, avg_sorrow), 1),
        "mood_score": mood_score,
    }
