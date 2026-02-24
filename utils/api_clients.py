"""
External API clients — Google Vision, Azure Face, AWS Rekognition.
Each returns face locations in a normalized format:
    [{"x1": int, "y1": int, "x2": int, "y2": int, "confidence": float, "source": str}, ...]
"""

import base64
import requests


# ── Normalized result format ────────────────────────────────────────────

def _normalize_google_faces(annotations: list) -> list:
    """Convert Google Vision faceAnnotations to normalized bounding boxes."""
    faces = []
    for ann in annotations:
        vertices = ann.get("boundingPoly", {}).get("vertices", [])
        if len(vertices) >= 4:
            faces.append({
                "x1": vertices[0].get("x", 0),
                "y1": vertices[0].get("y", 0),
                "x2": vertices[2].get("x", 0),
                "y2": vertices[2].get("y", 0),
                "confidence": _likelihood_to_score(
                    ann.get("detectionConfidence", 0)
                ),
                "joy": ann.get("joyLikelihood", "UNKNOWN"),
                "sorrow": ann.get("sorrowLikelihood", "UNKNOWN"),
                "source": "google_vision",
            })
    return faces


def _normalize_azure_faces(azure_faces: list) -> list:
    """Convert Azure Face API response to normalized bounding boxes."""
    faces = []
    for face in azure_faces:
        rect = face.get("faceRectangle", {})
        x1 = rect.get("left", 0)
        y1 = rect.get("top", 0)
        w = rect.get("width", 0)
        h = rect.get("height", 0)
        faces.append({
            "x1": x1,
            "y1": y1,
            "x2": x1 + w,
            "y2": y1 + h,
            "confidence": face.get("faceAttributes", {}).get("confidence", 0.9),
            "source": "azure_face",
        })
    return faces


def _likelihood_to_score(value) -> float:
    """Convert Google likelihood string or numeric confidence to 0-1 float."""
    if isinstance(value, (int, float)):
        return float(value)
    mapping = {
        "VERY_LIKELY": 0.95,
        "LIKELY": 0.80,
        "POSSIBLE": 0.50,
        "UNLIKELY": 0.20,
        "VERY_UNLIKELY": 0.05,
    }
    return mapping.get(str(value), 0.5)


# ── Google Cloud Vision ─────────────────────────────────────────────────

def detect_faces_google_vision(image_bytes: bytes, api_key: str) -> list:
    """Detect faces using Google Cloud Vision API.

    Returns normalized face list.
    """
    if not api_key:
        return []

    try:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
        payload = {
            "requests": [
                {
                    "image": {"content": encoded},
                    "features": [{"type": "FACE_DETECTION", "maxResults": 50}],
                }
            ]
        }
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if "responses" in result and result["responses"]:
                raw = result["responses"][0].get("faceAnnotations", [])
                return _normalize_google_faces(raw)
        return []
    except Exception:
        return []


# ── Azure Face API ──────────────────────────────────────────────────────

def detect_faces_azure(image_bytes: bytes, api_key: str, endpoint: str) -> list:
    """Detect faces using Azure Face API.

    Returns normalized face list.
    """
    if not api_key or not endpoint:
        return []

    try:
        url = f"{endpoint.rstrip('/')}/face/v1.0/detect"
        headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "Content-Type": "application/octet-stream",
        }
        params = {
            "returnFaceId": "false",
            "returnFaceLandmarks": "false",
            "detectionModel": "detection_03",
        }
        response = requests.post(
            url, headers=headers, params=params, data=image_bytes, timeout=30
        )
        if response.status_code == 200:
            return _normalize_azure_faces(response.json())
        return []
    except Exception:
        return []


# ── AWS Rekognition (stub) ──────────────────────────────────────────────

def detect_faces_aws_rekognition(
    image_bytes: bytes, access_key: str, secret_key: str, region: str
) -> list:
    """Detect faces using AWS Rekognition (stub for future implementation)."""
    # TODO: Implement via boto3 when needed
    return []


# ── OpenCV Haar Cascade (local fallback, no API needed) ─────────────────

def detect_faces_opencv(image_bytes: bytes) -> list:
    """Detect faces using OpenCV Haar cascade — offline fallback.

    Returns normalized face list.
    """
    try:
        import cv2
        import numpy as np

        # Decode image
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        classifier = cv2.CascadeClassifier(cascade_path)

        rects = classifier.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        faces = []
        for (x, y, w, h) in rects:
            faces.append({
                "x1": int(x),
                "y1": int(y),
                "x2": int(x + w),
                "y2": int(y + h),
                "confidence": 0.7,
                "source": "opencv_haar",
            })
        return faces
    except Exception:
        return []


# ── Unified detector with fallback chain ─────────────────────────────────

def detect_faces(
    image_bytes: bytes,
    google_api_key: str = None,
    azure_api_key: str = None,
    azure_endpoint: str = None,
) -> list:
    """Try face detection providers in priority order.

    Order: Google Vision -> Azure Face -> OpenCV Haar (offline fallback).
    Returns on first success (non-empty result).
    """
    # 1. Google Vision (most accurate, paid)
    if google_api_key:
        faces = detect_faces_google_vision(image_bytes, google_api_key)
        if faces:
            return faces

    # 2. Azure Face API (good accuracy, paid)
    if azure_api_key and azure_endpoint:
        faces = detect_faces_azure(image_bytes, azure_api_key, azure_endpoint)
        if faces:
            return faces

    # 3. OpenCV Haar cascade (free, local, less accurate)
    faces = detect_faces_opencv(image_bytes)
    return faces
