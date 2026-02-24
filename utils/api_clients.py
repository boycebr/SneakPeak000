"""
External API clients — Google Vision, Azure Face, AWS Rekognition.
"""

import base64
import requests


def detect_faces_google_vision(image_bytes: bytes, api_key: str) -> list:
    """Detect faces using Google Cloud Vision API."""
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
                return result["responses"][0].get("faceAnnotations", [])
        return []
    except Exception:
        return []


def detect_faces_azure(image_bytes: bytes, api_key: str, endpoint: str) -> list:
    """Detect faces using Azure Face API (stub for future implementation)."""
    # TODO: Implement Azure Face API integration
    return []


def detect_faces_aws_rekognition(image_bytes: bytes, access_key: str, secret_key: str, region: str) -> list:
    """Detect faces using AWS Rekognition (stub for future implementation)."""
    # TODO: Implement AWS Rekognition integration
    return []
