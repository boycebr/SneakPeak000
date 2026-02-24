"""
Supabase helpers — REST API, Storage uploads, CRUD for video_results.
"""

import requests
import uuid
from datetime import datetime


def get_supabase_headers(api_key: str) -> dict:
    """Build headers for Supabase REST API calls."""
    return {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def test_connection(supabase_url: str, api_key: str) -> bool:
    """Return True if the Supabase REST API responds."""
    try:
        url = f"{supabase_url}/rest/v1/"
        resp = requests.get(url, headers=get_supabase_headers(api_key), timeout=5)
        return resp.status_code in [200, 404]
    except Exception:
        return False


def save_video_results(supabase_url: str, api_key: str, data: dict) -> tuple:
    """Insert a row into video_results. Returns (success, response_or_error)."""
    try:
        url = f"{supabase_url}/rest/v1/video_results"
        resp = requests.post(
            url, headers=get_supabase_headers(api_key), json=data, timeout=10
        )
        if resp.status_code in [200, 201]:
            return True, resp.json()
        return False, f"Error {resp.status_code}: {resp.text}"
    except Exception as e:
        return False, str(e)


def update_video_result(supabase_url: str, api_key: str, row_id: int, data: dict) -> bool:
    """Update a video_results row by id."""
    try:
        url = f"{supabase_url}/rest/v1/video_results?id=eq.{row_id}"
        resp = requests.patch(url, headers=get_supabase_headers(api_key), json=data, timeout=10)
        return resp.status_code in [200, 204]
    except Exception:
        return False


def get_recent_venues(supabase_url: str, api_key: str, limit: int = 10) -> list:
    """Fetch the most recent venue results."""
    try:
        url = (
            f"{supabase_url}/rest/v1/video_results"
            f"?select=*&order=created_at.desc&limit={limit}"
        )
        resp = requests.get(url, headers=get_supabase_headers(api_key), timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception:
        return []


def get_venues_by_energy(supabase_url: str, api_key: str, order: str = "desc", limit: int = 20) -> list:
    """Fetch venues sorted by energy score."""
    try:
        url = (
            f"{supabase_url}/rest/v1/video_results"
            f"?select=*&order=energy_score.{order}&limit={limit}"
        )
        resp = requests.get(url, headers=get_supabase_headers(api_key), timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception:
        return []


# ── Supabase Storage ─────────────────────────────────────────────────────

STORAGE_BUCKET = "venue-media"


def upload_to_storage(
    supabase_url: str,
    api_key: str,
    file_bytes: bytes,
    file_path: str,
    content_type: str = "image/jpeg",
) -> tuple:
    """Upload a file to Supabase Storage.

    Args:
        file_path: Path within the bucket, e.g. "thumbnails/abc123.jpg"

    Returns:
        (success, public_url_or_error)
    """
    try:
        url = f"{supabase_url}/storage/v1/object/{STORAGE_BUCKET}/{file_path}"
        headers = {
            "apikey": api_key,
            "Authorization": f"Bearer {api_key}",
            "Content-Type": content_type,
        }
        resp = requests.post(url, headers=headers, data=file_bytes, timeout=30)

        if resp.status_code in [200, 201]:
            public_url = (
                f"{supabase_url}/storage/v1/object/public/{STORAGE_BUCKET}/{file_path}"
            )
            return True, public_url
        return False, f"Error {resp.status_code}: {resp.text}"
    except Exception as e:
        return False, str(e)


def generate_storage_path(venue_name: str, suffix: str = ".jpg") -> str:
    """Generate a unique storage path for a file."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    safe_name = "".join(c if c.isalnum() else "_" for c in venue_name)[:30]
    return f"thumbnails/{timestamp}_{safe_name}_{uid}{suffix}"
