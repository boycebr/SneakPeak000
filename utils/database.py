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


def search_venues(
    supabase_url: str,
    api_key: str,
    query: str = "",
    venue_type: str = "",
    sort: str = "recent",
    limit: int = 20,
) -> list:
    """Search venues with optional name filter, type filter, and sort order.

    Args:
        query: Case-insensitive substring match on venue_name.
        venue_type: Exact match on venue_type (empty = all).
        sort: "recent", "hot" (energy desc), or "chill" (energy asc).
    """
    try:
        params = "select=*"
        if query:
            params += f"&venue_name=ilike.*{query}*"
        if venue_type:
            params += f"&venue_type=eq.{venue_type}"
        if sort == "hot":
            params += "&order=energy_score.desc.nullslast"
        elif sort == "chill":
            params += "&order=energy_score.asc.nullslast"
        else:
            params += "&order=created_at.desc"
        params += f"&limit={limit}"

        url = f"{supabase_url}/rest/v1/video_results?{params}"
        resp = requests.get(url, headers=get_supabase_headers(api_key), timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception:
        return []


def get_nearby_venues(
    supabase_url: str,
    api_key: str,
    lat: float,
    lon: float,
    radius_km: float = 10.0,
    limit: int = 20,
) -> list:
    """Fetch venues near a GPS point, sorted by distance.

    Uses a simple bounding-box filter (PostgREST doesn't do Haversine natively).
    Distance is approximated client-side after fetch.
    """
    try:
        # ~0.009 degrees per km at mid-latitudes
        delta = radius_km * 0.009
        params = (
            f"select=*"
            f"&latitude=gte.{lat - delta}&latitude=lte.{lat + delta}"
            f"&longitude=gte.{lon - delta}&longitude=lte.{lon + delta}"
            f"&order=created_at.desc&limit={limit}"
        )
        url = f"{supabase_url}/rest/v1/video_results?{params}"
        resp = requests.get(url, headers=get_supabase_headers(api_key), timeout=10)
        if resp.status_code == 200:
            venues = resp.json()
            # Add approximate distance and sort by it
            import math
            for v in venues:
                vlat = float(v.get("latitude") or 0)
                vlon = float(v.get("longitude") or 0)
                dlat = math.radians(vlat - lat)
                dlon = math.radians(vlon - lon)
                a = (math.sin(dlat / 2) ** 2 +
                     math.cos(math.radians(lat)) * math.cos(math.radians(vlat)) *
                     math.sin(dlon / 2) ** 2)
                v["_distance_km"] = round(6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 2)
            venues.sort(key=lambda v: v.get("_distance_km", 999))
            return venues
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
    timestamp = datetime.now(tz=None).strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    safe_name = "".join(c if c.isalnum() else "_" for c in venue_name)[:30]
    return f"thumbnails/{timestamp}_{safe_name}_{uid}{suffix}"


# ── User Ratings ────────────────────────────────────────────────────────

def save_user_rating(supabase_url: str, api_key: str, data: dict) -> tuple:
    """Insert a row into user_ratings. Returns (success, response_or_error)."""
    try:
        url = f"{supabase_url}/rest/v1/user_ratings"
        resp = requests.post(
            url, headers=get_supabase_headers(api_key), json=data, timeout=10
        )
        if resp.status_code in [200, 201]:
            return True, resp.json()
        return False, f"Error {resp.status_code}: {resp.text}"
    except Exception as e:
        return False, str(e)


def get_ratings_for_venue(supabase_url: str, api_key: str, video_result_id: int) -> list:
    """Fetch all ratings for a specific venue result."""
    try:
        url = (
            f"{supabase_url}/rest/v1/user_ratings"
            f"?video_result_id=eq.{video_result_id}&order=created_at.desc"
        )
        resp = requests.get(url, headers=get_supabase_headers(api_key), timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception:
        return []


def get_user_submissions(supabase_url: str, api_key: str, user_id: str, limit: int = 20) -> list:
    """Fetch video_results submitted by a specific user."""
    try:
        url = (
            f"{supabase_url}/rest/v1/video_results"
            f"?user_id=eq.{user_id}&order=created_at.desc&limit={limit}"
        )
        resp = requests.get(url, headers=get_supabase_headers(api_key), timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception:
        return []


def get_user_rating_count(supabase_url: str, api_key: str, user_id: str) -> int:
    """Count how many ratings a user has submitted."""
    try:
        url = (
            f"{supabase_url}/rest/v1/user_ratings"
            f"?user_id=eq.{user_id}&select=id"
        )
        headers = get_supabase_headers(api_key)
        headers["Prefer"] = "count=exact"
        resp = requests.get(url, headers=headers, timeout=10)
        # count is in content-range header
        cr = resp.headers.get("content-range", "")
        if "/" in cr:
            return int(cr.split("/")[1])
        if resp.status_code == 200:
            return len(resp.json())
        return 0
    except Exception:
        return 0
