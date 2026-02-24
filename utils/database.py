"""
Supabase database helpers — connection test, CRUD for video_results.
"""

import requests


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
