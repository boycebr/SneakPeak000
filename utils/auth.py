"""
Supabase GoTrue authentication — signup, login, session management.

Uses the Supabase Auth REST API (GoTrue) directly via requests.
No Supabase Python SDK needed — just HTTP calls.
"""

import logging
import requests

logger = logging.getLogger("sneakpeak.auth")


def sign_up(supabase_url: str, api_key: str, email: str, password: str) -> dict:
    """Create a new user account.

    Returns:
        {"success": True, "user": {...}, "access_token": "...", "refresh_token": "..."}
        or {"success": False, "error": "..."}
    """
    try:
        url = f"{supabase_url}/auth/v1/signup"
        headers = {
            "apikey": api_key,
            "Content-Type": "application/json",
        }
        resp = requests.post(
            url, headers=headers,
            json={"email": email, "password": password},
            timeout=10,
        )
        data = resp.json()

        if resp.status_code in [200, 201]:
            # Supabase may return user without session if email confirmation is on
            if "access_token" in data:
                logger.info("sign_up: success email=%s", email)
                return {
                    "success": True,
                    "user": data.get("user", {}),
                    "access_token": data["access_token"],
                    "refresh_token": data.get("refresh_token", ""),
                }
            # Email confirmation required — user created but no session yet
            logger.info("sign_up: pending confirmation email=%s", email)
            return {
                "success": True,
                "user": data.get("user", data),
                "access_token": None,
                "refresh_token": None,
                "confirm_email": True,
            }
        logger.warning("sign_up: failed email=%s status=%s", email, resp.status_code)
        return {"success": False, "error": data.get("msg", data.get("error_description", str(data)))}
    except Exception as e:
        logger.exception("sign_up: error email=%s", email)
        return {"success": False, "error": str(e)}


def sign_in(supabase_url: str, api_key: str, email: str, password: str) -> dict:
    """Sign in with email + password.

    Returns:
        {"success": True, "user": {...}, "access_token": "...", "refresh_token": "..."}
        or {"success": False, "error": "..."}
    """
    try:
        url = f"{supabase_url}/auth/v1/token?grant_type=password"
        headers = {
            "apikey": api_key,
            "Content-Type": "application/json",
        }
        resp = requests.post(
            url, headers=headers,
            json={"email": email, "password": password},
            timeout=10,
        )
        data = resp.json()

        if resp.status_code == 200 and "access_token" in data:
            logger.info("sign_in: success email=%s", email)
            return {
                "success": True,
                "user": data.get("user", {}),
                "access_token": data["access_token"],
                "refresh_token": data.get("refresh_token", ""),
            }
        logger.warning("sign_in: failed email=%s status=%s", email, resp.status_code)
        return {"success": False, "error": data.get("error_description", data.get("msg", str(data)))}
    except Exception as e:
        logger.exception("sign_in: error email=%s", email)
        return {"success": False, "error": str(e)}


def get_user(supabase_url: str, api_key: str, access_token: str) -> dict | None:
    """Fetch the current user profile from an access token.

    Returns user dict or None.
    """
    try:
        url = f"{supabase_url}/auth/v1/user"
        headers = {
            "apikey": api_key,
            "Authorization": f"Bearer {access_token}",
        }
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def refresh_session(supabase_url: str, api_key: str, refresh_token: str) -> dict:
    """Refresh an expired access token.

    Returns same shape as sign_in on success.
    """
    try:
        url = f"{supabase_url}/auth/v1/token?grant_type=refresh_token"
        headers = {
            "apikey": api_key,
            "Content-Type": "application/json",
        }
        resp = requests.post(
            url, headers=headers,
            json={"refresh_token": refresh_token},
            timeout=10,
        )
        data = resp.json()

        if resp.status_code == 200 and "access_token" in data:
            return {
                "success": True,
                "user": data.get("user", {}),
                "access_token": data["access_token"],
                "refresh_token": data.get("refresh_token", refresh_token),
            }
        return {"success": False, "error": data.get("error_description", str(data))}
    except Exception as e:
        return {"success": False, "error": str(e)}


def sign_out(supabase_url: str, api_key: str, access_token: str) -> bool:
    """Sign out — invalidate the session on the server."""
    try:
        url = f"{supabase_url}/auth/v1/logout"
        headers = {
            "apikey": api_key,
            "Authorization": f"Bearer {access_token}",
        }
        resp = requests.post(url, headers=headers, timeout=10)
        return resp.status_code in [200, 204]
    except Exception:
        return False


# ── Authenticated headers ─────────────────────────────────────────────

def get_auth_headers(api_key: str, access_token: str) -> dict:
    """Build headers that use the user's JWT instead of the anon key.

    When a user is logged in, pass their access_token here so that
    Supabase RLS policies can identify the user.
    """
    return {
        "apikey": api_key,
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
