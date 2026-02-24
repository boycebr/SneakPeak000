"""
SneakPeak Configuration
Loads settings from Streamlit secrets -> environment variables -> .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _get(key, default=None):
    """Check Streamlit secrets first, then env vars."""
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val is not None:
            return val
    except Exception:
        pass
    return os.getenv(key, default)


# Supabase
SUPABASE_URL = _get("SUPABASE_URL")
SUPABASE_ANON_KEY = _get("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = _get("SUPABASE_SERVICE_KEY")

# Google Cloud Vision
GOOGLE_CLOUD_API_KEY = _get("GOOGLE_CLOUD_API_KEY")

# Azure Face API
AZURE_FACE_API_KEY = _get("AZURE_FACE_API_KEY")
AZURE_FACE_ENDPOINT = _get("AZURE_FACE_ENDPOINT")

# AWS Rekognition
AWS_ACCESS_KEY_ID = _get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = _get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = _get("AWS_REGION", "us-east-2")

# App settings
APP_ENV = _get("APP_ENV", "development")
DEBUG = _get("DEBUG", "false").lower() == "true"
MAX_UPLOAD_SIZE_MB = int(_get("MAX_UPLOAD_SIZE_MB", "100"))
VIDEO_MIN_DURATION = int(_get("VIDEO_MIN_DURATION", "5"))
VIDEO_MAX_DURATION = int(_get("VIDEO_MAX_DURATION", "60"))
