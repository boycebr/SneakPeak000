"""
SneakPeak MVP - Real-Time Venue Intelligence Platform
Version: MVP 1.0
Last Updated: February 2026
"""

import streamlit as st

# CRITICAL: Page config MUST be first Streamlit command
st.set_page_config(
    page_title="SneakPeak - Venue Pulse",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Standard imports
import numpy as np
import tempfile
import os
import requests
import json
import logging
from datetime import datetime
import uuid
import time

# ── Logging setup ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sneakpeak")

# Configuration — loaded from .env / Streamlit secrets (no hardcoded keys)
from config.settings import (
    SUPABASE_URL,
    SUPABASE_ANON_KEY,
    SUPABASE_SERVICE_KEY,
    GOOGLE_CLOUD_API_KEY,
    AZURE_FACE_API_KEY,
    AZURE_FACE_ENDPOINT,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    MAX_UPLOAD_SIZE_MB,
    VIDEO_MIN_DURATION,
    VIDEO_MAX_DURATION,
)

# Privacy pipeline — face detection (Google Vision -> Azure -> OpenCV) + blurring
from utils.video_processing import (
    process_frame_privacy,
    frame_to_jpeg_bytes,
    generate_thumbnail,
    validate_video,
    compute_motion_score,
    extract_mood_from_faces,
)
from utils.audio_analysis import analyze_audio_real
from utils.database import (
    upload_to_storage,
    generate_storage_path,
    get_venues_by_energy,
    save_user_rating,
    get_ratings_for_venue,
    get_user_submissions,
    get_user_rating_count,
    search_venues,
    get_nearby_venues,
)
from utils.auth import sign_up, sign_in, sign_out, get_user, refresh_session

# Video/Image processing
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

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ================================
# MOBILE-OPTIMIZED CSS
# ================================

st.markdown("""
<style>
/* Mobile-first responsive design */
.main > div {
    padding-top: 1rem;
    padding-left: 0.5rem;
    padding-right: 0.5rem;
    max-width: 100%;
}

/* Prevent iOS zoom on input focus */
input, select, textarea {
    font-size: 16px !important;
}

/* Energy score display */
.energy-score-card {
    text-align: center;
    padding: 1.5rem;
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 20px;
    margin: 1rem 0;
    color: white;
    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4);
}

.energy-score-card h1 {
    font-size: 4rem;
    margin: 0;
    font-weight: 700;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

.energy-score-card p {
    font-size: 1.1rem;
    margin: 0.5rem 0 0 0;
    opacity: 0.9;
}

/* Metric cards */
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    margin: 0.5rem 0;
    text-align: center;
    border-left: 4px solid #667eea;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #667eea;
    display: block;
}

.metric-label {
    font-size: 0.8rem;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Upload section */
.upload-section {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin: 1rem 0;
    border: 2px dashed #667eea;
}

/* Status indicators */
.status-success {
    background: #d4edda;
    color: #155724;
    padding: 0.75rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}

.status-warning {
    background: #fff3cd;
    color: #856404;
    padding: 0.75rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}

.status-error {
    background: #f8d7da;
    color: #721c24;
    padding: 0.75rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}

/* Venue card */
.venue-card {
    background: white;
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 2px 15px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
    transition: transform 0.2s;
}

.venue-card:hover {
    transform: translateY(-2px);
}

/* Touch-friendly buttons */
.stButton > button {
    min-height: 48px;
    font-size: 16px !important;
    border-radius: 10px;
    width: 100%;
}

/* Live indicator pulse */
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.4; }
    100% { opacity: 1; }
}
.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #28a745;
    border-radius: 50%;
    animation: pulse 1.5s infinite;
    margin-right: 4px;
}

/* Search input styling */
input[data-testid="stTextInput"] {
    border-radius: 20px !important;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Mobile adjustments */
@media (max-width: 768px) {
    .energy-score-card h1 {
        font-size: 3rem;
    }
    .metric-value {
        font-size: 1.5rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ================================
# SESSION STATE INITIALIZATION
# ================================

def init_session():
    """Initialize session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'user_name' not in st.session_state:
        st.session_state.user_name = 'Anonymous'
    if 'videos_processed' not in st.session_state:
        st.session_state.videos_processed = 0
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'discover'
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = None
    # Geolocation
    if 'user_lat' not in st.session_state:
        st.session_state.user_lat = None
    if 'user_lon' not in st.session_state:
        st.session_state.user_lon = None
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    # Rate limiting
    if 'upload_timestamps' not in st.session_state:
        st.session_state.upload_timestamps = []
    # Auth state
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
    if 'refresh_token' not in st.session_state:
        st.session_state.refresh_token = None
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None

init_session()


def is_logged_in() -> bool:
    """Check if the user has an active session."""
    return st.session_state.access_token is not None


def _set_auth_state(result: dict):
    """Store auth tokens + user info in session state."""
    st.session_state.access_token = result.get("access_token")
    st.session_state.refresh_token = result.get("refresh_token")
    user = result.get("user", {})
    st.session_state.user_id = user.get("id")
    st.session_state.user_email = user.get("email")


def _clear_auth_state():
    """Remove auth tokens from session state."""
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.user_id = None
    st.session_state.user_email = None


# ================================
# RATE LIMITING
# ================================

MAX_UPLOADS_PER_HOUR = 10


def check_rate_limit() -> bool:
    """Return True if the user is within the upload rate limit."""
    now = time.time()
    one_hour_ago = now - 3600
    # Prune old timestamps
    st.session_state.upload_timestamps = [
        ts for ts in st.session_state.upload_timestamps if ts > one_hour_ago
    ]
    return len(st.session_state.upload_timestamps) < MAX_UPLOADS_PER_HOUR


def record_upload():
    """Record a successful upload for rate limiting."""
    st.session_state.upload_timestamps.append(time.time())


def uploads_remaining() -> int:
    """How many uploads the user has left this hour."""
    now = time.time()
    one_hour_ago = now - 3600
    recent = [ts for ts in st.session_state.upload_timestamps if ts > one_hour_ago]
    return max(0, MAX_UPLOADS_PER_HOUR - len(recent))


# ================================
# GEOLOCATION
# ================================

def request_geolocation():
    """Inject JS to request browser GPS and store in query params.

    Returns (lat, lon) if available, else (None, None).
    Uses Streamlit's st.query_params as a bridge from JS → Python.
    """
    # Check if we already have coordinates in session state
    if st.session_state.get("user_lat") is not None:
        return st.session_state.user_lat, st.session_state.user_lon

    # Check query params (set by JS)
    qp = st.query_params
    lat_str = qp.get("lat")
    lon_str = qp.get("lon")
    if lat_str and lon_str:
        try:
            lat = float(lat_str)
            lon = float(lon_str)
            st.session_state.user_lat = lat
            st.session_state.user_lon = lon
            return lat, lon
        except (ValueError, TypeError):
            pass

    return None, None


def render_geolocation_button():
    """Render a button that triggers browser geolocation via JS."""
    import streamlit.components.v1 as components
    if st.button("📍 Use My Location", key="geo_btn", use_container_width=True):
        components.html("""
        <script>
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                function(pos) {
                    const lat = pos.coords.latitude.toFixed(6);
                    const lon = pos.coords.longitude.toFixed(6);
                    // Update URL query params so Streamlit can read them
                    const url = new URL(window.parent.location);
                    url.searchParams.set('lat', lat);
                    url.searchParams.set('lon', lon);
                    window.parent.history.replaceState({}, '', url);
                    // Trigger a rerun by clicking a hidden Streamlit element
                    window.parent.location.reload();
                },
                function(err) {
                    console.log('Geolocation error:', err.message);
                },
                {enableHighAccuracy: true, timeout: 10000}
            );
        }
        </script>
        """, height=0)


# ================================
# SUPABASE DATABASE FUNCTIONS
# ================================

def get_supabase_headers():
    """Get headers for Supabase API calls"""
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

def test_supabase_connection():
    """Test if Supabase is accessible"""
    try:
        url = f"{SUPABASE_URL}/rest/v1/"
        response = requests.get(url, headers=get_supabase_headers(), timeout=5)
        return response.status_code in [200, 404]  # 404 is OK, means API is responding
    except Exception as e:
        st.error(f"Supabase connection error: {e}")
        return False

def save_video_results(data):
    """Save video analysis results to Supabase"""
    try:
        url = f"{SUPABASE_URL}/rest/v1/video_results"
        response = requests.post(url, headers=get_supabase_headers(), json=data, timeout=10)
        
        if response.status_code in [200, 201]:
            return True, response.json()
        else:
            return False, f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return False, str(e)

def get_recent_venues(limit=10):
    """Get recent venue results from Supabase"""
    try:
        url = f"{SUPABASE_URL}/rest/v1/video_results?select=*&order=created_at.desc&limit={limit}"
        response = requests.get(url, headers=get_supabase_headers(), timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except Exception as e:
        st.warning(f"Could not fetch venues: {e}")
        return []

# ================================
# VIDEO PROCESSING FUNCTIONS
# ================================

def extract_video_metadata(video_path):
    """Extract basic metadata from video file"""
    metadata = {
        "duration": 0,
        "fps": 0,
        "width": 0,
        "height": 0,
        "frame_count": 0
    }
    
    if MOVIEPY_AVAILABLE:
        try:
            with VideoFileClip(video_path) as clip:
                metadata["duration"] = clip.duration
                metadata["fps"] = clip.fps
                metadata["width"] = clip.w
                metadata["height"] = clip.h
                metadata["frame_count"] = int(clip.duration * clip.fps)
        except Exception as e:
            st.warning(f"MoviePy extraction failed: {e}")
    
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
        except Exception as e:
            st.warning(f"OpenCV extraction failed: {e}")
    
    return metadata

def extract_frames(video_path, num_frames=5):
    """Extract frames from video for analysis"""
    frames = []
    
    if not CV2_AVAILABLE:
        return frames
    
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            return frames
        
        # Get evenly spaced frame indices
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
    except Exception as e:
        st.warning(f"Frame extraction error: {e}")
    
    return frames

    # Inline analysis functions replaced by utils imports:
    # - analyze_audio_real (from utils.audio_analysis)
    # - compute_motion_score, extract_mood_from_faces (from utils.video_processing)

# ================================
# ANALYSIS FUNCTIONS
# ================================

def analyze_visual(frames):
    """Analyze visual characteristics from frames"""
    if not frames or not CV2_AVAILABLE:
        return {
            "brightness_level": 50.0,
            "lighting_type": "Unknown",
            "color_scheme": "Unknown",
            "visual_energy": "medium",
            "color_saturation": 50.0,
        }

    try:
        brightness_values = []
        saturation_values = []

        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            brightness_values.append(np.mean(hsv[:, :, 2]))
            saturation_values.append(np.mean(hsv[:, :, 1]))

        avg_brightness = np.mean(brightness_values)
        avg_saturation = np.mean(saturation_values)

        if avg_brightness < 50:
            lighting_type = "Dark/Club"
        elif avg_brightness < 100:
            lighting_type = "Dim/Ambient"
        elif avg_brightness < 150:
            lighting_type = "Moderate"
        else:
            lighting_type = "Bright"

        # Use motion score for visual energy (injected by caller)
        brightness_variance = np.var(brightness_values)
        if brightness_variance > 500:
            visual_energy = "high"
        elif brightness_variance > 100:
            visual_energy = "medium"
        else:
            visual_energy = "low"

        color_scheme = "Warm" if avg_saturation > 100 else "Cool/Neutral"

        return {
            "brightness_level": round((avg_brightness / 255) * 100, 1),
            "lighting_type": lighting_type,
            "color_scheme": color_scheme,
            "visual_energy": visual_energy,
            "color_saturation": round((avg_saturation / 255) * 100, 1),
        }
    except Exception:
        return {
            "brightness_level": 50.0,
            "lighting_type": "Unknown",
            "color_scheme": "Unknown",
            "visual_energy": "medium",
            "color_saturation": 50.0,
        }

def analyze_crowd(frames, face_count):
    """Analyze crowd density and activity"""
    if face_count <= 3:
        density = "sparse"
        density_score = face_count * 2
    elif face_count <= 10:
        density = "moderate"
        density_score = 6 + (face_count - 3)
    elif face_count <= 25:
        density = "crowded"
        density_score = 13 + min((face_count - 10) * 0.3, 4)
    else:
        density = "packed"
        density_score = 17 + min((face_count - 25) * 0.1, 3)

    return {
        "crowd_density": density,
        "density_score": round(min(density_score, 20), 1),
        "estimated_people": face_count,
        "activity_level": "Active" if face_count > 5 else "Calm",
        "engagement_score": min(face_count * 5, 100),
    }

def calculate_energy_score(audio, visual, crowd, motion=None, mood=None):
    """Calculate overall venue energy score.

    Weights:
        Audio (BPM + volume)     — 25%
        Visual (brightness/light) — 15%
        Motion (frame diffs)      — 20%
        Crowd (density)           — 20%
        Mood (face expressions)   — 20%
    """
    try:
        # Audio component (25%)
        bpm_score = min(max((audio.get("bpm", 100) - 60) / 140 * 100, 0), 100)
        volume_score = min(max((audio.get("volume_level", 70) - 40) / 60 * 100, 0), 100)
        audio_energy = (bpm_score + volume_score) / 2

        # Visual component (15%)
        brightness = visual.get("brightness_level", 50)
        visual_energy_map = {"low": 30, "medium": 50, "high": 80}
        visual_activity = visual_energy_map.get(visual.get("visual_energy", "medium"), 50)
        visual_score = (brightness * 0.4 + visual_activity * 0.6)

        # Motion component (20%) — real frame-difference data
        if motion and "motion_score" in motion:
            motion_score = motion["motion_score"]
        else:
            motion_score = 50.0

        # Crowd component (20%)
        crowd_score = (crowd.get("density_score", 5) / 20) * 100

        # Mood component (20%) — from face expression analysis
        if mood and "mood_score" in mood:
            mood_score = mood["mood_score"]
        else:
            mood_score = 50.0

        energy_score = (
            audio_energy * 0.25 +
            visual_score * 0.15 +
            motion_score * 0.20 +
            crowd_score * 0.20 +
            mood_score * 0.20
        )

        return round(max(0, min(100, energy_score)), 1)
    except Exception:
        return 50.0

# ================================
# AUTH UI
# ================================

def render_auth_header():
    """Show logged-in user info or login prompt in a compact bar."""
    if is_logged_in():
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(
                f"<span style='color:#667eea; font-size:0.9rem;'>"
                f"Signed in as <b>{st.session_state.user_email}</b></span>",
                unsafe_allow_html=True,
            )
        with col2:
            if st.button("Sign Out", key="signout_header"):
                sign_out(SUPABASE_URL, SUPABASE_ANON_KEY, st.session_state.access_token)
                _clear_auth_state()
                st.rerun()


def render_auth_form(key_prefix=""):
    """Render login / signup form. Returns True if user just authenticated."""
    p = f"{key_prefix}_" if key_prefix else ""
    tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

    with tab_login:
        with st.form(f"{p}login_form"):
            email = st.text_input("Email", key=f"{p}login_email")
            password = st.text_input("Password", type="password", key=f"{p}login_pw")
            submitted = st.form_submit_button("Sign In", use_container_width=True)
        if submitted:
            if not email or not password:
                st.error("Please enter email and password")
                return False
            result = sign_in(SUPABASE_URL, SUPABASE_ANON_KEY, email, password)
            if result["success"]:
                _set_auth_state(result)
                st.success("Signed in!")
                st.rerun()
            else:
                st.error(result["error"])
            return False

    with tab_signup:
        with st.form(f"{p}signup_form"):
            new_email = st.text_input("Email", key=f"{p}signup_email")
            new_pw = st.text_input("Password", type="password", key=f"{p}signup_pw",
                                   help="Minimum 6 characters")
            confirm_pw = st.text_input("Confirm Password", type="password", key=f"{p}signup_pw2")
            submitted = st.form_submit_button("Create Account", use_container_width=True)
        if submitted:
            if not new_email or not new_pw:
                st.error("Please fill all fields")
                return False
            if new_pw != confirm_pw:
                st.error("Passwords don't match")
                return False
            if len(new_pw) < 6:
                st.error("Password must be at least 6 characters")
                return False
            result = sign_up(SUPABASE_URL, SUPABASE_ANON_KEY, new_email, new_pw)
            if result["success"]:
                if result.get("confirm_email"):
                    st.info("Account created! Check your email to confirm, then sign in.")
                else:
                    _set_auth_state(result)
                    st.success("Account created and signed in!")
                    st.rerun()
            else:
                st.error(result["error"])
            return False

    return False


# ================================
# UI COMPONENTS
# ================================

def render_energy_donut(score, title="Energy Score"):
    """Render energy score as donut chart"""
    if not PLOTLY_AVAILABLE:
        st.metric(title, f"{score}/100")
        return
    
    # Determine color based on score
    if score >= 80:
        color = "#28a745"  # Green - High energy
    elif score >= 60:
        color = "#667eea"  # Purple - Good energy
    elif score >= 40:
        color = "#ffc107"  # Yellow - Moderate
    else:
        color = "#dc3545"  # Red - Low energy
    
    fig = go.Figure(data=[go.Pie(
        values=[score, 100 - score],
        hole=0.7,
        marker_colors=[color, '#e9ecef'],
        textinfo='none',
        hoverinfo='skip'
    )])
    
    fig.update_layout(
        showlegend=False,
        annotations=[dict(
            text=f'<b>{score:.0f}</b>',
            x=0.5, y=0.5,
            font_size=36,
            font_color=color,
            showarrow=False
        )],
        margin=dict(l=20, r=20, t=20, b=20),
        height=200,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_venue_card(venue):
    """Render a venue card with optional thumbnail"""
    name = venue.get('venue_name', 'Unknown Venue')
    venue_type = venue.get('venue_type', 'Venue')
    energy = venue.get('energy_score', 0) or 0
    crowd = venue.get('crowd_density', 'Unknown') or 'Unknown'
    thumb = venue.get('thumbnail_url')
    genre = venue.get('genre', '')
    people = venue.get('estimated_people', 0) or 0
    created = venue.get('created_at', '')

    # Color coding for energy
    if energy >= 70:
        energy_color = "#28a745"
        energy_label = "🔥 High Energy"
    elif energy >= 50:
        energy_color = "#667eea"
        energy_label = "⚡ Good Vibes"
    else:
        energy_color = "#ffc107"
        energy_label = "😌 Chill"

    # Time ago
    time_label = "Updated recently"
    if created:
        try:
            dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
            diff = datetime.now(dt.tzinfo) - dt
            mins = int(diff.total_seconds() / 60)
            if mins < 60:
                time_label = f"{mins}m ago"
            elif mins < 1440:
                time_label = f"{mins // 60}h ago"
            else:
                time_label = f"{mins // 1440}d ago"
        except Exception:
            pass

    # Build thumbnail HTML
    thumb_html = ""
    if thumb:
        thumb_html = f'''
        <div style="margin-bottom: 0.5rem;">
            <img src="{thumb}" style="width:100%; height:120px; object-fit:cover; border-radius:8px;" />
        </div>'''

    # Build detail chips
    chips = f'👥 {crowd.title()}'
    if people > 0:
        chips += f' (~{people})'
    if genre:
        chips += f' | 🎵 {genre}'
    dist_label = venue.get('_distance_label')
    if dist_label:
        chips += f' | 📍 {dist_label}'
    chips += f' | {time_label}'

    venue_id = venue.get('id')
    mood = venue.get('dominant_mood', '')
    mood_chip = f' | {mood.title()}' if mood else ''

    st.markdown(f"""
    <div class="venue-card">
        {thumb_html}
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin: 0; color: #333;">{name}</h3>
                <p style="margin: 0.25rem 0; color: #666; font-size: 0.9rem;">{venue_type}</p>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 1.8rem; font-weight: bold; color: {energy_color};">{energy:.0f}</span>
                <p style="margin: 0; font-size: 0.8rem; color: {energy_color};">{energy_label}</p>
            </div>
        </div>
        <div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid #eee;">
            <span style="font-size: 0.85rem; color: #888;">{chips}{mood_chip}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Rating widget (only for real DB venues, only for logged-in users)
    if venue_id and is_logged_in():
        with st.expander("Rate this venue", expanded=False):
            render_rating_form(venue_id, name)


def render_rating_form(video_result_id: int, venue_name: str):
    """Inline rating form for a venue card."""
    form_key = f"rate_{video_result_id}"
    with st.form(form_key):
        rating = st.slider("Overall vibe (1-5)", 1, 5, 3, key=f"r_{video_result_id}")
        accuracy = st.slider("How accurate was the score? (1-5)", 1, 5, 3, key=f"a_{video_result_id}")
        vibe = st.selectbox("What's the actual vibe?", [
            "Buzzing", "Chill", "Dead", "Romantic", "Wild", "Cozy", "Loud",
        ], key=f"v_{video_result_id}")
        would_return = st.checkbox("Would you go back?", value=True, key=f"wr_{video_result_id}")
        comment = st.text_input("Quick comment (optional)", key=f"c_{video_result_id}")
        submitted = st.form_submit_button("Submit Rating")

    if submitted:
        data = {
            "video_result_id": video_result_id,
            "session_id": st.session_state.session_id,
            "user_id": st.session_state.user_id,
            "rating": rating,
            "accuracy_rating": accuracy,
            "actual_vibe": vibe,
            "would_return": would_return,
            "comment": comment if comment else None,
        }
        ok, resp = save_user_rating(SUPABASE_URL, SUPABASE_ANON_KEY, data)
        if ok:
            st.success(f"Thanks for rating {venue_name}!")
        else:
            st.error(f"Could not save rating: {resp}")


# ================================
# MAIN APPLICATION PAGES
# ================================

def page_discover():
    """Discover page - browse venues with search, filters, and geolocation"""
    st.markdown("## 🔍 Discover Venues")

    if 'discover_filter' not in st.session_state:
        st.session_state.discover_filter = "recent"

    # ── Search bar ──
    search_query = st.text_input(
        "Search venues",
        placeholder="Type a venue name...",
        key="venue_search",
        label_visibility="collapsed",
    )

    # ── Filter row ──
    filter_cols = st.columns([1, 1, 1, 1])
    filters = [
        ("🔥 Hot", "hot"),
        ("😌 Chill", "chill"),
        ("🆕 Recent", "recent"),
        ("📍 Nearby", "nearby"),
    ]
    for col, (label, key) in zip(filter_cols, filters):
        with col:
            is_active = st.session_state.discover_filter == key
            btn_label = f"**{label}**" if is_active else label
            if st.button(btn_label, key=f"filter_{key}", use_container_width=True):
                st.session_state.discover_filter = key

    # ── Type filter + auto-refresh row ──
    opt_col1, opt_col2 = st.columns([3, 1])
    with opt_col1:
        venue_type_filter = st.selectbox(
            "Venue type",
            ["All Types", "Bar", "Nightclub", "Lounge", "Restaurant",
             "Live Music", "Rooftop", "Pub", "Other"],
            key="type_filter",
            label_visibility="collapsed",
        )
    with opt_col2:
        auto_refresh = st.toggle("Live", value=st.session_state.auto_refresh, key="live_toggle")
        st.session_state.auto_refresh = auto_refresh

    active = st.session_state.discover_filter
    type_val = "" if venue_type_filter == "All Types" else venue_type_filter

    # Show active filter label
    label_map = {"hot": "🔥 Hottest", "chill": "😌 Chillest", "recent": "🆕 Most Recent", "nearby": "📍 Nearby"}
    st.caption(f"Showing: **{label_map.get(active, active.title())}**"
               + (f" matching \"{search_query}\"" if search_query else "")
               + (f" | {venue_type_filter}" if type_val else "")
               + (" | 🟢 Live" if auto_refresh else ""))
    st.markdown("---")

    # ── Geolocation for "Nearby" filter ──
    user_lat, user_lon = request_geolocation()

    if active == "nearby":
        if user_lat is None:
            st.info("Enable location to see nearby venues.")
            render_geolocation_button()
            venues = []
        else:
            venues = get_nearby_venues(
                SUPABASE_URL, SUPABASE_ANON_KEY, user_lat, user_lon, radius_km=10.0
            )
    elif search_query or type_val:
        venues = search_venues(
            SUPABASE_URL, SUPABASE_ANON_KEY,
            query=search_query, venue_type=type_val, sort=active,
        )
    elif active == "hot":
        venues = get_venues_by_energy(SUPABASE_URL, SUPABASE_ANON_KEY, order="desc", limit=20)
    elif active == "chill":
        venues = get_venues_by_energy(SUPABASE_URL, SUPABASE_ANON_KEY, order="asc", limit=20)
    else:
        venues = get_recent_venues(limit=20)

    # ── Render results ──
    if venues:
        for venue in venues:
            # Show distance chip if available
            dist = venue.get("_distance_km")
            if dist is not None:
                venue["_distance_label"] = f"{dist:.1f} km away"
            render_venue_card(venue)
        st.caption(f"{len(venues)} venue{'s' if len(venues) != 1 else ''} found")
    else:
        st.markdown(
            "<div class='metric-card' style='text-align:center; padding:2rem;'>"
            "<p style='font-size:1.2rem;'>No venues found</p>"
            "<p style='color:#888;'>Be the first to upload a video!</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        # Show sample data when DB is empty
        if not search_query and not type_val:
            st.markdown("### 📍 Sample Venues (Demo)")
            sample_venues = [
                {"venue_name": "The Basement", "venue_type": "Nightclub", "energy_score": 82, "crowd_density": "crowded"},
                {"venue_name": "Rooftop Lounge", "venue_type": "Bar", "energy_score": 65, "crowd_density": "moderate"},
                {"venue_name": "Jazz Corner", "venue_type": "Live Music", "energy_score": 58, "crowd_density": "sparse"},
            ]
            for venue in sample_venues:
                render_venue_card(venue)

    # ── Auto-refresh (rerun every 30s) ──
    if auto_refresh:
        time.sleep(30)
        st.rerun()

def page_upload():
    """Upload page - submit venue videos (login required)"""
    st.markdown("## 📹 Share a Venue")

    if not is_logged_in():
        st.info("Sign in to upload venue videos. Browsing is free!")
        render_auth_form(key_prefix="upload")
        return

    st.markdown("Help others by sharing what's happening right now!")
    remaining = uploads_remaining()
    if remaining <= 3:
        st.caption(f"Uploads remaining this hour: **{remaining}** / {MAX_UPLOADS_PER_HOUR}")

    # Venue details
    with st.form("upload_form"):
        venue_name = st.text_input("📍 Venue Name", placeholder="e.g., The Rooftop Bar")
        
        venue_type = st.selectbox("🏢 Venue Type", [
            "Bar",
            "Nightclub",
            "Lounge",
            "Restaurant",
            "Live Music",
            "Rooftop",
            "Pub",
            "Other"
        ])
        
        # GPS coordinates — pre-fill from browser geolocation if available
        default_lat = st.session_state.user_lat or 40.7128
        default_lon = st.session_state.user_lon or -74.0060
        st.markdown("##### 📍 Location")
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input("Latitude", value=default_lat, format="%.6f")
        with col2:
            longitude = st.number_input("Longitude", value=default_lon, format="%.6f")
        
        # Video upload
        uploaded_file = st.file_uploader(
            "🎬 Upload Video (max 30 seconds)",
            type=["mp4", "mov", "avi"],
            help="Record a short clip of the venue atmosphere"
        )
        
        submitted = st.form_submit_button("🚀 Analyze & Share", use_container_width=True)
    
    if submitted:
        if not venue_name:
            st.error("Please enter the venue name")
            return

        if not uploaded_file:
            st.error("Please upload a video")
            return

        if not check_rate_limit():
            st.error(f"Upload limit reached ({MAX_UPLOADS_PER_HOUR}/hour). Try again later.")
            return

        # Process the video
        process_video_upload(uploaded_file, venue_name, venue_type, latitude, longitude)

def process_video_upload(uploaded_file, venue_name, venue_type, latitude, longitude):
    """Process uploaded video and save results"""
    logger.info("Upload started: venue=%s type=%s size=%.1fMB user=%s",
                venue_name, venue_type, uploaded_file.size / 1024 / 1024,
                st.session_state.user_email or "anonymous")
    start_time = time.time()

    progress = st.progress(0)
    status = st.empty()

    try:
        # Step 1: Validate file size before saving
        file_size = uploaded_file.size
        size_mb = file_size / (1024 * 1024)
        if size_mb > MAX_UPLOAD_SIZE_MB:
            st.error(f"File too large ({size_mb:.1f}MB). Maximum is {MAX_UPLOAD_SIZE_MB}MB.")
            return

        status.info("📥 Saving video...")
        progress.progress(10)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        # Step 2: Extract metadata + validate duration
        status.info("📊 Extracting video metadata...")
        progress.progress(15)
        metadata = extract_video_metadata(video_path)

        errors = validate_video(
            file_size, metadata["duration"],
            VIDEO_MIN_DURATION, VIDEO_MAX_DURATION, MAX_UPLOAD_SIZE_MB,
        )
        if errors:
            for err in errors:
                st.error(err)
            os.unlink(video_path)
            return

        # Step 3: Extract frames
        status.info("🖼️ Extracting frames...")
        progress.progress(25)
        frames = extract_frames(video_path, num_frames=5)

        # Step 4: Detect and blur faces (privacy pipeline)
        status.info("👤 Detecting faces for privacy protection...")
        progress.progress(35)

        total_faces = 0
        detection_source = "none"
        blurred_thumbnail_frame = None
        if frames:
            # Process the middle frame for the thumbnail
            mid_idx = len(frames) // 2
            for i, frame in enumerate(frames[:3]):
                result = process_frame_privacy(
                    frame,
                    google_api_key=GOOGLE_CLOUD_API_KEY,
                    azure_api_key=AZURE_FACE_API_KEY,
                    azure_endpoint=AZURE_FACE_ENDPOINT,
                )
                total_faces = max(total_faces, result["face_count"])
                if result["source"] != "none":
                    detection_source = result["source"]
                if i == min(mid_idx, 2):
                    blurred_thumbnail_frame = result["blurred_frame"]

        # Step 5: Generate and upload thumbnail
        thumbnail_url = None
        if blurred_thumbnail_frame is not None:
            status.info("🖼️ Uploading thumbnail...")
            progress.progress(45)
            thumb_bytes = generate_thumbnail(blurred_thumbnail_frame)
            storage_path = generate_storage_path(venue_name)
            ok, url_or_err = upload_to_storage(
                SUPABASE_URL, SUPABASE_SERVICE_KEY,
                thumb_bytes, storage_path, "image/jpeg",
            )
            if ok:
                thumbnail_url = url_or_err

        # Step 6: Audio analysis (real via Librosa, fallback to simulated)
        status.info("🎵 Analyzing audio...")
        progress.progress(50)
        audio_results = analyze_audio_real(video_path)

        # Step 7: Visual analysis
        status.info("🎨 Analyzing visuals...")
        progress.progress(58)
        visual_results = analyze_visual(frames)

        # Step 8: Motion detection (frame differences)
        status.info("🏃 Measuring activity...")
        progress.progress(65)
        motion_results = compute_motion_score(frames)

        # Step 9: Mood estimation from face expressions
        status.info("😊 Reading the vibe...")
        progress.progress(72)
        all_detected_faces = []
        if frames:
            for frame in frames[:3]:
                r = process_frame_privacy(
                    frame,
                    google_api_key=GOOGLE_CLOUD_API_KEY,
                    azure_api_key=AZURE_FACE_API_KEY,
                    azure_endpoint=AZURE_FACE_ENDPOINT,
                )
                all_detected_faces.extend(r["faces"])
        mood_results = extract_mood_from_faces(all_detected_faces)

        # Step 10: Crowd analysis
        status.info("👥 Analyzing crowd...")
        progress.progress(80)
        crowd_results = analyze_crowd(frames, total_faces)

        # Step 11: Calculate energy score (all 5 components)
        status.info("⚡ Calculating energy score...")
        progress.progress(88)
        energy_score = calculate_energy_score(
            audio_results, visual_results, crowd_results,
            motion=motion_results, mood=mood_results,
        )

        # Step 12: Save to database
        status.info("💾 Saving results...")
        progress.progress(95)

        result_data = {
            # Venue
            "venue_name": venue_name,
            "venue_type": venue_type,
            "latitude": latitude,
            "longitude": longitude,
            # Audio
            "bpm": audio_results.get("bpm"),
            "volume_level": audio_results.get("volume_level"),
            "genre": audio_results.get("genre"),
            "energy_level": audio_results.get("energy_level"),
            "tempo_consistency": audio_results.get("tempo_consistency"),
            # Visual
            "brightness_level": visual_results.get("brightness_level"),
            "lighting_type": visual_results.get("lighting_type"),
            "color_scheme": visual_results.get("color_scheme"),
            "visual_energy": visual_results.get("visual_energy"),
            "color_saturation": visual_results.get("color_saturation"),
            # Motion
            "visual_activity_score": motion_results.get("motion_score"),
            "movement_patterns": motion_results.get("motion_level"),
            # Crowd
            "crowd_density": crowd_results.get("crowd_density"),
            "activity_level": crowd_results.get("activity_level"),
            "density_score": crowd_results.get("density_score"),
            "estimated_people": crowd_results.get("estimated_people"),
            "engagement_score": crowd_results.get("engagement_score"),
            # Mood
            "dominant_mood": mood_results.get("dominant_mood"),
            "mood_confidence": mood_results.get("mood_confidence"),
            "joy_percentage": mood_results.get("joy_percentage"),
            "overall_vibe": mood_results.get("dominant_mood"),
            # Privacy
            "face_count": total_faces,
            "faces_blurred": total_faces,
            "privacy_protected": True,
            # Media
            "thumbnail_url": thumbnail_url,
            # Scoring
            "energy_score": energy_score,
            "processing_complete": True,
            "session_id": st.session_state.session_id,
            "user_id": st.session_state.user_id,
            "video_duration": metadata.get("duration", 0),
        }
        
        success, response = save_video_results(result_data)
        
        progress.progress(100)
        
        # Cleanup temp file
        try:
            os.unlink(video_path)
        except:
            pass
        
        if success:
            elapsed = round(time.time() - start_time, 1)
            logger.info("Upload complete: venue=%s energy=%.1f time=%ss",
                        venue_name, energy_score, elapsed)
            status.empty()
            st.session_state.videos_processed += 1
            record_upload()

            # Show results
            st.success("✅ Video analyzed and shared!")
            
            st.markdown("### 🎯 Venue Energy Score")
            render_energy_donut(energy_score)
            
            # Detailed metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <span class="metric-value">{audio_results.get('bpm', 0)}</span>
                    <span class="metric-label">BPM</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <span class="metric-value">{crowd_results.get('crowd_density', 'N/A').title()}</span>
                    <span class="metric-label">Crowd</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <span class="metric-value">{visual_results.get('lighting_type', 'N/A')}</span>
                    <span class="metric-label">Lighting</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.balloons()
        else:
            status.empty()
            st.error(f"Failed to save results: {response}")
            
            # Still show the analysis results even if save failed
            st.markdown("### 🎯 Analysis Results (Not Saved)")
            render_energy_donut(energy_score)
    
    except Exception as e:
        logger.exception("Upload failed: venue=%s error=%s", venue_name, e)
        progress.progress(100)
        status.empty()
        st.error(f"Processing error: {str(e)}")

        # Cleanup
        try:
            os.unlink(video_path)
        except:
            pass

def page_profile():
    """Profile page - account, submission history, system status"""
    st.markdown("## 👤 Your Profile")

    # ── Account section ──
    if is_logged_in():
        st.markdown(f"""
        <div class="metric-card" style="text-align: left; padding: 1.5rem;">
            <h3 style="margin: 0 0 1rem 0;">Account</h3>
            <p><strong>Email:</strong> {st.session_state.user_email}</p>
            <p><strong>User ID:</strong> {st.session_state.user_id[:8]}...</p>
            <p><strong>Videos Shared (session):</strong> {st.session_state.videos_processed}</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Sign Out", key="signout_profile", use_container_width=True):
            sign_out(SUPABASE_URL, SUPABASE_ANON_KEY, st.session_state.access_token)
            _clear_auth_state()
            st.rerun()

        # ── Submission history ──
        st.markdown("---")
        st.markdown("### 📊 Your Submissions")
        submissions = get_user_submissions(
            SUPABASE_URL, SUPABASE_ANON_KEY, st.session_state.user_id
        )
        if submissions:
            for sub in submissions:
                render_venue_card(sub)
        else:
            st.info("You haven't uploaded any videos yet. Head to the Upload tab!")

        # ── Rating stats ──
        rating_count = get_user_rating_count(
            SUPABASE_URL, SUPABASE_ANON_KEY, st.session_state.user_id
        )
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Videos Uploaded", len(submissions) if submissions else 0)
        with col2:
            st.metric("Ratings Given", rating_count)

    else:
        st.markdown("""
        <div class="metric-card" style="text-align: left; padding: 1.5rem;">
            <h3 style="margin: 0 0 1rem 0;">Not signed in</h3>
            <p>Create an account or sign in to track your uploads, rate venues, and build your profile.</p>
        </div>
        """, unsafe_allow_html=True)
        render_auth_form(key_prefix="profile")

    # ── System status (always visible) ──
    st.markdown("---")
    st.markdown("### 🔧 System Status")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Test Supabase", use_container_width=True):
            if test_supabase_connection():
                st.success("Supabase connected!")
            else:
                st.error("Supabase connection failed")

    with col2:
        st.markdown(f"""
        **Libraries:**
        - MoviePy: {'✅' if MOVIEPY_AVAILABLE else '❌'}
        - OpenCV: {'✅' if CV2_AVAILABLE else '❌'}
        - Plotly: {'✅' if PLOTLY_AVAILABLE else '❌'}
        """)

# ================================
# MAIN APP
# ================================

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="margin: 0; font-size: 2rem;">🎯 SneakPeak</h1>
        <p style="margin: 0.5rem 0; color: #666;">Real-time venue vibes</p>
    </div>
    """, unsafe_allow_html=True)

    # Auth status bar
    render_auth_header()

    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Discover", "📹 Upload", "👤 Profile"])
    
    with tab1:
        page_discover()
    
    with tab2:
        page_upload()
    
    with tab3:
        page_profile()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.8rem; padding: 1rem 0;">
        SneakPeak MVP v1.0 | Made with ❤️
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
