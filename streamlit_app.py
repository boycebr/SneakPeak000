# -*- coding: utf-8 -*-
import os
import io
import time
import hmac
import json
import base64
import hashlib
import tempfile
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

# Optional libs with graceful fallbacks
try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

try:
    from moviepy.editor import VideoFileClip
    HAS_MOVIEPY = True
except Exception:
    VideoFileClip = None
    HAS_MOVIEPY = False

from supabase import create_client, Client
import librosa
import soundfile as sf
from PIL import Image
import cv2

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="SneakPeak Video Scorer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CONFIG & INIT with explicit checks
# ============================================

def _missing(name: str) -> bool:
    val = st.secrets.get(name)
    return (val is None) or (isinstance(val, str) and (not val.strip() or "REPLACE_ME_IN_SECRETS" in val))

# --- Required: Supabase credentials ---
if _missing("SUPABASE_URL") or _missing("SUPABASE_KEY"):
    st.error(
        "Supabase credentials are not configured.\n\n"
        "Add `SUPABASE_URL` and `SUPABASE_KEY` to your app **Secrets**.\n\n"
        "On Streamlit Cloud: **Manage app → Settings → Secrets**\n"
        "Locally: create `.streamlit/secrets.toml`."
    )
    st.stop()

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# Create client (this will fail fast if the key is invalid)
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Failed to initialize Supabase client. Check SUPABASE_URL/KEY.\n\nDetails: {e}")
    st.stop()

# --- Optional: External APIs (warn if absent) ---
GOOGLE_VISION_API_KEY = st.secrets.get("GOOGLE_VISION_API_KEY", "")
if not GOOGLE_VISION_API_KEY:
    st.warning("Google Vision API key is not set. Visual analysis will fail. Add `GOOGLE_VISION_API_KEY` to Secrets.")

ACRCLOUD_ACCESS_KEY = st.secrets.get("ACRCLOUD_ACCESS_KEY", "")
ACRCLOUD_SECRET_KEY = st.secrets.get("ACRCLOUD_SECRET_KEY", "")
ACRCLOUD_API_HOST = st.secrets.get("ACRCLOUD_API_HOST", "identify-eu-west-1.acrcloud.com")
ACRCLOUD_API_ENDPOINT = st.secrets.get("ACRCLOUD_API_ENDPOINT", "/v1/identify")
if not (ACRCLOUD_ACCESS_KEY and ACRCLOUD_SECRET_KEY):
    st.warning("ACRCloud keys are not set. Genre detection will fall back to 'Unknown'. Add `ACRCLOUD_ACCESS_KEY` and `ACRCLOUD_SECRET_KEY` to Secrets.")

# ============================================
# STYLE
# ============================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 { font-size: 2rem; margin-bottom: 0.5rem; font-weight: 700; }
    .main-header p { font-size: 1.1rem; opacity: 0.9; margin: 0; }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 12px; padding: 1rem 2rem;
        font-weight: 600; font-size: 1.1rem; width: 100%; min-height: 50px;
        box-shadow: 0 4px 12px rgba(102,126,234,0.3); transition: all 0.3s ease;
    }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102,126,234,0.4); }
    .metric-card {
        background: white; padding: 1.2rem; border-radius: 12px;
        border-left: 4px solid #667eea; margin: 0.8rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .stVideo > div { border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    @media (max-width: 768px) {
        .main .block-container { padding-top: 2rem; padding-left: 1rem; padding-right: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION
# ============================================
if "processed_videos" not in st.session_state:
    st.session_state.processed_videos = []
if "user" not in st.session_state:
    st.session_state.user = None
if "show_confirm_notice" not in st.session_state:
    st.session_state.show_confirm_notice = False
if "last_signup_email" not in st.session_state:
    st.session_state.last_signup_email = None

# Restore session if present
try:
    sb_session = supabase.auth.get_session()
    if sb_session and getattr(sb_session, "user", None) and not st.session_state.user:
        st.session_state.user = sb_session.user
except Exception:
    pass

# ============================================
# HELPERS
# ============================================
def calculate_energy_score(results):
    try:
        energy_score = (
            (float(results["audio_environment"]["bpm"]) / 160) * 0.3 +
            (float(results["audio_environment"]["volume_level"]) / 100) * 0.2 +
            (float(results["crowd_density"]["density_score"]) / 20) * 0.3 +
            float(results["mood_recognition"]["confidence"]) * 0.2
        ) * 100
        return float(min(100, max(0, energy_score)))
    except Exception as e:
        st.error(f"Error calculating energy score: {e}")
        return 50.0

def generate_acrcloud_signature(timestamp: str, data_type="audio", signature_version="1"):
    string_to_sign = f"POST\n{ACRCLOUD_API_ENDPOINT}\n{ACRCLOUD_ACCESS_KEY}\n{data_type}\n{signature_version}\n{timestamp}"
    h = hmac.new(ACRCLOUD_SECRET_KEY.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha1)
    return base64.b64encode(h.digest()).strip().decode("utf-8")

def extract_audio_features(video_path):
    if not HAS_MOVIEPY:
        st.warning("MoviePy/ffmpeg not detected. Using default audio features.")
        return {"bpm": 0, "volume_level": 0.0, "genre": "Unknown", "energy_level": "Unknown"}

    temp_audio_path = None
    try:
        video = VideoFileClip(video_path)
        duration = min(video.duration or 0, 10)
        audio = video.audio.subclip(0, duration)

        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.write_audiofile(temp_audio_file.name, verbose=False, logger=None)
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close()

        y, sr = librosa.load(temp_audio_path, sr=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        volume_level = float(np.mean(rms) * 1000)

        energy_level = "Medium"
        if tempo > 120 and volume_level > 30:
            energy_level = "High"
        elif tempo < 90 or volume_level < 10:
            energy_level = "Low"

        # ACRCloud genre (optional)
        if not (ACRCLOUD_ACCESS_KEY and ACRCLOUD_SECRET_KEY):
            return {"bpm": int(tempo), "volume_level": volume_level, "genre": "Unknown", "energy_level": energy_level}

        timestamp = str(int(time.time()))
        signature = generate_acrcloud_signature(timestamp, data_type="audio", signature_version="1")
        payload = {
            "access_key": ACRCLOUD_ACCESS_KEY,
            "timestamp": timestamp,
            "signature": signature,
            "data_type": "audio",
            "signature_version": "1",
            "sample_bytes": os.path.getsize(temp_audio_path),
        }
        with open(temp_audio_path, "rb") as f:
            files = {"sample": f}
            try:
                r = requests.post(f"https://{ACRCLOUD_API_HOST}{ACRCLOUD_API_ENDPOINT}", data=payload, files=files, timeout=15)
                r.raise_for_status()
                acr = r.json()
            except requests.exceptions.RequestException as e:
                st.warning(f"ACRCloud error: {e}. Falling back to 'Unknown' genre.")
                return {"bpm": int(tempo), "volume_level": volume_level, "genre": "Unknown", "energy_level": energy_level}

        genre = "Unknown"
        if acr.get("status", {}).get("code") == 0 and acr.get("metadata", {}).get("music"):
            first = acr["metadata"]["music"][0]
            if first.get("genres"):
                g = first["genres"][0].get("name")
                if g: genre = g
            elif first.get("external_metadata", {}).get("spotify", {}).get("genres"):
                arr = first["external_metadata"]["spotify"]["genres"]
                if arr: genre = arr[0]

        return {"bpm": int(tempo), "volume_level": volume_level, "genre": genre, "energy_level": energy_level}

    except Exception as e:
        st.error(f"Audio analysis error: {e}")
        return {"bpm": 0, "volume_level": 0.0, "genre": "Unknown", "energy_level": "Unknown"}
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)

def _vision_annotate(image_path, features):
    if not GOOGLE_VISION_API_KEY:
        raise RuntimeError("Google Vision API key missing.")
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    payload = {"requests": [{"image": {"content": base64.b64encode(image_bytes).decode("utf-8")}, "features": features}]}
    url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()

def analyze_visual_features_with_vision_api(image_path):
    try:
        res = _vision_annotate(image_path, [{"type": "IMAGE_PROPERTIES"}, {"type": "LABEL_DETECTION"}])
        ip = res["responses"][0].get("imagePropertiesAnnotation", {})
        colors = ip.get("dominantColors", {}).get("colors", [])
        if colors:
            dom = sorted(colors, key=lambda x: x.get("pixelFraction", 0), reverse=True)[0]
            r = dom["color"].get("red", 0); g = dom["color"].get("green", 0); b = dom["color"].get("blue", 0)
            brightness = b * 0.299 + g * 0.587 + r * 0.114
            color_scheme = f"RGB({r}, {g}, {b})"
        else:
            brightness = float(np.random.uniform(30, 90))
            color_scheme = "Unknown"
        labels = res["responses"][0].get("labelAnnotations", [])
        label_set = {l["description"].lower() for l in labels if "description" in l}
        visual_energy = "Medium"
        if {"crowd", "party", "dance", "celebration"} & label_set: visual_energy = "High"
        elif {"quiet", "calm", "indoor", "still"} & label_set: visual_energy = "Low"
        lighting_type = "Mixed Indoor"
        if "indoor" in label_set and brightness < 100: lighting_type = "Dark/Club Lighting"
        elif "outdoor" in label_set or brightness > 150: lighting_type = "Bright/Bar Lighting"
        return {"brightness_level": float(brightness), "lighting_type": lighting_type, "color_scheme": color_scheme, "visual_energy": visual_energy}
    except RuntimeError as e:
        st.warning(str(e))
        return {"brightness_level": 0.0, "lighting_type": "Unknown", "color_scheme": "Unknown", "visual_energy": "Unknown"}
    except requests.exceptions.HTTPError as err:
        st.error(f"Vision API Error: {err.response.text}")
        return {}
    except Exception as e:
        st.error(f"Visual analysis error: {e}")
        return {}

def analyze_crowd_features_with_vision_api(image_path):
    try:
        res = _vision_annotate(image_path, [{"type": "FACE_DETECTION"}])
        faces = res["responses"][0].get("faceAnnotations", [])
        num = len(faces)
        if num == 0: density = "Empty"
        elif num <= 2: density = "Sparse"
        elif num <= 5: density = "Moderate"
        elif num <= 10: density = "Busy"
        else: density = "Packed"
        activity = "Still/Seated"
        if any(f.get("joyLikelihood") == "VERY_LIKELY" or f.get("sorrowLikelihood") == "VERY_LIKELY" for f in faces):
            activity = "High Movement/Dancing"
        elif any(f.get("joyLikelihood") == "LIKELY" for f in faces):
            activity = "Social/Standing"
        density_score = float(num * 1.5 + np.random.uniform(0, 5))
        return {"crowd_density": density, "activity_level": activity, "density_score": density_score}
    except RuntimeError as e:
        st.warning(str(e))
        return {"crowd_density": "Unknown", "activity_level": "Unknown", "density_score": 0.0}
    except requests.exceptions.HTTPError as err:
        st.error(f"Vision API Error: {err.response.text}")
        return {}
    except Exception as e:
        st.error(f"Crowd analysis error: {e}")
        return {}

def analyze_mood_recognition_with_vision_api(image_path):
    try:
        res = _vision_annotate(image_path, [{"type": "FACE_DETECTION"}])
        faces = res["responses"][0].get("faceAnnotations", [])
        if not faces:
            return {"dominant_mood": "Calm", "confidence": 0.5, "mood_breakdown": {"Calm": 1.0}, "overall_vibe": "Neutral"}
        mood_counts = {"joy": 0, "sorrow": 0, "anger": 0, "surprise": 0, "undetermined": 0}
        for f in faces:
            def add(lik, key):
                mood_counts[key] += 3 if lik == "VERY_LIKELY" else 2 if lik == "LIKELY" else 1 if lik == "POSSIBLE" else 0
            add(f.get("joyLikelihood", "UNDETERMINED"), "joy")
            add(f.get("sorrowLikelihood", "UNDETERMINED"), "sorrow")
            add(f.get("angerLikelihood", "UNDETERMINED"), "anger")
            add(f.get("surpriseLikelihood", "UNDETERMINED"), "surprise")
        key = max(mood_counts, key=mood_counts.get) if sum(mood_counts.values()) else "undetermined"
        mood_map = {"joy": "Happy", "sorrow": "Calm", "anger": "Energetic", "surprise": "Excited", "undetermined": "Social"}
        dominant = mood_map.get(key, "Social")
        total = sum(mood_counts.values()) or 1
        conf = mood_counts.get(key, 0) / total
        vibe = "Positive"
        if "Calm" in dominant or key == "sorrow": vibe = "Mixed"
        return {"dominant_mood": dominant, "confidence": float(conf), "mood_breakdown": {mood_map.get(k,k): v/total for k,v in mood_counts.items()}, "overall_vibe": vibe}
    except RuntimeError as e:
        st.warning(str(e))
        return {"dominant_mood": "Unknown", "confidence": 0.0, "mood_breakdown": {"Unknown": 1.0}, "overall_vibe": "Unknown"}
    except requests.exceptions.HTTPError as err:
        st.error(f"Vision API Error: {err.response.text}")
        return {}
    except Exception as e:
        st.error(f"Mood analysis error: {e}")
        return {}

def get_single_frame_from_video(video_path):
    temp_image_path = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
            return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        mid = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            st.error("Error: Could not read frame from video.")
            return None
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(tmp.name, frame)
        temp_image_path = tmp.name
        tmp.close()
        return temp_image_path
    except Exception as e:
        st.error(f"Frame extraction error: {e}")
        return None

def save_to_supabase(results, uploaded_file=None):
    try:
        video_id = str(uuid.uuid4())

        def upload_video(uploaded_file, video_id):
            try:
                ext = uploaded_file.name.split(".")[-1].lower()
                filename = f"{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
                headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": uploaded_file.type or "application/octet-stream"}
                r = requests.post(f"{SUPABASE_URL}/storage/v1/object/videos/{filename}", headers=headers, data=uploaded_file.getvalue(), timeout=60)
                if r.status_code in (200, 201):
                    url = f"{SUPABASE_URL}/storage/v1/object/public/videos/{filename}"
                    return url, filename
                st.error(f"Video upload failed: {r.status_code} {r.text}")
                return None, None
            except Exception as e:
                st.error(f"Upload error: {e}")
                return None, None

        video_url = None
        video_filename = None
        if uploaded_file:
            video_url, video_filename = upload_video(uploaded_file, video_id)

        user_id = st.session_state.user.id if st.session_state.user else None
        if not user_id:
            st.error("❌ Cannot save results. You must be logged in.")
            return False, None

        gps = results.get("gps_data", {}) or {}
        db_data = {
            "id": video_id,
            "user_id": user_id,
            "venue_name": str(results["venue_name"])[:100],
            "venue_type": str(results["venue_type"])[:50],
            "video_url": video_url,
            "video_filename": video_filename,
            "video_stored": bool(video_url),
            "latitude": float(gps.get("latitude")) if gps.get("latitude") is not None else None,
            "longitude": float(gps.get("longitude")) if gps.get("longitude") is not None else None,
            "gps_accuracy": float(gps.get("accuracy")) if gps.get("accuracy") is not None else None,
            "venue_verified": bool(gps.get("venue_verified", False)),
            "bpm": int(results["audio_environment"]["bpm"]),
            "volume_level": float(results["audio_environment"]["volume_level"]),
            "genre": str(results["audio_environment"]["genre"])[:50],
            "energy_level": str(results["audio_environment"]["energy_level"])[:20],
            "brightness_level": float(results["visual_environment"]["brightness_level"]),
            "lighting_type": str(results["visual_environment"]["lighting_type"])[:50],
            "color_scheme": str(results["visual_environment"]["color_scheme"])[:50],
            "visual_energy": str(results["visual_environment"]["visual_energy"])[:20],
            "crowd_density": str(results["crowd_density"]["crowd_density"])[:20],
            "activity_level": str(results["crowd_density"]["activity_level"])[:50],
            "density_score": float(results["crowd_density"]["density_score"]),
            "dominant_mood": str(results["mood_recognition"]["dominant_mood"])[:30],
            "mood_confidence": float(results["mood_recognition"]["confidence"]),
            "overall_vibe": str(results["mood_recognition"]["overall_vibe"])[:30],
            "energy_score": float(calculate_energy_score(results)),
        }
        headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json", "Prefer": "return=minimal"}
        resp = requests.post(f"{SUPABASE_URL}/rest/v1/video_results", headers=headers, json=db_data, timeout=60)
        if resp.status_code == 201:
            st.success("✅ Results saved to database!")
            return True, video_id
        st.error(f"❌ Database save failed: {resp.status_code} {resp.text}")
        return False, None
    except Exception as e:
        st.error(f"Database error: {e}")
        return False, None

def load_user_results(user_id):
    if not user_id:
        return []
    try:
        data = supabase.from_("video_results").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        return data.data
    except Exception as e:
        st.error(f"Data load error: {e}")
        return []

def load_video_by_id(video_id):
    try:
        headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"}
        resp = requests.get(f"{SUPABASE_URL}/rest/v1/video_results?id=eq.{video_id}", headers=headers, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return data[0] if data else None
        st.error(f"Failed to load video {video_id}: {resp.status_code} {resp.text}")
        return None
    except Exception as e:
        st.error(f"Video load error: {e}")
        return None

# ---------- AUTH ----------
def handle_login(email, password):
    try:
        resp = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if resp and getattr(resp, "user", None):
            st.session_state.user = resp.user
            st.success("Logged in successfully!")
            st.toast("Logged in ✅", icon="✅")
        else:
            st.error("Login failed. Please check your credentials.")
    except Exception as e:
        # Common causes: invalid API key, email not confirmed, wrong URL
        st.error(f"Login failed: {e}")

def handle_signup(email, password):
    try:
        resp = supabase.auth.sign_up({"email": email, "password": password})
        st.session_state.last_signup_email = email
        st.session_state.show_confirm_notice = True
        if resp and getattr(resp, "user", None):
            st.success("Account created!")
        st.info("We’ve sent a confirmation email. Please click the link to activate your account.")
        st.toast("Check your email to confirm your signup", icon="✉️")
    except Exception as e:
        st.error(f"Sign up failed: {e}")

def handle_logout():
    try: supabase.auth.sign_out()
    except Exception: pass
    st.session_state.user = None
    st.success("Logged out successfully!")
    st.toast("Logged out", icon="👋")

# ---------- UI ----------
def display_results(results):
    st.subheader(f"📊 Analysis Results for {results['venue_name']}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="metric-card"><h4>Overall Vibe</h4><p>{results.get("overall_vibe","N/A")}</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h4>Energy Score</h4><p>{float(results.get("energy_score",0)):.2f}/100</p></div>', unsafe_allow_html=True)

    with st.expander("🔊 Audio Environment"):
        st.markdown(f"**BPM:** {int(results.get('bpm', results['audio_environment']['bpm']))} BPM")
        vol = results.get("volume_level", results["audio_environment"]["volume_level"])
        st.markdown(f"**Volume Level:** {float(vol):.2f}%")
        st.markdown(f"**Genre:** {results.get('genre', results['audio_environment']['genre'])}")
        st.markdown(f"**Energy Level:** {results.get('energy_level', results['audio_environment']['energy_level'])}")

    with st.expander("💡 Visual Environment"):
        bright = results.get("brightness_level", results["visual_environment"]["brightness_level"])
        st.markdown(f"**Brightness:** {float(bright):.2f}/255")
        st.markdown(f"**Lighting Type:** {results.get('lighting_type', results['visual_environment']['lighting_type'])}")
        st.markdown(f"**Color Scheme:** {results.get('color_scheme', results['visual_environment']['color_scheme'])}")
        st.markdown(f"**Visual Energy:** {results.get('visual_energy', results['visual_environment']['visual_energy'])}")

    with st.expander("🕺 Crowd & Mood"):
        st.markdown(f"**Crowd Density:** {results.get('crowd_density',{}).get('crowd_density', results.get('crowd_density','N/A'))}")
        st.markdown(f"**Activity Level:** {results.get('activity_level', results.get('crowd_density',{}).get('activity_level','N/A'))}")
        st.markdown(f"**Dominant Mood:** {results.get('dominant_mood', results['mood_recognition']['dominant_mood'])} (Confidence: {float(results.get('mood_confidence', results['mood_recognition']['confidence'])):.2f})")

        if "mood_recognition" in results and "mood_breakdown" in results["mood_recognition"]:
            items = list(results["mood_recognition"]["mood_breakdown"].items())
        else:
            items = [("n/a", 1.0)]
        mood_df = pd.DataFrame(items, columns=["Mood", "Confidence"]).sort_values(by="Confidence", ascending=False)
        fig, ax = plt.subplots()
        if HAS_SEABORN:
            sns.barplot(x="Confidence", y="Mood", data=mood_df, ax=ax)
        else:
            ax.barh(mood_df["Mood"], mood_df["Confidence"])
        ax.set_title("Mood Breakdown"); ax.set_xlabel("Confidence"); ax.set_ylabel("")
        st.pyplot(fig)

def display_all_results_page():
    st.subheader("Your Uploaded Videos")
    if st.session_state.user:
        vids = load_user_results(st.session_state.user.id)
        if vids:
            st.write(f"Showing {len(vids)} videos uploaded by you.")
            q = st.text_input("Search your videos by venue name...", "")
            filt = [v for v in vids if q.lower() in (v.get("venue_name","") or "").lower()]
            if not filt:
                st.info("No videos match your search criteria.")
            for v in filt:
                if v.get("id"):
                    with st.expander(f"**{v.get('venue_name','?')}** ({v.get('venue_type','?')}) - {v.get('created_at','')[:10]}"):
                        if v.get("video_url"): st.video(v["video_url"])
                        else: st.info("No video file was stored for this entry.")
                        c1, c2 = st.columns(2)
                        c1.metric("Overall Vibe", v.get("overall_vibe", "N/A"))
                        c2.metric("Energy Score", f"{float(v.get('energy_score', 0)):.2f}/100")
                        rating = st.slider("Rate this video (1-5):", 1, 5, 3, key=f"slider_{v['id']}")
                        if st.button(f"Submit Rating for {v.get('venue_name','this venue')}", key=f"btn_{v['id']}"):
                            # You can implement save_user_rating again later if needed
                            st.success("Thank you for your rating!")
                        st.json(v)
                else:
                    st.error("❌ A video record was found but is missing a unique ID. Skipping display.")
        else:
            st.info("You have not uploaded any videos yet. Upload one from the 'Upload & Analyze' page!")
    else:
        st.warning("Please log in to view your uploaded videos.")

# ---------- MAIN ----------
def main():
    st.markdown('<div class="main-header"><h1>SneakPeak Video Scorer</h1><p>A tool for real-time venue intelligence</p></div>', unsafe_allow_html=True)

    if not HAS_MOVIEPY:
        st.warning("MoviePy/ffmpeg not detected. Audio analysis will be limited.")

    # Post-signup banner
    if st.session_state.show_confirm_notice:
        st.warning("Finish setting up your account: **check your email and click the confirmation link** to activate your login.")
        with st.expander("Having trouble confirming your email?"):
            st.markdown("""
- If the link doesn't open, copy the URL from the email and paste it directly into your browser.
- If you see “**this site can't be reached**,” the project’s Auth redirect URL may be misconfigured.
  - In Supabase Dashboard → **Authentication → URL Configuration**, set **Site URL** (and **Redirect URLs**, if used) to your app’s deployed URL (or `http://localhost:8501` for local testing).
- You can also try sending the email again with the button below.
            """)
        if st.session_state.last_signup_email and st.button("Resend confirmation email"):
            try:
                supabase.auth.resend({"type": "signup", "email": st.session_state.last_signup_email})
                st.success("Confirmation email re-sent. Please check your inbox.")
                st.toast("Confirmation email sent ✉️", icon="✉️")
            except Exception as e:
                st.error(f"Could not resend confirmation email: {e}")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload & Analyze", "View My Videos"])

    st.sidebar.header("User Account")
    if st.session_state.user:
        st.sidebar.success(f"Logged in as {st.session_state.user.email}")
        if st.sidebar.button("Log Out"):
            handle_logout()
    else:
        auth_tab = st.sidebar.tabs(["Log In", "Sign Up"])
        with auth_tab[0]:
            with st.form("login_form"):
                lemail = st.text_input("Email", key="login_email")
                lpass = st.text_input("Password", type="password", key="login_pass")
                if st.form_submit_button("Log In"):
                    handle_login(lemail, lpass)
        with auth_tab[1]:
            with st.form("signup_form"):
                semail = st.text_input("Email", key="signup_email")
                spass = st.text_input("Password", type="password", key="signup_pass")
                if st.form_submit_button("Sign Up"):
                    handle_signup(semail, spass)

    if page == "Upload & Analyze":
        st.header("Upload a Video")
        if not st.session_state.user:
            st.warning("Please log in to upload and analyze a video.")
            return

        st.info(f"You are logged in as {st.session_state.user.email}.")

        with st.form("analysis_form"):
            st.subheader("Enter Venue Details")
            venue_name = st.text_input("Venue Name", "Demo Nightclub", key="venue_name_input")
            venue_type = st.selectbox(
                "Venue Type",
                ["Club", "Bar", "Restaurant", "Lounge", "Rooftop", "Outdoors Space", "Concert Hall", "Event Space", "Dive Bar", "Speakeasy", "Sports Bar", "Brewery", "Other"],
                key="venue_type_input"
            )

            # GPS skipped for now; placeholders kept for DB schema compatibility
            latitude = longitude = accuracy = None

            uploaded_file = st.file_uploader("Choose a video file (max 200MB)...", type=["mp4", "mov", "avi"])
            submitted = st.form_submit_button("Start Analysis")

            if submitted:
                if not uploaded_file:
                    st.error("Please upload a video file to proceed with the analysis.")
                    return
                if (uploaded_file.size / (1024 * 1024)) > 200:
                    st.error("File size exceeds 200MB limit. Please upload a smaller video.")
                    return

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                    tfile.write(uploaded_file.getvalue())
                    temp_video_path = tfile.name

                temp_image_path = None
                progress_bar = st.progress(0, text="Initializing analysis...")

                try:
                    progress_bar.progress(10, text="Extracting video frame...")
                    temp_image_path = get_single_frame_from_video(temp_video_path)
                    if not temp_image_path:
                        st.error("Failed to extract a video frame. Analysis aborted.")
                        return

                    progress_bar.progress(30, text="Analyzing audio...")
                    audio_features = extract_audio_features(temp_video_path)

                    progress_bar.progress(50, text="Analyzing visual environment...")
                    visual_features = analyze_visual_features_with_vision_api(temp_image_path)

                    progress_bar.progress(70, text="Analyzing crowd density and activity...")
                    crowd_features = analyze_crowd_features_with_vision_api(temp_image_path)

                    progress_bar.progress(90, text="Detecting dominant mood...")
                    mood_features = analyze_mood_recognition_with_vision_api(temp_image_path)

                    results = {
                        "venue_name": venue_name,
                        "venue_type": venue_type,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "gps_data": {
                            "latitude": latitude,
                            "longitude": longitude,
                            "accuracy": accuracy,
                            "venue_verified": False
                        },
                        "audio_environment": audio_features,
                        "visual_environment": visual_features,
                        "crowd_density": crowd_features,
                        "mood_recognition": mood_features
                    }
                    results["energy_score"] = calculate_energy_score(results)

                    progress_bar.progress(100, text="Saving results to database...")
                    ok, video_id = save_to_supabase(results, uploaded_file)
                    if ok:
                        saved = load_video_by_id(video_id)
                        if saved:
                            st.session_state.processed_videos.append(saved)
                            st.success("Analysis complete!")
                            st.toast("Analysis complete ✅", icon="✅")
                            display_results(saved)

                except Exception as e:
                    st.error(f"Unexpected error during analysis: {e}")
                finally:
                    if temp_video_path and os.path.exists(temp_video_path): os.unlink(temp_video_path)
                    if temp_image_path and os.path.exists(temp_image_path): os.unlink(temp_image_path)
                    try: progress_bar.empty()
                    except Exception: pass

    elif page == "View My Videos":
        display_all_results_page()

if __name__ == "__main__":
    main()
