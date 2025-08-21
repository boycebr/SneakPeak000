# streamlit_app.py
# SneakPeak Video Scorer — clean build (no SQL inside this file)

import os
import io
import uuid
import time
import base64
import hmac
import hashlib
import tempfile
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Optional video/audio stack
try:
    from moviepy.editor import VideoFileClip  # requires system ffmpeg to do audio
    MOVIEPY_OK = True
except Exception:
    MOVIEPY_OK = False

try:
    import cv2
    OPENCV_OK = True
except Exception:
    OPENCV_OK = False

try:
    import librosa
    import soundfile as sf
    LIBROSA_OK = True
except Exception:
    LIBROSA_OK = False

# Supabase client (auth + simple selects)
try:
    from supabase import create_client, Client
    SUPABASE_LIB_OK = True
except Exception:
    SUPABASE_LIB_OK = False


# --------------------------------------------------------------------------------------
# Configuration / Secrets (read from Streamlit Secrets, with safe fallbacks for testing)
# --------------------------------------------------------------------------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "https://tmmheslzkqiveylrnpal.supabase.co")
SUPABASE_KEY = st.secrets.get("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRtbWhlc2x6a3FpdmV5bHJucGFsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQzMzI5MjAsImV4cCI6MjA2OTkwODkyMH0.U-10R707xIs6rH-Vd5lBgh2INylFu6zn_EyoJYx_zpI")

GOOGLE_VISION_API_KEY = st.secrets.get("GOOGLE_VISION_API_KEY", "AIzaSyCcwH6w-3AglhEUmegXlWOtABZzJ1MrSiQ")
ACRCLOUD_ACCESS_KEY = st.secrets.get("ACRCLOUD_ACCESS_KEY", "b1f7b901a4f15b99aba0efac395f6848")
ACRCLOUD_SECRET_KEY = st.secrets.get("ACRCLOUD_SECRET_KEY", "tIVqMBQwOYGkCjkXAyY2wPiM5wxS5UrNwqMwMQjA")
ACRCLOUD_API_HOST = "identify-eu-west-1.acrcloud.com"
ACRCLOUD_API_ENDPOINT = "/v1/identify"

# Supabase Client
supabase: "Client | None" = None
if SUPABASE_LIB_OK:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.warning(f"Supabase client init failed: {e}")
else:
    st.warning("supabase-py not installed; auth/reads via client may be limited.")

# --------------------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="SneakPeak Video Scorer", page_icon="🎯", layout="wide")

# Visible diagnostics (you asked to keep warnings visible)
if not GOOGLE_VISION_API_KEY or GOOGLE_VISION_API_KEY.startswith("AIzaSyC"):
    st.warning("Google Vision API key is set (testing default). If this is a placeholder, set GOOGLE_VISION_API_KEY in Secrets.")
else:
    st.info("Google Vision API is configured.")

if not ACRCLOUD_ACCESS_KEY or not ACRCLOUD_SECRET_KEY:
    st.warning("ACRCloud keys are not set. Genre detection will fall back to 'Unknown'. Add ACRCLOUD_ACCESS_KEY and ACRCLOUD_SECRET_KEY to Secrets.")
else:
    st.info("ACRCloud API is configured.")

if not MOVIEPY_OK:
    st.warning("MoviePy is not importable. Audio analysis will be limited.")
else:
    # MoviePy still needs system ffmpeg; check quickly
    try:
        # lightweight check without actually encoding
        _ = VideoFileClip
        st.info("MoviePy is available. If you still see 'ffmpeg not found' later, install ffmpeg on the host.")
    except Exception:
        st.warning("MoviePy is present, but ffmpeg may be missing on the system. Audio analysis may be limited.")

# --------------------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "processed_videos" not in st.session_state:
    st.session_state.processed_videos = []

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def generate_acrcloud_signature(timestamp: str) -> bytes:
    string_to_sign = f"POST\n{ACRCLOUD_API_ENDPOINT}\n{ACRCLOUD_ACCESS_KEY}\n{ACRCLOUD_SECRET_KEY}\n{timestamp}"
    h = hmac.new(ACRCLOUD_SECRET_KEY.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha1)
    return base64.b64encode(h.digest()).strip()

def upload_video_to_supabase(uploaded_file, video_id: str):
    """Upload to the 'videos' storage bucket via REST."""
    try:
        ext = uploaded_file.name.split(".")[-1].lower()
        filename = f"{video_id}_{int(time.time())}.{ext}"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/octet-stream",
        }
        url = f"{SUPABASE_URL}/storage/v1/object/videos/{filename}"
        r = requests.post(url, headers=headers, data=uploaded_file.getvalue(), timeout=60)
        if r.status_code in (200, 201, 204):
            public_url = f"{SUPABASE_URL}/storage/v1/object/public/videos/{filename}"
            return public_url, filename
        else:
            st.error(f"Video upload failed: {r.status_code} {r.text}")
            return None, None
    except Exception as e:
        st.error(f"Video upload exception: {e}")
        return None, None

def load_user_results(user_id: str):
    if not user_id or not supabase:
        return []
    try:
        data = supabase.table("video_results").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        return data.data or []
    except Exception as e:
        st.error(f"Load results failed: {e}")
        return []

def load_video_by_id(video_id: str):
    try:
        headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
        url = f"{SUPABASE_URL}/rest/v1/video_results?id=eq.{video_id}"
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200 and isinstance(r.json(), list) and r.json():
            return r.json()[0]
        st.error(f"Failed to fetch video by id: {r.status_code} {r.text}")
        return None
    except Exception as e:
        st.error(f"Fetch by id exception: {e}")
        return None

def calculate_energy_score(results: dict) -> float:
    try:
        score = (
            (float(results["audio_environment"]["bpm"]) / 160) * 0.3 +
            (float(results["audio_environment"]["volume_level"]) / 100) * 0.2 +
            (float(results["crowd_density"]["density_score"]) / 20) * 0.3 +
            float(results["mood_recognition"]["confidence"]) * 0.2
        ) * 100.0
        return float(min(100.0, max(0.0, score)))
    except Exception:
        return 50.0

def extract_audio_features(video_path: str) -> dict:
    """Very defensive audio analysis: works even without ffmpeg/librosa."""
    if not MOVIEPY_OK:
        return {"bpm": 0, "volume_level": 0.0, "genre": "Unknown", "energy_level": "Unknown"}

    temp_audio_path = None
    try:
        clip = VideoFileClip(video_path)
        duration = min(clip.duration or 0, 10)  # limit to first 10s
        audio = clip.audio.subclip(0, duration) if clip.audio else None
        if not audio:
            return {"bpm": 0, "volume_level": 0.0, "genre": "Unknown", "energy_level": "Unknown"}

        t = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio_path = t.name
        t.close()
        audio.write_audiofile(temp_audio_path, verbose=False, logger=None)

        if LIBROSA_OK:
            y, sr = librosa.load(temp_audio_path, sr=None)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)
            vol = float(np.mean(rms) * 1000.0)
        else:
            tempo, vol = 0, 0.0

        energy = "Medium"
        if tempo > 120 and vol > 30:
            energy = "High"
        elif tempo < 90 or vol < 10:
            energy = "Low"

        # Genre via ACRCloud (best-effort)
        genre = "Unknown"
        if ACRCLOUD_ACCESS_KEY and ACRCLOUD_SECRET_KEY:
            try:
                ts = str(int(time.time()))
                sig = generate_acrcloud_signature(ts)
                payload = {
                    "access_key": ACRCLOUD_ACCESS_KEY,
                    "timestamp": ts,
                    "signature": sig,
                    "data_type": "audio",
                    "signature_version": "1",
                    "sample_bytes": os.path.getsize(temp_audio_path),
                }
                with open(temp_audio_path, "rb") as f:
                    files = {"sample": f}
                    resp = requests.post(
                        f"https://{ACRCLOUD_API_HOST}{ACRCLOUD_API_ENDPOINT}",
                        data=payload,
                        files=files,
                        timeout=15,
                    )
                if resp.ok:
                    data = resp.json()
                    if data.get("status", {}).get("code") == 0:
                        md = (data.get("metadata") or {}).get("music") or []
                        if md:
                            g = (md[0].get("genres") or [])
                            if g and g[0].get("name"):
                                genre = g[0]["name"]
            except Exception as e:
                st.warning(f"ACRCloud request failed (using Unknown): {e}")

        return {"bpm": int(tempo), "volume_level": float(vol), "genre": genre, "energy_level": energy}
    except Exception as e:
        st.error(f"Audio extraction error: {e}")
        return {"bpm": 0, "volume_level": 0.0, "genre": "Unknown", "energy_level": "Unknown"}
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception:
                pass

def get_single_frame(video_path: str) -> str | None:
    if not OPENCV_OK:
        st.error("OpenCV is not available; cannot extract frame.")
        return None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not open video.")
            return None
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = max(0, count // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            st.error("Could not grab frame.")
            return None
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(t.name, frame)
        t.close()
        return t.name
    except Exception as e:
        st.error(f"Frame extraction error: {e}")
        return None

def vision_visual_features(image_path: str) -> dict:
    if not GOOGLE_VISION_API_KEY:
        return {}
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        payload = {
            "requests": [
                {
                    "image": {"content": img_b64},
                    "features": [{"type": "IMAGE_PROPERTIES"}, {"type": "LABEL_DETECTION"}],
                }
            ]
        }
        r = requests.post(
            f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}",
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        props = (data["responses"][0].get("imagePropertiesAnnotation") or {})
        colors = (props.get("dominantColors") or {}).get("colors") or []
        if colors:
            c = sorted(colors, key=lambda x: x.get("pixelFraction", 0), reverse=True)[0]
            r_, g_, b_ = c["color"].get("red", 0), c["color"].get("green", 0), c["color"].get("blue", 0)
            brightness = b_ * 0.299 + g_ * 0.587 + r_ * 0.114
            color_scheme = f"RGB({r_}, {g_}, {b_})"
        else:
            brightness = np.random.uniform(30, 90)
            color_scheme = "Unknown"

        labels = [l.get("description", "").lower() for l in (data["responses"][0].get("labelAnnotations") or [])]
        visual_energy = "Medium"
        if any(x in labels for x in ["crowd", "party", "dance", "celebration"]):
            visual_energy = "High"
        elif any(x in labels for x in ["quiet", "calm", "indoor", "still"]):
            visual_energy = "Low"

        lighting = "Mixed Indoor"
        if "indoor" in labels and brightness < 100:
            lighting = "Dark/Club Lighting"
        elif "outdoor" in labels or brightness > 150:
            lighting = "Bright/Bar Lighting"

        return {
            "brightness_level": float(brightness),
            "lighting_type": lighting,
            "color_scheme": color_scheme,
            "visual_energy": visual_energy,
        }
    except Exception as e:
        st.error(f"Vision visual analysis error: {e}")
        return {}

def vision_crowd_features(image_path: str) -> dict:
    if not GOOGLE_VISION_API_KEY:
        return {}
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        payload = {
            "requests": [{"image": {"content": img_b64}, "features": [{"type": "FACE_DETECTION"}]}]
        }
        r = requests.post(
            f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}",
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        faces = data["responses"][0].get("faceAnnotations") or []
        n = len(faces)
        if n == 0:
            density = "Empty"
        elif n <= 2:
            density = "Sparse"
        elif n <= 5:
            density = "Moderate"
        elif n <= 10:
            density = "Busy"
        else:
            density = "Packed"

        activity = "Still/Seated"
        if any(f.get("joyLikelihood") == "VERY_LIKELY" for f in faces):
            activity = "High Movement/Dancing"
        elif any(f.get("joyLikelihood") == "LIKELY" for f in faces):
            activity = "Social/Standing"

        score = float(n * 1.5 + np.random.uniform(0, 5))
        return {"crowd_density": density, "activity_level": activity, "density_score": score}
    except Exception as e:
        st.error(f"Vision crowd analysis error: {e}")
        return {}

def vision_mood(image_path: str) -> dict:
    if not GOOGLE_VISION_API_KEY:
        return {"dominant_mood": "Calm", "confidence": 0.5, "mood_breakdown": {"Calm": 1.0}, "overall_vibe": "Neutral"}
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        payload = {
            "requests": [{"image": {"content": img_b64}, "features": [{"type": "FACE_DETECTION"}]}]
        }
        r = requests.post(
            f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}",
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        faces = data["responses"][0].get("faceAnnotations") or []

        if not faces:
            return {"dominant_mood": "Calm", "confidence": 0.5, "mood_breakdown": {"Calm": 1.0}, "overall_vibe": "Neutral"}

        weights = {"VERY_LIKELY": 3, "LIKELY": 2, "POSSIBLE": 1}
        buckets = {"joy": 0, "sorrow": 0, "anger": 0, "surprise": 0}
        for f in faces:
            for k, key in [("joy", "joyLikelihood"), ("sorrow", "sorrowLikelihood"),
                           ("anger", "angerLikelihood"), ("surprise", "surpriseLikelihood")]:
                buckets[k] += weights.get(f.get(key, ""), 0)

        dom_key = max(buckets, key=buckets.get) if sum(buckets.values()) else "joy"
        mood_map = {"joy": "Happy", "sorrow": "Calm", "anger": "Energetic", "surprise": "Excited"}
        dominant = mood_map.get(dom_key, "Social")
        total = sum(buckets.values()) or 1
        conf = buckets.get(dom_key, 0) / total
        overall = "Positive"
        if dominant == "Calm":
            overall = "Mixed"

        breakdown = {mood_map.get(k, k): v / total for k, v in buckets.items()}
        return {"dominant_mood": dominant, "confidence": float(conf), "mood_breakdown": breakdown, "overall_vibe": overall}
    except Exception as e:
        st.error(f"Vision mood analysis error: {e}")
        return {"dominant_mood": "Calm", "confidence": 0.5, "mood_breakdown": {"Calm": 1.0}, "overall_vibe": "Neutral"}

def save_to_supabase(results: dict, uploaded_file=None):
    """Insert into public.video_results via REST, and upload video to Storage if provided."""
    if not st.session_state.user:
        st.error("You must be logged in to save results.")
        return False, None

    video_id = str(uuid.uuid4())
    video_url = None
    video_filename = None
    if uploaded_file is not None:
        video_url, video_filename = upload_video_to_supabase(uploaded_file, video_id)

    payload = {
        "id": video_id,
        "user_id": st.session_state.user.id,  # UUID string
        "venue_name": str(results["venue_name"])[:100],
        "venue_type": str(results["venue_type"])[:50],
        "video_url": video_url,
        "video_filename": video_filename,
        "video_stored": bool(video_url),
        "latitude": None,
        "longitude": None,
        "gps_accuracy": None,
        "venue_verified": False,
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
        "energy_score": float(results["energy_score"]),
    }

    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        r = requests.post(f"{SUPABASE_URL}/rest/v1/video_results", headers=headers, json=payload, timeout=30)
        if r.status_code == 201:
            st.success("Saved to database.")
            return True, video_id
        st.error(f"Database save failed: {r.status_code} {r.text}")
        return False, None
    except Exception as e:
        st.error(f"Save exception: {e}")
        return False, None

def display_results(results_row: dict):
    st.subheader(f"📊 Analysis Results for {results_row.get('venue_name','Unknown')}")
    col1, col2 = st.columns(2)
    col1.metric("Overall Vibe", results_row.get("overall_vibe", "N/A"))
    col2.metric("Energy Score", f"{float(results_row.get('energy_score', 0)):.2f}/100")

    with st.expander("🔊 Audio Environment"):
        st.write(
            f"**BPM:** {results_row.get('bpm','N/A')} | "
            f"**Volume:** {float(results_row.get('volume_level',0)):.2f}% | "
            f"**Genre:** {results_row.get('genre','Unknown')} | "
            f"**Energy:** {results_row.get('energy_level','Unknown')}"
        )

    with st.expander("💡 Visual Environment"):
        st.write(
            f"**Brightness:** {float(results_row.get('brightness_level',0)):.2f}/255 | "
            f"**Lighting:** {results_row.get('lighting_type','N/A')} | "
            f"**Color:** {results_row.get('color_scheme','N/A')} | "
            f"**Visual Energy:** {results_row.get('visual_energy','N/A')}"
        )

    with st.expander("🕺 Crowd & Mood"):
        st.write(
            f"**Crowd Density:** {results_row.get('crowd_density','N/A')} | "
            f"**Activity:** {results_row.get('activity_level','N/A')} | "
            f"**Dominant Mood:** {results_row.get('dominant_mood','N/A')} "
            f"(Confidence: {float(results_row.get('mood_confidence',0)):.2f})"
        )

def handle_login(email: str, password: str):
    if not supabase:
        st.error("Supabase client not initialized.")
        return
    try:
        user = supabase.auth.sign_in_with_password({"email": email, "password": password}).user
        st.session_state.user = user
        st.success("Logged in.")
        st.rerun()
    except Exception as e:
        st.error(f"Login failed: {e}")

def handle_signup(email: str, password: str):
    if not supabase:
        st.error("Supabase client not initialized.")
        return
    try:
        user = supabase.auth.sign_up({"email": email, "password": password}).user
        st.session_state.user = user
        st.info("Sign-up created. Please check your email and click the confirmation link, then return to log in.")
        st.rerun()
    except Exception as e:
        st.error(f"Sign up failed: {e}")

def handle_logout():
    if supabase:
        try:
            supabase.auth.sign_out()
        except Exception:
            pass
    st.session_state.user = None
    st.success("Logged out.")
    st.rerun()

# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
st.title("SneakPeak Video Scorer")

st.sidebar.header("Account")
if st.session_state.user:
    st.sidebar.success(f"Logged in as {st.session_state.user.email}")
    if st.sidebar.button("Log Out"):
        handle_logout()
else:
    with st.sidebar.form("auth"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        login = st.form_submit_button("Log In")
        signup = st.form_submit_button("Sign Up")
        if login:
            handle_login(email, password)
        elif signup:
            handle_signup(email, password)
    st.sidebar.caption("After signing up, confirm your email from the message you receive. If the link fails, re-open the Streamlit app and try logging in again after a minute.")

page = st.sidebar.radio("Go to", ["Upload & Analyze", "View My Videos"])

if page == "Upload & Analyze":
    st.header("Upload a Video")
    if not st.session_state.user:
        st.warning("Please log in to upload and analyze a video.")
    else:
        uploaded = st.file_uploader("Choose a video (max 200MB)", type=["mp4", "mov", "avi", "m4v"])
        venue_name = st.text_input("Venue Name", "Demo Nightclub")
        venue_type = st.selectbox(
            "Venue Type",
            ["Club", "Bar", "Restaurant", "Lounge", "Rooftop", "Outdoors Space", "Concert Hall", "Event Space", "Dive Bar", "Speakeasy", "Sports Bar", "Brewery", "Other"],
        )

        if st.button("Start Analysis", disabled=uploaded is None):
            if not uploaded:
                st.error("Please choose a video file.")
            else:
                size_mb = uploaded.size / (1024 * 1024)
                if size_mb > 200:
                    st.error("File exceeds 200MB limit.")
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as t:
                        t.write(uploaded.getvalue())
                        video_path = t.name

                    image_path = None
                    try:
                        prog = st.progress(0, text="Extracting frame...")
                        image_path = get_single_frame(video_path)
                        if not image_path:
                            st.error("Could not extract frame; aborting.")
                            raise RuntimeError("no-frame")

                        prog.progress(20, text="Analyzing audio…")
                        audio = extract_audio_features(video_path)

                        prog.progress(50, text="Analyzing visual…")
                        visual = vision_visual_features(image_path)

                        prog.progress(70, text="Analyzing crowd…")
                        crowd = vision_crowd_features(image_path)

                        prog.progress(85, text="Analyzing mood…")
                        mood = vision_mood(image_path)

                        results = {
                            "venue_name": venue_name,
                            "venue_type": venue_type,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "gps_data": {},
                            "audio_environment": audio,
                            "visual_environment": visual,
                            "crowd_density": crowd,
                            "mood_recognition": mood,
                        }
                        results["energy_score"] = calculate_energy_score(results)

                        prog.progress(95, text="Saving to database…")
                        ok, vid = save_to_supabase(results, uploaded)
                        prog.progress(100, text="Done")

                        if ok and vid:
                            row = load_video_by_id(vid)
                            if row:
                                st.success("Analysis complete.")
                                display_results(row)
                    finally:
                        try:
                            if video_path and os.path.exists(video_path):
                                os.unlink(video_path)
                        except Exception:
                            pass
                        try:
                            if image_path and os.path.exists(image_path):
                                os.unlink(image_path)
                        except Exception:
                            pass

elif page == "View My Videos":
    st.header("Your Uploaded Videos")
    if not st.session_state.user:
        st.warning("Please log in to view your uploads.")
    else:
        rows = load_user_results(st.session_state.user.id)
        if not rows:
            st.info("No uploads yet.")
        else:
            q = st.text_input("Search by venue name")
            filtered = [r for r in rows if q.lower() in (r.get("venue_name","").lower())]
            for r in filtered:
                with st.expander(f"{r.get('venue_name','(Unknown)')} — {r.get('venue_type','')}" ):
                    if r.get("video_url"):
                        st.video(r["video_url"])
                    display_results(r)
