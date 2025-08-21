# streamlit_app.py
# SneakPeak Video Scorer — uses authed Supabase client for Storage + DB (fixes 403)
# and does not send 'id' on inserts (fixes bigint/UUID error).
# UPDATED: persists/restores Supabase session so the client is truly "authed".

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

# Optional libs (graceful fallbacks)
try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_OK = True
except Exception:
    MOVIEPY_OK = False

try:
    import cv2
    OPENCV_OK = True
except Exception:
    OPENCV_OK = False

try:
    import librosa, soundfile as sf
    LIBROSA_OK = True
except Exception:
    LIBROSA_OK = False

try:
    from supabase import create_client, Client
    SUPABASE_LIB_OK = True
except Exception:
    SUPABASE_LIB_OK = False

# --------------------------------------------------------------------------------------
# Config / Secrets
# --------------------------------------------------------------------------------------
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "https://tmmheslzkqiveylrnpal.supabase.co")
SUPABASE_KEY = st.secrets.get("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRtbWhlc2x6a3FpdmV5bHJucGFsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQzMzI5MjAsImV4cCI6MjA2OTkwODkyMH0.U-10R707xIs6rH-Vd5lBgh2INylFu6zn_EyoJYx_zpI")

GOOGLE_VISION_API_KEY = st.secrets.get("GOOGLE_VISION_API_KEY", "AIzaSyCcwH6w-3AglhEUmegXlWOtABZzJ1MrSiQ")
ACRCLOUD_ACCESS_KEY   = st.secrets.get("ACRCLOUD_ACCESS_KEY", "b1f7b901a4f15b99aba0efac395f6848")
ACRCLOUD_SECRET_KEY   = st.secrets.get("ACRCLOUD_SECRET_KEY", "tIVqMBQwOYGkCjkXAyY2wPiM5wxS5UrNwqMwMQjA")
ACRCLOUD_API_HOST     = "identify-eu-west-1.acrcloud.com"
ACRCLOUD_API_ENDPOINT = "/v1/identify"

# Supabase client (single global client reused everywhere)
supabase: "Client | None" = None
if SUPABASE_LIB_OK:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.error(f"Supabase init failed: {e}")
        supabase = None
else:
    st.error("supabase>=2.x is not installed; please add it to requirements.txt")

# --- NEW: restore a saved session so the client stays authed across reruns -------------
if supabase:
    if "access_token" in st.session_state and st.session_state.access_token:
        try:
            supabase.auth.set_session(
                access_token=st.session_state.access_token,
                refresh_token=st.session_state.get("refresh_token", "")
            )
        except Exception:
            pass

# --------------------------------------------------------------------------------------
# Page + diagnostics (keep visible while testing)
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="SneakPeak Video Scorer", page_icon="🎯", layout="wide")

st.title("SneakPeak Video Scorer")
if GOOGLE_VISION_API_KEY:
    st.info("Google Vision API is configured.")
else:
    st.warning("Google Vision API key is not set; visual analysis will be limited.")

if ACRCLOUD_ACCESS_KEY and ACRCLOUD_SECRET_KEY:
    st.info("ACRCloud API is configured.")
else:
    st.warning("ACRCloud keys are not set. Genre detection will fall back to 'Unknown'.")

if not MOVIEPY_OK:
    st.warning("MoviePy not importable. Audio analysis may be limited (install ffmpeg on the host).")

# --------------------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------------------
if "user" not in st.session_state:
    st.session_state.user = None

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
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

def generate_acrcloud_signature(timestamp: str) -> str:
    string_to_sign = f"POST\n{ACRCLOUD_API_ENDPOINT}\n{ACRCLOUD_ACCESS_KEY}\n{timestamp}"
    h = hmac.new(ACRCLOUD_SECRET_KEY.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha1)
    return base64.b64encode(h.digest()).decode("utf-8")

def extract_audio_features(video_path: str) -> dict:
    if not MOVIEPY_OK:
        return {"bpm": 0, "volume_level": 0.0, "genre": "Unknown", "energy_level": "Unknown"}

    temp_audio_path = None
    try:
        clip = VideoFileClip(video_path)
        dur = min(clip.duration or 0, 10)
        if not clip.audio:
            return {"bpm": 0, "volume_level": 0.0, "genre": "Unknown", "energy_level": "Unknown"}

        t = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio_path = t.name
        t.close()
        clip.audio.subclip(0, dur).write_audiofile(temp_audio_path, verbose=False, logger=None)

        bpm, vol = 0, 0.0
        if LIBROSA_OK:
            y, sr = librosa.load(temp_audio_path, sr=None)
            bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)
            vol = float(np.mean(rms) * 1000.0)

        energy = "Medium"
        if bpm > 120 and vol > 30: energy = "High"
        elif bpm < 90 or vol < 10: energy = "Low"

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
                    r = requests.post(f"https://{ACRCLOUD_API_HOST}{ACRCLOUD_API_ENDPOINT}",
                                      data=payload, files=files, timeout=15)
                if r.ok:
                    data = r.json()
                    if data.get("status", {}).get("code") == 0:
                        music = (data.get("metadata") or {}).get("music") or []
                        if music:
                            g = (music[0].get("genres") or [])
                            if g and g[0].get("name"):
                                genre = g[0]["name"]
            except Exception as e:
                st.warning(f"ACRCloud request failed; genre=Unknown. {e}")

        return {"bpm": int(bpm), "volume_level": vol, "genre": genre, "energy_level": energy}
    except Exception as e:
        st.error(f"Audio analysis error: {e}")
        return {"bpm": 0, "volume_level": 0.0, "genre": "Unknown", "energy_level": "Unknown"}
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try: os.unlink(temp_audio_path)
            except Exception: pass

def grab_middle_frame(video_path: str) -> str | None:
    if not OPENCV_OK:
        st.error("OpenCV not available; cannot extract frame.")
        return None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Cannot open video.")
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = max(0, total // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            st.error("Failed to read frame.")
            return None
        t = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(t.name, frame)
        t.close()
        return t.name
    except Exception as e:
        st.error(f"Frame extraction error: {e}")
        return None

def vision_visual(image_path: str) -> dict:
    if not GOOGLE_VISION_API_KEY:
        return {}
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        payload = {"requests":[{"image":{"content":b64},"features":[{"type":"IMAGE_PROPERTIES"},{"type":"LABEL_DETECTION"}]}]}
        r = requests.post(f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}",
                          json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        props = (data["responses"][0].get("imagePropertiesAnnotation") or {})
        colors = (props.get("dominantColors") or {}).get("colors") or []
        if colors:
            c = sorted(colors, key=lambda x: x.get("pixelFraction",0), reverse=True)[0]
            r_, g_, b_ = c["color"].get("red",0), c["color"].get("green",0), c["color"].get("blue",0)
            brightness = b_*0.299 + g_*0.587 + r_*0.114
            color_scheme = f"RGB({r_}, {g_}, {b_})"
        else:
            brightness = np.random.uniform(30, 90)
            color_scheme = "Unknown"

        labels = [l.get("description","").lower() for l in (data["responses"][0].get("labelAnnotations") or [])]
        visual_energy = "Medium"
        if any(x in labels for x in ["crowd","party","dance","celebration"]): visual_energy = "High"
        elif any(x in labels for x in ["quiet","calm","indoor","still"]): visual_energy = "Low"

        lighting = "Mixed Indoor"
        if "indoor" in labels and brightness < 100: lighting = "Dark/Club Lighting"
        elif "outdoor" in labels or brightness > 150: lighting = "Bright/Bar Lighting"

        return {"brightness_level": float(brightness), "lighting_type": lighting,
                "color_scheme": color_scheme, "visual_energy": visual_energy}
    except Exception as e:
        st.error(f"Vision visual error: {e}")
        return {}

def vision_crowd(image_path: str) -> dict:
    if not GOOGLE_VISION_API_KEY:
        return {}
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        payload = {"requests":[{"image":{"content":b64},"features":[{"type":"FACE_DETECTION"}]}]}
        r = requests.post(f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}",
                          json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        faces = data["responses"][0].get("faceAnnotations") or []
        n = len(faces)
        if n == 0: dens = "Empty"
        elif n <= 2: dens = "Sparse"
        elif n <= 5: dens = "Moderate"
        elif n <= 10: dens = "Busy"
        else: dens = "Packed"

        activity = "Still/Seated"
        if any(f.get("joyLikelihood") == "VERY_LIKELY" for f in faces):
            activity = "High Movement/Dancing"
        elif any(f.get("joyLikelihood") == "LIKELY" for f in faces):
            activity = "Social/Standing"

        score = float(n*1.5 + np.random.uniform(0,5))
        return {"crowd_density": dens, "activity_level": activity, "density_score": score}
    except Exception as e:
        st.error(f"Vision crowd error: {e}")
        return {}

def vision_mood(image_path: str) -> dict:
    if not GOOGLE_VISION_API_KEY:
        return {"dominant_mood":"Calm","confidence":0.5,"mood_breakdown":{"Calm":1.0},"overall_vibe":"Neutral"}
    try:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        payload = {"requests":[{"image":{"content":b64},"features":[{"type":"FACE_DETECTION"}]}]}
        r = requests.post(f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}",
                          json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        faces = data["responses"][0].get("faceAnnotations") or []
        if not faces:
            return {"dominant_mood":"Calm","confidence":0.5,"mood_breakdown":{"Calm":1.0},"overall_vibe":"Neutral"}

        weights = {"VERY_LIKELY":3,"LIKELY":2,"POSSIBLE":1}
        cnt = {"joy":0,"sorrow":0,"anger":0,"surprise":0}
        for f in faces:
            cnt["joy"]     += weights.get(f.get("joyLikelihood",""),0)
            cnt["sorrow"]  += weights.get(f.get("sorrowLikelihood",""),0)
            cnt["anger"]   += weights.get(f.get("angerLikelihood",""),0)
            cnt["surprise"]+= weights.get(f.get("surpriseLikelihood",""),0)

        key = max(cnt, key=cnt.get) if sum(cnt.values()) else "joy"
        mood_map = {"joy":"Happy","sorrow":"Calm","anger":"Energetic","surprise":"Excited"}
        dom = mood_map.get(key,"Social")
        total = sum(cnt.values()) or 1
        conf  = cnt.get(key,0)/total
        vibe = "Positive" if dom != "Calm" else "Mixed"
        breakdown = {mood_map.get(k,k): v/total for k,v in cnt.items()}
        return {"dominant_mood":dom,"confidence":float(conf),"mood_breakdown":breakdown,"overall_vibe":vibe}
    except Exception as e:
        st.error(f"Vision mood error: {e}")
        return {"dominant_mood":"Calm","confidence":0.5,"mood_breakdown":{"Calm":1.0},"overall_vibe":"Neutral"}

# -------- STORAGE & DB (uses authed Supabase client) ----------------------------------
def upload_video_to_storage(uploaded_file, user_id: str) -> tuple[str|None, str|None]:
    """Upload to 'videos' bucket using the user's authenticated client (RLS-safe)."""
    if not supabase or not st.session_state.user:
        st.error("Not authenticated; please log in.")
        return None, None
    try:
        ext = (uploaded_file.name.split(".")[-1] or "mp4").lower()
        filename = f"{user_id}/{uuid.uuid4().hex}.{ext}"  # user-scoped path
        data = uploaded_file.getvalue()
        content_type = uploaded_file.type or "application/octet-stream"
        resp = supabase.storage.from_("videos").upload(
            path=filename,
            file=data,
            file_options={"content-type": content_type, "x-upsert": "false"}
        )
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/videos/{filename}"
        return public_url, filename
    except Exception as e:
        st.error(f"Video upload failed: {e}")
        return None, None

def save_results_row(results: dict, uploaded_file=None):
    """Insert into video_results via the authed client. Do NOT send 'id' (bigint)."""
    if not supabase or not st.session_state.user:
        st.error("Not authenticated.")
        return False, None, None

    user_id = st.session_state.user.id
    video_url, video_path = (None, None)
    if uploaded_file is not None:
        video_url, video_path = upload_video_to_storage(uploaded_file, user_id)

    row = {
        "user_id": user_id,  # UUID
        "venue_name": str(results["venue_name"])[:100],
        "venue_type": str(results["venue_type"])[:50],
        "video_url": video_url,
        "video_filename": video_path,
        "video_stored": bool(video_url),
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
        ins = supabase.table("video_results").insert(row).execute()
        if getattr(ins, "data", None):
            return True, ins.data[0], video_url
        st.error("Insert returned no data.")
        return False, None, video_url
    except Exception as e:
        st.error(f"Database save failed: {e}")
        return False, None, video_url

def load_user_results(user_id: str):
    try:
        res = supabase.table("video_results").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        return res.data or []
    except Exception as e:
        st.error(f"Load results error: {e}")
        return []

# --------------------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------------------
def handle_login(email: str, password: str):
    if not supabase:
        st.error("Supabase client not initialized.")
        return
    try:
        resp = supabase.auth.sign_in_with_password({"email": email, "password": password})
        # Save user + tokens in session state and set client session (NEW)
        st.session_state.user = resp.user
        if resp.session:
            st.session_state.access_token = resp.session.access_token
            st.session_state.refresh_token = resp.session.refresh_token
            supabase.auth.set_session(
                access_token=st.session_state.access_token,
                refresh_token=st.session_state.refresh_token,
            )
        st.success("Logged in.")
        st.rerun()
    except Exception as e:
        st.error(f"Login failed: {e}")

def handle_signup(email: str, password: str):
    if not supabase:
        st.error("Supabase client not initialized.")
        return
    try:
        resp = supabase.auth.sign_up({"email": email, "password": password})
        st.info("Account created. Please confirm your email, then log in.")
    except Exception as e:
        st.error(f"Sign up failed: {e}")

def handle_logout():
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    for k in ("user", "access_token", "refresh_token"):
        st.session_state.pop(k, None)
    st.success("Logged out.")
    st.rerun()

# --------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------
st.sidebar.header("Account")
if st.session_state.user:
    sess = supabase.auth.get_session() if supabase else None
    st.sidebar.success(f"Logged in as {st.session_state.user.email}")
    st.sidebar.caption(f"Authed client: {bool(sess and getattr(sess, 'access_token', None))}")  # sanity check
    if st.sidebar.button("Log Out"):
        handle_logout()
else:
    with st.sidebar.form("auth_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Log In")
        signup_btn = st.form_submit_button("Sign Up")
        if login_btn:
            handle_login(email, password)
        elif signup_btn:
            handle_signup(email, password)

page = st.sidebar.radio("Go to", ["Upload & Analyze", "View My Videos"])

def display_results(row: dict):
    st.subheader(f"📊 Analysis Results for {row.get('venue_name','Unknown')}")
    c1, c2 = st.columns(2)
    c1.metric("Overall Vibe", row.get("overall_vibe","N/A"))
    c2.metric("Energy Score", f"{float(row.get('energy_score',0)):.2f}/100")
    with st.expander("🔊 Audio Environment"):
        st.write(f"BPM: {row.get('bpm','N/A')} | Volume: {float(row.get('volume_level',0)):.2f}% | Genre: {row.get('genre','Unknown')} | Energy: {row.get('energy_level','Unknown')}")
    with st.expander("💡 Visual Environment"):
        st.write(f"Brightness: {float(row.get('brightness_level',0)):.2f}/255 | Lighting: {row.get('lighting_type','Unknown')} | Color: {row.get('color_scheme','Unknown')} | Visual Energy: {row.get('visual_energy','Unknown')}")
    with st.expander("🕺 Crowd & Mood"):
        st.write(f"Crowd Density: {row.get('crowd_density','Unknown')} | Activity: {row.get('activity_level','Unknown')} | Dominant Mood: {row.get('dominant_mood','Unknown')} (conf {float(row.get('mood_confidence',0)):.2f})")
    if row.get("video_url"):
        st.video(row["video_url"])

if page == "Upload & Analyze":
    st.header("Upload a Video")
    if not st.session_state.user:
        st.warning("Please log in to upload and analyze a video.")
    else:
        up = st.file_uploader("Choose a video (max 200MB)", type=["mp4","mov","avi","m4v"])
        venue_name = st.text_input("Venue Name", "Demo Nightclub")
        venue_type = st.selectbox("Venue Type", ["Club","Bar","Restaurant","Lounge","Rooftop","Outdoors Space","Concert Hall","Event Space","Dive Bar","Speakeasy","Sports Bar","Brewery","Other"])

        if st.button("Start Analysis", disabled=up is None):
            if not up:
                st.error("Please choose a video file.")
            elif up.size/(1024*1024) > 200:
                st.error("File exceeds 200MB limit.")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as t:
                    t.write(up.getvalue())
                    video_path = t.name

                try:
                    prog = st.progress(5, text="Extracting frame…")
                    frame_path = grab_middle_frame(video_path)
                    if not frame_path:
                        raise RuntimeError("No frame extracted.")

                    prog.progress(25, text="Analyzing audio…")
                    audio = extract_audio_features(video_path)

                    prog.progress(45, text="Analyzing visual…")
                    visual = vision_visual(frame_path)

                    prog.progress(65, text="Analyzing crowd…")
                    crowd = vision_crowd(frame_path)

                    prog.progress(80, text="Analyzing mood…")
                    mood  = vision_mood(frame_path)

                    results = {
                        "venue_name": venue_name,
                        "venue_type": venue_type,
                        "audio_environment": audio,
                        "visual_environment": visual,
                        "crowd_density": crowd,
                        "mood_recognition": mood
                    }
                    results["energy_score"] = calculate_energy_score(results)

                    prog.progress(95, text="Saving results…")
                    ok, inserted_row, public_url = save_results_row(results, uploaded_file=up)
                    prog.progress(100, text="Done")

                    if ok and inserted_row:
                        if public_url: inserted_row["video_url"] = public_url
                        st.success("Analysis complete.")
                        display_results(inserted_row)
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
                finally:
                    try:
                        if os.path.exists(video_path): os.unlink(video_path)
                    except Exception:
                        pass
                    try:
                        if frame_path and os.path.exists(frame_path): os.unlink(frame_path)
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
                with st.expander(f"{r.get('venue_name','(Unknown)')} — {r.get('venue_type','')}  [{r.get('created_at','')[:10]}]"):
                    display_results(r)
