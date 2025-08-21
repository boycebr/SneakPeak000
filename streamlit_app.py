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

# MoviePy (we’ll also verify ffmpeg below)
try:
    from moviepy.editor import VideoFileClip
    from moviepy.config import change_settings as mvpy_change_settings
    HAS_MOVIEPY_IMPORT = True
except Exception:
    VideoFileClip = None
    mvpy_change_settings = None
    HAS_MOVIEPY_IMPORT = False

# Try to detect a bundled ffmpeg from imageio-ffmpeg and make MoviePy use it
HAS_FFMPEG = False
FFMPEG_BINARY = None
try:
    import imageio_ffmpeg
    FFMPEG_BINARY = imageio_ffmpeg.get_ffmpeg_exe()  # downloads/locates ffmpeg if needed
    if FFMPEG_BINARY:
        os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_BINARY
        if mvpy_change_settings:
            mvpy_change_settings({"FFMPEG_BINARY": FFMPEG_BINARY})
        HAS_FFMPEG = True
except Exception:
    FFMPEG_BINARY = None

HAS_MOVIEPY = bool(HAS_MOVIEPY_IMPORT and HAS_FFMPEG)

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
# CONFIG & INIT
# ============================================

# ---- Embedded defaults (as requested) ----
_EMBEDDED_SUPABASE_URL  = "https://tmmheslzkqiveylrnpal.supabase.co"
_EMBEDDED_SUPABASE_KEY  = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRtbWhlc2x6a3FpdmV5bHJucGFsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQzMzI5MjAsImV4cCI6MjA2OTkwODkyMH0.U-10R707xIs6rH-Vd5lBgh2INylFu6zn_EyoJYx_zpI"
_EMBEDDED_VISION_KEY    = "AIzaSyCcwH6w-3AglhEUmegXlWOtABZzJ1MrSiQ"
_EMBEDDED_ACR_ACCESS    = "b1f7b901a4f15b99aba0efac395f6848"
_EMBEDDED_ACR_SECRET    = "tIVqMBQwOYGkCjkXAyY2wPiM5wxS5UrNwqMwMQjA"
_EMBEDDED_ACR_HOST      = "identify-eu-west-1.acrcloud.com"
_EMBEDDED_ACR_ENDPOINT  = "/v1/identify"

# ---- Load actual config (Secrets override embedded) ----
SUPABASE_URL = st.secrets.get("SUPABASE_URL", _EMBEDDED_SUPABASE_URL)
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", _EMBEDDED_SUPABASE_KEY)

GOOGLE_VISION_API_KEY = st.secrets.get("GOOGLE_VISION_API_KEY", _EMBEDDED_VISION_KEY)
ACRCLOUD_ACCESS_KEY   = st.secrets.get("ACRCLOUD_ACCESS_KEY", _EMBEDDED_ACR_ACCESS)
ACRCLOUD_SECRET_KEY   = st.secrets.get("ACRCLOUD_SECRET_KEY", _EMBEDDED_ACR_SECRET)
ACRCLOUD_API_HOST     = st.secrets.get("ACRCLOUD_API_HOST", _EMBEDDED_ACR_HOST)
ACRCLOUD_API_ENDPOINT = st.secrets.get("ACRCLOUD_API_ENDPOINT", _EMBEDDED_ACR_ENDPOINT)

# ---- Initialize Supabase early (and fail loudly if broken) ----
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Failed to initialize Supabase client. Check SUPABASE_URL/KEY.\n\nDetails: {e}")
    st.stop()

# ---- Health messages (do NOT hide: you asked to see them) ----
if GOOGLE_VISION_API_KEY:
    st.info("Google Vision API is configured.")
else:
    st.warning("Google Vision API key is not set. Visual analysis will fail.")

if ACRCLOUD_ACCESS_KEY and ACRCLOUD_SECRET_KEY:
    st.info("ACRCloud API is configured.")
else:
    st.warning("ACRCloud keys are not set. Genre detection will fall back to 'Unknown'.")

if HAS_MOVIEPY:
    st.info(f"MoviePy/ffmpeg detected. Using ffmpeg at: {FFMPEG_BINARY}")
else:
    st.warning("MoviePy/ffmpeg not detected. Audio analysis will be limited.")

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
# CONSTANTS (tunable)
# ============================================
# Sampling
BASE_FPS_STANDARD = 1.0      # 1 fps (default)
BASE_FPS_SLOW     = 0.5      # 0.5 fps (1 frame every 2s)
MAX_FRAMES        = 60       # hard cap
MIN_COVERAGE_SEC  = 4.0      # warmup before early-stop can trigger

# Motion adaptation (very static)
STATIC_THRESH       = 0.02   # rolling median below this => static
WAKE_THRESH         = 0.05   # rolling median above this => wake / return to faster rate
STATIC_CONSEC_NEED  = 5      # seconds below threshold to engage slow mode
WAKE_CONSEC_NEED    = 3      # seconds above wake threshold to release slow mode
ROLL_WINDOW         = 5      # seconds window for rolling median

# Unique convergence (no new faces for 2s)
NO_NEW_SEC          = 2.0    # required idle secs before early-stop
CONFIRM_FRAMES_NEED = 2      # frames needed to confirm a new track
MAX_MISSES          = 5      # frames before deleting a track
IOU_THRESH          = 0.3    # gating
COLOR_WEIGHT        = 0.3    # in matching cost (1 - IoU)*0.7 + color*0.3

# Vision cadence
VISION_MIN_GAP_SEC  = 4.0    # minimum gap between Vision mood calls

# ============================================
# HELPERS: Audio (unchanged behavior)
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

# ============================================
# HELPERS: Vision API (unchanged endpoints)
# ============================================
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

# ============================================
# LOCAL DETECTION + TRACKING (people count)
# ============================================

# HOG person detector (local, no API)
_HOG = cv2.HOGDescriptor()
_HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# For optional face-only local fallback (not used for count; Vision is used for mood)
_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_persons(frame_bgr):
    # Downscale for speed
    scale = 0.75
    h, w = frame_bgr.shape[:2]
    resized = cv2.resize(frame_bgr, (int(w*scale), int(h*scale)))
    rects, weights = _HOG.detectMultiScale(resized, winStride=(8,8), padding=(8,8), scale=1.05)
    boxes = []
    for (x,y,wc,hc) in rects:
        x1 = int(x/scale); y1 = int(y/scale); x2 = int((x+wc)/scale); y2 = int((y+hc)/scale)
        boxes.append([x1,y1,x2,y2])
    return boxes

def detect_faces_local(frame_gray):
    faces = _FACE_CASCADE.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    boxes = []
    for (x,y,w,h) in faces:
        boxes.append([x, y, x+w, y+h])
    return boxes

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0: return 0.0
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    return inter / float(area_a + area_b - inter + 1e-6)

def crop_and_hist(frame_bgr, box, bins=(16,16,16)):
    x1,y1,x2,y2 = [int(v) for v in box]
    x1 = max(0,x1); y1=max(0,y1); x2=min(frame_bgr.shape[1]-1,x2); y2=min(frame_bgr.shape[0]-1,y2)
    if x2<=x1 or y2<=y1:
        return np.zeros((sum(bins),), dtype=np.float32)
    crop = frame_bgr[y1:y2, x1:x2]
    hist = []
    for ch in range(3):
        h = cv2.calcHist([crop],[ch],None,[bins[ch]],[0,256])
        h = cv2.normalize(h, None).flatten()
        hist.append(h)
    return np.concatenate(hist).astype(np.float32)

def hist_distance(h1, h2):
    # Bhattacharyya distance via OpenCV compareHist
    if h1 is None or h2 is None: return 1.0
    # split back to 3 channels (not necessary; compare as 1D)
    d = cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
    return float(np.clip(d, 0.0, 1.0))

class Track:
    __slots__ = ("tid","bbox","hist","hits_consec","total_hits","misses","confirmed","first_confirm_s","last_seen_s","centroids")
    def __init__(self, tid, bbox, hist, now_s):
        self.tid = tid
        self.bbox = bbox
        self.hist = hist
        self.hits_consec = 1
        self.total_hits = 1
        self.misses = 0
        self.confirmed = False
        self.first_confirm_s = None
        self.last_seen_s = now_s
        self.centroids = [self._centroid(bbox)]
    def _centroid(self, b):
        x1,y1,x2,y2 = b
        return ((x1+x2)/2.0, (y1+y2)/2.0)
    def update(self, bbox, hist, now_s):
        self.bbox = bbox
        self.hist = hist
        self.hits_consec += 1
        self.total_hits += 1
        self.misses = 0
        self.last_seen_s = now_s
        self.centroids.append(self._centroid(bbox))
        if not self.confirmed and self.hits_consec >= CONFIRM_FRAMES_NEED:
            self.confirmed = True
            self.first_confirm_s = now_s
    def mark_missed(self):
        self.misses += 1
        self.hits_consec = 0

class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.unique_confirmed = 0
        self.new_confirmed_this_frame = 0

    def _match(self, frame_bgr, dets, now_s):
        # Greedy matching by lowest cost
        used = set()
        self.new_confirmed_this_frame = 0
        for tr in self.tracks:
            best_idx = -1
            best_cost = 1e9
            for i, d in enumerate(dets):
                if i in used: continue
                iou_v = iou(tr.bbox, d)
                if iou_v < IOU_THRESH: 
                    continue
                hist_d = hist_distance(tr.hist, crop_and_hist(frame_bgr, d))
                cost = 0.7*(1.0 - iou_v) + COLOR_WEIGHT*hist_d
                if cost < best_cost:
                    best_cost = cost
                    best_idx = i
            if best_idx >= 0:
                # match found
                used.add(best_idx)
                tr.update(dets[best_idx], crop_and_hist(frame_bgr, dets[best_idx]), now_s)
                if tr.confirmed and tr.first_confirm_s == now_s:
                    self.unique_confirmed += 1
                    self.new_confirmed_this_frame += 1
            else:
                tr.mark_missed()

        # Create tracks for unmatched detections
        for i, d in enumerate(dets):
            if i in used: continue
            t = Track(self.next_id, d, crop_and_hist(frame_bgr, d), now_s)
            self.next_id += 1
            self.tracks.append(t)
            if t.confirmed:
                self.unique_confirmed += 1
                self.new_confirmed_this_frame += 1

        # Cleanup
        self.tracks = [t for t in self.tracks if t.misses <= MAX_MISSES]

    def step(self, frame_bgr, person_boxes, now_s):
        self._match(frame_bgr, person_boxes, now_s)

    def current_confirmed_in_frame(self):
        # count confirmed tracks that were seen "now" (misses == 0)
        return sum(1 for t in self.tracks if t.confirmed and t.misses == 0)

    def movement_index(self):
        mov = []
        for t in self.tracks:
            if len(t.centroids) >= 2:
                dists = [np.linalg.norm(np.array(t.centroids[i])-np.array(t.centroids[i-1])) for i in range(1,len(t.centroids))]
                if dists:
                    mov.append(np.mean(dists))
        return float(np.mean(mov)) if mov else 0.0

    def avg_dwell_time(self):
        # approximate dwell = number of frames seen for confirmed tracks (hits), scaled by sampling interval
        confirmed = [t for t in self.tracks if t.confirmed]
        if not confirmed: return 0.0
        frames_seen = [t.total_hits for t in confirmed]
        return float(np.mean(frames_seen))

# ============================================
# MULTI-FRAME SAMPLER with adaptations
# ============================================
def mean_abs_diff(prev_gray, curr_gray):
    if prev_gray is None or curr_gray is None: return 1.0
    # resize to stabilize computation
    target_w = 320
    scale = target_w / max(1, curr_gray.shape[1])
    curr_small = cv2.resize(curr_gray, (int(curr_gray.shape[1]*scale), int(curr_gray.shape[0]*scale)))
    prev_small = cv2.resize(prev_gray, (curr_small.shape[1], curr_small.shape[0]))
    diff = cv2.absdiff(curr_small, prev_small)
    return float(np.mean(diff) / 255.0)

def process_video_multiframe(video_path, base_fps_choice="Standard"):
    """
    Sampling:
      - Base cadence: 1 fps (Standard) or 0.5 fps (Slow)
      - Auto-slow if very static (rolling median < 0.02 for 5s)
      - Early-stop if no NEW confirmed tracks for >= 2s (guarded to >=2 frames)
      - Hard cap: 60 frames
    Returns:
      crowd_time_series, unique_people, telemetry, representative_frame_path (for Vision)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file for multi-frame processing.")
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = total_frames / max(1e-6, fps_src)

    if base_fps_choice == "Slow":
        base_fps = BASE_FPS_SLOW
    else:
        base_fps = BASE_FPS_STANDARD

    interval_s = 1.0 / base_fps
    now_s = 0.0
    processed = 0

    # Adaptation state
    motion_scores = []
    motion_rolling = []
    static_below_count = 0
    wake_above_count = 0
    auto_slow = False
    auto_slow_engaged_at = None
    auto_slow_released_at = None

    # Convergence state
    tracker = Tracker()
    last_new_confirm_time = None
    no_new_frames_consec = 0
    required_no_new_frames = max(2, int(np.ceil(NO_NEW_SEC * base_fps)))

    # Outputs
    times = []
    people_in_frame_series = []
    representative_frame_path = None

    prev_gray = None
    last_vision_call_time = -1e9  # so first call can happen ASAP after warmup

    while processed < MAX_FRAMES and now_s <= duration_s + 1e-6:
        # Seek & read
        target_frame_idx = int(now_s * fps_src)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        # Convert for motion & faces
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.GaussianBlur(frame_gray, (5,5), 0, dst=frame_gray)

        # Motion score
        m = mean_abs_diff(prev_gray, frame_gray)
        motion_scores.append(m)
        prev_gray = frame_gray

        # Update rolling median over last ROLL_WINDOW seconds (≈ last N frames at current cadence)
        window_len = int(max(1, ROLL_WINDOW * (1.0/interval_s)))
        motion_rolling = motion_scores[-window_len:]
        roll_med = float(np.median(motion_rolling))

        # Person detection (local, HOG)
        person_boxes = detect_persons(frame)

        # Tracking
        tracker.step(frame, person_boxes, now_s)
        people_now = tracker.current_confirmed_in_frame()
        times.append(now_s)
        people_in_frame_series.append(people_now)
        processed += 1

        # Representative frame for Vision (middle-ish if not set)
        if representative_frame_path is None and 0.3*duration_s <= now_s <= 0.7*duration_s:
            tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(tmp_img.name, frame)
            representative_frame_path = tmp_img.name
            tmp_img.close()

        # Update "no new" logic
        if tracker.new_confirmed_this_frame > 0:
            last_new_confirm_time = now_s
            no_new_frames_consec = 0
        else:
            no_new_frames_consec += 1

        # Adaptive: very static => downshift to 0.5 fps (after warmup 5s)
        if now_s >= 5.0:
            if roll_med < STATIC_THRESH:
                static_below_count += 1
                wake_above_count = 0
            elif roll_med > WAKE_THRESH:
                wake_above_count += 1
                static_below_count = 0

            if (not auto_slow) and static_below_count >= STATIC_CONSEC_NEED:
                auto_slow = True
                auto_slow_engaged_at = now_s
                interval_s = 1.0 / BASE_FPS_SLOW
                # Recompute guard for NO_NEW based on slower cadence
                required_no_new_frames = max(2, int(np.ceil(NO_NEW_SEC * (1.0/interval_s))))
            elif auto_slow and wake_above_count >= WAKE_CONSEC_NEED:
                auto_slow = False
                auto_slow_released_at = now_s
                interval_s = 1.0 / base_fps
                required_no_new_frames = max(2, int(np.ceil(NO_NEW_SEC * (1.0/interval_s))))

        # Early-stop: after warmup coverage, if NO new faces for >= 2s (guarded to >=2 frames)
        if now_s >= MIN_COVERAGE_SEC:
            if (last_new_confirm_time is None and no_new_frames_consec >= required_no_new_frames) or \
               (last_new_confirm_time is not None and (now_s - last_new_confirm_time) >= NO_NEW_SEC and no_new_frames_consec >= required_no_new_frames):
                break

        # Vision mood cadence (every ~4s)
        # (We keep current per-video Vision calls outside this loop to minimize cost; mood summary uses representative frame.)
        # If you later want multi-sample mood, gate here with time gap and store temp frames.

        # Advance time
        now_s += interval_s

    cap.release()

    telemetry = {
        "processed_frames": processed,
        "processed_seconds_est": times[-1] if times else 0.0,
        "auto_slow_engaged_at_s": auto_slow_engaged_at,
        "auto_slow_released_at_s": auto_slow_released_at,
        "early_stop_reason": "no_new_faces_2s" if (now_s >= MIN_COVERAGE_SEC and (last_new_confirm_time is None or (now_s - (last_new_confirm_time or 0)) >= NO_NEW_SEC)) else None,
        "base_mode": base_fps_choice,
    }

    return {
        "times": times,
        "people_in_frame": people_in_frame_series,
        "unique_people": tracker.unique_confirmed,
        "movement_index": tracker.movement_index(),
        "avg_dwell_frames": tracker.avg_dwell_time(),  # presented in frames (we'll convert to sec)
        "interval_s": interval_s,  # final sampling interval
        "telemetry": telemetry,
        "representative_frame_path": representative_frame_path
    }

# ============================================
# STORAGE & DB
# ============================================
def save_user_rating(venue_id, user_id, rating, venue_name, venue_type):
    try:
        data = {
            "venue_id": str(venue_id),
            "user_id": str(user_id)[:64],
            "rating": int(rating),
            "venue_name": str(venue_name)[:100],
            "venue_type": str(venue_type)[:50],
            "rated_at": datetime.utcnow().isoformat()
        }
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        r = requests.post(f"{SUPABASE_URL}/rest/v1/user_ratings", headers=headers, json=data, timeout=30)
        return r.status_code == 201
    except Exception as e:
        st.error(f"Network/DB error saving rating: {e}")
        return False

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
def display_people_analytics(pa):
    st.subheader("👥 People Analytics (multi-frame)")
    unique_people = pa["unique_people"]
    people_series = pa["people_in_frame"]
    times = pa["times"]
    interval_s = pa["interval_s"]
    dwell_frames_avg = pa["avg_dwell_frames"]
    movement_idx = pa["movement_index"]
    tele = pa["telemetry"]

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Unique People (confirmed)", f"{unique_people}")
    if people_series:
        c2.metric("People in Frame (avg)", f"{np.mean(people_series):.1f}")
        c3.metric("People in Frame (max)", f"{np.max(people_series)}")
    else:
        c2.metric("People in Frame (avg)", "0")
        c3.metric("People in Frame (max)", "0")

    # Dwell & movement
    c4, c5 = st.columns(2)
    c4.metric("Avg Dwell (frames @ sample rate)", f"{dwell_frames_avg:.1f}")
    c5.metric("Movement Index (px/frame)", f"{movement_idx:.1f}")

    # Timeline chart
    if people_series and times:
        fig, ax = plt.subplots()
        ax.plot(times, people_series, marker="o")
        ax.set_title("People in Frame Over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # Adaptation summary
    chips = []
    if tele.get("base_mode"): chips.append(f"Base: {tele['base_mode']}")
    if tele.get("auto_slow_engaged_at_s") is not None:
        chips.append(f"Auto-slow engaged @ {tele['auto_slow_engaged_at_s']:.1f}s")
    if tele.get("auto_slow_released_at_s") is not None:
        chips.append(f"Auto-slow released @ {tele['auto_slow_released_at_s']:.1f}s")
    if tele.get("early_stop_reason"):
        chips.append(f"Early-stop ({tele['early_stop_reason']}) @ {tele['processed_seconds_est']:.1f}s")
    st.info(" | ".join(chips) if chips else "No adaptive events triggered.")

def display_results(results, people_analytics=None):
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

    if people_analytics:
        display_people_analytics(people_analytics)

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
                            ok = save_user_rating(v["id"], st.session_state.user.id, rating, v.get("venue_name",""), v.get("venue_type",""))
                            st.success("Your rating has been submitted!" if ok else "There was an error submitting your rating.")
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

    # Show consolidated health message about MoviePy/ffmpeg
    if HAS_MOVIEPY:
        st.info(f"MoviePy/ffmpeg is ready. Binary: {FFMPEG_BINARY}")
    else:
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

            # Processing rate (your spec)
            rate_choice = st.selectbox("Processing rate", ["Standard (1 fps)", "Slow (0.5 fps)"], index=0)
            base_choice = "Standard" if rate_choice.startswith("Standard") else "Slow"

            # GPS intentionally skipped for now (schema fields kept as None)
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
                    # ---- Multi-frame analytics (people counting/tracking) ----
                    progress_bar.progress(15, text="Sampling frames & counting people...")
                    pa = process_video_multiframe(temp_video_path, base_fps_choice=base_choice)

                    # Use representative frame for Vision analytics (to control cost)
                    rep_frame = pa["representative_frame_path"]
                    if not rep_frame:
                        # Fallback: grab mid-frame if none selected
                        cap = cv2.VideoCapture(temp_video_path)
                        mid_idx = int((cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)//2)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
                        ok, frm = cap.read()
                        cap.release()
                        if ok:
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                            cv2.imwrite(tmp.name, frm)
                            rep_frame = tmp.name
                            tmp.close()

                    progress_bar.progress(45, text="Analyzing audio (if available)...")
                    audio_features = extract_audio_features(temp_video_path)

                    progress_bar.progress(65, text="Analyzing visual environment...")
                    visual_features = analyze_visual_features_with_vision_api(rep_frame) if rep_frame else {}

                    progress_bar.progress(80, text="Analyzing crowd density and activity...")
                    crowd_features = analyze_crowd_features_with_vision_api(rep_frame) if rep_frame else {"crowd_density":"Unknown","activity_level":"Unknown","density_score":0.0}

                    progress_bar.progress(90, text="Detecting dominant mood...")
                    mood_features = analyze_mood_recognition_with_vision_api(rep_frame) if rep_frame else {"dominant_mood":"Unknown","confidence":0.0,"mood_breakdown":{"Unknown":1.0},"overall_vibe":"Unknown"}

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
                            display_results(saved, people_analytics=pa)

                except Exception as e:
                    st.error(f"Unexpected error during analysis: {e}")
                finally:
                    if temp_video_path and os.path.exists(temp_video_path): os.unlink(temp_video_path)
                    if temp_image_path and os.path.exists(temp_image_path): os.unlink(temp_image_path)
                    if pa and pa.get("representative_frame_path") and os.path.exists(pa["representative_frame_path"]):
                        try: os.unlink(pa["representative_frame_path"])
                        except Exception: pass
                    try: progress_bar.empty()
                    except Exception: pass

    elif page == "View My Videos":
        display_all_results_page()

if __name__ == "__main__":
    main()
