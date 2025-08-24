# streamlit_app.py
# SneakPeak Video Scorer — 480p proxy, basic AV heuristics, optional demographics (InsightFace / AWS Rekognition)
# Security: NO hard-coded credentials. Load secrets from environment or st.secrets.

import os, io, time, uuid, hashlib, tempfile, subprocess
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image

# Point MoviePy (and our fallback) at the bundled ffmpeg
import imageio_ffmpeg
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

import streamlit as st
st.set_page_config(page_title="SneakPeak Video Scorer", page_icon="🎥", layout="wide")

# ----------------------------
# Init message queue (so we don't call UI before page_config)
# ----------------------------
INIT_MESSAGES: list[tuple[str, str]] = []   # ("success"|"warning"|"error", "message")
def _init_msg(level: str, text: str):
    INIT_MESSAGES.append((level, text))

# ----------------------------
# Optional dependencies
# ----------------------------
MOVIEPY_AVAILABLE = True
try:
    from moviepy.editor import VideoFileClip
except Exception as e:
    MOVIEPY_AVAILABLE = False
    _init_msg("warning", f"MoviePy not available; will use raw ffmpeg. ({e})")

# Is an ffmpeg binary available?
try:
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    VIDEO_PROCESSING_OK = bool(FFMPEG_EXE and os.path.isfile(FFMPEG_EXE))
    if not VIDEO_PROCESSING_OK:
        _init_msg("error", "ffmpeg binary not found on disk.")
except Exception as e:
    FFMPEG_EXE = ""
    VIDEO_PROCESSING_OK = False
    _init_msg("error", f"ffmpeg not available: {e}")

INSIGHTFACE_OK = True
try:
    import insightface
    from insightface.app import FaceAnalysis
except Exception:
    INSIGHTFACE_OK = False
    _init_msg("warning", "InsightFace not available (gender/age via InsightFace disabled).")

AWS_REKOGNITION_OK = True
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except Exception:
    AWS_REKOGNITION_OK = False
    _init_msg("warning", "boto3 not available (AWS Rekognition disabled).")

# ----------------------------
# Secrets & clients
# ----------------------------
def _get_secret(name: str, default: str = "") -> str:
    try:
        return os.getenv(name) or (st.secrets.get(name) if hasattr(st, "secrets") else "") or default
    except Exception:
        return os.getenv(name) or default

AWS_ACCESS_KEY_ID     = _get_secret("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = _get_secret("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION    = _get_secret("AWS_DEFAULT_REGION", "us-east-1")

SUPABASE_URL      = _get_secret("SUPABASE_URL").strip()
SUPABASE_ANON_KEY = _get_secret("SUPABASE_ANON_KEY").strip()

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    _init_msg("warning", "Supabase URL/Anon Key not found. Set SUPABASE_URL and SUPABASE_ANON_KEY.")

from supabase import create_client, Client
supabase: Optional[Client] = None
try:
    if SUPABASE_URL and SUPABASE_ANON_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
except Exception as e:
    _init_msg("error", f"Could not initialize Supabase client: {e}")

@st.cache_resource(show_spinner=False)
def get_rekognition_client():
    if not (AWS_REKOGNITION_OK and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY):
        return None
    try:
        return boto3.client(
            "rekognition",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_DEFAULT_REGION,
        )
    except Exception as e:
        _init_msg("warning", f"AWS Rekognition setup failed: {e}")
        return None

@st.cache_resource(show_spinner=False)
def get_insightface_app():
    if not INSIGHTFACE_OK:
        return None
    try:
        app = FaceAnalysis(
            providers=["CPUExecutionProvider"],  # switch to GPU provider if configured
            allowed_modules=["detection", "genderage"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app
    except Exception as e:
        _init_msg("warning", f"InsightFace setup failed: {e}")
        return None

rekog_client = get_rekognition_client()
insightface_app = get_insightface_app()

# ----------------------------
# Session state & auth
# ----------------------------
def ensure_session_state():
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("event_log", [])
    st.session_state.setdefault("metrics", {"success":0, "fail":0, "analysis_ms":[], "zero_conf":0})
ensure_session_state()

def sign_in(email: str, password: str):
    if not supabase:
        st.error("Supabase not configured.")
        return
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if res.user:
            st.session_state["user"] = res.user
            st.success(f"Signed in as {res.user.email}")
        else:
            st.error("Sign‑in failed (no user in response).")
    except Exception as e:
        st.error(f"Sign‑in error: {e}")

def sign_out():
    if not supabase:
        return
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    st.session_state["user"] = None

# ----------------------------
# Helpers
# ----------------------------
import colorsys

BRIGHTNESS_BINS = [
    (0, 50, "Very Dark"),
    (51, 100, "Dim"),
    (101, 150, "Moderate"),
    (151, 200, "Bright"),
    (201, 255, "Very Bright"),
]

def brightness_label_and_pct(brightness_level: float) -> Tuple[str, float]:
    try:
        v = max(0.0, min(255.0, float(brightness_level)))
    except Exception:
        v = 0.0
    pct = round(v / 255.0 * 100.0, 2)
    for lo, hi, name in BRIGHTNESS_BINS:
        if lo <= v <= hi:
            return name, pct
    return "Very Dark", pct

RCW_12 = [
    ("Red",         345,  15),
    ("Red-Orange",   15,  45),
    ("Orange",       45,  75),
    ("Yellow-Orange",75, 105),
    ("Yellow",      105, 135),
    ("Yellow-Green",135, 165),
    ("Green",       165, 195),
    ("Blue-Green",  195, 225),
    ("Blue",        225, 255),
    ("Blue-Violet", 255, 285),
    ("Violet",      285, 315),
    ("Red-Violet",  315, 345),
]

def rcw_color_name_from_rgb(r: int, g: int, b: int, gray_sat_threshold: float = 0.20) -> str:
    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    if s < gray_sat_threshold:
        return "Gray"
    deg = (h * 360.0) % 360.0
    for name, lo, hi in RCW_12:
        if lo <= hi and lo <= deg < hi:
            return name
        if lo > hi:
            if deg >= lo or deg < hi:
                return name
    return "Gray"

def confidence_bucket(conf: float) -> str:
    try:
        v = float(conf)
    except Exception:
        v = 0.0
    if v <= 0.0: return "No Confidence"
    if v <= 0.33: return "Low"
    if v <= 0.66: return "Medium"
    return "High"

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256(); h.update(data); return h.hexdigest()

def log_event(stage: str, correlation_id: str, **fields):
    entry = {"ts": datetime.utcnow().isoformat()+"Z", "stage": stage, "cid": correlation_id, **fields}
    st.session_state.setdefault("event_log", []).append(entry)
    m = st.session_state.setdefault("metrics", {"success":0, "fail":0, "analysis_ms":[], "zero_conf":0})
    if stage == "analysis_done" and fields.get("ok"):
        m["success"] += 1
        if "elapsed_ms" in fields: m["analysis_ms"].append(fields["elapsed_ms"])
        if fields.get("mood_conf_zero"): m["zero_conf"] += 1
    if stage == "analysis_failed":
        m["fail"] += 1

# ----------------------------
# Video ops (MoviePy if available, else raw ffmpeg)
# ----------------------------
def make_480p_proxy(file_bytes: bytes, max_seconds: int = 60) -> tuple[str, float]:
    # Save source to a temp file
    src_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    src_tmp.write(file_bytes); src_tmp.flush(); src_tmp.close()

    if MOVIEPY_AVAILABLE:
        clip = VideoFileClip(src_tmp.name)
        duration = float(clip.duration or 0.0)
        end = min(duration, float(max_seconds))
        sub = clip.subclip(0, end).resize(height=480)
        out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_480p.mp4"); out_tmp.close()
        sub.write_videofile(out_tmp.name, codec="libx264", audio_codec="aac",
                            threads=2, verbose=False, logger=None)
        clip.close(); sub.close()
        return out_tmp.name, end

    if not VIDEO_PROCESSING_OK:
        raise RuntimeError("No video processing backend available (MoviePy and ffmpeg both unavailable).")

    end = float(max_seconds)
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_480p.mp4"); out_tmp.close()
    cmd = (
        f'"{FFMPEG_EXE}" -y -hide_banner -loglevel error '
        f'-i "{src_tmp.name}" -t {end} -vf scale=-2:480 '
        f'-c:v libx264 -preset veryfast -crf 23 -c:a aac -movflags +faststart "{out_tmp.name}"'
    )
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0 or not os.path.exists(out_tmp.name):
        raise RuntimeError("ffmpeg failed to create 480p proxy.")
    return out_tmp.name, end

@st.cache_data(show_spinner=False)
def grab_middle_frame(local_video_path: str) -> str:
    try:
        if not MOVIEPY_AVAILABLE:
            raise RuntimeError("MoviePy not available")
        clip = VideoFileClip(local_video_path)
        t = (clip.duration or 0) / 2.0
        frame = clip.get_frame(t); clip.close()
        img = Image.fromarray(frame[:, :, :3])
    except Exception:
        img = Image.new("RGB", (640, 360), color=(64,64,64))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png"); img.save(tmp.name)
    return tmp.name

def extract_audio_features(local_video_path: str):
    # Placeholder — wire up librosa/ffmpeg if desired
    return {"bpm": 0, "volume_level": 0.0, "genre": "Unknown", "energy_level": "Unknown"}

def vision_visual(frame_path: str):
    try:
        img = Image.open(frame_path).convert("RGB")
        arr = np.array(img)
        hsv = np.array([colorsys.rgb_to_hsv(*(px/255.0)) for px in arr.reshape(-1,3)])
        v = hsv[:,2].mean()
        brightness = float(max(0.0, min(255.0, v*255.0)))
        mean_rgb = arr.reshape(-1,3).mean(axis=0).astype(int)
        hexcode = f"#{mean_rgb[0]:02x}{mean_rgb[1]:02x}{mean_rgb[2]:02x}"
        sat = hsv[:,1].mean()
        visual_energy = "High" if sat > 0.5 else ("Medium" if sat > 0.25 else "Low")
        return {
            "brightness_level": brightness,
            "lighting_type": "Unknown",
            "color_scheme": hexcode,
            "dominant_rgb": {"r": int(mean_rgb[0]), "g": int(mean_rgb[1]), "b": int(mean_rgb[2])},
            "visual_energy": visual_energy,
        }
    except Exception:
        return {"brightness_level": 0.0, "lighting_type":"Unknown", "color_scheme":"#000000", "visual_energy":"Unknown"}

def vision_crowd(_frame_path: str):
    return {"crowd_density":"Unknown", "activity_level":"Unknown", "density_score": 0.0}

def vision_mood(_frame_path: str):
    return {"dominant_mood":"Unknown", "confidence": 0.0, "overall_vibe":"Unknown", "faces_detected": 0}

def calculate_energy_score(results: dict) -> float:
    a = results.get("audio_environment", {})
    v = results.get("visual_environment", {})
    base = 50.0
    base += min(50.0, float(a.get("volume_level", 0.0))*10.0)
    base += (float(v.get("brightness_level", 0.0))/255.0)*20.0
    return round(max(0.0, min(100.0, base)), 2)

# ----------------------------
# Demographics (InsightFace / Rekognition / Mock)
# ----------------------------
def extract_frames_for_analysis(video_path: str) -> list:
    try:
        if not MOVIEPY_AVAILABLE:
            raise RuntimeError("MoviePy not available")
        clip = VideoFileClip(video_path); duration = clip.duration or 0
        sample_times = [duration/2.0] if duration < 1.0 else [duration*0.25, duration*0.50, duration*0.75]
        frames = []
        for t in sample_times:
            try:
                fr = clip.get_frame(t)
                frames.append(fr[:, :, :3])
            except Exception as e:
                st.warning(f"Could not extract frame at {t:.1f}s: {e}")
        clip.close()
        return frames
    except Exception as e:
        st.warning(f"Frame extraction failed: {e}")
        return []

def process_insightface_results(faces: list) -> Dict[str, Any]:
    if not faces:
        raise Exception("No faces to process")
    total_faces = len(faces)
    male_count=female_count=0
    ages=[]; male_ages=[]; female_ages=[]
    det_confs=[]
    for f in faces:
        det_confs.append(getattr(f, "det_score", 0.5))
        gender = getattr(f, "sex", None)   # 0=female, 1=male
        age = getattr(f, "age", None)
        if gender == 1: male_count += 1
        elif gender == 0: female_count += 1
        if age is not None:
            ages.append(age)
            (male_ages if gender==1 else female_ages if gender==0 else ages).append(age)
    male_pct = (male_count/total_faces*100) if total_faces else 0
    female_pct = (female_count/total_faces*100) if total_faces else 0
    avg_conf = float(np.mean(det_confs)) if det_confs else 0.5
    return format_demographic_results(total_faces, male_count, female_count, male_pct, female_pct,
                                      ages, male_ages, female_ages, avg_conf)

def process_aws_rekognition_results(faces: list) -> Dict[str, Any]:
    if not faces:
        raise Exception("No faces to process")
    total_faces = len(faces)
    male_count=female_count=0
    ages=[]; male_ages=[]; female_ages=[]; gender_confs=[]
    for face in faces:
        g = face.get("Gender", {})
        val = g.get("Value"); conf = float(g.get("Confidence", 0.0))/100.0
        gender_confs.append(conf)
        if val == "Male": male_count += 1
        elif val == "Female": female_count += 1
        ar = face.get("AgeRange", {})
        if "Low" in ar and "High" in ar:
            est = (ar["Low"]+ar["High"])/2.0
            ages.append(est)
            (male_ages if val=="Male" else female_ages if val=="Female" else ages).append(est)
    male_pct = (male_count/total_faces*100) if total_faces else 0
    female_pct = (female_count/total_faces*100) if total_faces else 0
    avg_conf = float(np.mean(gender_confs)) if gender_confs else 0.5
    return format_demographic_results(total_faces, male_count, female_count, male_pct, female_pct,
                                      ages, male_ages, female_ages, avg_conf)

def format_demographic_results(total_faces, male_count, female_count, male_pct, female_pct,
                               ages, male_ages, female_ages, confidence):
    def generate_age_summary(xs: list) -> str:
        if not xs: return "No age data"
        avg = float(np.mean(xs)); rng = (max(xs)-min(xs)) if len(xs) > 1 else 0
        if avg < 25: base = "Mostly early 20s"
        elif avg < 30: base = "Mostly mid‑to‑late 20s"
        elif avg < 35: base = "Mostly late 20s to early 30s"
        elif avg < 40: base = "Mostly 30s"
        else: base = "Mixed ages, 30s and up"
        if rng > 15: base += " (wide age range)"
        return base

    def gender_conf(count: int, base: float) -> float:
        if count == 0: return 0.0
        return round(base * (0.7 + 0.3 * min(1.0, count/5.0)), 2)

    def age_conf(xs: list, base: float) -> float:
        if not xs: return 0.0
        return round(base * 0.8 * (0.7 + 0.3 * min(1.0, len(xs)/5.0)), 2)

    male_avg = float(np.mean(male_ages)) if male_ages else None
    female_avg = float(np.mean(female_ages)) if female_ages else None
    male_range = f"{min(male_ages):.0f}-{max(male_ages):.0f}" if male_ages else None
    female_range = f"{min(female_ages):.0f}-{max(female_ages):.0f}" if female_ages else None

    return {
        "total_faces": total_faces,
        "male_count": male_count,
        "female_count": female_count,
        "male_percentage": round(male_pct, 1),
        "female_percentage": round(female_pct, 1),
        "confidence": round(float(confidence or 0.0), 2),
        "age_analysis": generate_age_summary(ages) if ages else "No age data",
        "male_age_analysis": generate_age_summary(male_ages) if male_ages else "No male faces",
        "female_age_analysis": generate_age_summary(female_ages) if female_ages else "No female faces",
        "age_male_average": round(male_avg, 1) if male_avg is not None else None,
        "age_female_average": round(female_avg, 1) if female_avg is not None else None,
        "age_male_range": male_range,
        "age_female_range": female_range,
        "gender_male_confidence": gender_conf(male_count, confidence or 0.0),
        "gender_female_confidence": gender_conf(female_count, confidence or 0.0),
        "age_male_confidence": age_conf(male_ages, confidence or 0.0),
        "age_female_confidence": age_conf(female_ages, confidence or 0.0),
        "overall_analysis_confidence": round(float(confidence or 0.0), 2),
    }

def generate_mock_demographics() -> Dict[str, Any]:
    total = int(np.random.randint(3, 20))
    male_pct = float(np.clip(np.random.normal(55, 15), 20, 80))
    male = int(total * male_pct / 100.0)
    female = total - male
    base_age = float(np.random.uniform(24, 32))
    male_ages   = [base_age + np.random.normal(0, 3)  for _ in range(male)]
    female_ages = [base_age + np.random.normal(-1, 3) for _ in range(female)]
    all_ages = male_ages + female_ages
    res = format_demographic_results(total, male, female, male_pct, 100.0 - male_pct,
                                     all_ages, male_ages, female_ages, 0.3)
    res.update({"analysis_method": "Mock Data", "confidence_level": "Low", "is_mock_data": True})
    return res

def analyze_demographics_with_fallback(video_path: str, allow: bool) -> Dict[str, Any]:
    if not allow:
        return {"analysis_method":"Disabled", "total_faces":0, "male_count":0, "female_count":0,
                "male_percentage":0.0, "female_percentage":0.0, "confidence":0.0,
                "age_analysis":"Disabled", "male_age_analysis":"Disabled", "female_age_analysis":"Disabled",
                "overall_analysis_confidence":0.0, "is_mock_data": False, "confidence_level":"N/A"}
    # Tier 1
    try:
        if insightface_app:
            st.info("🔍 InsightFace analysis running…")
            frames = extract_frames_for_analysis(video_path)
            if not frames: raise Exception("No frames")
            faces_all = []
            for i, fr in enumerate(frames):
                try:
                    fs = insightface_app.get(fr); faces_all.extend(fs)
                    st.caption(f"Frame {i+1}/{len(frames)}: {len(fs)} faces")
                except Exception as e:
                    st.warning(f"Frame {i+1} analysis failed: {e}")
            if not faces_all: raise Exception("No faces")
            dem = process_insightface_results(faces_all)
            dem["analysis_method"] = "InsightFace"
            dem["confidence_level"] = "High"
            st.success("✅ InsightFace complete.")
            return dem
    except Exception as e:
        st.warning(f"InsightFace failed: {e}")
    # Tier 2
    try:
        if rekog_client:
            st.info("🔍 AWS Rekognition analysis running…")
            frames = extract_frames_for_analysis(video_path)
            if not frames: raise Exception("No frames")
            faces_all = []
            for i, fr in enumerate(frames):
                try:
                    pil = Image.fromarray(fr.astype("uint8"))
                    buf = io.BytesIO(); pil.save(buf, format="JPEG", quality=85)
                    resp = rekog_client.detect_faces(Image={"Bytes": buf.getvalue()}, Attributes=["ALL"])
                    faces = resp.get("FaceDetails", []); faces_all.extend(faces)
                    st.caption(f"Frame {i+1}/{len(frames)}: {len(faces)} faces")
                except Exception as e:
                    st.warning(f"Frame {i+1} analysis failed: {e}")
            if not faces_all: raise Exception("No faces")
            dem = process_aws_rekognition_results(faces_all)
            dem["analysis_method"] = "AWS Rekognition"
            dem["confidence_level"] = "High"
            st.success("✅ Rekognition complete.")
            return dem
    except Exception as e:
        st.warning(f"AWS Rekognition failed: {e}")
    # Tier 3
    st.info("⚠️ Falling back to mock demographic data.")
    return generate_mock_demographics()

# ----------------------------
# Venue types & user‑specific sort
# ----------------------------
ALLOWED_VENUES = ["Club","Bar","Restaurant","Lounge","Rooftop","Outdoor Space","Speakeasy","Other"]

def get_venue_type_options(user_id: str) -> list:
    if not supabase:
        return ALLOWED_VENUES
    try:
        # simple frequency ordering
        resp = supabase.table("video_results") \
            .select("venue_type") \
            .eq("user_id", user_id).execute()
        counts: dict[str,int] = {}
        for r in (resp.data or []):
            v = r.get("venue_type")
            if v: counts[v] = counts.get(v, 0) + 1
    except Exception:
        counts = {}
    present = [v for v in ALLOWED_VENUES if v in counts]
    present.sort(key=lambda v: counts.get(v, 0), reverse=True)
    missing = [v for v in ALLOWED_VENUES if v not in present]
    return present + missing

# ----------------------------
# Storage & DB
# ----------------------------
def upload_video_to_storage(uploaded_file, user_id: str, content_hash: str) -> tuple[Optional[dict], Optional[str]]:
    if not supabase or not st.session_state.user:
        st.error("Not authenticated; please log in.")
        return None, None
    try:
        data = uploaded_file.getvalue()
        if not data:
            st.error("Empty file."); return None, None

        proxy_path, _trim = make_480p_proxy(data, max_seconds=60)

        ext = (uploaded_file.name.split(".")[-1] or "mp4").lower()
        base = f"{user_id}/{content_hash}"
        orig_key = f"{base}.mp4" if ext == "mp4" else f"{base}.{ext}"
        proxy_key = f"{base}_480p.mp4"

        supabase.storage.from_("videos").upload(
            path=orig_key, file=data,
            file_options={"content-type": uploaded_file.type or "video/mp4", "x-upsert": "false"}
        )
        with open(proxy_path, "rb") as pf:
            supabase.storage.from_("videos").upload(
                path=proxy_key, file=pf.read(),
                file_options={"content-type": "video/mp4", "x-upsert": "true"}
            )

        base_public = f"{SUPABASE_URL}/storage/v1/object/public/videos/"
        urls = {"original_url": base_public + orig_key, "proxy_url": base_public + proxy_key}
        try:
            os.unlink(proxy_path)
        except Exception:
            pass
        return urls, base
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None, None

def save_results_row(results: dict, uploaded_file=None, content_hash: Optional[str] = None):
    if not supabase or not st.session_state.user:
        st.error("Not authenticated.")
        return False, None, None

    user_id = st.session_state.user.id

    # idempotency
    if content_hash:
        try:
            existing = supabase.table("video_results").select("*") \
                .eq("user_id", user_id).eq("content_hash", content_hash).limit(1).execute()
            if existing.data:
                return True, existing.data[0], existing.data[0].get("public_url")
        except Exception:
            pass

    if uploaded_file is None:
        st.error("Missing file for upload.")
        return False, None, None

    urls, storage_base = upload_video_to_storage(uploaded_file, user_id, content_hash or uuid.uuid4().hex)
    if not urls:
        return False, None, None

    venv = results.get("visual_environment", {})
    aenv = results.get("audio_environment", {})
    crowd = results.get("crowd_density", {})
    mood  = results.get("mood_recognition", {})
    demo  = results.get("demographics", {})

    brightness = float(venv.get("brightness_level", 0.0))
    b_label, b_pct = brightness_label_and_pct(brightness)

    # dominant color name
    dc_name = "Gray"
    rgb = venv.get("dominant_rgb")
    if isinstance(rgb, dict) and {"r","g","b"} <= set(rgb):
        dc_name = rcw_color_name_from_rgb(int(rgb["r"]), int(rgb["g"]), int(rgb["b"]))
    else:
        hexcode = str(venv.get("color_scheme","")).strip()
        if hexcode.startswith("#") and len(hexcode) == 7:
            try:
                r = int(hexcode[1:3], 16); g = int(hexcode[3:5], 16); b = int(hexcode[5:7], 16)
                dc_name = rcw_color_name_from_rgb(r,g,b)
            except Exception:
                pass

    mood_conf = float(mood.get("confidence", 0.0))
    faces_detected = int(mood.get("faces_detected", 0))
    mood_note = "Faces detected; emotion unclear" if (faces_detected > 0 and mood_conf == 0.0) else None

    row = {
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat()+"Z",
        "venue_name": str(results.get("venue_name",""))[:100],
        "venue_type": str(results.get("venue_type","Other"))[:50],
        "storage_path": f"videos/{storage_base}",
        "public_url": urls["proxy_url"],
        "video_url": urls["proxy_url"],
        "video_stored": True,
        "content_hash": content_hash or "",

        # audio
        "bpm": int(aenv.get("bpm", 0) or 0),
        "volume_level": float(aenv.get("volume_level", 0.0) or 0.0),
        "genre": str(aenv.get("genre","Unknown"))[:50],
        "energy_level": str(aenv.get("energy_level","Unknown"))[:20],

        # visual
        "brightness_level": brightness,
        "brightness_pct": b_pct,
        "brightness_label": b_label,
        "lighting_type": str(venv.get("lighting_type","Unknown"))[:50],
        "color_scheme": str(venv.get("color_scheme",""))[:50],
        "dominant_color_name": dc_name,
        "visual_energy": str(venv.get("visual_energy","Unknown"))[:20],

        # crowd & mood
        "crowd_density": str(crowd.get("crowd_density","Unknown"))[:20],
        "activity_level": str(crowd.get("activity_level","Unknown"))[:50],
        "density_score": float(crowd.get("density_score", 0.0) or 0.0),
        "dominant_mood": str(mood.get("dominant_mood","Unknown"))[:30],
        "mood_confidence": mood_conf,
        "mood_note": mood_note,
        "overall_vibe": str(mood.get("overall_vibe","Unknown"))[:30],

        # aggregate
        "energy_score": float(results.get("energy_score", 0.0) or 0.0),

        # demographics
        "face_count": int(demo.get("total_faces", 0)),
        "gender_male_count": int(demo.get("male_count", 0)),
        "gender_female_count": int(demo.get("female_count", 0)),
        "gender_male_percentage": float(demo.get("male_percentage", 0.0)),
        "gender_female_percentage": float(demo.get("female_percentage", 0.0)),
        "gender_confidence": float(demo.get("confidence", 0.0)),
        "age_analysis": str(demo.get("age_analysis", ""))[:100],
        "demographics_raw": demo,

        "age_male_analysis": str(demo.get("male_age_analysis", ""))[:100],
        "age_female_analysis": str(demo.get("female_age_analysis", ""))[:100],
        "age_male_average": demo.get("age_male_average"),
        "age_female_average": demo.get("age_female_average"),
        "age_male_range": str(demo.get("age_male_range", "") or "")[:20],
        "age_female_range": str(demo.get("age_female_range", "") or "")[:20],

        "gender_male_confidence": float(demo.get("gender_male_confidence", 0.0)),
        "gender_female_confidence": float(demo.get("gender_female_confidence", 0.0)),
        "age_male_confidence": float(demo.get("age_male_confidence", 0.0)),
        "age_female_confidence": float(demo.get("age_female_confidence", 0.0)),
        "overall_analysis_confidence": float(demo.get("overall_analysis_confidence", 0.0)),
    }

    try:
        resp = supabase.table("video_results").insert(row).select("*").execute()
        inserted = resp.data[0] if resp.data else None
        return True, inserted, urls["proxy_url"]
    except Exception as e:
        st.error(f"DB insert failed: {e}")
        return False, None, None

# ----------------------------
# UI helpers
# ----------------------------
def display_results(row: dict):
    st.subheader(f"📊 Analysis Results for {row.get('venue_name','Unknown')}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Vibe", row.get("overall_vibe","N/A"))
    try:
        c2.metric("Energy Score", f"{float(row.get('energy_score',0)):.2f}/100")
    except Exception:
        c2.metric("Energy Score", str(row.get('energy_score',"N/A")))
    c3.metric("People Detected", row.get("face_count", 0))

    demo_raw = row.get("demographics_raw", {})
    if demo_raw and row.get("face_count", 0) > 0:
        with st.expander("👥 Demographics Analysis", expanded=True):
            is_mock = bool(demo_raw.get("is_mock_data", False))
            method = demo_raw.get("analysis_method", "Unknown")
            conf_level = demo_raw.get("confidence_level", "Unknown")
            if is_mock:
                st.warning("⚠️ Mock data shown (analysis systems unavailable).")
            else:
                st.info(f"Method: **{method}** | Confidence: **{conf_level}**")

            m_count = row.get("gender_male_count", 0)
            f_count = row.get("gender_female_count", 0)
            mp = row.get("gender_male_percentage", 0.0)
            fp = row.get("gender_female_percentage", 0.0)
            gconf = row.get("gender_confidence", 0.0)

            col1, col2, col3 = st.columns(3)
            col1.metric("Male", f"{m_count} ({mp:.1f}%)")
            col2.metric("Female", f"{f_count} ({fp:.1f}%)")
            col3.metric("Gender Confidence", f"{gconf:.0%}")

            st.write(f"**Overall Age Distribution:** {row.get('age_analysis', 'No age data')}")

            colA, colB = st.columns(2)
            with colA:
                maa = row.get("age_male_analysis")
                if maa: st.write(f"**Male Ages:** {maa}")
                mav = row.get("age_male_average")
                if mav is not None: st.write(f"Average: {float(mav):.1f} years")
            with colB:
                faa = row.get("age_female_analysis")
                if faa: st.write(f"**Female Ages:** {faa}")
                fav = row.get("age_female_average")
                if fav is not None: st.write(f"Average: {float(fav):.1f} years")

    with st.expander("🎵 Audio Environment", expanded=False):
        bpm = row.get('bpm','N/A')
        vol = float(row.get('volume_level', 0.0) or 0.0)
        genre = row.get('genre','Unknown')
        energy = row.get('energy_level','Unknown')
        st.write(f"**BPM:** {bpm}  |  **Volume:** {vol:.2f}  |  **Genre:** {genre}  |  **Energy:** {energy}")

    with st.expander("💡 Visual Environment", expanded=True):
        b = float(row.get('brightness_level', 0.0) or 0.0)
        b_label, b_pct = brightness_label_and_pct(b)
        dc_name = row.get("dominant_color_name", "Gray")
        v_energy = row.get('visual_energy','Unknown')
        st.write(f"**Brightness:** {b_label} • {b_pct:.2f}%")
        st.write(f"**Color:** {dc_name}")
        st.write(f"**Visual Energy:** {v_energy}")

    with st.expander("🕺 Crowd & Mood", expanded=True):
        cdens = row.get('crowd_density','Unknown')
        act = row.get('activity_level','Unknown')
        mood = row.get('dominant_mood','Unknown')
        conf = float(row.get('mood_confidence',0.0) or 0.0)
        st.write(f"**Crowd Density:** {cdens}  |  **Activity:** {act}")
        st.write(f"**Dominant Mood:** {mood}  |  **Confidence:** {confidence_bucket(conf)} ({conf:.0%})")
        extra = row.get("mood_note")
        if extra: st.caption(extra)

    url = row.get("public_url") or row.get("video_url")
    if url: st.video(url)

# ----------------------------
# Main UI
# ----------------------------
st.title("SneakPeak Video Scorer")

with st.sidebar:
    st.markdown("### Account")
    if st.session_state.user:
        st.success(f"Authed client: True\n\n{st.session_state.user.email}")
        if st.button("Sign out"):
            sign_out()
            st.rerun()
    else:
        st.info("Authed client: False")
        with st.form("login"):
            email = st.text_input("Email")
            pw = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign in")
        if submit:
            sign_in(email, pw)
            st.rerun()

    st.markdown("---")
    st.markdown("### System Status")
    for level, msg in INIT_MESSAGES:
        getattr(st, level)(msg)

    st.markdown("---")
    st.markdown("### Analysis Engines")
    st.write("InsightFace:", "✅ Ready" if insightface_app else "❌ Unavailable")
    st.write("AWS Rekognition:", "✅ Ready" if rekog_client else "❌ Unavailable")
    st.write("Video Processing:", "✅ Ready" if VIDEO_PROCESSING_OK else "❌ Unavailable")

    st.markdown("---")
    st.markdown("### Observability (session)")
    met = st.session_state.get("metrics", {})
    st.write(met)
    if met.get("analysis_ms"):
        st.caption(f"Avg analysis time: {sum(met['analysis_ms'])/len(met['analysis_ms']):.0f} ms")
    if st.checkbox("Show raw event log"):
        st.json(st.session_state.get("event_log", []))

tab_analyze, tab_history = st.tabs(["Analyze a Video", "View My Videos"])

with tab_analyze:
    st.subheader("Upload & Analyze")
    if not st.session_state.user:
        st.warning("Please sign in to analyze.")

    demographics_enabled = st.toggle("Run demographic analysis (gender/age)", value=True, help="Disable to skip gender/age inference.")
    up = st.file_uploader("Choose a video (≤60s processed; longer will be trimmed)", type=["mp4","mov","m4v","avi","webm"])
    venue_name = st.text_input("Venue Name", "")
    if st.session_state.user:
        opts = get_venue_type_options(st.session_state.user.id)
    else:
        opts = ALLOWED_VENUES
    venue_type = st.selectbox("Venue Type", opts, index=0)

    disabled = not (up and st.session_state.user)
    if st.button("Start Analysis", disabled=disabled):
        if not up:
            st.error("Please choose a video file."); st.stop()
        if not st.session_state.user:
            st.error("Please log in."); st.stop()

        cid = uuid.uuid4().hex
        data = up.getvalue() or b""
        if not data:
            st.error("Empty file."); st.stop()

        log_event("start", cid, filename=up.name, size=len(data))

        chash = sha256_bytes(data)
        try:
            dup = supabase.table("video_results").select("id") \
                .eq("user_id", st.session_state.user.id).eq("content_hash", chash) \
                .limit(1).execute()
            if dup.data:
                st.info("This video was analyzed before. Showing existing result.")
                row = supabase.table("video_results").select("*").eq("id", dup.data[0]["id"]).single().execute().data
                display_results(row); st.stop()
        except Exception:
            pass

        # Validate ffmpeg/proxy early
        try:
            _proxy_path, _dur = make_480p_proxy(data, max_seconds=60)
        except Exception as e:
            st.error(f"Could not prepare analysis proxy: {e}")
            log_event("analysis_failed", cid, reason="proxy_error")
            st.stop()

        prog = st.progress(5, text="Analyzing…")
        t0 = time.time()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as t:
            t.write(data); t.flush(); local_video_path = t.name

        audio = extract_audio_features(local_video_path) or {}
        prog.progress(20, text="Analyzing audio…")

        mid_frame_path = grab_middle_frame(local_video_path)
        visual = vision_visual(mid_frame_path) or {}
        crowd  = vision_crowd(mid_frame_path) or {}
        mood   = vision_mood(mid_frame_path) or {}
        prog.progress(40, text="Analyzing visuals & mood…")

        demographics = analyze_demographics_with_fallback(local_video_path, allow=demographics_enabled)
        prog.progress(70, text="Analyzing demographics…")

        results = {
            "venue_name": venue_name,
            "venue_type": venue_type or "Other",
            "audio_environment": audio,
            "visual_environment": visual,
            "crowd_density": crowd,
            "mood_recognition": mood,
            "demographics": demographics,
        }
        results["energy_score"] = calculate_energy_score(results)
        prog.progress(85, text="Saving…")

        ok, inserted_row, public_url = save_results_row(results, uploaded_file=up, content_hash=chash)
        elapsed_ms = int((time.time()-t0)*1000)
        if ok and inserted_row:
            if public_url:
                inserted_row["public_url"] = public_url
                inserted_row["video_url"]  = public_url
            log_event("analysis_done", cid, ok=True, elapsed_ms=elapsed_ms,
                      mood_conf_zero=(float(inserted_row.get("mood_confidence",0))==0.0))
            prog.progress(100, text="Done")
            display_results(inserted_row)
        else:
            log_event("analysis_failed", cid, reason="insert_failed")
            st.error("Could not save results.")

        # Cleanup temps
        try:
            os.unlink(local_video_path)
            os.unlink(mid_frame_path)
        except Exception:
            pass

with tab_history:
    st.subheader("My Videos")
    if not st.session_state.user:
        st.info("Please sign in to view your history.")
    else:
        page = st.number_input("Page", min_value=1, value=1, step=1)
        page_size = 10
        from_idx = (page-1)*page_size
        try:
            rows = supabase.table("video_results") \
                .select("*") \
                .eq("user_id", st.session_state.user.id) \
                .order("created_at", desc=True) \
                .range(int(from_idx), int(from_idx + page_size - 1)) \
                .execute().data
        except Exception as e:
            st.error(f"Failed to fetch results: {e}")
            rows = []

        if not rows:
            st.info("No results yet.")
        else:
            for r in rows:
                st.markdown("---")
                if not r.get("public_url") and not r.get("video_url"):
                    st.warning("No video available for playback for this row.")
                    upfix = st.file_uploader("Reattach original file for this row",
                                             type=["mp4","mov","m4v","avi","webm"], key=f"fix_{r['id']}")
                    if upfix:
                        ch = sha256_bytes(upfix.getvalue())
                        if ch != r.get("content_hash"):
                            st.error("That file doesn't match this row's original content hash.")
                        else:
                            urls, base = upload_video_to_storage(upfix, st.session_state.user.id, ch)
                            if urls:
                                try:
                                    supabase.table("video_results").update({
                                        "storage_path": f"videos/{base}",
                                        "public_url": urls["proxy_url"],
                                        "video_url": urls["proxy_url"],
                                        "video_stored": True
                                    }).eq("id", r["id"]).execute()
                                    st.success("Video reattached. Refresh to play.")
                                except Exception as e:
                                    st.error(f"Failed to update row: {e}")

                display_results(r)
