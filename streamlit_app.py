# Streamlit App: SneakPeak Video Scorer with Gender Detection
# Implements: InsightFace + AWS Rekognition + demographic analysis
# Added: Gender detection, age analysis, face counting, demographic confidence scoring

import os, io, time, uuid, hashlib, tempfile
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

import streamlit as st
import numpy as np
from PIL import Image

# ---------- Optional deps ----------
MOVIEPY_OK = True
try:
    from moviepy.editor import VideoFileClip
except Exception:
    MOVIEPY_OK = False

# ---------- InsightFace Integration ----------
INSIGHTFACE_OK = True
try:
    import insightface
    from insightface.app import FaceAnalysis
except Exception:
    INSIGHTFACE_OK = False

# ---------- AWS Rekognition Integration ----------
AWS_REKOGNITION_OK = True
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except Exception:
    AWS_REKOGNITION_OK = False

# AWS Credentials (embedded as requested)
AWS_ACCESS_KEY_ID = "AKIAV6VAEUEIJ2SQ6HXC"
AWS_SECRET_ACCESS_KEY = "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"
AWS_DEFAULT_REGION = "us-east-1"

# Initialize AWS Rekognition client
aws_rekognition_client = None
if AWS_REKOGNITION_OK:
    try:
        aws_rekognition_client = boto3.client(
            'rekognition',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_DEFAULT_REGION
        )
    except Exception as e:
        AWS_REKOGNITION_OK = False
        st.sidebar.warning(f"AWS Rekognition setup failed: {e}")

# Initialize InsightFace
insightface_app = None
if INSIGHTFACE_OK:
    try:
        insightface_app = FaceAnalysis(
            providers=['CPUExecutionProvider'],  # Start with CPU, can upgrade to GPU
            allowed_modules=['detection', 'genderage']
        )
        insightface_app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as e:
        INSIGHTFACE_OK = False
        st.sidebar.warning(f"InsightFace setup failed: {e}")

# ---------- Supabase ----------
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.warning("Supabase URL/Anon Key not found in environment. Set SUPABASE_URL and SUPABASE_ANON_KEY.")
supabase: Optional[Client] = None
try:
    if SUPABASE_URL and SUPABASE_ANON_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
except Exception as e:
    st.error(f"Could not initialize Supabase client: {e}")

# ---------------------------------------
# Auth utilities
# ---------------------------------------
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
            st.error("Sign-in failed (no user in response).")
    except Exception as e:
        st.error(f"Sign-in error: {e}")

def sign_out():
    if not supabase:
        return
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    st.session_state["user"] = None

# ---------------------------------------
# Helpers: brightness, RCW, confidence, hashing, logs
# ---------------------------------------
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

# 12-hue RCW buckets (+ Gray)
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
        if lo > hi:  # wrap-around bucket
            if deg >= lo or deg < hi:
                return name
    return "Gray"

def confidence_bucket(conf: float) -> str:
    try:
        v = float(conf)
    except Exception:
        v = 0.0
    if v <= 0.0:
        return "No Confidence"
    if v <= 0.33:
        return "Low"
    if v <= 0.66:
        return "Medium"
    return "High"

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def log_event(stage: str, correlation_id: str, **fields):
    entry = {"ts": datetime.utcnow().isoformat()+"Z", "stage": stage, "cid": correlation_id, **fields}
    st.session_state.setdefault("event_log", []).append(entry)
    m = st.session_state.setdefault("metrics", {"success":0, "fail":0, "analysis_ms":[], "zero_conf":0})
    if stage == "analysis_done" and fields.get("ok"):
        m["success"] += 1
        if "elapsed_ms" in fields:
            m["analysis_ms"].append(fields["elapsed_ms"])
        if fields.get("mood_conf_zero"):
            m["zero_conf"] += 1
    if stage == "analysis_failed":
        m["fail"] += 1

# ---------------------------------------
# NEW: Gender Detection and Demographics Analysis
# ---------------------------------------

def extract_frames_for_analysis(video_path: str) -> list:
    """Extract 3 frames from video at 25%, 50%, 75% for demographic analysis"""
    try:
        if not MOVIEPY_OK:
            raise RuntimeError("MoviePy not available")
        
        clip = VideoFileClip(video_path)
        duration = clip.duration or 0
        
        if duration < 1.0:  # Very short video
            sample_times = [duration / 2.0]  # Just middle frame
        else:
            sample_times = [
                duration * 0.25,  # 25%
                duration * 0.50,  # 50%
                duration * 0.75   # 75%
            ]
        
        frames = []
        for t in sample_times:
            try:
                frame = clip.get_frame(t)
                # Convert to RGB format for analysis
                if frame.shape[-1] == 3:  # RGB
                    frames.append(frame)
                else:  # Handle other formats
                    frames.append(frame[:, :, :3])
            except Exception as e:
                st.warning(f"Could not extract frame at {t:.1f}s: {e}")
        
        clip.close()
        return frames
        
    except Exception as e:
        st.warning(f"Frame extraction failed: {e}")
        return []

def analyze_demographics_insightface(video_path: str) -> Dict[str, Any]:
    """Primary demographic analysis using InsightFace"""
    try:
        if not INSIGHTFACE_OK or not insightface_app:
            raise Exception("InsightFace not available")
        
        st.info("🔍 Analyzing demographics with InsightFace...")
        
        frames = extract_frames_for_analysis(video_path)
        if not frames:
            raise Exception("No frames extracted from video")
        
        all_faces = []
        for i, frame in enumerate(frames):
            try:
                faces = insightface_app.get(frame)
                all_faces.extend(faces)
                st.info(f"Frame {i+1}/{len(frames)}: Found {len(faces)} faces")
            except Exception as e:
                st.warning(f"Frame {i+1} analysis failed: {e}")
        
        if not all_faces:
            raise Exception("No faces detected in any frame")
        
        demographics = process_insightface_results(all_faces)
        demographics['analysis_method'] = 'InsightFace'
        demographics['confidence_level'] = 'High'
        
        st.success(f"✅ InsightFace: Analyzed {len(all_faces)} faces across {len(frames)} frames")
        return demographics
        
    except Exception as e:
        st.warning(f"InsightFace analysis failed: {e}")
        raise

def analyze_demographics_aws_rekognition(video_path: str) -> Dict[str, Any]:
    """Secondary demographic analysis using AWS Rekognition"""
    try:
        if not AWS_REKOGNITION_OK or not aws_rekognition_client:
            raise Exception("AWS Rekognition not available")
        
        st.info("🔍 Analyzing demographics with AWS Rekognition...")
        
        frames = extract_frames_for_analysis(video_path)
        if not frames:
            raise Exception("No frames extracted from video")
        
        all_faces = []
        for i, frame in enumerate(frames):
            try:
                # Convert frame to JPEG bytes for AWS API
                pil_image = Image.fromarray(frame.astype('uint8'))
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='JPEG', quality=85)
                img_bytes = img_byte_arr.getvalue()
                
                # Call AWS Rekognition
                response = aws_rekognition_client.detect_faces(
                    Image={'Bytes': img_bytes},
                    Attributes=['ALL']
                )
                
                faces = response.get('FaceDetails', [])
                all_faces.extend(faces)
                st.info(f"Frame {i+1}/{len(frames)}: Found {len(faces)} faces")
                
            except Exception as e:
                st.warning(f"Frame {i+1} analysis failed: {e}")
        
        if not all_faces:
            raise Exception("No faces detected in any frame")
        
        demographics = process_aws_rekognition_results(all_faces)
        demographics['analysis_method'] = 'AWS Rekognition'
        demographics['confidence_level'] = 'High'
        
        st.success(f"✅ AWS Rekognition: Analyzed {len(all_faces)} faces across {len(frames)} frames")
        return demographics
        
    except Exception as e:
        st.warning(f"AWS Rekognition analysis failed: {e}")
        raise

def process_insightface_results(faces: list) -> Dict[str, Any]:
    """Process InsightFace results into SneakPeak demographic format"""
    if not faces:
        raise Exception("No faces to process")
    
    total_faces = len(faces)
    male_count = 0
    female_count = 0
    ages = []
    male_ages = []
    female_ages = []
    detection_confidences = []
    
    for face in faces:
        # Detection confidence
        det_confidence = getattr(face, 'det_score', 0.5)
        detection_confidences.append(det_confidence)
        
        # Gender analysis (face.sex: 0=female, 1=male)
        gender = getattr(face, 'sex', None)
        if gender is not None:
            if gender == 1:  # Male
                male_count += 1
            else:  # Female
                female_count += 1
        
        # Age analysis
        age = getattr(face, 'age', None)
        if age is not None:
            ages.append(age)
            if gender == 1:  # Male
                male_ages.append(age)
            else:  # Female
                female_ages.append(age)
    
    # Calculate percentages
    male_percentage = (male_count / total_faces * 100) if total_faces > 0 else 0
    female_percentage = (female_count / total_faces * 100) if total_faces > 0 else 0
    
    # Calculate confidence scores
    avg_detection_confidence = np.mean(detection_confidences) if detection_confidences else 0.5
    
    return format_demographic_results(
        total_faces, male_count, female_count, male_percentage, female_percentage,
        ages, male_ages, female_ages, avg_detection_confidence
    )

def process_aws_rekognition_results(faces: list) -> Dict[str, Any]:
    """Process AWS Rekognition results into SneakPeak demographic format"""
    if not faces:
        raise Exception("No faces to process")
    
    total_faces = len(faces)
    male_count = 0
    female_count = 0
    ages = []
    male_ages = []
    female_ages = []
    gender_confidences = []
    
    for face in faces:
        # Gender analysis
        gender_data = face.get('Gender', {})
        gender_value = gender_data.get('Value', '')
        gender_confidence = gender_data.get('Confidence', 0.0) / 100.0  # Convert to 0-1 scale
        gender_confidences.append(gender_confidence)
        
        if gender_value == 'Male':
            male_count += 1
        elif gender_value == 'Female':
            female_count += 1
        
        # Age analysis (AWS gives age ranges)
        age_range = face.get('AgeRange', {})
        age_low = age_range.get('Low', 0)
        age_high = age_range.get('High', 100)
        estimated_age = (age_low + age_high) / 2.0  # Use midpoint of range
        
        ages.append(estimated_age)
        if gender_value == 'Male':
            male_ages.append(estimated_age)
        elif gender_value == 'Female':
            female_ages.append(estimated_age)
    
    # Calculate percentages
    male_percentage = (male_count / total_faces * 100) if total_faces > 0 else 0
    female_percentage = (female_count / total_faces * 100) if total_faces > 0 else 0
    
    # Use gender confidence as overall confidence
    avg_confidence = np.mean(gender_confidences) if gender_confidences else 0.5
    
    return format_demographic_results(
        total_faces, male_count, female_count, male_percentage, female_percentage,
        ages, male_ages, female_ages, avg_confidence
    )

def format_demographic_results(total_faces, male_count, female_count, male_percentage, 
                             female_percentage, ages, male_ages, female_ages, confidence):
    """Format demographic results into standard format"""
    
    # Generate age analysis summaries
    age_analysis = generate_age_summary(ages) if ages else "No age data"
    male_age_analysis = generate_age_summary(male_ages) if male_ages else "No male faces"
    female_age_analysis = generate_age_summary(female_ages) if female_ages else "No female faces"
    
    # Calculate averages and ranges
    male_avg = np.mean(male_ages) if male_ages else None
    female_avg = np.mean(female_ages) if female_ages else None
    male_range = f"{min(male_ages):.0f}-{max(male_ages):.0f}" if male_ages else None
    female_range = f"{min(female_ages):.0f}-{max(female_ages):.0f}" if female_ages else None
    
    return {
        'total_faces': total_faces,
        'male_count': male_count,
        'female_count': female_count,
        'male_percentage': round(male_percentage, 1),
        'female_percentage': round(female_percentage, 1),
        'confidence': round(confidence, 2),
        'age_analysis': age_analysis,
        'male_age_analysis': male_age_analysis,
        'female_age_analysis': female_age_analysis,
        'age_male_average': round(male_avg, 1) if male_avg else None,
        'age_female_average': round(female_avg, 1) if female_avg else None,
        'age_male_range': male_range,
        'age_female_range': female_range,
        # Estimate separate confidences (since APIs don't provide them)
        'gender_male_confidence': estimate_gender_confidence(male_count, confidence),
        'gender_female_confidence': estimate_gender_confidence(female_count, confidence),
        'age_male_confidence': estimate_age_confidence(male_ages, confidence),
        'age_female_confidence': estimate_age_confidence(female_ages, confidence),
        'overall_analysis_confidence': confidence
    }

def generate_age_summary(ages: list) -> str:
    """Generate human-readable age distribution summary"""
    if not ages:
        return "No age data available"
    
    avg_age = np.mean(ages)
    age_range = max(ages) - min(ages) if len(ages) > 1 else 0
    
    if avg_age < 25:
        base = "Mostly early 20s"
    elif avg_age < 30:
        base = "Mostly mid-to-late 20s"
    elif avg_age < 35:
        base = "Mostly late 20s to early 30s"
    elif avg_age < 40:
        base = "Mostly 30s"
    else:
        base = "Mixed ages, 30s and up"
    
    if age_range > 15:
        base += " (wide age range)"
    
    return base

def estimate_gender_confidence(count: int, base_confidence: float) -> float:
    """Estimate gender confidence based on count and base confidence"""
    if count == 0:
        return 0.0
    # Higher counts generally mean more reliable gender classification
    count_factor = min(1.0, count / 5.0)  # Normalize count factor
    return round(base_confidence * (0.7 + 0.3 * count_factor), 2)

def estimate_age_confidence(ages: list, base_confidence: float) -> float:
    """Estimate age confidence based on age data and base confidence"""
    if not ages:
        return 0.0
    # Age confidence typically lower than gender confidence
    age_factor = 0.8  # Age is generally less reliable
    count_factor = min(1.0, len(ages) / 5.0)
    return round(base_confidence * age_factor * (0.7 + 0.3 * count_factor), 2)

def generate_mock_demographics() -> Dict[str, Any]:
    """Generate realistic mock demographic data when analysis fails"""
    # Generate realistic venue demographics
    total_faces = np.random.randint(3, 20)
    male_percentage = np.random.normal(55, 15)  # Slight male bias typical in nightlife
    male_percentage = max(20, min(80, male_percentage))
    
    male_count = int(total_faces * male_percentage / 100)
    female_count = total_faces - male_count
    
    # Generate age data
    base_age = np.random.uniform(24, 32)
    male_ages = [base_age + np.random.normal(0, 3) for _ in range(male_count)]
    female_ages = [base_age + np.random.normal(-1, 3) for _ in range(female_count)]
    all_ages = male_ages + female_ages
    
    return format_demographic_results(
        total_faces, male_count, female_count, male_percentage, 100 - male_percentage,
        all_ages, male_ages, female_ages, 0.3  # Low confidence for mock data
    ) | {
        'analysis_method': 'Mock Data',
        'confidence_level': 'Low',
        'is_mock_data': True
    }

def analyze_demographics_with_fallback(video_path: str) -> Dict[str, Any]:
    """Multi-tier demographic analysis with fallback strategy"""
    
    # Tier 1: Try InsightFace
    try:
        if INSIGHTFACE_OK:
            return analyze_demographics_insightface(video_path)
    except Exception as e:
        st.warning(f"InsightFace failed: {e}")
    
    # Tier 2: Try AWS Rekognition
    try:
        if AWS_REKOGNITION_OK:
            return analyze_demographics_aws_rekognition(video_path)
    except Exception as e:
        st.warning(f"AWS Rekognition failed: {e}")
    
    # Tier 3: Mock fallback
    st.info("⚠️ Using mock demographic data - analysis systems unavailable")
    return generate_mock_demographics()

# ---------------------------------------
# Video proxy (480p) + auto-trim to 60s
# ---------------------------------------
def make_480p_proxy(file_bytes: bytes, max_seconds: int = 60) -> tuple[str, float]:
    """
    Returns (proxy_path, duration_seconds). Writes 480p mp4 trimmed to <= max_seconds.
    """
    if not MOVIEPY_OK:
        raise RuntimeError("MoviePy/ffmpeg not available for proxy generation.")
    src_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    src_tmp.write(file_bytes)
    src_tmp.flush(); src_tmp.close()

    clip = VideoFileClip(src_tmp.name)
    duration = float(clip.duration or 0.0)
    end = min(duration, float(max_seconds))
    sub = clip.subclip(0, end)
    # Resize to height=480 preserving aspect
    sub = sub.resize(height=480)
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix="_480p.mp4")
    out_tmp.close()
    sub.write_videofile(out_tmp.name, codec="libx264", audio_codec="aac", threads=2, verbose=False, logger=None)
    clip.close(); sub.close()
    return out_tmp.name, end

# ---------------------------------------
# Minimal analysis fallbacks (replace with your real analyzers if present)
# ---------------------------------------
def grab_middle_frame(local_video_path: str) -> str:
    """
    Extract a middle frame as PNG for visual analysis.
    If MoviePy is unavailable, returns a blank image path.
    """
    try:
        if not MOVIEPY_OK:
            raise RuntimeError("MoviePy not available")
        clip = VideoFileClip(local_video_path)
        t = (clip.duration or 0) / 2.0
        frame = clip.get_frame(t)
        clip.close()
        img = Image.fromarray(frame[:, :, ::-1] if frame.shape[-1]==3 else frame)
    except Exception:
        img = Image.new("RGB", (640, 360), color=(64,64,64))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    img.save(tmp.name)
    return tmp.name

def extract_audio_features(local_video_path: str):
    """
    Placeholder: returns conservative defaults.
    Hook up to librosa/ffmpeg or your existing pipeline if available.
    """
    return {"bpm": 0, "volume_level": 0.0, "genre": "Unknown", "energy_level": "Unknown"}

def vision_visual(frame_path: str):
    """
    Simple brightness & color estimator from a frame.
    Replace with Google Vision outputs if you already have them.
    """
    try:
        img = Image.open(frame_path).convert("RGB")
        arr = np.array(img)
        # brightness: mean of V in HSV, scaled to 0..255
        hsv = np.array([colorsys.rgb_to_hsv(*(px/255.0)) for px in arr.reshape(-1,3)])
        v = hsv[:,2].mean()
        brightness = float(max(0.0, min(255.0, v*255.0)))
        # dominant color approx: mean RGB (quick & cheap)
        mean_rgb = arr.reshape(-1,3).mean(axis=0).astype(int)
        hexcode = f"#{mean_rgb[0]:02x}{mean_rgb[1]:02x}{mean_rgb[2]:02x}"
        # visual energy heuristic
        sat = hsv[:,1].mean()
        visual_energy = "High" if sat > 0.5 else ("Medium" if sat > 0.25 else "Low")
        return {
            "brightness_level": brightness,
            "lighting_type": "Unknown",
            "color_scheme": hexcode,     # keep backend code
            "dominant_rgb": {"r": int(mean_rgb[0]), "g": int(mean_rgb[1]), "b": int(mean_rgb[2])},
            "visual_energy": visual_energy,
        }
    except Exception:
        return {"brightness_level": 0.0, "lighting_type":"Unknown", "color_scheme":"#000000", "visual_energy":"Unknown"}

def vision_crowd(frame_path: str):
    # Placeholder
    return {"crowd_density":"Unknown", "activity_level":"Unknown", "density_score": 0.0}

def vision_mood(frame_path: str):
    # Placeholder with face count = 0 so confidence won't be misleading
    return {"dominant_mood":"Unknown", "confidence": 0.0, "overall_vibe":"Unknown", "faces_detected": 0}

def calculate_energy_score(results: dict) -> float:
    # Simple blend of audio/visual proxies
    a = results.get("audio_environment", {})
    v = results.get("visual_environment", {})
    base = 50.0
    base += min(50.0, float(a.get("volume_level", 0.0))*10.0)
    base += (float(v.get("brightness_level", 0.0))/255.0)*20.0
    return round(max(0.0, min(100.0, base)), 2)

# ---------------------------------------
# Venue types (filtered) & frequency sort
# ---------------------------------------
ALLOWED_VENUES = ["Club","Bar","Restaurant","Lounge","Rooftop","Outdoors Space","Speakeasy","Other"]

def get_venue_type_options(user_id: str) -> list:
    try:
        resp = supabase.table("video_results") \
            .select("venue_type, count:venue_type") \
            .eq("user_id", user_id) \
            .group("venue_type") \
            .execute()
        counts = {row["venue_type"]: row["count"] for row in (resp.data or []) if row.get("venue_type")}
    except Exception:
        counts = {}
    present = [v for v in ALLOWED_VENUES if v in counts]
    present.sort(key=lambda v: counts.get(v, 0), reverse=True)
    missing = [v for v in ALLOWED_VENUES if v not in present]
    return present + missing

# ---------------------------------------
# Storage upload (original+proxy) and DB insert (atomic)
# ---------------------------------------
def upload_video_to_storage(uploaded_file, user_id: str, content_hash: str) -> tuple[Optional[dict], Optional[str]]:
    """
    Upload original and 480p proxy. Returns (urls, storage_base) or (None, None) on failure.
    urls: {"original_url","proxy_url"}
    storage_base: f"{user_id}/{content_hash}"
    """
    if not supabase or not st.session_state.user:
        st.error("Not authenticated; please log in.")
        return None, None
    try:
        data = uploaded_file.getvalue()
        if not data:
            st.error("Empty file.")
            return None, None

        # 480p proxy (auto-trim to 60s)
        proxy_path, trimmed_seconds = make_480p_proxy(data, max_seconds=60)

        ext = (uploaded_file.name.split(".")[-1] or "mp4").lower()
        base = f"{user_id}/{content_hash}"
        orig_key = f"{base}.mp4" if ext == "mp4" else f"{base}.{ext}"
        proxy_key = f"{base}_480p.mp4"

        # Upload original
        supabase.storage.from_("videos").upload(
            path=orig_key,
            file=data,
            file_options={"content-type": uploaded_file.type or "video/mp4", "x-upsert": "false"}
        )
        # Upload proxy
        with open(proxy_path, "rb") as pf:
            supabase.storage.from_("videos").upload(
                path=proxy_key,
                file=pf.read(),
                file_options={"content-type": "video/mp4", "x-upsert": "true"}
            )

        base_public = f"{SUPABASE_URL}/storage/v1/object/public/videos/"
        urls = {
            "original_url": base_public + orig_key,
            "proxy_url":    base_public + proxy_key,
        }
        return urls, base
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None, None

def save_results_row(results: dict, uploaded_file=None, content_hash: Optional[str] = None):
    """
    Inserts into video_results ONLY after storage upload succeeds.
    Adds: storage_path, public_url (proxy), content_hash, brightness_pct, dominant_color_name, mood_note.
    UPDATED: Now includes all 19 demographic columns.
    """
    if not supabase or not st.session_state.user:
        st.error("Not authenticated.")
        return False, None, None

    user_id = st.session_state.user.id

    # idempotency
    if content_hash:
        try:
            existing = supabase.table("video_results").select("*").eq("user_id", user_id).eq("content_hash", content_hash).limit(1).execute()
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
    demographics = results.get("demographics", {})  # NEW: Demographics data

    brightness = float(venv.get("brightness_level", 0.0))
    b_label, b_pct = brightness_label_and_pct(brightness)

    dc_name = None
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
    if not dc_name:
        dc_name = "Gray"

    mood_conf = float(mood.get("confidence", 0.0))
    faces_detected = int(mood.get("faces_detected", 0))
    mood_note = None
    if faces_detected > 0 and mood_conf == 0.0:
        mood_note = "Faces detected; emotion unclear"

    row = {
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat()+"Z",
        "venue_name": str(results.get("venue_name",""))[:100],
        "venue_type": str(results.get("venue_type","Other"))[:50],
        "storage_path": f"videos/{storage_base}",
        "public_url": urls["proxy_url"],      # proxy for playback
        "video_url": urls["proxy_url"],       # keep legacy field working
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
        
        # overall
        "overall_vibe": str(mood.get("overall_vibe","Unknown"))[:30],
        "energy_score": float(results.get("energy_score", 0.0) or 0.0),
        
        # NEW: Demographics columns (all 19 fields)
        "face_count": int(demographics.get("total_faces", 0)),
        "gender_male_count": int(demographics.get("male_count", 0)),
        "gender_female_count": int(demographics.get("female_count", 0)),
        "gender_male_percentage": float(demographics.get("male_percentage", 0.0)),
        "gender_female_percentage": float(demographics.get("female_percentage", 0.0)),
        "gender_confidence": float(demographics.get("confidence", 0.0)),
        "age_analysis": str(demographics.get("age_analysis", ""))[:100],
        "demographics_raw": demographics,  # Store full data as JSONB
        
        "age_male_analysis": str(demographics.get("male_age_analysis", ""))[:100],
        "age_female_analysis": str(demographics.get("female_age_analysis", ""))[:100],
        "age_male_average": demographics.get("age_male_average"),
        "age_female_average": demographics.get("age_female_average"),
        "age_male_range": str(demographics.get("age_male_range", "") or "")[:20],
        "age_female_range": str(demographics.get("age_female_range", "") or "")[:20],
        
        "gender_male_confidence": float(demographics.get("gender_male_confidence", 0.0)),
        "gender_female_confidence": float(demographics.get("gender_female_confidence", 0.0)),
        "age_male_confidence": float(demographics.get("age_male_confidence", 0.0)),
        "age_female_confidence": float(demographics.get("age_female_confidence", 0.0)),
        "overall_analysis_confidence": float(demographics.get("overall_analysis_confidence", 0.0)),
    }

    try:
        resp = supabase.table("video_results").insert(row).select("*").execute()
        inserted = resp.data[0] if resp.data else None
        return True, inserted, urls["proxy_url"]
    except Exception as e:
        st.error(f"DB insert failed: {e}")
        return False, None, None

# ---------------------------------------
# UI: rendering results
# ---------------------------------------
def display_results(row: dict):
    st.subheader(f"📊 Analysis Results for {row.get('venue_name','Unknown')}")
    
    # Overall metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Vibe", row.get("overall_vibe","N/A"))
    try:
        c2.metric("Energy Score", f"{float(row.get('energy_score',0)):.2f}/100")
    except Exception:
        c2.metric("Energy Score", str(row.get('energy_score',"N/A")))
    
    # NEW: Demographics summary
    face_count = row.get('face_count', 0)
    c3.metric("People Detected", face_count)

    # NEW: Demographics Analysis Section
    demographics_raw = row.get('demographics_raw', {})
    if demographics_raw and face_count > 0:
        with st.expander("👥 Demographics Analysis", expanded=True):
            # Check if this is mock data
            is_mock = demographics_raw.get('is_mock_data', False)
            analysis_method = demographics_raw.get('analysis_method', 'Unknown')
            confidence_level = demographics_raw.get('confidence_level', 'Unknown')
            
            if is_mock:
                st.warning("⚠️ **Mock Data Warning**: Demographic analysis systems were unavailable. Data shown is simulated for demonstration purposes.")
            else:
                st.info(f"✅ **Analysis Method**: {analysis_method} | **Confidence**: {confidence_level}")
            
            # Gender breakdown
            male_count = row.get('gender_male_count', 0)
            female_count = row.get('gender_female_count', 0)
            male_pct = row.get('gender_male_percentage', 0.0)
            female_pct = row.get('gender_female_percentage', 0.0)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Male", f"{male_count} ({male_pct:.1f}%)")
            col2.metric("Female", f"{female_count} ({female_pct:.1f}%)")
            
            gender_conf = row.get('gender_confidence', 0.0)
            col3.metric("Gender Confidence", f"{gender_conf:.1%}")
            
            # Age analysis
            age_analysis = row.get('age_analysis', 'No age data')
            st.write(f"**Overall Age Distribution**: {age_analysis}")
            
            # Detailed age breakdown if available
            male_age_analysis = row.get('age_male_analysis')
            female_age_analysis = row.get('age_female_analysis')
            
            if male_age_analysis and female_age_analysis:
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Male Ages**: {male_age_analysis}")
                    male_avg = row.get('age_male_average')
                    if male_avg:
                        st.write(f"Average: {male_avg:.1f} years")
                
                with col2:
                    st.write(f"**Female Ages**: {female_age_analysis}")
                    female_avg = row.get('age_female_average')
                    if female_avg:
                        st.write(f"Average: {female_avg:.1f} years")

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
        conf_text = confidence_bucket(conf)
        extra = row.get("mood_note")
        st.write(f"**Crowd Density:** {cdens}  |  **Activity:** {act}")
        st.write(f"**Dominant Mood:** {mood}  |  **Confidence:** {conf_text} ({conf:.0%})")
        if extra:
            st.caption(extra)

    url = row.get("public_url") or row.get("video_url")
    if url:
        st.video(url)

# ---------------------------------------
# App UI
# ---------------------------------------
st.set_page_config(page_title="SneakPeak Video Scorer", page_icon="🎥", layout="wide")
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
            st.experimental_rerun()

    st.markdown("---")
    st.markdown("### System Status")
    
    # NEW: Show analysis system availability
    if INSIGHTFACE_OK:
        st.success("✅ InsightFace: Ready")
    else:
        st.error("❌ InsightFace: Not Available")
    
    if AWS_REKOGNITION_OK:
        st.success("✅ AWS Rekognition: Ready")
    else:
        st.error("❌ AWS Rekognition: Not Available")
    
    if MOVIEPY_OK:
        st.success("✅ Video Processing: Ready")
    else:
        st.error("❌ Video Processing: Not Available")

    st.markdown("---")
    st.markdown("### Observability (session)")
    met = st.session_state.get("metrics", {})
    st.write(met)
    if met.get("analysis_ms"):
        st.caption(f"Avg analysis time: {sum(met['analysis_ms'])/len(met['analysis_ms']):.0f} ms")
    if st.checkbox("Show raw event log"):
        st.json(st.session_state.get("event_log", []))

# Tabs
tab_analyze, tab_history = st.tabs(["Analyze a Video", "View My Videos"])

# -------- Analyze Tab --------
with tab_analyze:
    st.subheader("Upload & Analyze")
    if not st.session_state.user:
        st.warning("Please sign in to analyze.")
    up = st.file_uploader("Choose a video (<= 60s recommended; longer will be auto-trimmed for analysis)", type=["mp4","mov","m4v","avi"])
    venue_name = st.text_input("Venue Name", "")
    if st.session_state.user:
        opts = get_venue_type_options(st.session_state.user.id)
    else:
        opts = ALLOWED_VENUES
    venue_type = st.selectbox("Venue Type", opts, index=0)

    disabled = not (up and st.session_state.user)
    if st.button("Start Analysis", disabled=disabled):
        if not up:
            st.error("Please choose a video file.")
            st.stop()
        if not st.session_state.user:
            st.error("Please log in.")
            st.stop()

        cid = uuid.uuid4().hex
        data = up.getvalue() or b""
        if not data:
            st.error("Empty file.")
            st.stop()

        log_event("start", cid, filename=up.name, size=len(data))

        # Idempotency check
        chash = sha256_bytes(data)
        try:
            dup = supabase.table("video_results").select("id").eq("user_id", st.session_state.user.id).eq("content_hash", chash).limit(1).execute()
            if dup.data:
                st.info("This video was analyzed before. Showing existing result.")
                row = supabase.table("video_results").select("*").eq("id", dup.data[0]["id"]).single().execute().data
                display_results(row)
                st.stop()
        except Exception:
            pass

        # Proxy pre-check (auto-trim + resize) to validate ffmpeg availability early
        try:
            _proxy_path, _dur = make_480p_proxy(data, max_seconds=60)
        except Exception as e:
            st.error(f"Could not prepare analysis proxy: {e}")
            log_event("analysis_failed", cid, reason="proxy_error")
            st.stop()

        prog = st.progress(5, text="Analyzing…")
        t0 = time.time()

        # Save temp original for analyzers
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as t:
            t.write(data); t.flush(); local_video_path = t.name

        # Audio Analysis
        audio = extract_audio_features(local_video_path) or {}
        prog.progress(20, text="Analyzing audio…")

        # Visual Analysis (legacy single frame approach)
        mid_frame_path = grab_middle_frame(local_video_path)
        visual = vision_visual(mid_frame_path) or {}
        crowd  = vision_crowd(mid_frame_path) or {}
        mood   = vision_mood(mid_frame_path) or {}
        prog.progress(40, text="Analyzing visuals & mood…")

        # NEW: Demographics Analysis (multi-frame approach)
        demographics = analyze_demographics_with_fallback(local_video_path)
        prog.progress(70, text="Analyzing demographics…")

        results = {
            "venue_name": venue_name,
            "venue_type": venue_type or "Other",
            "audio_environment": audio,
            "visual_environment": visual,
            "crowd_density": crowd,
            "mood_recognition": mood,
            "demographics": demographics  # NEW
        }
        results["energy_score"] = calculate_energy_score(results)
        prog.progress(85, text="Saving…")

        ok, inserted_row, public_url = save_results_row(results, uploaded_file=up, content_hash=chash)
        elapsed_ms = int((time.time()-t0)*1000)
        if ok and inserted_row:
            if public_url:
                inserted_row["public_url"] = public_url
                inserted_row["video_url"]  = public_url
            log_event("analysis_done", cid, ok=True, elapsed_ms=elapsed_ms, mood_conf_zero=(float(inserted_row.get("mood_confidence",0))==0.0))
            prog.progress(100, text="Done")
            display_results(inserted_row)
        else:
            log_event("analysis_failed", cid, reason="insert_failed")
            st.error("Could not save results.")

        # Cleanup temp files
        try:
            os.unlink(local_video_path)
            os.unlink(mid_frame_path)
        except Exception:
            pass

# -------- History Tab --------
with tab_history:
    st.subheader("My Videos")
    if not st.session_state.user:
        st.info("Please sign in to view your history.")
    else:
        # Simple pagination
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
                # Legacy rows: allow reattach
                if not r.get("public_url") and not r.get("video_url"):
                    st.warning("No video available for playback for this row.")
                    upfix = st.file_uploader("Reattach original file for this row", type=["mp4","mov","m4v","avi"], key=f"fix_{r['id']}")
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
