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
