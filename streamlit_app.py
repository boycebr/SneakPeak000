# Streamlit App: SneakPeak Video Scorer (Brent edition)
# Implements: atomic upload->insert, 480p proxy (<=60s), RCW color naming, brightness labels,
# confidence buckets, venue list sorting by frequency, idempotency, observability,
# and legacy "Reattach Video" flow.

import os, io, time, uuid, hashlib, tempfile
from datetime import datetime
from typing import Tuple, Optional

import streamlit as st
import numpy as np
from PIL import Image

# ---------- Optional deps ----------
MOVIEPY_OK = True
try:
    from moviepy.editor import VideoFileClip
except Exception:
    MOVIEPY_OK = False

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

# 12‑hue RCW buckets (+ Gray)
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

        # 480p proxy (auto‑trim to 60s)
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
    c1, c2 = st.columns(2)
    c1.metric("Overall Vibe", row.get("overall_vibe","N/A"))
    try:
        c2.metric("Energy Score", f"{float(row.get('energy_score',0)):.2f}/100")
    except Exception:
        c2.metric("Energy Score", str(row.get('energy_score',"N/A")))

    with st.expander("🔊 Audio Environment", expanded=False):
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
    up = st.file_uploader("Choose a video (<= 60s recommended; longer will be auto‑trimmed for analysis)", type=["mp4","mov","m4v","avi"])
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

        # Audio / Visual / Crowd / Mood
        audio = extract_audio_features(local_video_path) or {}
        prog.progress(25, text="Analyzing audio…")

        mid_frame_path = grab_middle_frame(local_video_path)
        visual = vision_visual(mid_frame_path) or {}
        crowd  = vision_crowd(mid_frame_path) or {}
        mood   = vision_mood(mid_frame_path) or {}
        prog.progress(60, text="Analyzing visuals & mood…")

        results = {
            "venue_name": venue_name,
            "venue_type": venue_type or "Other",
            "audio_environment": audio,
            "visual_environment": visual,
            "crowd_density": crowd,
            "mood_recognition": mood
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
                            st.error("That file doesn't match this row’s original content hash.")
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
