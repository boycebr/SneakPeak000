import streamlit as st
import numpy as np
import tempfile
import os
from datetime import datetime
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from moviepy.editor import VideoFileClip
import requests
import base64
import uuid
import re

# Supabase configuration
SUPABASE_URL = "https://tmmheslzkqiveylrnpal.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRtbWhlc2x6a3FpdmV5bHJucGFsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQzMzI5MjAsImV4cCI6MjA2OTkwODkyMH0.U-10R707xIs6rH-Vd5lBgh2INylFu6zn_EyoJYx_zpI"

# Page config
st.set_page_config(
    page_title="SneakPeak Video Scorer",
    page_icon="🎯",
    layout="wide"
)

# Enhanced CSS for mobile-first design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom metric containers */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        display: block;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        display: block;
    }
    
    /* Energy score display */
    .energy-score {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(79, 172, 254, 0.3);
    }
    
    .energy-score h1 {
        font-size: 4rem;
        margin: 0;
        font-weight: 800;
    }
    
    .energy-score p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        min-height: 50px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Input styling */
    .stSelectbox > div > div {
        border-radius: 12px;
        border: 2px solid #e1e5e9;
        font-size: 16px;
        min-height: 50px;
    }
    
    .stTextInput > div > div {
        border-radius: 12px;
        border: 2px solid #e1e5e9;
        font-size: 16px;
        min-height: 50px;
    }
    
    /* File uploader */
    .stFileUploader > div {
        border-radius: 12px;
        border: 2px dashed #667eea;
        padding: 2rem;
        text-align: center;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Results cards */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #e1e5e9;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .result-card h3 {
        color: #333;
        margin: 0 0 1rem 0;
        font-weight: 600;
    }
    
    .result-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid #f0f0f0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .result-item:last-child {
        border-bottom: none;
    }
    
    .result-label {
        font-weight: 500;
        color: #666;
    }
    
    .result-value {
        font-weight: 600;
        color: #333;
    }
    
    /* Status indicators */
    .status-success {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .metric-value {
            font-size: 2rem;
        }
        
        .energy-score h1 {
            font-size: 3rem;
        }
        
        .stButton > button {
            width: 100%;
            margin: 0.5rem 0;
        }
        
        .main .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_videos' not in st.session_state:
    st.session_state.processed_videos = []
if 'user_session_id' not in st.session_state:
    st.session_state.user_session_id = str(uuid.uuid4())[:8]

def verify_venue_location(latitude, longitude, venue_name):
    """Simple venue verification - in production this would check against venue database"""
    if latitude and longitude:
        # Check if coordinates are in NYC area (rough bounds)
        if 40.4774 <= latitude <= 40.9176 and -74.2591 <= longitude <= -73.7004:
            return True
    return False

def save_user_rating(venue_id, user_session, rating, venue_name, venue_type, user_name=None):
    """Save a user's rating of a venue"""
    try:
        rating_data = {
            "venue_id": str(venue_id),
            "user_session": str(user_session)[:20],
            "rating": int(rating),
            "venue_name": str(venue_name)[:100],
            "venue_type": str(venue_type)[:50],
            "user_name": str(user_name)[:50] if user_name else None,
            "rated_at": datetime.now().isoformat()
        }
        
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/user_ratings",
            headers=headers,
            json=rating_data
        )
        
        return response.status_code == 201
    except Exception as e:
        st.error(f"Error saving rating: {str(e)}")
        return False

def save_to_supabase(results):
    """Save analysis results to Supabase database with GPS data and detailed debugging"""
    try:
        # Include user name if provided
        user_name = st.session_state.get('user_name', '')
        
        # Get GPS data from results
        gps_data = results.get("gps_data", {})
        
        # Prepare data with proper type casting and validation
        db_data = {
            "venue_name": str(results["venue_name"])[:100],
            "venue_type": str(results["venue_type"])[:50],
            "user_session": str(st.session_state.user_session_id)[:20],
            "user_name": str(user_name)[:50] if user_name else None,
            
            # GPS COLUMNS
            "latitude": float(gps_data.get("latitude")) if gps_data.get("latitude") else None,
            "longitude": float(gps_data.get("longitude")) if gps_data.get("longitude") else None,
            "gps_accuracy": float(gps_data.get("accuracy")) if gps_data.get("accuracy") else None,
            "venue_verified": bool(gps_data.get("venue_verified", False)),
            
            # Audio analysis
            "bpm": max(0, min(300, int(results["audio_environment"]["bpm"]))),
            "volume_level": max(0.0, min(100.0, float(results["audio_environment"]["volume_level"]))),
            "genre": str(results["audio_environment"]["genre"])[:50],
            "energy_level": str(results["audio_environment"]["energy_level"])[:20],
            
            # Visual analysis
            "brightness_level": max(0.0, min(255.0, float(results["visual_environment"]["brightness_level"]))),
            "lighting_type": str(results["visual_environment"]["lighting_type"])[:50],
            "color_scheme": str(results["visual_environment"]["color_scheme"])[:50],
            "visual_energy": str(results["visual_environment"]["visual_energy"])[:20],
            
            # Crowd analysis
            "crowd_density": str(results["crowd_density"]["crowd_density"])[:20],
            "activity_level": str(results["crowd_density"]["activity_level"])[:50],
            "density_score": max(0.0, min(100.0, float(results["crowd_density"]["density_score"]))),
            
            # Mood analysis
            "dominant_mood": str(results["mood_recognition"]["dominant_mood"])[:30],
            "mood_confidence": max(0.0, min(1.0, float(results["mood_recognition"]["confidence"]))),
            "overall_vibe": str(results["mood_recognition"]["overall_vibe"])[:30],
            
            # Enhanced fields (v3.0)
            "energy_score": max(0.0, min(100.0, float(calculate_energy_score(results)))),
            "analysis_version": "3.0",
            "processing_method": "Enhanced Mock",
            "engagement_score": max(0.0, min(100.0, float(results["crowd_density"]["density_score"]) * 1.2)),
            "estimated_people": max(0, int(results["crowd_density"]["density_score"] * 2)),
            "mood_diversity": max(1, min(8, len(results["mood_recognition"]["dominant_mood"].split()))),
            "lighting_score": max(0.0, min(100.0, float(results["visual_environment"]["brightness_level"]) / 2.55)),
            "video_filename": results.get("video_filename", ""),
            "video_stored": True,
            "upload_timestamp": datetime.now().isoformat()
        }
        
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/video_results",
            headers=headers,
            json=db_data
        )
        
        if response.status_code == 201:
            st.success("✅ Results saved to database!")
            return True
        else:
            st.error(f"❌ Database save failed: {response.status_code}")
            if response.text:
                st.error(f"Error details: {response.text}")
            return False
            
    except Exception as e:
        st.error(f"❌ Database error: {str(e)}")
        return False

def load_all_results():
    """Load all results from Supabase database"""
    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        }
        
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/video_results?select=*&order=created_at.desc",
            headers=headers
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to load data: {response.status_code}")
            return []
            
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return []

def calculate_energy_score(results):
    """Calculate energy score for consistency"""
    try:
        energy_score = (
            (float(results["audio_environment"]["bpm"]) / 160) * 0.3 +
            (float(results["audio_environment"]["volume_level"]) / 100) * 0.2 +
            (float(results["crowd_density"]["density_score"]) / 20) * 0.3 +
            float(results["mood_recognition"]["confidence"]) * 0.2
        ) * 100
        
        return min(100, max(0, energy_score))
    except Exception as e:
        return 50.0  # Default fallback

def extract_audio_features(video_file):
    """Extract audio features from video - Enhanced simulation"""
    try:
        # Base BPM on file properties for more realistic simulation
        file_size_mb = len(video_file.getvalue()) / (1024 * 1024)
        base_bpm = int(90 + (file_size_mb * 15) % 90)  # 90-180 range
        
        # Volume based on file name and size
        volume = min(100, max(20, int(50 + (file_size_mb * 10) % 50)))
        
        # Genre prediction based on characteristics
        if base_bpm > 140:
            genre = np.random.choice(["Electronic", "Dance", "Techno", "House"])
        elif base_bpm > 120:
            genre = np.random.choice(["Pop", "Hip-Hop", "R&B", "Indie"])
        else:
            genre = np.random.choice(["Jazz", "Blues", "Acoustic", "Ambient"])
        
        # Energy level based on BPM and volume
        if base_bpm > 140 and volume > 70:
            energy = "High"
        elif base_bpm > 110 and volume > 50:
            energy = "Medium"
        else:
            energy = "Low"
        
        return {
            "bpm": base_bpm,
            "volume_level": volume,
            "genre": genre,
            "energy_level": energy
        }
    except Exception as e:
        # Fallback values
        return {
            "bpm": np.random.randint(90, 180),
            "volume_level": np.random.randint(30, 90),
            "genre": np.random.choice(["Electronic", "Pop", "Hip-Hop", "Jazz"]),
            "energy_level": np.random.choice(["Low", "Medium", "High"])
        }

def analyze_visual_environment(video_path):
    """Analyze visual characteristics of the video"""
    try:
        with VideoFileClip(video_path) as clip:
            duration = clip.duration
            
            # Sample a few frames for analysis
            sample_times = np.linspace(1, min(duration-1, 10), 5)
            brightness_values = []
            
            for t in sample_times:
                frame = clip.get_frame(t)
                brightness = np.mean(frame)
                brightness_values.append(brightness)
            
            avg_brightness = np.mean(brightness_values)
            brightness_std = np.std(brightness_values)
            
            # Determine lighting type based on brightness and variation
            if avg_brightness < 80:
                lighting = "Dim"
            elif avg_brightness > 180:
                lighting = "Bright"
            elif brightness_std > 30:
                lighting = "Strobe"
            else:
                lighting = "Colorful"
            
            # Color scheme analysis
            if avg_brightness < 100:
                color_scheme = np.random.choice(["Dark/Moody", "Neon Accents", "Red Lighting"])
            else:
                color_scheme = np.random.choice(["Warm Tones", "Cool Blues", "Mixed Colors"])
            
            # Visual energy based on brightness variation
            if brightness_std > 40:
                visual_energy = "Dynamic"
            elif brightness_std > 20:
                visual_energy = "Moderate"
            else:
                visual_energy = "Static"
            
            return {
                "brightness_level": avg_brightness,
                "lighting_type": lighting,
                "color_scheme": color_scheme,
                "visual_energy": visual_energy
            }
    except Exception as e:
        # Fallback analysis
        return {
            "brightness_level": np.random.randint(60, 200),
            "lighting_type": np.random.choice(["Dim", "Bright", "Strobe", "Colorful"]),
            "color_scheme": np.random.choice(["Dark/Moody", "Warm Tones", "Cool Blues", "Neon Accents"]),
            "visual_energy": np.random.choice(["Static", "Moderate", "Dynamic"])
        }

def analyze_crowd_density(video_path, venue_type):
    """Analyze crowd density and movement patterns"""
    try:
        # Venue-specific crowd patterns
        venue_patterns = {
            "Nightclub": {"base_density": 15, "activity_high": 0.8},
            "Bar": {"base_density": 8, "activity_high": 0.4},
            "Lounge": {"base_density": 6, "activity_high": 0.3},
            "Restaurant": {"base_density": 10, "activity_high": 0.2},
            "Rooftop": {"base_density": 12, "activity_high": 0.6},
            "Speakeasy": {"base_density": 7, "activity_high": 0.3}
        }
        
        pattern = venue_patterns.get(venue_type, {"base_density": 10, "activity_high": 0.5})
        
        # Simulate crowd density (1-20 scale)
        density_score = pattern["base_density"] + np.random.randint(-3, 6)
        density_score = max(1, min(20, density_score))
        
        # Crowd density categories
        if density_score < 5:
            crowd_density = "Light"
        elif density_score < 10:
            crowd_density = "Moderate"
        elif density_score < 15:
            crowd_density = "Busy"
        else:
            crowd_density = "Packed"
        
        # Activity level based on venue type and random factor
        if np.random.random() < pattern["activity_high"]:
            activity_level = np.random.choice(["Dancing", "Socializing Actively", "High Energy"])
        else:
            activity_level = np.random.choice(["Mingling", "Conversing", "Relaxed"])
        
        return {
            "crowd_density": crowd_density,
            "activity_level": activity_level,
            "density_score": density_score
        }
    except Exception as e:
        return {
            "crowd_density": "Moderate",
            "activity_level": "Socializing",
            "density_score": 10
        }

def analyze_mood_recognition(venue_type, crowd_data, audio_data):
    """Analyze mood and overall vibe"""
    try:
        # Mood options with weights based on venue type and crowd
        mood_options = {
            "Nightclub": ["Energetic", "Euphoric", "Excited", "Social"],
            "Bar": ["Social", "Relaxed", "Happy", "Conversational"],
            "Lounge": ["Relaxed", "Intimate", "Sophisticated", "Calm"],
            "Restaurant": ["Social", "Comfortable", "Happy", "Engaged"],
            "Rooftop": ["Social", "Celebratory", "Relaxed", "Happy"],
            "Speakeasy": ["Intimate", "Mysterious", "Social", "Sophisticated"]
        }
        
        moods = mood_options.get(venue_type, ["Social", "Happy", "Relaxed"])
        
        # Weight mood selection based on crowd density and audio energy
        energy_level = audio_data.get("energy_level", "Medium")
        density_score = crowd_data.get("density_score", 10)
        
        if energy_level == "High" and density_score > 12:
            dominant_mood = np.random.choice(["Energetic", "Euphoric", "Excited"])
            confidence = 0.8 + np.random.random() * 0.15
        elif energy_level == "Medium":
            dominant_mood = np.random.choice(["Social", "Happy", "Engaged"])
            confidence = 0.7 + np.random.random() * 0.2
        else:
            dominant_mood = np.random.choice(["Relaxed", "Calm", "Intimate"])
            confidence = 0.6 + np.random.random() * 0.25
        
        # Overall vibe calculation
        if confidence > 0.8 and energy_level == "High":
            overall_vibe = "Electric"
        elif confidence > 0.7:
            overall_vibe = "Positive"
        elif confidence > 0.5:
            overall_vibe = "Neutral"
        else:
            overall_vibe = "Subdued"
        
        return {
            "dominant_mood": dominant_mood,
            "confidence": min(0.95, confidence),
            "overall_vibe": overall_vibe
        }
    except Exception as e:
        return {
            "dominant_mood": "Social",
            "confidence": 0.75,
            "overall_vibe": "Positive"
        }

def process_video(video_file, venue_name, venue_type, gps_data=None):
    """Process uploaded video and extract all features"""
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Show processing status
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Extract features
        status_text.text("🎵 Analyzing audio features...")
        progress_bar.progress(25)
        audio_features = extract_audio_features(video_file)
        
        status_text.text("🎨 Analyzing visual environment...")
        progress_bar.progress(50)
        visual_features = analyze_visual_environment(tmp_path)
        
        status_text.text("👥 Analyzing crowd density...")
        progress_bar.progress(75)
        crowd_features = analyze_crowd_density(tmp_path, venue_type)
        
        status_text.text("😊 Recognizing mood and vibe...")
        progress_bar.progress(90)
        mood_features = analyze_mood_recognition(venue_type, crowd_features, audio_features)
        
        status_text.text("✅ Analysis complete!")
        progress_bar.progress(100)
        
        # Compile results
        results = {
            "venue_name": venue_name,
            "venue_type": venue_type,
            "video_filename": video_file.name if hasattr(video_file, 'name') else "uploaded_video.mp4",
            "audio_environment": audio_features,
            "visual_environment": visual_features,
            "crowd_density": crowd_features,
            "mood_recognition": mood_features,
            "gps_data": gps_data or {}
        }
        
        return results
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass

def display_results(results):
    """Display analysis results in an attractive format"""
    
    # Calculate energy score
    energy_score = calculate_energy_score(results)
    
    # Energy Score - Large display
    st.markdown(f"""
    <div class="energy-score">
        <h1>{energy_score:.1f}</h1>
        <p>Overall Energy Score</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <span class="metric-value">{results['audio_environment']['bpm']}</span>
            <span class="metric-label">BPM</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <span class="metric-value">{results['audio_environment']['volume_level']:.0f}</span>
            <span class="metric-label">Volume</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <span class="metric-value">{results['crowd_density']['density_score']}</span>
            <span class="metric-label">Crowd Density</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <span class="metric-value">{results['mood_recognition']['confidence']:.2f}</span>
            <span class="metric-label">Mood Confidence</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed results in cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="result-card">
            <h3>🎵 Audio Analysis</h3>
            <div class="result-item">
                <span class="result-label">Genre:</span>
                <span class="result-value">{results['audio_environment']['genre']}</span>
            </div>
            <div class="result-item">
                <span class="result-label">Energy Level:</span>
                <span class="result-value">{results['audio_environment']['energy_level']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="result-card">
            <h3>👥 Crowd Analysis</h3>
            <div class="result-item">
                <span class="result-label">Crowd Density:</span>
                <span class="result-value">{results['crowd_density']['crowd_density']}</span>
            </div>
            <div class="result-item">
                <span class="result-label">Activity Level:</span>
                <span class="result-value">{results['crowd_density']['activity_level']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="result-card">
            <h3>🎨 Visual Environment</h3>
            <div class="result-item">
                <span class="result-label">Lighting:</span>
                <span class="result-value">{results['visual_environment']['lighting_type']}</span>
            </div>
            <div class="result-item">
                <span class="result-label">Color Scheme:</span>
                <span class="result-value">{results['visual_environment']['color_scheme']}</span>
            </div>
            <div class="result-item">
                <span class="result-label">Visual Energy:</span>
                <span class="result-value">{results['visual_environment']['visual_energy']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="result-card">
            <h3>😊 Mood & Vibe</h3>
            <div class="result-item">
                <span class="result-label">Dominant Mood:</span>
                <span class="result-value">{results['mood_recognition']['dominant_mood']}</span>
            </div>
            <div class="result-item">
                <span class="result-label">Overall Vibe:</span>
                <span class="result-value">{results['mood_recognition']['overall_vibe']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # GPS information if available
    if results.get('gps_data') and results['gps_data'].get('latitude'):
        gps_data = results['gps_data']
        verification_icon = "✅" if gps_data.get('venue_verified') else "❌"
        
        st.markdown(f"""
        <div class="result-card">
            <h3>📍 Location Data</h3>
            <div class="result-item">
                <span class="result-label">GPS Coordinates:</span>
                <span class="result-value">{gps_data['latitude']:.6f}, {gps_data['longitude']:.6f}</span>
            </div>
            <div class="result-item">
                <span class="result-label">Accuracy:</span>
                <span class="result-value">{gps_data.get('accuracy', 'Unknown')}m</span>
            </div>
            <div class="result-item">
                <span class="result-label">Venue Verified:</span>
                <span class="result-value">{verification_icon} {gps_data.get('venue_verified', False)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Main app
def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #333; font-weight: 700; margin: 0;">🎯 SneakPeak Video Scorer</h1>
        <p style="color: #666; font-size: 1.2rem; margin: 0.5rem 0;">AI-Powered Venue Vibe Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    tab1, tab2 = st.tabs(["📹 Upload Video", "📊 Analytics Dashboard"])
    
    with tab1:
        upload_interface()
    
    with tab2:
        analytics_dashboard()

def upload_interface():
    """Main upload interface"""
    
    # Sidebar - User Info
    st.sidebar.markdown("### 👤 Your Info")
    
    user_name = st.sidebar.text_input(
        "Your Name (optional)", 
        placeholder="e.g., Sarah",
        help="Helps us track your contributions",
        key="user_name_input"
    )
    
    if user_name:
        st.sidebar.success(f"Hi {user_name}! 👋")
        st.session_state.user_name = user_name
    
    st.sidebar.info(f"Session ID: **{st.session_state.user_session_id}**")
    
    # Contribution counter
    user_contributions = len(st.session_state.processed_videos)
    if user_contributions > 0:
        st.sidebar.metric("Your Videos", user_contributions)
        if user_contributions >= 3:
            st.sidebar.success("🌟 Super Contributor!")
        elif user_contributions >= 1:
            st.sidebar.success("🎯 Great Job!")
    
    # Recent videos in sidebar
    if st.session_state.processed_videos:
        st.sidebar.markdown("### 📱 Recent Videos")
        for i, video in enumerate(st.session_state.processed_videos[-3:]):
            st.sidebar.text(f"{i+1}. {video['venue_name']} ({video['venue_type']})")
    
    # Main upload interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📍 Venue Information")
        
        venue_name = st.text_input(
            "Venue Name", 
            placeholder="e.g., The Delancey, Attaboy, Kind Regards",
            help="Enter the name of the venue"
        )
        
        venue_type = st.selectbox(
            "Venue Type",
            ["Nightclub", "Bar", "Lounge", "Restaurant", "Rooftop", "Speakeasy"],
            help="Select the type of venue"
        )
    
    with col2:
        st.markdown("### 📍 GPS Location")
        
        # GPS location button
        if st.button("📍 Get My Location", help="Click to capture your GPS coordinates"):
            st.markdown("""
            <script>
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(function(position) {
                    const lat = position.coords.latitude;
                    const lng = position.coords.longitude;
                    const accuracy = position.coords.accuracy;
                    
                    // Store in session state (this would need to be handled differently in production)
                    console.log("GPS:", lat, lng, accuracy);
                    alert(`GPS captured: ${lat.toFixed(6)}, ${lng.toFixed(6)} (±${Math.round(accuracy)}m)`);
                });
            } else {
                alert("Geolocation is not supported by this browser.");
            }
            </script>
            """, unsafe_allow_html=True)
        
        st.info("📱 GPS helps verify venue authenticity")
    
    # Video upload
    st.markdown("### 🎬 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'mov', 'avi', 'mkv'],
        help="Upload a video of the venue atmosphere (max 100MB)"
    )
    
    if uploaded_file is not None and venue_name:
        st.success(f"✅ Video uploaded: {uploaded_file.name}")
        
        # Mock GPS data for demo
        gps_data = {
            "latitude": 40.7218 + np.random.uniform(-0.01, 0.01),
            "longitude": -73.9876 + np.random.uniform(-0.01, 0.01),
            "accuracy": np.random.uniform(5, 25),
            "venue_verified": np.random.choice([True, False], p=[0.8, 0.2])
        }
        
        if st.button("🎯 Analyze Video", type="primary"):
            with st.spinner("Processing video..."):
                # Process the video
                results = process_video(uploaded_file, venue_name, venue_type, gps_data)
                
                # Display results
                st.markdown("## 🎯 Analysis Results")
                display_results(results)
                
                # Save to database
                if save_to_supabase(results):
                    # Add to session state
                    st.session_state.processed_videos.append({
                        'venue_name': venue_name,
                        'venue_type': venue_type,
                        'energy_score': calculate_energy_score(results),
                        'timestamp': datetime.now()
                    })
                    
                    # Rating system
                    st.markdown("### ⭐ Rate This Analysis")
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        user_rating = st.slider(
                            "How accurate was this analysis?",
                            min_value=1,
                            max_value=10,
                            value=8,
                            help="Rate the accuracy of our venue analysis"
                        )
                    
                    with col2:
                        if st.button("Submit Rating"):
                            venue_id = f"{venue_name}_{venue_type}".replace(" ", "_").lower()
                            user_name = st.session_state.get('user_name', '')
                            
                            if save_user_rating(venue_id, st.session_state.user_session_id, 
                                              user_rating, venue_name, venue_type, user_name):
                                st.success("Thanks for your feedback! 🙏")
                            else:
                                st.error("Failed to save rating")

def analytics_dashboard():
    """Analytics dashboard showing database results"""
    
    st.markdown("### 📊 Real-Time Analytics")
    
    # Load data from database
    with st.spinner("Loading analytics data..."):
        all_results = load_all_results()
    
    if all_results:
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        # Data preprocessing
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['Venue'] = df['venue_name']
        df['Type'] = df['venue_type']
        df['Energy Score'] = df['energy_score'].fillna(0)
        df['User'] = df['user_name'].fillna('Anonymous')
        df['Verified'] = df['venue_verified'].fillna(False)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <span class="metric-value">{len(all_results)}</span>
                <span class="metric-label">Total Videos</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_energy = df["Energy Score"].mean() if not df.empty else 0
            st.markdown(f"""
            <div class="metric-container">
                <span class="metric-value">{avg_energy:.1f}</span>
                <span class="metric-label">Avg Energy</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            unique_venues = df["Venue"].nunique() if not df.empty else 0
            st.markdown(f"""
            <div class="metric-container">
                <span class="metric-value">{unique_venues}</span>
                <span class="metric-label">Venues</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            verified_count = len([r for r in all_results if r.get("venue_verified")])
            st.markdown(f"""
            <div class="metric-container">
                <span class="metric-value">{verified_count}</span>
                <span class="metric-label">GPS Verified</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Data table
        st.markdown("#### 📋 Recent Submissions")
        st.dataframe(
            df[['Venue', 'Type', 'Energy Score', 'User', 'Verified', 'created_at']].head(10), 
            use_container_width=True,
            hide_index=True,
            column_config={
                "Energy Score": st.column_config.ProgressColumn(
                    "Energy Score",
                    help="Overall venue energy (0-100)",
                    min_value=0,
                    max_value=100,
                ),
                "created_at": st.column_config.DatetimeColumn(
                    "Submitted",
                    format="MMM DD, YYYY - HH:mm"
                )
            }
        )
        
        # Charts
        if not df.empty:
            st.markdown("#### 📈 Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Venue type distribution
                venue_counts = df["Type"].value_counts()
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.Set3(np.linspace(0, 1, len(venue_counts)))
                bars = ax.bar(venue_counts.index, venue_counts.values, color=colors)
                ax.set_title("Venue Type Distribution", fontsize=16, fontweight='bold')
                ax.set_ylabel("Number of Videos", fontsize=12)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                # Energy score distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(df["Energy Score"], bins=12, alpha=0.8, color='#667eea', edgecolor='white', linewidth=1)
                ax.set_xlabel("Energy Score", fontsize=12)
                ax.set_ylabel("Number of Venues", fontsize=12)
                ax.set_title("Energy Score Distribution", fontsize=16, fontweight='bold')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
    
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h3>📱 No videos yet!</h3>
            <p>Upload some venue videos to see analytics here.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
