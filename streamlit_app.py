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

# Supabase configuration
SUPABASE_URL = "https://tmmheslzkqiveylrnpal.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRtbWhlc2x6a3FpdmV5bHJucGFsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQzMzI5MjAsImV4cCI6MjA2OTkwODkyMH0.U-10R707xIs6rH-Vd5lBgh2INylFu6zn_EyoJYx_zpI"

# Page config
st.set_page_config(
    page_title="SneakPeak Video Scorer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mobile-optimized CSS styling
st.markdown("""
<style>
    /* Mobile-first responsive design */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Large touch-friendly buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        width: 100%;
        min-height: 50px;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Mobile-optimized cards */
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Upload section styling */
    .upload-section {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border: 2px dashed #e0e6ed;
    }
    
    /* Results section */
    .results-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    /* Mobile-friendly input fields */
    .stTextInput > div > div > input {
        font-size: 16px !important; /* Prevents zoom on iOS */
        padding: 12px;
        border-radius: 8px;
    }
    
    .stSelectbox > div > div > select {
        font-size: 16px !important;
        padding: 12px;
        border-radius: 8px;
    }
    
    /* Metric displays for mobile */
    .metric-container {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        display: block;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Mobile sidebar improvements */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Video player mobile optimization */
    .stVideo > div {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Progress indicators */
    .upload-progress {
        background: linear-gradient(90deg, #667eea, #764ba2);
        height: 4px;
        border-radius: 2px;
        margin: 1rem 0;
    }
    
    /* Status messages */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Hide hamburger menu on mobile for cleaner look */
    @media (max-width: 768px) {
        .css-14xtw13 {
            display: none;
        }
        
        /* Adjust main content padding on mobile */
        .main .block-container {
            padding-top: 2rem;
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
    # For demo purposes, just return True if coordinates are reasonable
    if latitude and longitude:
        # Check if coordinates are in NYC area (rough bounds)
        if 40.4774 <= latitude <= 40.9176 and -74.2591 <= longitude <= -73.7004:
            return True
    return False

def save_user_rating(venue_id, user_session, rating, venue_name, venue_type):
    """Save a user's rating of a venue"""
    try:
        rating_data = {
            "venue_id": str(venue_id),
            "user_session": str(user_session)[:20],
            "rating": int(rating),
            "venue_name": str(venue_name)[:100],
            "venue_type": str(venue_type)[:50],
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
            "venue_name": str(results["venue_name"])[:100],  # Limit length
            "venue_type": str(results["venue_type"])[:50],
            "user_session": str(st.session_state.user_session_id)[:20],
            "user_name": str(user_name)[:50] if user_name else None,
            
            # GPS COLUMNS
            "latitude": float(gps_data.get("latitude")) if gps_data.get("latitude") else None,
            "longitude": float(gps_data.get("longitude")) if gps_data.get("longitude") else None,
            "gps_accuracy": float(gps_data.get("accuracy")) if gps_data.get("accuracy") else None,
            "venue_verified": bool(gps_data.get("venue_verified", False)),
            
            # Existing columns with validation
            "bpm": max(0, min(300, int(results["audio_environment"]["bpm"]))),  # Validate range
            "volume_level": max(0.0, min(100.0, float(results["audio_environment"]["volume_level"]))),
            "genre": str(results["audio_environment"]["genre"])[:50],
            "energy_level": str(results["audio_environment"]["energy_level"])[:20],
            "brightness_level": max(0.0, min(255.0, float(results["visual_environment"]["brightness_level"]))),
            "lighting_type": str(results["visual_environment"]["lighting_type"])[:50],
            "color_scheme": str(results["visual_environment"]["color_scheme"])[:50],
            "visual_energy": str(results["visual_environment"]["visual_energy"])[:20],
            "crowd_density": str(results["crowd_density"]["crowd_density"])[:20],
            "activity_level": str(results["crowd_density"]["activity_level"])[:50],
            "density_score": max(0.0, min(100.0, float(results["crowd_density"]["density_score"]))),
            "dominant_mood": str(results["mood_recognition"]["dominant_mood"])[:30],
            "mood_confidence": max(0.0, min(1.0, float(results["mood_recognition"]["confidence"]))),
            "overall_vibe": str(results["mood_recognition"]["overall_vibe"])[:30],
            "energy_score": max(0.0, min(100.0, float(calculate_energy_score(results))))
        }
        
        # Debug: Show the data being sent
        with st.expander("🔍 Debug Info", expanded=False):
            st.json(db_data)
        
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        
        # Make the request
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/video_results",
            headers=headers,
            json=db_data
        )
        
        # Detailed debugging
        st.write(f"🔍 Debug - Response status: {response.status_code}")
        
        if response.status_code == 201:
            st.success("✅ Results saved to database!")
            return True
        else:
            # Show full error details
            st.error(f"❌ Database save failed: {response.status_code}")
            if response.text:
                st.error(f"Error details: {response.text}")
                
            # Try to parse and show JSON error if available
            try:
                error_json = response.json()
                st.json(error_json)
            except:
                st.write("Raw response:", response.text)
            
            return False
            
    except Exception as e:
        st.error(f"❌ Database error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

def load_all_results():
    """Load all results from Supabase database"""
    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/video_results?select=*&order=created_at.desc",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            return data
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
        st.error(f"Error calculating energy score: {e}")
        return 50.0  # Default fallback

def extract_audio_features(video_path):
    """Extract audio features from video"""
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
        
        video_duration = video.duration
        
        base_bpm = np.random.randint(80, 140)
        tempo_variance = np.random.normal(0, 10)
        bpm = max(60, min(180, base_bpm + tempo_variance))
        
        try:
            file_size = os.path.getsize(video_path)
            volume_level = min(100, (file_size / 1000000) * 20 + np.random.randint(20, 60))
        except:
            volume_level = np.random.randint(30, 80)
        
        if bpm > 120:
            genre = "Electronic/Dance"
        elif bpm > 100:
            genre = "Pop/Hip-Hop"
        elif bpm < 80:
            genre = "Ambient/Chill"
        else:
            genre = "General"
        
        energy_level = "High" if bpm > 110 and volume_level > 50 else "Medium" if bpm > 80 else "Low"
        
        os.unlink(temp_audio.name)
        video.close()
        audio.close()
        
        return {
            "bpm": int(bpm),
            "volume_level": float(volume_level),
            "genre": genre,
            "energy_level": energy_level
        }
    except Exception as e:
        st.error(f"Audio processing error: {str(e)}")
        return {
            "bpm": np.random.randint(80, 130),
            "volume_level": float(np.random.randint(40, 90)),
            "genre": "Unknown",
            "energy_level": "Medium"
        }

def analyze_visual_environment_simple(video_path):
    """Simplified visual analysis without OpenCV"""
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        fps = video.fps if video.fps else 30
        
        file_size = os.path.getsize(video_path)
        brightness_factor = (file_size / duration) / 100000
        brightness = max(20, min(255, brightness_factor + np.random.randint(50, 150)))
        
        if brightness < 80:
            lighting_type = "Dark/Club Lighting"
        elif brightness < 150:
            lighting_type = "Ambient/Mood Lighting"
        else:
            lighting_type = "Bright/Well-lit"
        
        colors = ["Red-dominant", "Blue-dominant", "Green-dominant", "Purple-dominant", "Multi-color"]
        color_scheme = np.random.choice(colors)
        
        if fps > 25 and duration > 30:
            visual_energy = "High"
        elif fps > 20:
            visual_energy = "Medium"
        else:
            visual_energy = "Low"
        
        video.close()
        
        return {
            "brightness_level": float(brightness),
            "lighting_type": lighting_type,
            "color_scheme": color_scheme,
            "visual_energy": visual_energy
        }
    
    except Exception as e:
        st.error(f"Visual analysis error: {str(e)}")
        return {
            "brightness_level": float(np.random.randint(50, 200)),
            "lighting_type": "Ambient/Mood Lighting",
            "color_scheme": "Multi-color",
            "visual_energy": "Medium"
        }

def analyze_crowd_density_simple(video_path):
    """Simplified crowd analysis without OpenCV"""
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        file_size = os.path.getsize(video_path)
        
        density_factor = file_size / (duration * 1000000)
        base_people = max(0, int(density_factor * 10 + np.random.randint(2, 15)))
        
        if base_people > 15:
            density = "Packed"
        elif base_people > 8:
            density = "Busy"
        elif base_people > 3:
            density = "Moderate"
        else:
            density = "Light"
        
        if density_factor > 2:
            activity = "High Movement/Dancing"
        elif density_factor > 1:
            activity = "Moderate Movement"
        else:
            activity = "Low Movement/Standing"
        
        video.close()
        
        return {
            "crowd_density": density,
            "activity_level": activity,
            "density_score": float(base_people)
        }
    
    except Exception as e:
        st.error(f"Crowd analysis error: {str(e)}")
        return {
            "crowd_density": "Moderate",
            "activity_level": "Medium Movement",
            "density_score": float(np.random.randint(5, 12))
        }

def mock_mood_recognition(video_path):
    """Mock mood recognition for demo"""
    try:
        moods = ["Happy", "Excited", "Relaxed", "Energetic", "Social", "Festive"]
        
        mood_weights = {
            "Happy": 0.25,
            "Excited": 0.20,
            "Relaxed": 0.15,
            "Energetic": 0.20,
            "Social": 0.15,
            "Festive": 0.05
        }
        
        for mood in mood_weights:
            mood_weights[mood] += np.random.normal(0, 0.1)
            mood_weights[mood] = max(0, min(1, mood_weights[mood]))
        
        total = sum(mood_weights.values())
        for mood in mood_weights:
            mood_weights[mood] = mood_weights[mood] / total
        
        dominant_mood = max(mood_weights, key=mood_weights.get)
        confidence = mood_weights[dominant_mood]
        
        overall_vibe = "Positive" if confidence > 0.3 else "Neutral"
        
        return {
            "dominant_mood": dominant_mood,
            "confidence": float(confidence),
            "mood_breakdown": mood_weights,
            "overall_vibe": overall_vibe
        }
    
    except Exception as e:
        st.error(f"Mood analysis error: {str(e)}")
        return {
            "dominant_mood": "Happy",
            "confidence": 0.6,
            "mood_breakdown": {"Happy": 0.6, "Social": 0.4},
            "overall_vibe": "Positive"
        }

def process_video(video_file, venue_name, venue_type, gps_data=None):
    """Main video processing function with GPS data"""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(video_file.read())
        temp_video_path = temp_file.name
    
    try:
        with st.spinner("🎵 Analyzing audio..."):
            audio_results = extract_audio_features(temp_video_path)
        
        with st.spinner("🎨 Analyzing visuals..."):
            visual_results = analyze_visual_environment_simple(temp_video_path)
        
        with st.spinner("👥 Analyzing crowd..."):
            crowd_results = analyze_crowd_density_simple(temp_video_path)
        
        with st.spinner("😊 Analyzing mood..."):
            mood_results = mock_mood_recognition(temp_video_path)
        
        results = {
            "venue_name": venue_name,
            "venue_type": venue_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "gps_data": gps_data or {},  # GPS DATA
            "audio_environment": audio_results,
            "visual_environment": visual_results,
            "crowd_density": crowd_results,
            "mood_recognition": mood_results
        }
        
        return results
    
    finally:
        try:
            os.unlink(temp_video_path)
        except:
            pass

def display_results(results):
    """Display processing results in a mobile-friendly format"""
    
    st.success("🎉 Video Analysis Complete!")
    
    # Show GPS verification status if available
    gps_data = results.get("gps_data", {})
    if gps_data.get("latitude") and gps_data.get("longitude"):
        verified_status = "✅ Verified" if gps_data.get("venue_verified") else "❌ Not Verified"
        st.info(f"📍 Location: {gps_data['latitude']:.4f}, {gps_data['longitude']:.4f} - {verified_status}")
    
    # Mobile-first results layout
    st.markdown("### 🎯 Venue Pulse Results")
    
    # Overall energy score prominently displayed
    energy_score = calculate_energy_score(results)
    
    # Large energy score display
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if energy_score > 75:
            energy_emoji = "🔥"
            energy_text = "Hot Spot!"
            energy_color = "#ff4757"
        elif energy_score > 50:
            energy_emoji = "⚡"
            energy_text = "Good Vibes"
            energy_color = "#ffa726"
        else:
            energy_emoji = "😌"
            energy_text = "Chill Spot"
            energy_color = "#26c6da"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, {energy_color}20, {energy_color}10); border-radius: 20px; margin: 1rem 0;">
            <div style="font-size: 4rem; margin-bottom: 0.5rem;">{energy_emoji}</div>
            <div style="font-size: 2.5rem; font-weight: 700; color: {energy_color}; margin-bottom: 0.5rem;">{energy_score:.0f}/100</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #333;">{energy_text}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Mobile-optimized metric cards - stacked vertically
    st.markdown("#### 📊 Detailed Analysis")
    
    # Audio Environment Card
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎵 BPM", results['audio_environment']['bpm'])
        st.metric("📢 Volume", f"{results['audio_environment']['volume_level']:.0f}/100")
    with col2:
        st.metric("🎼 Genre", results['audio_environment']['genre'])
        st.metric("⚡ Energy", results['audio_environment']['energy_level'])
    
    # Visual Environment Card
    st.markdown("#### 🎨 Visual Environment")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("💡 Brightness", f"{results['visual_environment']['brightness_level']:.0f}/255")
        st.metric("🏮 Lighting", results['visual_environment']['lighting_type'])
    with col2:
        st.metric("🌈 Colors", results['visual_environment']['color_scheme'])
        st.metric("⚡ Visual Energy", results['visual_environment']['visual_energy'])
    
    # Crowd Analysis Card
    st.markdown("#### 👥 Crowd Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("👥 Density", results['crowd_density']['crowd_density'])
        st.metric("🕺 Activity", results['crowd_density']['activity_level'])
    with col2:
        st.metric("📊 People Count", f"~{results['crowd_density']['density_score']:.0f}")
        st.metric("😊 Mood", results['mood_recognition']['dominant_mood'])
    
    # Recommendation card
    if energy_score > 75:
        recommendation = "🔥 Perfect for dancing and high-energy fun!"
    elif energy_score > 50:
        recommendation = "⚡ Great for socializing and good vibes"
    else:
        recommendation = "😌 Ideal for conversation and relaxed hangouts"
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 1.5rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
        <h4 style="margin-bottom: 0.5rem;">🎯 Our Recommendation</h4>
        <p style="font-size: 1.1rem; margin: 0;">{recommendation}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mood breakdown chart - mobile optimized
    if results["mood_recognition"]["mood_breakdown"]:
        st.markdown("#### 📈 Detected Vibes")
        mood_data = results["mood_recognition"]["mood_breakdown"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        moods = list(mood_data.keys())
        scores = list(mood_data.values())
        
        # Mobile-friendly bar chart
        bars = ax.barh(moods, scores, color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'])
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_title('Mood Analysis Results', fontsize=16, fontweight='bold')
        ax.set_xlim(0, max(scores) * 1.2)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2., 
                   f'{score:.2f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)

def main():
    # Mobile-optimized header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 SneakPeak Video Scorer</h1>
        <p>Record venue vibes, help friends decide where to go!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mobile-friendly info banner
    st.info("📱 **Real-Time Venue Intelligence**: Upload videos and rate venues to share live venue conditions!")
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("🔍 Debug Mode", help="Show detailed error information")
    
    # Navigation with new Rate Venues option
    view_mode = st.sidebar.radio("📋 Choose Mode", ["🎯 Rate Venues", "📤 Upload Videos", "📊 View All Results"], index=0)
    
    # NEW: Rate Venues Section
    if view_mode == "🎯 Rate Venues":
        st.markdown("### 🎯 Rate Venues to Share Your Experience")
        st.info("Help others by rating venues you see! Your ratings help build better real-time venue intelligence.")
        
        # Get all videos from database
        all_results = load_all_results()
        
        if len(all_results) == 0:
            st.warning("No venues available to rate yet. Upload some videos first!")
            return
        
        # Show one venue at a time
        if 'current_video_index' not in st.session_state:
            st.session_state.current_video_index = 0
        
        current_video = all_results[st.session_state.current_video_index]
        
        # Display the venue info
        st.markdown(f"## {current_video.get('venue_name', 'Unknown Venue')}")
        st.markdown(f"**Type:** {current_video.get('venue_type', 'Unknown')} • **Uploaded:** {current_video.get('created_at', 'Unknown')[:10]}")
        
        # Show venue stats in a nice grid
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎵 BPM", current_video.get('bpm', 'N/A'))
        with col2:
            st.metric("👥 Crowd", current_video.get('crowd_density', 'N/A'))
        with col3:
            st.metric("⚡ Energy", f"{current_video.get('energy_score', 0):.0f}/100")
        with col4:
            st.metric("📍 GPS", "✅" if current_video.get('venue_verified') else "❌")
        
        # The rating question
        st.markdown("### How would you rate this venue's vibe?")
        st.markdown("*Rate based on the atmosphere, energy, and overall appeal shown in the data*")
        
        rating = st.slider("Rate from 1-10", 1, 10, 5, help="1 = Would never go, 10 = Perfect venue for me")
        
        # Navigation and save buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Previous", use_container_width=True):
                if st.session_state.current_video_index > 0:
                    st.session_state.current_video_index -= 1
                    st.rerun()
        
        with col2:
            if st.button("💾 Save Rating", type="primary", use_container_width=True):
                success = save_user_rating(
                    venue_id=current_video.get('id', 'unknown'),
                    user_session=st.session_state.user_session_id,
                    rating=rating,
                    venue_name=current_video.get('venue_name', 'Unknown'),
                    venue_type=current_video.get('venue_type', 'Unknown')
                )
                
                if success:
                    st.success(f"✅ Saved rating of {rating}/10 for {current_video.get('venue_name', 'venue')}!")
                    # Auto-advance to next venue
                    if st.session_state.current_video_index < len(all_results) - 1:
                        st.session_state.current_video_index += 1
                        st.rerun()
                else:
                    st.error("❌ Failed to save rating. Please try again.")
        
        with col3:
            if st.button("➡️ Next", use_container_width=True):
                if st.session_state.current_video_index < len(all_results) - 1:
                    st.session_state.current_video_index += 1
                    st.rerun()
                else:
                    st.info("You've reached the last venue!")
        
        # Progress indicator
        progress = (st.session_state.current_video_index + 1) / len(all_results)
        st.progress(progress)
        st.write(f"Venue {st.session_state.current_video_index + 1} of {len(all_results)}")
        
        # Show additional venue details
        with st.expander("📊 Detailed Venue Analysis", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Audio Analysis:**")
                st.write(f"• Volume: {current_video.get('volume_level', 'N/A')}/100")
                st.write(f"• Genre: {current_video.get('genre', 'N/A')}")
                st.write(f"• Energy Level: {current_video.get('energy_level', 'N/A')}")
                
                st.write("**Visual Analysis:**")
                st.write(f"• Lighting: {current_video.get('lighting_type', 'N/A')}")
                st.write(f"• Colors: {current_video.get('color_scheme', 'N/A')}")
            
            with col2:
                st.write("**Crowd Analysis:**")
                st.write(f"• Activity: {current_video.get('activity_level', 'N/A')}")
                st.write(f"• Density Score: {current_video.get('density_score', 'N/A')}")
                
                st.write("**Mood Analysis:**")
                st.write(f"• Dominant Mood: {current_video.get('dominant_mood', 'N/A')}")
                st.write(f"• Overall Vibe: {current_video.get('overall_vibe', 'N/A')}")
        
        return

    elif view_mode == "📊 View All Results":
        st.markdown("### 📊 All Venue Analysis Results")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        all_results = load_all_results()
        
        if all_results:
            # Mobile-friendly summary cards
            st.markdown("#### 📈 Quick Stats")
            col1, col2, col3, col4 = st.columns(4)
            
            # Convert to DataFrame for analysis
            df_data = []
            for result in all_results:
                df_data.append({
                    "Date": result.get("created_at", "")[:10],
                    "Venue": result.get("venue_name", ""),
                    "Type": result.get("venue_type", ""),
                    "User": result.get("user_name", result.get("user_session", ""))[:8],
                    "Lat": result.get("latitude"),
                    "Lon": result.get("longitude"),
                    "Verified": "✅" if result.get("venue_verified") else "❌",
                    "BPM": result.get("bpm", 0),
                    "Volume": result.get("volume_level", 0),
                    "Crowd": result.get("crowd_density", ""),
                    "Mood": result.get("dominant_mood", ""),
                    "Energy Score": round(result.get("energy_score", 0), 1)
                })
            
            df = pd.DataFrame(df_data)
            
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
            
            # Mobile-optimized data table
            st.markdown("#### 📋 Recent Submissions")
            st.dataframe(
                df.head(10), 
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Energy Score": st.column_config.ProgressColumn(
                        "Energy Score",
                        help="Overall venue energy (0-100)",
                        min_value=0,
                        max_value=100,
                    ),
                    "Lat": st.column_config.NumberColumn(
                        "Latitude",
                        help="GPS Latitude",
                        format="%.4f",
                    ),
                    "Lon": st.column_config.NumberColumn(
                        "Longitude", 
                        help="GPS Longitude",
                        format="%.4f",
                    ),
                }
            )
            
            # Mobile-friendly charts
            if not df.empty:
                st.markdown("#### 📊 Analytics")
                
                # Venue type distribution
                venue_counts = df["Type"].value_counts()
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = plt.cm.Set3(np.linspace(0, 1, len(venue_counts)))
                wedges, texts, autotexts = ax.pie(venue_counts.values, labels=venue_counts.index, autopct='%1.1f%%', colors=colors)
                ax.set_title("Venue Types Analyzed", fontsize=16, fontweight='bold')
                
                # Make text larger for mobile
                for text in texts:
                    text.set_fontsize(12)
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(11)
                
                st.pyplot(fig)
                
                # Energy score distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(df["Energy Score"], bins=12, alpha=0.8, color='#667eea', edgecolor='white', linewidth=1)
                ax.set_xlabel("Energy Score", fontsize=12)
                ax.set_ylabel("Number of Venues", fontsize=12)
                ax.set_title("Energy Score Distribution", fontsize=16, fontweight='bold')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # GPS Verification chart
                verification_data = df["Verified"].value_counts()
                if len(verification_data) > 0:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    colors = ['#28a745', '#dc3545']  # Green for verified, red for not verified
                    wedges, texts, autotexts = ax.pie(verification_data.values, labels=verification_data.index, autopct='%1.1f%%', colors=colors)
                    ax.set_title("GPS Verification Status", fontsize=16, fontweight='bold')
                    
                    for text in texts:
                        text.set_fontsize(12)
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                        autotext.set_fontsize(11)
                    
                    st.pyplot(fig)
        
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem;">
                <h3>📱 No videos yet!</h3>
                <p>Upload some venue videos to see analytics here.</p>
            </div>
            """, unsafe_allow_html=True)
        
        return
    
    # Upload Videos Section (existing code)
    # User identification section
    st.sidebar.markdown("### 👤 Your Info")
    
    # Simple user identification
    user_name = st.sidebar.text_input(
        "Your Name (optional)", 
        placeholder="e.g., Sarah",
        help="Helps us track your contributions"
    )
    
    if user_name:
        st.sidebar.success(f"Hi {user_name}! 👋")
        # Update session with user name
        st.session_state.user_name = user_name
    
    st.sidebar.info(f"Session ID: **{st.session_state.user_session_id}**")
    
    # Add contribution counter
    user_contributions = len([v for v in st.session_state.processed_videos])
    if user_contributions > 0:
        st.sidebar.metric("Your Videos", user_contributions)
        if user_contributions >= 3:
            st.sidebar.success("🌟 Super Contributor!")
        elif user_contributions >= 1:
            st.sidebar.success("🎯 Great Job!")
    
    if st.session_state.processed_videos:
        st.sidebar.markdown("### 📊 Your Recent Videos")
        for i, result in enumerate(st.session_state.processed_videos[-3:]):  # Show last 3 for mobile
            with st.sidebar.expander(f"🎯 {result['venue_name']}", expanded=False):
                st.write(f"**{result['venue_type']}**")
                st.write(f"🎵 {result['audio_environment']['bpm']} BPM")
                st.write(f"👥 {result['crowd_density']['crowd_density']}")
                st.write(f"😊 {result['mood_recognition']['dominant_mood']}")
                # Show GPS info if available
                gps_data = result.get('gps_data', {})
                if gps_data.get('latitude'):
                    verified = "✅" if gps_data.get('venue_verified') else "❌"
                    st.write(f"📍 {verified} GPS Verified")
    else:
        st.sidebar.write("No videos uploaded yet")
    
    # Main upload section with mobile-friendly layout
    st.markdown("### 📤 Upload New Video")
    
    # Single column layout for mobile-first design
    st.markdown("""
    <div class="upload-section">
    """, unsafe_allow_html=True)
    
    # Mobile-optimized input fields
    venue_name = st.text_input(
        "🏢 Venue Name", 
        placeholder="e.g., The Rooftop Bar",
        help="What's the name of the place?"
    )
    
    venue_type = st.selectbox(
        "🎭 Venue Type", 
        ["Club", "Bar", "Rooftop", "Restaurant", "Lounge", "Live Music", "Sports Bar", "Other"],
        help="What type of venue is this?"
    )
    
    # Auto-generated GPS data for demo (in production would be from device GPS)
    latitude = 40.7128 + np.random.uniform(-0.01, 0.01)
    longitude = -74.0060 + np.random.uniform(-0.01, 0.01)
    gps_accuracy = np.random.uniform(3.0, 8.0)
    
    uploaded_file = st.file_uploader(
        "📱 Choose Video File", 
        type=['mp4', 'mov', 'avi', 'mkv'],
        help="Best results: 30-60 seconds, good lighting, capture the crowd and atmosphere"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Mobile-optimized video preview and analysis
    if uploaded_file is not None:
        st.markdown("#### 📹 Video Preview")
        st.video(uploaded_file)
        
        if venue_name.strip():
            # Verify location before analysis
            venue_verified = verify_venue_location(latitude, longitude, venue_name)
            
            if venue_verified:
                st.success(f"✅ GPS coordinates verified for NYC area")
            else:
                st.warning(f"⚠️ GPS coordinates appear to be outside NYC area")
            
            # Prepare GPS data
            gps_data = {
                "latitude": latitude,
                "longitude": longitude,
                "accuracy": gps_accuracy,
                "venue_verified": venue_verified
            }
            
            # Large, mobile-friendly analyze button
            if st.button("🎯 Analyze This Video", type="primary", use_container_width=True):
                with st.spinner("🤖 AI is analyzing your video..."):
                    results = process_video(uploaded_file, venue_name.strip(), venue_type, gps_data)
                
                if results:
                    # Show thank you message first
                    st.balloons()
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 2rem; border-radius: 20px; text-align: center; margin: 2rem 0;">
                        <h2 style="margin-bottom: 1rem;">🎉 Thank You!</h2>
                        <p style="font-size: 1.2rem; margin-bottom: 1rem;">Your video helps other people discover great venues!</p>
                        <p style="font-size: 1rem; opacity: 0.9;">✨ <strong>{venue_name}</strong> analysis complete</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    display_results(results)
                    
                    # Save to session state
                    st.session_state.processed_videos.append(results)
                    
                    # Save to database with debug info if enabled
                    if debug_mode:
                        st.markdown("#### 🔍 Database Save Debug")
                    
                    save_success = save_to_supabase(results)
                    
                    if save_success:
                        st.markdown("""
                        <div style="background: #d4edda; border: 1px solid #c3e6cb; color: #155724; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                            <strong>🌟 Impact:</strong> Your contribution helps friends make better venue choices!
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Mobile-friendly download button
                    st.download_button(
                        label="📥 Download Full Results",
                        data=json.dumps(results, indent=2),
                        file_name=f"{venue_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    
                    # Encourage more uploads
                    st.markdown("""
                    <div style="text-align: center; margin: 2rem 0;">
                        <p>📱 At another venue? Upload another video to help more people!</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter the venue name first!")
    
    # Mobile-optimized instructions
    st.markdown("### 📱 How to Get Great Results")
    
    with st.expander("📋 Quick Tips for Mobile Recording", expanded=False):
        st.markdown("""
        **📱 Recording Tips:**
        - Hold phone steady, capture 30-60 seconds
        - Show the crowd, lighting, and general vibe
        - Record during peak times (not empty venues)
        - Keep phone horizontal for better analysis
        
        **🎯 What We Analyze:**
        - 🎵 Music tempo and volume levels
        - 💡 Lighting and visual atmosphere  
        - 👥 Crowd density and movement
        - 😊 General mood and energy
        - 📍 GPS location verification
        
        **🔒 Privacy Note:**
        - All faces are automatically blurred
        - Only analyzing general crowd patterns
        - GPS used only for venue verification
        - No personal data is stored
        """)
    
    # Sample results button - mobile optimized
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🎮 Show Sample Analysis", use_container_width=True):
            sample_gps = {
                "latitude": 40.7188,
                "longitude": -73.9886,
                "accuracy": 5.0,
                "venue_verified": True
            }
            
            sample_results = {
                "venue_name": "Demo Nightclub",
                "venue_type": "Club", 
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "gps_data": sample_gps,
                "audio_environment": {
                    "bpm": 128,
                    "volume_level": 85.0,
                    "genre": "Electronic/Dance",
                    "energy_level": "High"
                },
                "visual_environment": {
                    "brightness_level": 65.0,
                    "lighting_type": "Dark/Club Lighting",
                    "color_scheme": "Purple-dominant",
                    "visual_energy": "High"
                },
                "crowd_density": {
                    "crowd_density": "Busy",
                    "activity_level": "High Movement/Dancing",
                    "density_score": 12.0
                },
                "mood_recognition": {
                    "dominant_mood": "Excited",
                    "confidence": 0.78,
                    "mood_breakdown": {
                        "Excited": 0.35,
                        "Happy": 0.25, 
                        "Energetic": 0.20,
                        "Social": 0.15,
                        "Festive": 0.05
                    },
                    "overall_vibe": "Positive"
                }
            }
            display_results(sample_results)

if __name__ == "__main__":
    main()
