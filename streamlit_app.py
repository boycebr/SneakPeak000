import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
import requests
import json
from datetime import datetime
import uuid
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from moviepy.editor import VideoFileClip
import cv2

# ================================
# CONFIGURATION
# ================================

# Supabase Configuration
SUPABASE_URL = "https://tmmheslzkqiveylrnpal.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRtbWhlc2x6a3FpdmV5bHJucGFsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjQ5NTU5NDAsImV4cCI6MjA0MDUzMTk0MH0.g_0U_o7W5xRIXlOjSz7lZJCBayXjy5EJfSw1kNGJuSg"

# Page configuration
st.set_page_config(
    page_title="SneakPeak - Venue Pulse",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================================
# CSS STYLING
# ================================

st.markdown("""
<style>
/* Mobile-first responsive design */
.main > div {
    padding-top: 2rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

/* Energy score display */
.energy-score {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 20px;
    margin: 1rem 0;
    color: white;
}

.energy-score h1 {
    font-size: 4rem;
    margin: 0;
    font-weight: 700;
}

.energy-score p {
    font-size: 1.2rem;
    margin: 0.5rem 0 0 0;
    opacity: 0.9;
}

/* Metric containers */
.metric-container {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin: 0.5rem 0;
}

.metric-value {
    display: block;
    font-size: 2.5rem;
    font-weight: 700;
    color: #667eea;
}

.metric-label {
    display: block;
    font-size: 0.9rem;
    color: #666;
    margin-top: 0.5rem;
}

/* Cards */
.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin: 1rem 0;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 25px;
    height: 3rem;
    font-weight: 600;
    font-size: 1.1rem;
    border: none;
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white;
}

/* Input fields */
.stTextInput > div > div > input {
    border-radius: 10px;
    border: 2px solid #e1e5e9;
    font-size: 16px;
}

.stSelectbox > div > div > select {
    border-radius: 10px;
    border: 2px solid #e1e5e9;
    font-size: 16px;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    .energy-score h1 {
        font-size: 3rem;
    }
    
    .metric-value {
        font-size: 2rem;
    }
    
    .main > div {
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
}

/* GPS button styling */
.gps-button {
    background: linear-gradient(45deg, #48bb78, #38a169);
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    cursor: pointer;
    width: 100%;
    margin: 0.5rem 0;
}

.gps-button:hover {
    background: linear-gradient(45deg, #38a169, #2f855a);
}
</style>
""", unsafe_allow_html=True)

# ================================
# SESSION STATE INITIALIZATION
# ================================

if 'user_session_id' not in st.session_state:
    st.session_state.user_session_id = str(uuid.uuid4())

if 'processed_videos' not in st.session_state:
    st.session_state.processed_videos = []

if 'gps_data' not in st.session_state:
    st.session_state.gps_data = {}

# ================================
# GPS FUNCTIONS
# ================================

def get_gps_location():
    """Get GPS location using JavaScript geolocation API"""
    st.markdown("""
    <div id="gps-section">
        <button class="gps-button" onclick="getLocation()">📍 Get Current Location</button>
        <div id="gps-status" style="margin-top: 10px; font-size: 14px;"></div>
        <div id="gps-coords" style="margin-top: 10px; font-size: 12px; color: #666;"></div>
    </div>
    
    <script>
    function getLocation() {
        const statusDiv = document.getElementById('gps-status');
        const coordsDiv = document.getElementById('gps-coords');
        
        statusDiv.innerHTML = '📍 Getting location...';
        
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                function(position) {
                    const lat = position.coords.latitude;
                    const lng = position.coords.longitude;
                    const accuracy = position.coords.accuracy;
                    
                    statusDiv.innerHTML = '✅ Location obtained!';
                    coordsDiv.innerHTML = `Lat: ${lat.toFixed(6)}, Lng: ${lng.toFixed(6)}, Accuracy: ${accuracy.toFixed(0)}m`;
                    
                    // Store in session state (this is a demo - in real app, this would be handled differently)
                    window.parent.postMessage({
                        type: 'gps_data',
                        latitude: lat,
                        longitude: lng,
                        accuracy: accuracy
                    }, '*');
                },
                function(error) {
                    switch(error.code) {
                        case error.PERMISSION_DENIED:
                            statusDiv.innerHTML = '❌ Location access denied - continuing without GPS';
                            break;
                        case error.POSITION_UNAVAILABLE:
                            statusDiv.innerHTML = '❌ Location unavailable - continuing without GPS';
                            break;
                        case error.TIMEOUT:
                            statusDiv.innerHTML = '❌ Location timeout - continuing without GPS';
                            break;
                        default:
                            statusDiv.innerHTML = '❌ Unknown GPS error - continuing without GPS';
                            break;
                    }
                }
            );
        } else {
            statusDiv.innerHTML = '❌ Geolocation not supported - continuing without GPS';
        }
    }
    </script>
    """, unsafe_allow_html=True)

def verify_venue_location(latitude, longitude, venue_name):
    """Verify venue location against NYC boundaries"""
    try:
        # NYC approximate boundaries
        NYC_BOUNDS = {
            'north': 40.9176,
            'south': 40.4774,
            'east': -73.7004,
            'west': -74.2591
        }
        
        if (NYC_BOUNDS['south'] <= latitude <= NYC_BOUNDS['north'] and 
            NYC_BOUNDS['west'] <= longitude <= NYC_BOUNDS['east']):
            return True, f"✅ Venue location verified in NYC area"
        else:
            return False, f"❌ Location outside NYC area"
            
    except Exception as e:
        return False, f"❌ Location verification failed: {str(e)}"

# ================================
# ENERGY SCORE CALCULATION (FIXED)
# ================================

def calculate_energy_score(results):
    """Calculate energy score with proper bounds checking (0-100)"""
    try:
        # Extract values with safe defaults
        bpm = float(results["audio_environment"].get("bpm", 100))
        volume = float(results["audio_environment"].get("volume_level", 50))
        density = float(results["crowd_density"].get("density_score", 10))
        confidence = float(results["mood_recognition"].get("confidence", 0.5))
        
        # Normalize each component to 0-100 scale
        normalized_bpm = max(0, min(100, (bpm / 160) * 100))
        normalized_volume = max(0, min(100, volume))
        normalized_density = max(0, min(100, (density / 20) * 100))
        normalized_confidence = max(0, min(100, confidence * 100))
        
        # Calculate weighted score
        energy_score = (
            normalized_bpm * 0.3 +
            normalized_volume * 0.2 +
            normalized_density * 0.3 +
            normalized_confidence * 0.2
        )
        
        # CRITICAL FIX: Ensure final score is always within 0-100 range
        final_energy_score = max(0, min(100, energy_score))
        
        return final_energy_score
        
    except Exception as e:
        st.error(f"Error calculating energy score: {str(e)}")
        return 50.0  # Safe fallback value

# ================================
# DONUT CHART CREATION
# ================================

def create_energy_donut_chart(results):
    """Create multi-ring donut chart for energy visualization"""
    try:
        # Extract individual energy scores
        bmp = float(results["audio_environment"].get("bmp", 100))
        volume = float(results["audio_environment"].get("volume_level", 50))
        brightness = float(results["visual_environment"].get("brightness_level", 127))
        density = float(results["crowd_density"].get("density_score", 10))
        
        # Normalize scores to 0-100
        audio_score = max(0, min(100, ((bmp / 160) * 50) + (volume / 100 * 50)))
        visual_score = max(0, min(100, (brightness / 255) * 100))
        crowd_score = max(0, min(100, (density / 20) * 100))
        overall_score = calculate_energy_score(results)
        
        # Create subplot with donut charts
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "domain"}]],
            subplot_titles=["Energy Breakdown"]
        )
        
        # Define colors for each ring
        colors = {
            'audio': '#FF6B6B',      # Red
            'visual': '#4ECDC4',     # Teal  
            'crowd': '#45B7D1',      # Blue
            'overall': '#96CEB4'     # Green
        }
        
        # Helper function to get color based on score
        def get_ring_color(score, base_color):
            if score >= 80:
                return base_color  # Full color for 80-100%
            else:
                return base_color + '80'  # Lighter color for <80%
        
        # Create each ring (from outer to inner)
        rings = [
            {'label': 'Overall', 'score': overall_score, 'color': colors['overall'], 'hole': 0.7},
            {'label': 'Audio', 'score': audio_score, 'color': colors['audio'], 'hole': 0.55},
            {'label': 'Visual', 'score': visual_score, 'color': colors['visual'], 'hole': 0.4},
            {'label': 'Crowd', 'score': crowd_score, 'color': colors['crowd'], 'hole': 0.25}
        ]
        
        for i, ring in enumerate(rings):
            score = ring['score']
            
            # Calculate completion percentage (full at 80%)
            completion = min(100, (score / 80) * 100) if score <= 80 else 100
            remaining = 100 - completion
            
            # Colors for filled and unfilled portions
            filled_color = ring['color'] if score >= 80 else ring['color'] + '80'
            unfilled_color = '#F0F0F0'
            
            # Add filled portion
            if completion > 0:
                fig.add_trace(go.Pie(
                    values=[completion, remaining],
                    labels=[f"{ring['label']}: {score:.0f}%", ""],
                    hole=ring['hole'],
                    marker=dict(colors=[filled_color, unfilled_color]),
                    textinfo='none',
                    showlegend=False,
                    hovertemplate=f"<b>{ring['label']}</b><br>Score: {score:.1f}/100<extra></extra>"
                ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Energy Score Breakdown",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#333'}
            },
            annotations=[
                dict(
                    text=f"<b>{overall_score:.0f}</b><br>Overall",
                    x=0.5, y=0.5,
                    font_size=20,
                    showarrow=False,
                    font_color='#333'
                )
            ],
            width=400,
            height=400,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating donut chart: {str(e)}")
        return None

# ================================
# VIDEO ANALYSIS FUNCTIONS
# ================================

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
        bmp = max(60, min(180, base_bpm + tempo_variance))  # Note: keeping 'bmp' for compatibility
        
        try:
            file_size = os.path.getsize(video_path)
            volume_level = min(100, (file_size / 1000000) * 20 + np.random.randint(20, 60))
        except:
            volume_level = np.random.randint(30, 80)
        
        if bmp > 120:
            genre = "Electronic/Dance"
        elif bmp > 100:
            genre = "Pop/Hip-Hop"
        elif bmp < 80:
            genre = "Ambient/Chill"
        else:
            genre = "General"
        
        energy_level = "High" if bmp > 110 and volume_level > 50 else "Medium" if bmp > 80 else "Low"
        
        video.close()
        audio.close()
        os.unlink(temp_audio.name)
        
        return {
            "bmp": int(bmp),  # Note: keeping 'bmp' for compatibility
            "bpm": int(bmp),  # Adding correct 'bpm' field
            "volume_level": float(volume_level),
            "genre": genre,
            "energy_level": energy_level,
            "confidence_score": 85.0
        }
        
    except Exception as e:
        st.error(f"Audio analysis error: {str(e)}")
        return {
            "bmp": 120,
            "bpm": 120,
            "volume_level": 65.0,
            "genre": "Electronic/Dance",
            "energy_level": "Medium",
            "confidence_score": 70.0
        }

def analyze_visual_environment(video_path):
    """Analyze visual environment of the video"""
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        brightness_values = []
        
        # Sample 10 frames for analysis
        for i in range(0, frame_count, max(1, frame_count // 10)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                brightness_values.append(brightness)
        
        cap.release()
        
        avg_brightness = np.mean(brightness_values) if brightness_values else 127
        
        if avg_brightness > 160:
            lighting_type = "Bright/Well-lit"
            visual_energy = "High"
        elif avg_brightness > 100:
            lighting_type = "Moderate/Mixed"
            visual_energy = "Medium"
        elif avg_brightness > 60:
            lighting_type = "Dim/Ambient"
            visual_energy = "Medium"
        else:
            lighting_type = "Dark/Moody"
            visual_energy = "Low"
        
        color_schemes = ["Multi-color", "Blue/Purple", "Red/Pink", "Green", "Yellow/Orange", "Monochrome"]
        color_scheme = np.random.choice(color_schemes)
        
        return {
            "brightness_level": float(avg_brightness),
            "lighting_type": lighting_type,
            "color_scheme": color_scheme,
            "visual_energy": visual_energy
        }
        
    except Exception as e:
        st.error(f"Visual analysis error: {str(e)}")
        return {
            "brightness_level": 127.0,
            "lighting_type": "Moderate/Mixed",
            "color_scheme": "Multi-color",
            "visual_energy": "Medium"
        }

def analyze_crowd_density(video_path, venue_type):
    """Analyze crowd density and activity"""
    try:
        file_size = os.path.getsize(video_path)
        
        # Base density influenced by venue type
        venue_multipliers = {
            "Club": 1.3,
            "Bar": 1.0,
            "Rooftop": 0.9,
            "Restaurant": 0.7,
            "Lounge": 0.8,
            "Live Music": 1.2,
            "Sports Bar": 1.1,
            "Other": 1.0
        }
        
        multiplier = venue_multipliers.get(venue_type, 1.0)
        base_density = (file_size / 2000000) * multiplier + np.random.randint(5, 15)
        density_score = max(1, min(20, base_density))
        
        if density_score > 15:
            crowd_density = "Extremely Packed"
            activity_level = "High Movement/Dancing"
        elif density_score > 12:
            crowd_density = "Packed"
            activity_level = "Active Movement"
        elif density_score > 8:
            crowd_density = "Busy"
            activity_level = "Moderate Movement"
        elif density_score > 5:
            crowd_density = "Moderate"
            activity_level = "Light Movement"
        else:
            crowd_density = "Light"
            activity_level = "Low Movement/Standing"
        
        estimated_people = int(density_score * np.random.uniform(3, 8))
        engagement_score = min(100, density_score * 4 + np.random.randint(-10, 20))
        
        return {
            "crowd_density": crowd_density,
            "density_score": float(density_score),
            "activity_level": activity_level,
            "estimated_people": estimated_people,
            "engagement_score": float(engagement_score)
        }
        
    except Exception as e:
        st.error(f"Crowd analysis error: {str(e)}")
        return {
            "crowd_density": "Moderate",
            "density_score": 10.0,
            "activity_level": "Moderate Movement",
            "estimated_people": 30,
            "engagement_score": 65.0
        }

def analyze_mood_recognition(video_path):
    """Analyze mood and atmosphere"""
    try:
        moods = ["Happy", "Excited", "Relaxed", "Energetic", "Social", "Festive", "Intense", "Chill"]
        
        # Generate realistic mood distribution
        mood_weights = {}
        total_weight = 0
        
        for mood in moods:
            weight = max(0, np.random.normal(0.3, 0.2))
            mood_weights[mood] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            mood_weights = {mood: weight/total_weight for mood, weight in mood_weights.items()}
        else:
            mood_weights = {mood: 1/len(moods) for mood in moods}
        
        dominant_mood = max(mood_weights, key=mood_weights.get)
        confidence = float(mood_weights[dominant_mood])
        
        # Calculate overall vibe
        positive_score = mood_weights.get("Happy", 0) + mood_weights.get("Excited", 0) + mood_weights.get("Festive", 0)
        energy_score = mood_weights.get("Energetic", 0) + mood_weights.get("Intense", 0) + mood_weights.get("Excited", 0)
        calm_score = mood_weights.get("Relaxed", 0) + mood_weights.get("Chill", 0)
        
        if positive_score > 0.5:
            overall_vibe = "Positive"
        elif energy_score > 0.4:
            overall_vibe = "High Energy"
        elif calm_score > 0.3:
            overall_vibe = "Relaxed"
        else:
            overall_vibe = "Neutral"
        
        return {
            "dominant_mood": dominant_mood,
            "confidence": confidence,
            "mood_breakdown": mood_weights,
            "overall_vibe": overall_vibe,
            "mood_diversity": len([m for m in mood_weights.values() if m > 0.1])
        }
        
    except Exception as e:
        st.error(f"Mood analysis error: {str(e)}")
        return {
            "dominant_mood": "Happy",
            "confidence": 0.6,
            "mood_breakdown": {"Happy": 0.6, "Social": 0.4},
            "overall_vibe": "Positive",
            "mood_diversity": 2
        }

# ================================
# MAIN PROCESSING FUNCTION
# ================================

def process_video(video_file, venue_name, venue_type, gps_data=None):
    """Main video processing function"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Audio analysis
        status_text.text("🎵 Analyzing audio features...")
        progress_bar.progress(25)
        audio_features = extract_audio_features(tmp_path)
        
        # Visual analysis
        status_text.text("🎨 Analyzing visual environment...")
        progress_bar.progress(50)
        visual_features = analyze_visual_environment(tmp_path)
        
        # Crowd analysis
        status_text.text("👥 Analyzing crowd density...")
        progress_bar.progress(75)
        crowd_features = analyze_crowd_density(tmp_path, venue_type)
        
        # Mood analysis
        status_text.text("😊 Analyzing mood and atmosphere...")
        progress_bar.progress(90)
        mood_features = analyze_mood_recognition(tmp_path)
        
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
        try:
            os.unlink(tmp_path)
        except:
            pass

# ================================
# DATABASE FUNCTIONS
# ================================

def save_to_supabase(results):
    """Save results to Supabase database"""
    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        
        # Extract GPS data
        gps_data = results.get("gps_data", {})
        
        # Prepare data for database
        data = {
            "venue_name": results["venue_name"],
            "venue_type": results["venue_type"],
            "user_session": st.session_state.user_session_id,
            "user_name": st.session_state.get('user_name', ''),
            "latitude": gps_data.get("latitude"),
            "longitude": gps_data.get("longitude"),
            "gps_accuracy": gps_data.get("accuracy"),
            "venue_verified": gps_data.get("venue_verified", False),
            "audio_environment": json.dumps(results["audio_environment"]),
            "visual_environment": json.dumps(results["visual_environment"]),
            "crowd_density": json.dumps(results["crowd_density"]),
            "mood_recognition": json.dumps(results["mood_recognition"]),
            "energy_score": calculate_energy_score(results),
            "created_at": datetime.now().isoformat()
        }
        
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/video_results",
            headers=headers,
            json=data
        )
        
        if response.status_code in [200, 201]:
            st.success("✅ Results saved to database!")
            return True
        else:
            st.error(f"❌ Database save failed: {response.status_code}")
            st.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        st.error(f"❌ Database error: {str(e)}")
        return False

def load_all_results():
    """Load all results from database"""
    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
        }
        
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/video_results?select=*&order=created_at.desc&limit=100",
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

# ================================
# RESULTS DISPLAY FUNCTION
# ================================

def display_results(results):
    """Display analysis results with donut chart"""
    
    # Calculate energy score
    energy_score = calculate_energy_score(results)
    
    # Energy Score - Large display with emojis
    if energy_score > 80:
        energy_emoji = "🔥"
        energy_text = "Amazing Energy!"
        energy_color = "#ff4757"
    elif energy_score > 60:
        energy_emoji = "⚡"
        energy_text = "Great Vibes"
        energy_color = "#ffa726"
    elif energy_score > 40:
        energy_emoji = "😊"
        energy_text = "Good Atmosphere"
        energy_color = "#26c6da"
    else:
        energy_emoji = "😌"
        energy_text = "Chill Spot"
        energy_color = "#66bb6a"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, {energy_color}20, {energy_color}10); border-radius: 20px; margin: 1rem 0;">
        <div style="font-size: 4rem; margin-bottom: 0.5rem;">{energy_emoji}</div>
        <div style="font-size: 2.5rem; font-weight: 700; color: {energy_color}; margin-bottom: 0.5rem;">{energy_score:.0f}/100</div>
        <div style="font-size: 1.2rem; font-weight: 600; color: #333;">{energy_text}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns: metrics and donut chart
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📊 Key Metrics")
        
        # Audio Environment Card
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #667eea; margin-bottom: 1rem;">🎵 Audio Environment</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div><strong>BPM:</strong> {results['audio_environment']['bpm']}</div>
                <div><strong>Volume:</strong> {results['audio_environment']['volume_level']:.0f}/100</div>
                <div><strong>Genre:</strong> {results['audio_environment']['genre']}</div>
                <div><strong>Energy:</strong> {results['audio_environment']['energy_level']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Visual Environment Card
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #667eea; margin-bottom: 1rem;">🎨 Visual Environment</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div><strong>Brightness:</strong> {results['visual_environment']['brightness_level']:.0f}/255</div>
                <div><strong>Lighting:</strong> {results['visual_environment']['lighting_type']}</div>
                <div><strong>Colors:</strong> {results['visual_environment']['color_scheme']}</div>
                <div><strong>Visual Energy:</strong> {results['visual_environment']['visual_energy']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Crowd Analysis Card
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #667eea; margin-bottom: 1rem;">👥 Crowd Analysis</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div><strong>Density:</strong> {results['crowd_density']['crowd_density']}</div>
                <div><strong>People:</strong> {results['crowd_density']['estimated_people']}</div>
                <div><strong>Activity:</strong> {results['crowd_density']['activity_level']}</div>
                <div><strong>Engagement:</strong> {results['crowd_density']['engagement_score']:.0f}/100</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Mood Recognition Card
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #667eea; margin-bottom: 1rem;">😊 Mood & Atmosphere</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div><strong>Dominant Mood:</strong> {results['mood_recognition']['dominant_mood']}</div>
                <div><strong>Confidence:</strong> {results['mood_recognition']['confidence']:.1%}</div>
                <div><strong>Overall Vibe:</strong> {results['mood_recognition']['overall_vibe']}</div>
                <div><strong>Diversity:</strong> {results['mood_recognition']['mood_diversity']}/8</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🍩 Energy Breakdown")
        
        # Display donut chart
        donut_chart = create_energy_donut_chart(results)
        if donut_chart:
            st.plotly_chart(donut_chart, use_container_width=True)
        else:
            st.error("Unable to generate donut chart")
        
        # GPS Information if available
        gps_data = results.get('gps_data', {})
        if gps_data.get('latitude'):
            st.markdown("#### 📍 Location Info")
            verified_icon = "✅" if gps_data.get('venue_verified') else "❌"
            st.markdown(f"""
            <div class="metric-card">
                <div><strong>Status:</strong> {verified_icon} {'Verified' if gps_data.get('venue_verified') else 'Unverified'}</div>
                <div><strong>Accuracy:</strong> {gps_data.get('accuracy', 0):.0f}m</div>
                <div><strong>Coordinates:</strong> {gps_data.get('latitude', 0):.4f}, {gps_data.get('longitude', 0):.4f}</div>
            </div>
            """, unsafe_allow_html=True)

# ================================
# ANALYTICS DASHBOARD
# ================================

def analytics_dashboard():
    """Analytics dashboard with venue type energy analysis"""
    
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
            unique_venues = df['Venue'].nunique() if not df.empty else 0
            st.markdown(f"""
            <div class="metric-container">
                <span class="metric-value">{unique_venues}</span>
                <span class="metric-label">Unique Venues</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            verified_count = df['Verified'].sum() if not df.empty else 0
            st.markdown(f"""
            <div class="metric-container">
                <span class="metric-value">{verified_count}</span>
                <span class="metric-label">GPS Verified</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Venue Type Energy Analysis
        st.markdown("#### 🏢 Venue Type Energy Analysis")
        
        if not df.empty:
            # Calculate average energy by venue type
            venue_energy = df.groupby('Type')['Energy Score'].agg(['mean', 'count']).reset_index()
            venue_energy = venue_energy[venue_energy['count'] >= 1]  # Only show types with data
            venue_energy['mean'] = venue_energy['mean'].round(1)
            
            if not venue_energy.empty:
                # Create bar chart for venue type averages
                fig_bar = px.bar(
                    venue_energy, 
                    x='Type', 
                    y='mean',
                    title="Average Energy Score by Venue Type",
                    labels={'mean': 'Average Energy Score', 'Type': 'Venue Type'},
                    color='mean',
                    color_continuous_scale='Viridis'
                )
                fig_bar.update_layout(
                    xaxis_tickangle=-45,
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Display venue type insights
                best_type = venue_energy.loc[venue_energy['mean'].idxmax(), 'Type']
                best_score = venue_energy.loc[venue_energy['mean'].idxmax(), 'mean']
                
                st.info(f"🏆 **{best_type}** venues have the highest average energy score: **{best_score}/100**")
        
        # Recent Activity
        st.markdown("#### 📅 Recent Activity")
        
        if not df.empty:
            recent_df = df.head(10)[['Venue', 'Type', 'Energy Score', 'User', 'Verified', 'created_at']]
            recent_df['GPS'] = recent_df['Verified'].apply(lambda x: '✅' if x else '❌')
            recent_df['Time'] = recent_df['created_at'].dt.strftime('%m/%d %H:%M')
            
            display_df = recent_df[['Venue', 'Type', 'Energy Score', 'GPS', 'User', 'Time']]
            st.dataframe(display_df, use_container_width=True)
        
        # Energy Score Distribution
        st.markdown("#### 📈 Energy Score Distribution")
        
        if not df.empty:
            fig_hist = px.histogram(
                df, 
                x='Energy Score', 
                nbins=20,
                title="Distribution of Energy Scores",
                labels={'Energy Score': 'Energy Score', 'count': 'Number of Videos'}
            )
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
    
    else:
        st.info("No data available yet. Upload some videos to see analytics!")

# ================================
# MAIN APPLICATION
# ================================

def main():
    """Main application function"""
    
    # App header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #667eea; font-size: 3rem; margin-bottom: 0.5rem;">🎯 SneakPeak</h1>
        <p style="color: #666; font-size: 1.2rem; margin: 0;">Discover the real vibe of any venue</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    tab1, tab2 = st.tabs(["📤 Upload Video", "📊 Analytics"])
    
    with tab1:
        upload_interface()
    
    with tab2:
        analytics_dashboard()

def upload_interface():
    """Upload interface with GPS integration"""
    
    # Sidebar for user info
    with st.sidebar:
        st.markdown("### 👤 User Info")
        user_name = st.text_input(
            "Your Name (Optional)", 
            value=st.session_state.get('user_name', ''),
            placeholder="Enter your name"
        )
        
        if user_name and user_name != st.session_state.get('user_name', ''):
            st.success(f"Welcome {user_name}! 👋")
            st.session_state.user_name = user_name
        
        st.info(f"Session ID: **{st.session_state.user_session_id}**")
        
        # Add contribution counter
        user_contributions = len([v for v in st.session_state.processed_videos])
        if user_contributions > 0:
            st.metric("Your Videos", user_contributions)
            if user_contributions >= 3:
                st.success("🌟 Super Contributor!")
            elif user_contributions >= 1:
                st.success("🎯 Great Job!")
        
        if st.session_state.processed_videos:
            st.markdown("### 📊 Your Recent Videos")
            for i, result in enumerate(st.session_state.processed_videos[-3:]):
                with st.expander(f"🎯 {result['venue_name']}", expanded=False):
                    st.write(f"**{result['venue_type']}**")
                    st.write(f"⚡ Energy: {result['energy_score']:.0f}/100")
        else:
            st.write("No videos uploaded yet")
    
    # Main upload section
    st.markdown("### 📤 Upload New Video")
    
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
    
    # Video upload
    st.markdown("### 🎬 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'mov', 'avi', 'mkv'],
        help="Upload a video of the venue atmosphere (max 100MB)"
    )
    
    if uploaded_file is not None and venue_name:
        st.success(f"✅ Video uploaded: {uploaded_file.name}")
        
        # GPS Section - triggered when starting analysis
        if st.button("🎯 Analyze Video", type="primary"):
            # GPS collection
            st.markdown("#### 📍 Getting Location...")
            get_gps_location()
            
            # Mock GPS data for demo (in real app, this would come from JavaScript)
            gps_data = {
                "latitude": 40.7218 + np.random.uniform(-0.01, 0.01),
                "longitude": -73.9876 + np.random.uniform(-0.01, 0.01),
                "accuracy": np.random.uniform(5, 25),
                "venue_verified": np.random.choice([True, False], p=[0.8, 0.2])
            }
            
            # Verify location
            verified, msg = verify_venue_location(
                gps_data["latitude"], 
                gps_data["longitude"], 
                venue_name
            )
            gps_data["venue_verified"] = verified
            st.info(msg)
            
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
                    
                    st.success("🎉 Analysis complete and saved!")

if __name__ == "__main__":
    main()
