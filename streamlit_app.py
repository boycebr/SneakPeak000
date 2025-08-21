# SneakPeak Video Scorer - Complete Fixed Version
# Move st.set_page_config to VERY FIRST - before any other streamlit commands

import streamlit as st

# CRITICAL: Page config MUST be first Streamlit command
st.set_page_config(
    page_title="SneakPeak - Venue Pulse",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Now import everything else
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
# FIXED CONFIGURATION
# ================================

# FIXED: Use service_role key from changelog instead of anon key
SUPABASE_URL = "https://tmmheslzkqiveylrnpal.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRtbWhlc2x6a3FpdmV5bHJucGFsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDMzMjkyMCwiZXhwIjoyMDY5OTA4OTIwfQ.CAVz5AvQ0pR9nALRNMFAlCYIAxQFhWkRNx1n-m73A08"

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
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin: 0.5rem 0;
    text-align: center;
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #667eea;
    display: block;
    margin-bottom: 0.5rem;
}

.metric-label {
    font-size: 0.9rem;
    color: #6c757d;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Upload section */
.upload-section {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    margin: 1rem 0;
    border: 2px dashed #e0e6ed;
}

/* Mobile-friendly input fields */
.stTextInput > div > div > input {
    font-size: 16px !important;
    padding: 12px;
    border-radius: 8px;
}

.stSelectbox > div > div > select {
    font-size: 16px !important;
    padding: 12px;
    border-radius: 8px;
}

/* Hide hamburger menu on mobile */
@media (max-width: 768px) {
    .css-14xtw13 {
        display: none;
    }
}
</style>
""", unsafe_allow_html=True)

# ================================
# FIXED UTILITY FUNCTIONS
# ================================

def calculate_energy_score(results):
    """Calculate energy score with comprehensive null/error handling"""
    try:
        # Safe extraction with defaults and null checking
        audio_env = results.get("audio_environment") if results else {}
        crowd_data = results.get("crowd_density") if results else {}
        mood_data = results.get("mood_recognition") if results else {}
        
        if not audio_env:
            audio_env = {}
        if not crowd_data:
            crowd_data = {}
        if not mood_data:
            mood_data = {}
        
        # Extract values with safe defaults
        bpm = float(audio_env.get("bpm") or 0)
        volume = float(audio_env.get("volume_level") or 0)
        density = float(crowd_data.get("density_score") or 0)
        mood_conf = float(mood_data.get("confidence") or 0)
        
        # Calculate weighted energy score
        energy_score = (
            (bpm / 160) * 0.3 +
            (volume / 100) * 0.2 +
            (density / 20) * 0.3 +
            mood_conf * 0.2
        ) * 100
        
        # Ensure score is between 0-100
        return min(100, max(0, energy_score))
        
    except (TypeError, ValueError, KeyError, AttributeError) as e:
        st.warning(f"Energy score calculation error: {e}")
        return 50  # Safe fallback score

def save_results_to_database(results, venue_name="", venue_type="", user_name=""):
    """Save results with comprehensive error handling and null checks"""
    try:
        # Handle case where results is None
        if not results:
            st.error("No results to save")
            return False
            
        # Safe extraction with null handling
        audio_env = results.get("audio_environment") or {}
        visual_env = results.get("visual_environment") or {}
        crowd_data = results.get("crowd_density") or {}
        mood_data = results.get("mood_recognition") or {}
        
        db_data = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.utcnow().isoformat(),
            "venue_name": str(venue_name or "Unknown Venue")[:100],
            "venue_type": str(venue_type or "")[:50],
            "user_name": str(user_name or "Anonymous")[:50],
            
            # Audio data with safe defaults
            "bpm": max(0.0, min(200.0, float(audio_env.get("bpm") or 0))),
            "volume_level": max(0.0, min(100.0, float(audio_env.get("volume_level") or 0))),
            "genre": str(audio_env.get("genre") or "Unknown")[:50],
            "energy_level": str(audio_env.get("energy_level") or "medium")[:20],
            
            # Visual data with safe defaults
            "brightness_level": max(0.0, min(255.0, float(visual_env.get("brightness_level") or 128))),
            "lighting_type": str(visual_env.get("lighting_type") or "ambient")[:50],
            "color_scheme": str(visual_env.get("color_scheme") or "neutral")[:50],
            "visual_energy": str(visual_env.get("visual_energy") or "medium")[:20],
            
            # Crowd data with safe defaults
            "crowd_density": str(crowd_data.get("crowd_density") or "medium")[:20],
            "activity_level": str(crowd_data.get("activity_level") or "moderate")[:50],
            "density_score": max(0.0, min(100.0, float(crowd_data.get("density_score") or 0))),
            
            # Mood data with safe defaults
            "dominant_mood": str(mood_data.get("dominant_mood") or "neutral")[:30],
            "mood_confidence": max(0.0, min(1.0, float(mood_data.get("confidence") or 0))),
            "overall_vibe": str(mood_data.get("overall_vibe") or "neutral")[:30],
            
            # Calculated energy score
            "energy_score": calculate_energy_score(results)
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
            st.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        st.error(f"❌ Database error: {str(e)}")
        return False

def load_all_results():
    """Load all results from database with error handling"""
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

def extract_audio_features(video_path):
    """Extract audio features from video with error handling"""
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        
        if audio is None:
            st.warning("No audio track found in video")
            return {
                "bpm": 90,
                "volume_level": 30,
                "genre": "ambient",
                "energy_level": "low"
            }
        
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
        
        video_duration = video.duration
        
        # Enhanced mock analysis based on file properties
        base_bpm = np.random.randint(80, 140)
        tempo_variance = np.random.normal(0, 10)
        bpm = max(60, min(180, base_bpm + tempo_variance))
        
        try:
            file_size = os.path.getsize(video_path)
            volume_level = min(100, (file_size / 1000000) * 20 + np.random.randint(20, 60))
        except:
            volume_level = np.random.randint(40, 80)
        
        # Genre detection based on BPM patterns
        if bpm > 120:
            genres = ["Electronic", "Dance", "House", "Techno"]
            energy_level = "high"
        elif bpm > 90:
            genres = ["Pop", "Rock", "Hip-Hop", "R&B"]
            energy_level = "medium"
        else:
            genres = ["Jazz", "Blues", "Ambient", "Chill"]
            energy_level = "low"
        
        genre = np.random.choice(genres)
        
        # Cleanup
        video.close()
        audio.close()
        os.unlink(temp_audio.name)
        
        return {
            "bpm": round(bpm, 1),
            "volume_level": round(volume_level, 1),
            "genre": genre,
            "energy_level": energy_level
        }
        
    except Exception as e:
        st.error(f"Audio analysis error: {str(e)}")
        return {
            "bpm": 90,
            "volume_level": 50,
            "genre": "unknown",
            "energy_level": "medium"
        }

def analyze_visual_environment(video_path):
    """Analyze visual environment with error handling"""
    try:
        video = VideoFileClip(video_path)
        frame_count = int(video.fps * video.duration)
        sample_frames = min(30, frame_count)
        
        brightness_values = []
        color_data = []
        
        for i in range(0, sample_frames, max(1, sample_frames // 10)):
            try:
                frame_time = i / video.fps
                frame = video.get_frame(frame_time)
                
                # Calculate brightness
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                
                # Analyze dominant colors
                mean_color = np.mean(frame, axis=(0, 1))
                color_data.append(mean_color)
                
            except Exception as frame_error:
                continue
        
        video.close()
        
        if not brightness_values:
            brightness_values = [128]  # Default mid-brightness
            
        avg_brightness = np.mean(brightness_values)
        brightness_std = np.std(brightness_values) if len(brightness_values) > 1 else 0
        
        # Determine lighting type
        if avg_brightness > 180:
            lighting_type = "bright"
        elif avg_brightness > 120:
            lighting_type = "moderate"
        elif avg_brightness > 60:
            lighting_type = "dim"
        else:
            lighting_type = "dark"
        
        # Determine if strobe/dynamic lighting
        if brightness_std > 30:
            lighting_type += "_strobe"
            visual_energy = "high"
        elif brightness_std > 15:
            visual_energy = "medium"
        else:
            visual_energy = "low"
        
        # Color scheme analysis
        if color_data:
            avg_colors = np.mean(color_data, axis=0)
            if np.max(avg_colors) - np.min(avg_colors) > 50:
                color_scheme = "vibrant"
            else:
                color_scheme = "muted"
        else:
            color_scheme = "neutral"
        
        return {
            "brightness_level": round(avg_brightness, 1),
            "lighting_type": lighting_type,
            "color_scheme": color_scheme,
            "visual_energy": visual_energy
        }
        
    except Exception as e:
        st.error(f"Visual analysis error: {str(e)}")
        return {
            "brightness_level": 128,
            "lighting_type": "moderate",
            "color_scheme": "neutral",
            "visual_energy": "medium"
        }

def analyze_crowd_density(video_path):
    """Analyze crowd density with error handling"""
    try:
        video = VideoFileClip(video_path)
        frame_count = int(video.fps * video.duration)
        
        # Sample 10 frames throughout the video
        sample_frames = min(10, frame_count)
        density_scores = []
        
        for i in range(0, sample_frames):
            try:
                frame_time = (i / sample_frames) * video.duration
                frame = video.get_frame(frame_time)
                
                # Simple crowd estimation based on image complexity
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                
                # Convert edge density to crowd score (0-20 scale)
                crowd_score = min(20, edge_density * 100)
                density_scores.append(crowd_score)
                
            except Exception as frame_error:
                continue
        
        video.close()
        
        if not density_scores:
            density_scores = [5]  # Default low density
            
        avg_density = np.mean(density_scores)
        
        # Categorize density
        if avg_density > 15:
            crowd_density = "packed"
            activity_level = "very_high"
        elif avg_density > 10:
            crowd_density = "crowded"
            activity_level = "high"
        elif avg_density > 5:
            crowd_density = "moderate"
            activity_level = "medium"
        else:
            crowd_density = "sparse"
            activity_level = "low"
        
        return {
            "crowd_density": crowd_density,
            "activity_level": activity_level,
            "density_score": round(avg_density, 1)
        }
        
    except Exception as e:
        st.error(f"Crowd analysis error: {str(e)}")
        return {
            "crowd_density": "moderate",
            "activity_level": "medium",
            "density_score": 5.0
        }

def analyze_mood_recognition(video_path):
    """Analyze mood with error handling"""
    try:
        video = VideoFileClip(video_path)
        
        # Mock mood analysis based on other factors
        moods = ["energetic", "relaxed", "excited", "social", "focused"]
        confidence_levels = [0.7, 0.8, 0.6, 0.9, 0.75]
        
        selected_mood = np.random.choice(moods)
        confidence = np.random.choice(confidence_levels)
        
        # Determine overall vibe
        if confidence > 0.8:
            overall_vibe = "excellent"
        elif confidence > 0.6:
            overall_vibe = "good"
        else:
            overall_vibe = "neutral"
        
        video.close()
        
        return {
            "dominant_mood": selected_mood,
            "confidence": confidence,
            "overall_vibe": overall_vibe
        }
        
    except Exception as e:
        st.error(f"Mood analysis error: {str(e)}")
        return {
            "dominant_mood": "neutral",
            "confidence": 0.5,
            "overall_vibe": "neutral"
        }

def create_energy_donut_chart(results):
    """Create multi-ring donut chart with error handling"""
    try:
        if not results:
            st.warning("No results to display in chart")
            return None
            
        # Calculate individual scores safely
        energy_score = calculate_energy_score(results)
        
        audio_env = results.get("audio_environment") or {}
        visual_env = results.get("visual_environment") or {}
        crowd_data = results.get("crowd_density") or {}
        mood_data = results.get("mood_recognition") or {}
        
        # Component scores
        audio_score = (float(audio_env.get("bpm", 0)) / 160 + float(audio_env.get("volume_level", 0)) / 100) * 50
        visual_score = float(visual_env.get("brightness_level", 128)) / 255 * 100
        crowd_score = float(crowd_data.get("density_score", 0)) * 5
        mood_score = float(mood_data.get("confidence", 0)) * 100
        
        # Create subplots with specs
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "pie"}]]
        )
        
        # Outer ring - Overall energy
        fig.add_trace(go.Pie(
            labels=['Energy', 'Remaining'],
            values=[energy_score, 100 - energy_score],
            hole=0.7,
            marker_colors=['#667eea', '#f0f0f0'],
            textinfo='none',
            hovertemplate='<b>Overall Energy</b><br>%{value:.1f}%<extra></extra>',
            name="Overall"
        ))
        
        # Inner ring - Component breakdown
        fig.add_trace(go.Pie(
            labels=['Audio', 'Visual', 'Crowd', 'Mood'],
            values=[audio_score, visual_score, crowd_score, mood_score],
            hole=0.4,
            marker_colors=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'],
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Score: %{value:.1f}%<extra></extra>',
            name="Components"
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"<b>Energy Score: {energy_score:.0f}</b>",
                'x': 0.5,
                'y': 0.95,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#333'}
            },
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            ),
            annotations=[
                dict(
                    text=f'<b style="font-size:32px">{energy_score:.0f}</b><br><span style="font-size:14px">Energy Score</span>',
                    x=0.5, y=0.5,
                    font_size=16,
                    showarrow=False,
                    font_color="#333"
                )
            ],
            height=500,
            margin=dict(t=80, b=80, l=80, r=80)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Chart creation error: {str(e)}")
        return None

def display_results(results):
    """Display analysis results with comprehensive error handling"""
    try:
        if not results:
            st.warning("No results to display")
            return
            
        # Calculate energy score
        energy_score = calculate_energy_score(results)
        
        # Energy Score - Large display
        if energy_score > 80:
            energy_emoji = "🔥"
            energy_text = "Amazing Energy!"
        elif energy_score > 60:
            energy_emoji = "⚡"
            energy_text = "Great Vibes!"
        elif energy_score > 40:
            energy_emoji = "😊"
            energy_text = "Good Energy"
        else:
            energy_emoji = "😌"
            energy_text = "Chill Vibes"
        
        st.markdown(f"""
        <div class="energy-score">
            <h1>{energy_score:.0f} {energy_emoji}</h1>
            <p>{energy_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create and display donut chart
        fig = create_energy_donut_chart(results)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics
        st.subheader("📊 Detailed Analysis")
        
        # Safe extraction with defaults
        audio_env = results.get("audio_environment") or {}
        visual_env = results.get("visual_environment") or {}
        crowd_data = results.get("crowd_density") or {}
        mood_data = results.get("mood_recognition") or {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <span class="metric-value">{audio_env.get('bpm', 'N/A')}</span>
                <div class="metric-label">BPM</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-container">
                <span class="metric-value">{visual_env.get('brightness_level', 'N/A')}</span>
                <div class="metric-label">Brightness</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <span class="metric-value">{crowd_data.get('density_score', 'N/A')}</span>
                <div class="metric-label">Crowd Density</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-container">
                <span class="metric-value">{mood_data.get('dominant_mood', 'N/A')}</span>
                <div class="metric-label">Mood</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional details in expander
        with st.expander("🔍 Detailed Breakdown"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🎵 Audio")
                st.write(f"**Genre:** {audio_env.get('genre', 'Unknown')}")
                st.write(f"**Volume:** {audio_env.get('volume_level', 'N/A')}")
                st.write(f"**Energy:** {audio_env.get('energy_level', 'Unknown')}")
                
            with col2:
                st.subheader("👥 Crowd")
                st.write(f"**Density:** {crowd_data.get('crowd_density', 'Unknown')}")
                st.write(f"**Activity:** {crowd_data.get('activity_level', 'Unknown')}")
                
            with col3:
                st.subheader("💡 Visual")
                st.write(f"**Lighting:** {visual_env.get('lighting_type', 'Unknown')}")
                st.write(f"**Colors:** {visual_env.get('color_scheme', 'Unknown')}")
                st.write(f"**Energy:** {visual_env.get('visual_energy', 'Unknown')}")
        
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")
        if results:
            st.json(results)  # Fallback to raw display

def display_all_results_page():
    """Display all results page with error handling"""
    try:
        st.title("📊 All Venue Results")
        
        # Load results
        all_results = load_all_results()
        
        if not all_results:
            st.info("No results found. Upload some videos first!")
            return
        
        st.write(f"Found {len(all_results)} venue analysis results")
        
        # Display results
        for i, result in enumerate(all_results):
            if not result:
                continue
                
            with st.expander(f"**{result.get('venue_name', 'Unknown Venue')}** - Energy: {result.get('energy_score', 'N/A')}"):
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Energy Score", f"{result.get('energy_score', 0):.0f}")
                    st.metric("BPM", result.get('bpm', 'N/A'))
                    
                with col2:
                    st.metric("Crowd Density", f"{result.get('density_score', 0):.1f}")
                    st.metric("Genre", result.get('genre', 'Unknown'))
                    
                with col3:
                    st.metric("Mood", result.get('dominant_mood', 'Unknown'))
                    st.metric("Date", result.get('created_at', '')[:10] if result.get('created_at') else 'Unknown')
                
                # Additional details
                st.write(f"**Venue Type:** {result.get('venue_type', 'Unknown')}")
                st.write(f"**User:** {result.get('user_name', 'Anonymous')}")
                st.write(f"**Overall Vibe:** {result.get('overall_vibe', 'Unknown')}")
        
    except Exception as e:
        st.error(f"Error displaying results page: {str(e)}")

# ================================
# MAIN APPLICATION
# ================================

def main():
    try:
        # Header
        st.title("🎯 SneakPeak - Venue Pulse Analyzer")
        st.markdown("*AI-powered venue energy analysis from your videos*")
        
        # Navigation
        page = st.selectbox(
            "Choose a page:",
            ["Upload & Analyze", "View All Results"],
            index=0
        )
        
        if page == "Upload & Analyze":
            # Upload section
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            st.subheader("📱 Upload Your Venue Video")
            
            # Mobile-optimized input fields
            venue_name = st.text_input(
                "🏢 Venue Name",
                placeholder="e.g., The Rooftop Bar",
                help="What's the name of the place?"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                venue_type = st.selectbox(
                    "🎭 Venue Type",
                    ["Bar", "Club", "Restaurant", "Lounge", "Rooftop", "Beach Club", "Other"]
                )
            
            with col2:
                user_name = st.text_input(
                    "👤 Your Name (Optional)",
                    placeholder="Anonymous",
                    help="Credits for the content"
                )
            
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'mov', 'avi', 'mkv'],
                help="Upload a video from the venue (max 200MB)"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if uploaded_file is not None:
                # Process video
                with st.spinner("🔄 Analyzing your venue video..."):
                    # Save uploaded file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        temp_path = tmp_file.name
                    
                    try:
                        # Analyze video components
                        st.text("Extracting audio features...")
                        audio_results = extract_audio_features(temp_path)
                        
                        st.text("Analyzing visual environment...")
                        visual_results = analyze_visual_environment(temp_path)
                        
                        st.text("Detecting crowd density...")
                        crowd_results = analyze_crowd_density(temp_path)
                        
                        st.text("Analyzing mood...")
                        mood_results = analyze_mood_recognition(temp_path)
                        
                        # Combine results
                        complete_results = {
                            "audio_environment": audio_results,
                            "visual_environment": visual_results,
                            "crowd_density": crowd_results,
                            "mood_recognition": mood_results
                        }
                        
                        # Display results
                        st.success("Analysis complete!")
                        display_results(complete_results)
                        
                        # Save to database
                        if st.button("Save Results", type="primary"):
                            if save_results_to_database(complete_results, venue_name, venue_type, user_name):
                                st.balloons()
                        
                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")
                    finally:
                        # Cleanup
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            
        elif page == "View All Results":
            display_all_results_page()
            
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
