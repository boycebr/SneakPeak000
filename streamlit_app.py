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
    layout="wide"
)

# Initialize session state
if 'processed_videos' not in st.session_state:
    st.session_state.processed_videos = []
if 'user_session_id' not in st.session_state:
    st.session_state.user_session_id = str(uuid.uuid4())[:8]

def save_to_supabase(results):
    """Save analysis results to Supabase database"""
    try:
        # Prepare data for database
        db_data = {
            "venue_name": results["venue_name"],
            "venue_type": results["venue_type"],
            "user_session": st.session_state.user_session_id,
            "bpm": results["audio_environment"]["bpm"],
            "volume_level": results["audio_environment"]["volume_level"],
            "genre": results["audio_environment"]["genre"],
            "energy_level": results["audio_environment"]["energy_level"],
            "brightness_level": results["visual_environment"]["brightness_level"],
            "lighting_type": results["visual_environment"]["lighting_type"],
            "color_scheme": results["visual_environment"]["color_scheme"],
            "visual_energy": results["visual_environment"]["visual_energy"],
            "crowd_density": results["crowd_density"]["crowd_density"],
            "activity_level": results["crowd_density"]["activity_level"],
            "density_score": results["crowd_density"]["density_score"],
            "dominant_mood": results["mood_recognition"]["dominant_mood"],
            "mood_confidence": results["mood_recognition"]["confidence"],
            "overall_vibe": results["mood_recognition"]["overall_vibe"],
            "energy_score": calculate_energy_score(results)
        }
        
        # Send to Supabase
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
    energy_score = (
        (results["audio_environment"]["bpm"] / 160) * 0.3 +
        (results["audio_environment"]["volume_level"] / 100) * 0.2 +
        (results["crowd_density"]["density_score"] / 20) * 0.3 +
        results["mood_recognition"]["confidence"] * 0.2
    ) * 100
    
    return min(100, max(0, energy_score))

def extract_audio_features(video_path):
    """Extract audio features from video"""
    try:
        # Extract audio from video
        video = VideoFileClip(video_path)
        audio = video.audio
        
        # Save temp audio file
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
        
        # For demo purposes - simulate audio analysis
        # In production, you'd use librosa or similar
        video_duration = video.duration
        
        # Mock BPM based on video length and random factors
        base_bpm = np.random.randint(80, 140)
        tempo_variance = np.random.normal(0, 10)
        bpm = max(60, min(180, base_bpm + tempo_variance))
        
        # Mock volume based on file size (rough proxy)
        try:
            file_size = os.path.getsize(video_path)
            volume_level = min(100, (file_size / 1000000) * 20 + np.random.randint(20, 60))
        except:
            volume_level = np.random.randint(30, 80)
        
        # Simple genre classification based on BPM
        if bpm > 120:
            genre = "Electronic/Dance"
        elif bpm > 100:
            genre = "Pop/Hip-Hop"
        elif bpm < 80:
            genre = "Ambient/Chill"
        else:
            genre = "General"
        
        # Energy level
        energy_level = "High" if bpm > 110 and volume_level > 50 else "Medium" if bpm > 80 else "Low"
        
        # Cleanup
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
            "volume_level": np.random.randint(40, 90),
            "genre": "Unknown",
            "energy_level": "Medium"
        }

def analyze_visual_environment_simple(video_path):
    """Simplified visual analysis without OpenCV"""
    try:
        # Get basic video info
        video = VideoFileClip(video_path)
        duration = video.duration
        fps = video.fps if video.fps else 30
        
        # Mock analysis based on video properties
        # In production, you'd analyze actual frames
        
        # Simulate brightness based on file size and duration
        file_size = os.path.getsize(video_path)
        brightness_factor = (file_size / duration) / 100000
        brightness = max(20, min(255, brightness_factor + np.random.randint(50, 150)))
        
        # Lighting type based on brightness
        if brightness < 80:
            lighting_type = "Dark/Club Lighting"
        elif brightness < 150:
            lighting_type = "Ambient/Mood Lighting"
        else:
            lighting_type = "Bright/Well-lit"
        
        # Random color scheme for demo
        colors = ["Red-dominant", "Blue-dominant", "Green-dominant", "Purple-dominant", "Multi-color"]
        color_scheme = np.random.choice(colors)
        
        # Visual energy based on fps and duration
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
            "brightness_level": np.random.randint(50, 200),
            "lighting_type": "Ambient/Mood Lighting",
            "color_scheme": "Multi-color",
            "visual_energy": "Medium"
        }

def analyze_crowd_density_simple(video_path):
    """Simplified crowd analysis without OpenCV"""
    try:
        # Get video properties
        video = VideoFileClip(video_path)
        duration = video.duration
        file_size = os.path.getsize(video_path)
        
        # Mock crowd density based on file size and duration
        # Larger files often indicate more movement/activity
        density_factor = file_size / (duration * 1000000)  # MB per second
        
        # Simulate person count
        base_people = max(0, int(density_factor * 10 + np.random.randint(2, 15)))
        
        # Crowd density classification
        if base_people > 15:
            density = "Packed"
        elif base_people > 8:
            density = "Busy"
        elif base_people > 3:
            density = "Moderate"
        else:
            density = "Light"
        
        # Activity level based on file complexity
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
            "density_score": np.random.randint(5, 12)
        }

def mock_mood_recognition(video_path):
    """Mock mood recognition for demo"""
    try:
        # Simulate mood analysis based on other factors
        moods = ["Happy", "Excited", "Relaxed", "Energetic", "Social", "Festive"]
        
        # Weight moods based on simulated crowd and energy
        mood_weights = {
            "Happy": 0.25,
            "Excited": 0.20,
            "Relaxed": 0.15,
            "Energetic": 0.20,
            "Social": 0.15,
            "Festive": 0.05
        }
        
        # Add some randomness
        for mood in mood_weights:
            mood_weights[mood] += np.random.normal(0, 0.1)
            mood_weights[mood] = max(0, min(1, mood_weights[mood]))
        
        # Normalize
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

def process_video(video_file, venue_name, venue_type):
    """Main video processing function"""
    
    # Save uploaded video to temp file
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
        
        # Compile final results
        results = {
            "venue_name": venue_name,
            "venue_type": venue_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "audio_environment": audio_results,
            "visual_environment": visual_results,
            "crowd_density": crowd_results,
            "mood_recognition": mood_results
        }
        
        return results
    
    finally:
        # Cleanup temp file
        try:
            os.unlink(temp_video_path)
        except:
            pass

def display_results(results):
    """Display processing results in a nice format"""
    
    st.success("✅ Video processed successfully!")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("🎵 Audio Environment")
        st.metric("BPM", results["audio_environment"]["bpm"])
        st.metric("Volume Level", f"{results['audio_environment']['volume_level']:.1f}")
        st.write(f"**Genre:** {results['audio_environment']['genre']}")
        st.write(f"**Energy:** {results['audio_environment']['energy_level']}")
    
    with col2:
        st.subheader("🎨 Visual Environment")
        st.metric("Brightness", f"{results['visual_environment']['brightness_level']:.1f}")
        st.write(f"**Lighting:** {results['visual_environment']['lighting_type']}")
        st.write(f"**Colors:** {results['visual_environment']['color_scheme']}")
        st.write(f"**Visual Energy:** {results['visual_environment']['visual_energy']}")
    
    with col3:
        st.subheader("👥 Crowd Density")
        st.metric("Density Score", f"{results['crowd_density']['density_score']:.1f}")
        st.write(f"**Density:** {results['crowd_density']['crowd_density']}")
        st.write(f"**Activity:** {results['crowd_density']['activity_level']}")
    
    with col4:
        st.subheader("😊 Mood Recognition")
        st.metric("Dominant Mood", results["mood_recognition"]["dominant_mood"])
        st.metric("Confidence", f"{results['mood_recognition']['confidence']:.2f}")
        st.write(f"**Overall Vibe:** {results['mood_recognition']['overall_vibe']}")
    
    # Overall venue pulse score
    st.subheader("🎯 Overall Venue Pulse")
    
    # Calculate composite scores
    energy_score = (
        (results["audio_environment"]["bpm"] / 160) * 0.3 +
        (results["audio_environment"]["volume_level"] / 100) * 0.2 +
        (results["crowd_density"]["density_score"] / 20) * 0.3 +
        results["mood_recognition"]["confidence"] * 0.2
    ) * 100
    
    energy_score = min(100, max(0, energy_score))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Energy Score", f"{energy_score:.0f}/100")
    with col2:
        vibe_rating = "🔥 Hot" if energy_score > 75 else "⚡ Good" if energy_score > 50 else "😌 Chill"
        st.metric("Vibe Rating", vibe_rating)
    with col3:
        recommendation = "Perfect for dancing!" if energy_score > 75 else "Good for socializing" if energy_score > 50 else "Relaxed atmosphere"
        st.write(f"**Recommendation:** {recommendation}")
    
    # Mood breakdown chart
    if results["mood_recognition"]["mood_breakdown"]:
        st.subheader("📊 Mood Breakdown")
        mood_data = results["mood_recognition"]["mood_breakdown"]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        moods = list(mood_data.keys())
        scores = list(mood_data.values())
        
        bars = ax.bar(moods, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3'])
        ax.set_ylabel('Confidence Score')
        ax.set_title('Detected Moods in Venue')
        ax.set_ylim(0, max(scores) * 1.2)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.2f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

def main():
    st.title("🎯 SneakPeak Video Scorer")
    st.markdown("Upload venue videos to analyze the vibe and get real-time pulse metrics!")
    
    st.info("🔧 **Demo Mode**: This is a simplified version that works without OpenCV. Results are simulated but follow realistic patterns.")
    
    # Add admin dashboard option
    view_mode = st.sidebar.radio("View Mode", ["Upload Videos", "Admin Dashboard"])
    
    if view_mode == "Admin Dashboard":
        st.subheader("📊 Admin Dashboard - All Results")
        
        if st.button("🔄 Refresh Data"):
            st.rerun()
        
        all_results = load_all_results()
        
        if all_results:
            st.success(f"Found {len(all_results)} total submissions")
            
            # Convert to DataFrame for better display
            df_data = []
            for result in all_results:
                df_data.append({
                    "Date": result.get("created_at", "")[:10],
                    "Venue": result.get("venue_name", ""),
                    "Type": result.get("venue_type", ""),
                    "User": result.get("user_session", "")[:8],
                    "BPM": result.get("bpm", 0),
                    "Volume": result.get("volume_level", 0),
                    "Crowd": result.get("crowd_density", ""),
                    "Mood": result.get("dominant_mood", ""),
                    "Energy Score": round(result.get("energy_score", 0), 1)
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Submissions", len(all_results))
            with col2:
                avg_energy = df["Energy Score"].mean() if not df.empty else 0
                st.metric("Avg Energy Score", f"{avg_energy:.1f}")
            with col3:
                unique_venues = df["Venue"].nunique() if not df.empty else 0
                st.metric("Unique Venues", unique_venues)
            with col4:
                unique_users = df["User"].nunique() if not df.empty else 0
                st.metric("Unique Users", unique_users)
            
            # Charts
            if not df.empty:
                st.subheader("📈 Analytics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Venue type distribution
                    venue_counts = df["Type"].value_counts()
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.pie(venue_counts.values, labels=venue_counts.index, autopct='%1.1f%%')
                    ax.set_title("Venue Types Tested")
                    st.pyplot(fig)
                
                with col2:
                    # Energy score distribution
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(df["Energy Score"], bins=10, alpha=0.7, color='skyblue')
                    ax.set_xlabel("Energy Score")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Energy Score Distribution")
                    st.pyplot(fig)
        
        else:
            st.info("No results found in database yet. Upload some videos first!")
        
        return
    
    # Regular upload interface
    # Sidebar for previous results
    st.sidebar.title("📊 Previous Results")
    if st.session_state.processed_videos:
        for i, result in enumerate(st.session_state.processed_videos[-5:]):  # Show last 5
            with st.sidebar.expander(f"{result['venue_name']} - {result['timestamp'][:10]}"):
                st.write(f"**Type:** {result['venue_type']}")
                st.write(f"**BPM:** {result['audio_environment']['bpm']}")
                st.write(f"**Crowd:** {result['crowd_density']['crowd_density']}")
                st.write(f"**Mood:** {result['mood_recognition']['dominant_mood']}")
    
    # Display user session ID
    st.sidebar.info(f"Your session ID: {st.session_state.user_session_id}")
    
    # Main upload interface
    st.subheader("📤 Upload Video")
    
    col1, col2 = st.columns(2)
    with col1:
        venue_name = st.text_input("Venue Name", placeholder="e.g., The Rooftop Bar")
    with col2:
        venue_type = st.selectbox("Venue Type", 
                                 ["Club", "Bar", "Rooftop", "Restaurant", "Lounge", "Other"])
    
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=['mp4', 'mov', 'avi'],
        help="Upload a 30-60 second video of the venue"
    )
    
    if uploaded_file is not None and venue_name:
        st.video(uploaded_file)
        
        if st.button("🎯 Analyze Video", type="primary"):
            results = process_video(uploaded_file, venue_name, venue_type)
            
            if results:
                display_results(results)
                
                # Save to session state
                st.session_state.processed_videos.append(results)
                
                # Save to Supabase database
                save_to_supabase(results)
                
                # Download results as JSON
                st.download_button(
                    label="📥 Download Results (JSON)",
                    data=json.dumps(results, indent=2),
                    file_name=f"{venue_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    elif uploaded_file is not None and not venue_name:
        st.warning("Please enter a venue name before analyzing.")
    
    # Instructions
    st.subheader("📋 Instructions for Friends")
    st.markdown("""
    **How to submit videos:**
    1. Record 30-60 second video at the venue
    2. Capture the general atmosphere (crowd, lighting, sound)
    3. Upload here with venue name and type
    4. Get instant venue pulse analysis!
    
    **What we're testing:**
    - Audio analysis (BPM, volume, genre detection)
    - Visual environment (lighting, colors, brightness)
    - Crowd density and movement patterns
    - Mood recognition from facial expressions
    
    **Note:** This demo version simulates the analysis algorithms. The production version will use actual computer vision and audio processing.
    """)
    
    # Sample results for demo
    if st.button("🎮 Show Sample Results"):
        sample_results = {
            "venue_name": "Demo Nightclub",
            "venue_type": "Club",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
