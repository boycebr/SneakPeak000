import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime
import pandas as pd
import json
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from moviepy.editor import VideoFileClip
import requests
import base64

# Page config
st.set_page_config(
    page_title="SneakPeak Video Scorer",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
if 'processed_videos' not in st.session_state:
    st.session_state.processed_videos = []

def extract_audio_features(video_path):
    """Extract audio features from video"""
    try:
        # Extract audio from video
        video = VideoFileClip(video_path)
        audio = video.audio
        
        # Save temp audio file
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
        
        # Load audio with librosa
        y, sr = librosa.load(temp_audio.name)
        
        # Calculate features
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)[0]
        avg_volume = np.mean(rms)
        
        # Spectral features for genre detection
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Simple genre classification based on features
        if tempo > 120 and spectral_centroid > 2000:
            genre = "Electronic/Dance"
        elif tempo > 100 and spectral_centroid > 1500:
            genre = "Pop/Hip-Hop"
        elif tempo < 80:
            genre = "Ambient/Chill"
        else:
            genre = "General"
        
        # Cleanup
        os.unlink(temp_audio.name)
        video.close()
        audio.close()
        
        return {
            "bpm": int(tempo),
            "volume_level": float(avg_volume * 100),  # Scale to 0-100
            "genre": genre,
            "energy_level": "High" if tempo > 110 and avg_volume > 0.1 else "Medium" if tempo > 80 else "Low"
        }
    except Exception as e:
        st.error(f"Audio processing error: {str(e)}")
        return {
            "bpm": 0,
            "volume_level": 0,
            "genre": "Unknown",
            "energy_level": "Unknown"
        }

def analyze_visual_environment(video_path):
    """Analyze visual aspects of the video"""
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        brightness_values = []
        color_analysis = []
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = max(1, total_frames // 30)  # Sample every nth frame
        
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                # Calculate brightness
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                brightness_values.append(brightness)
                
                # Color analysis
                mean_colors = np.mean(frame_rgb.reshape(-1, 3), axis=0)
                color_analysis.append(mean_colors)
            
            frame_count += 1
        
        cap.release()
        
        if not brightness_values:
            return {
                "brightness_level": 0,
                "lighting_type": "Unknown",
                "color_scheme": "Unknown",
                "visual_energy": "Unknown"
            }
        
        avg_brightness = np.mean(brightness_values)
        brightness_variance = np.var(brightness_values)
        
        # Determine lighting type
        if avg_brightness < 50:
            lighting_type = "Dark/Club Lighting"
        elif avg_brightness < 120:
            lighting_type = "Ambient/Mood Lighting"
        else:
            lighting_type = "Bright/Well-lit"
        
        # Color scheme analysis
        avg_colors = np.mean(color_analysis, axis=0)
        dominant_color = ["Red", "Green", "Blue"][np.argmax(avg_colors)]
        
        # Visual energy based on brightness variance
        visual_energy = "High" if brightness_variance > 500 else "Medium" if brightness_variance > 200 else "Low"
        
        return {
            "brightness_level": float(avg_brightness),
            "lighting_type": lighting_type,
            "color_scheme": f"{dominant_color}-dominant",
            "visual_energy": visual_energy
        }
    
    except Exception as e:
        st.error(f"Visual analysis error: {str(e)}")
        return {
            "brightness_level": 0,
            "lighting_type": "Unknown",
            "color_scheme": "Unknown",
            "visual_energy": "Unknown"
        }

def analyze_crowd_density(video_path):
    """Analyze crowd density and movement"""
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Load pre-trained person detection model
        # Using Haar cascades as a simple alternative to YOLO for demo
        person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        
        person_counts = []
        movement_values = []
        prev_gray = None
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_rate = max(1, total_frames // 20)
        
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Person detection (simplified)
                persons = person_cascade.detectMultiScale(gray, 1.1, 3)
                person_counts.append(len(persons))
                
                # Movement detection
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowPyrLK(prev_gray, gray, None, None)[0]
                    if flow is not None:
                        movement = np.mean(np.linalg.norm(flow, axis=1)) if len(flow) > 0 else 0
                        movement_values.append(movement)
                
                prev_gray = gray.copy()
            
            frame_count += 1
        
        cap.release()
        
        if not person_counts:
            return {
                "crowd_density": "Unknown",
                "activity_level": "Unknown",
                "density_score": 0
            }
        
        avg_people = np.mean(person_counts) if person_counts else 0
        avg_movement = np.mean(movement_values) if movement_values else 0
        
        # Crowd density classification
        if avg_people > 15:
            density = "Packed"
        elif avg_people > 8:
            density = "Busy"
        elif avg_people > 3:
            density = "Moderate"
        else:
            density = "Light"
        
        # Activity level
        if avg_movement > 5:
            activity = "High Movement/Dancing"
        elif avg_movement > 2:
            activity = "Moderate Movement"
        else:
            activity = "Low Movement/Standing"
        
        return {
            "crowd_density": density,
            "activity_level": activity,
            "density_score": float(avg_people)
        }
    
    except Exception as e:
        st.error(f"Crowd analysis error: {str(e)}")
        return {
            "crowd_density": "Unknown",
            "activity_level": "Unknown",
            "density_score": 0
        }

def mock_mood_recognition(video_path):
    """Mock mood recognition - replace with actual API call"""
    # This is a placeholder for actual mood recognition API
    # In production, you'd use Azure Emotion API or similar
    
    try:
        # For demo purposes, generate realistic mock data
        moods = ["Happy", "Excited", "Relaxed", "Energetic", "Social"]
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        
        mood_scores = {}
        for mood, weight in zip(moods, weights):
            # Add some randomness but keep it realistic
            score = weight + np.random.normal(0, 0.1)
            mood_scores[mood] = max(0, min(1, score))
        
        dominant_mood = max(mood_scores, key=mood_scores.get)
        confidence = mood_scores[dominant_mood]
        
        return {
            "dominant_mood": dominant_mood,
            "confidence": float(confidence),
            "mood_breakdown": mood_scores,
            "overall_vibe": "Positive" if confidence > 0.6 else "Neutral"
        }
    
    except Exception as e:
        st.error(f"Mood analysis error: {str(e)}")
        return {
            "dominant_mood": "Unknown",
            "confidence": 0,
            "mood_breakdown": {},
            "overall_vibe": "Unknown"
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
            visual_results = analyze_visual_environment(temp_video_path)
        
        with st.spinner("👥 Analyzing crowd..."):
            crowd_results = analyze_crowd_density(temp_video_path)
        
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
        os.unlink(temp_video_path)

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

def main():
    st.title("🎯 SneakPeak Video Scorer")
    st.markdown("Upload venue videos to analyze the vibe and get real-time pulse metrics!")
    
    # Sidebar for previous results
    st.sidebar.title("📊 Previous Results")
    if st.session_state.processed_videos:
        for i, result in enumerate(st.session_state.processed_videos[-5:]):  # Show last 5
            with st.sidebar.expander(f"{result['venue_name']} - {result['timestamp'][:10]}"):
                st.write(f"**Type:** {result['venue_type']}")
                st.write(f"**BPM:** {result['audio_environment']['bpm']}")
                st.write(f"**Crowd:** {result['crowd_density']['crowd_density']}")
                st.write(f"**Mood:** {result['mood_recognition']['dominant_mood']}")
    
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
    """)

if __name__ == "__main__":
    main()
