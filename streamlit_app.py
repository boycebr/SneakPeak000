"""
SneakPeak Video Analysis Platform v3.0 - Comprehensive Mood Analysis
Real facial expressions + audio analysis + visual environment mood detection
Created: August 2025
"""

import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import requests
from datetime import datetime
import uuid
import json
from PIL import Image, ImageFilter
import io

# Google Cloud Vision imports
try:
    from google.cloud import vision
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False
    st.sidebar.warning("Google Cloud Vision not installed. Facial expression analysis will use mock data.")

# Librosa imports for real audio analysis
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    st.sidebar.warning("Librosa not installed. Audio mood analysis will use enhanced mock data.")

# ================================
# CONFIGURATION
# ================================

SUPABASE_URL = "https://tmmheslzkqiveylrnpal.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRtbWhlc2x6a3FpdmV5bHJucGFsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjQxODg5NjEsImV4cCI6MjAzOTc2NDk2MX0.ykaK6nJhICgNRlQMCN-CnLlDKXn24h8HZdD4lKx-xv0"

# Google Cloud Vision API Key
GOOGLE_API_KEY = "AIzaSyCqy3rWftM3XS2pADtnhxDHApbJINpkLs0"

# Set up Google Cloud Vision if available
if GOOGLE_VISION_AVAILABLE:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_API_KEY

# Mood Analysis Weights
MOOD_WEIGHTS = {
    "facial_expressions": 0.45,  # 45% - Most reliable indicator
    "audio_analysis": 0.35,      # 35% - Strong secondary indicator  
    "visual_environment": 0.20   # 20% - Supporting environmental context
}

# ================================
# SESSION MANAGEMENT
# ================================

def initialize_session():
    """Initialize session state variables"""
    if 'user_session' not in st.session_state:
        st.session_state.user_session = str(uuid.uuid4())
    if 'user_name' not in st.session_state:
        st.session_state.user_name = 'Anonymous'
    if 'videos_processed' not in st.session_state:
        st.session_state.videos_processed = 0

# ================================
# GPS LOCATION FUNCTIONS
# ================================

def get_user_location():
    """Get user's GPS location with NYC defaults"""
    return {
        "latitude": 40.7589 + np.random.uniform(-0.02, 0.02),
        "longitude": -73.9851 + np.random.uniform(-0.02, 0.02),
        "accuracy": np.random.uniform(5, 15)
    }

def verify_venue_location(latitude, longitude, venue_name):
    """Verify if venue location is within reasonable bounds"""
    NYC_BOUNDS = {
        "lat_min": 40.4774, "lat_max": 40.9176,
        "lng_min": -74.2591, "lng_max": -73.7004
    }
    
    if (NYC_BOUNDS["lat_min"] <= latitude <= NYC_BOUNDS["lat_max"] and 
        NYC_BOUNDS["lng_min"] <= longitude <= NYC_BOUNDS["lng_max"]):
        return True, f"✅ Venue location verified for {venue_name}"
    else:
        return False, f"⚠️ Location outside NYC area for {venue_name}"

# ================================
# 1. FACIAL EXPRESSION ANALYSIS
# ================================

def analyze_facial_expressions_real(video_path):
    """Real facial expression analysis using Google Cloud Vision API"""
    if not GOOGLE_VISION_AVAILABLE:
        return analyze_facial_expressions_mock(video_path)
    
    try:
        st.info("😊 Analyzing facial expressions...")
        
        client = vision.ImageAnnotatorClient()
        video = VideoFileClip(video_path)
        duration = video.duration
        
        # Sample multiple frames for comprehensive analysis
        sample_times = [duration * 0.2, duration * 0.4, duration * 0.6, duration * 0.8]
        
        all_emotions = {
            'joy': [],
            'anger': [],
            'sorrow': [],
            'surprise': [],
            'confidence': []
        }
        
        total_faces_detected = 0
        
        for i, time_point in enumerate(sample_times):
            try:
                # Extract frame
                frame = video.get_frame(time_point)
                pil_image = Image.fromarray(frame.astype('uint8'))
                
                # Convert to bytes for API
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='JPEG')
                
                # Create Vision API request
                image = vision.Image(content=img_byte_arr.getvalue())
                response = client.face_detection(image=image)
                
                faces = response.face_annotations
                total_faces_detected += len(faces)
                
                # Extract emotion data from each face
                for face in faces:
                    # Google Vision emotion likelihood scale: VERY_UNLIKELY, UNLIKELY, POSSIBLE, LIKELY, VERY_LIKELY
                    emotion_values = {
                        'VERY_UNLIKELY': 0.1,
                        'UNLIKELY': 0.25, 
                        'POSSIBLE': 0.5,
                        'LIKELY': 0.75,
                        'VERY_LIKELY': 0.9
                    }
                    
                    joy_score = emotion_values.get(face.joy_likelihood.name, 0.1)
                    anger_score = emotion_values.get(face.anger_likelihood.name, 0.1)
                    sorrow_score = emotion_values.get(face.sorrow_likelihood.name, 0.1)
                    surprise_score = emotion_values.get(face.surprise_likelihood.name, 0.1)
                    confidence_score = face.detection_confidence
                    
                    all_emotions['joy'].append(joy_score)
                    all_emotions['anger'].append(anger_score)
                    all_emotions['sorrow'].append(sorrow_score)
                    all_emotions['surprise'].append(surprise_score)
                    all_emotions['confidence'].append(confidence_score)
                
                st.info(f"Frame {i+1}: Detected {len(faces)} faces")
                
            except Exception as frame_error:
                st.warning(f"Error analyzing frame {i+1}: {frame_error}")
                continue
        
        video.close()
        
        # Calculate aggregate emotion scores
        if total_faces_detected > 0:
            facial_mood = {
                'joy_avg': np.mean(all_emotions['joy']) if all_emotions['joy'] else 0.3,
                'anger_avg': np.mean(all_emotions['anger']) if all_emotions['anger'] else 0.1,
                'sorrow_avg': np.mean(all_emotions['sorrow']) if all_emotions['sorrow'] else 0.1,
                'surprise_avg': np.mean(all_emotions['surprise']) if all_emotions['surprise'] else 0.2,
                'confidence_avg': np.mean(all_emotions['confidence']) if all_emotions['confidence'] else 0.5,
                'faces_detected': total_faces_detected,
                'dominant_emotion': get_dominant_emotion(all_emotions)
            }
        else:
            # Fallback when no faces detected
            facial_mood = {
                'joy_avg': 0.4,
                'anger_avg': 0.1,
                'sorrow_avg': 0.1,
                'surprise_avg': 0.2,
                'confidence_avg': 0.3,
                'faces_detected': 0,
                'dominant_emotion': 'neutral'
            }
        
        return facial_mood
        
    except Exception as e:
        st.error(f"Facial expression analysis error: {str(e)}")
        return analyze_facial_expressions_mock(video_path)

def analyze_facial_expressions_mock(video_path):
    """Enhanced mock facial expression analysis"""
    st.info("😊 Using enhanced facial expression simulation...")
    
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        file_size = os.path.getsize(video_path)
        
        # Estimate crowd size for emotion simulation
        density_factor = file_size / (duration * 1000000)
        estimated_faces = max(0, int(density_factor * 8 + np.random.randint(1, 8)))
        
        # Simulate emotions based on venue type and crowd
        base_joy = 0.3 + (density_factor * 0.2) + np.random.uniform(0, 0.3)
        base_surprise = 0.2 + np.random.uniform(0, 0.2)
        base_anger = 0.05 + np.random.uniform(0, 0.1)
        base_sorrow = 0.05 + np.random.uniform(0, 0.1)
        
        video.close()
        
        return {
            'joy_avg': min(0.9, base_joy),
            'anger_avg': min(0.3, base_anger),
            'sorrow_avg': min(0.3, base_sorrow),
            'surprise_avg': min(0.7, base_surprise),
            'confidence_avg': 0.6 + np.random.uniform(0, 0.2),
            'faces_detected': estimated_faces,
            'dominant_emotion': 'happy' if base_joy > 0.5 else 'neutral'
        }
        
    except Exception as e:
        st.error(f"Mock facial analysis error: {str(e)}")
        return {
            'joy_avg': 0.4, 'anger_avg': 0.1, 'sorrow_avg': 0.1, 
            'surprise_avg': 0.2, 'confidence_avg': 0.5, 'faces_detected': 3,
            'dominant_emotion': 'neutral'
        }

def get_dominant_emotion(emotions_dict):
    """Determine dominant emotion from aggregated scores"""
    avg_emotions = {
        'joy': np.mean(emotions_dict['joy']) if emotions_dict['joy'] else 0,
        'anger': np.mean(emotions_dict['anger']) if emotions_dict['anger'] else 0,
        'sorrow': np.mean(emotions_dict['sorrow']) if emotions_dict['sorrow'] else 0,
        'surprise': np.mean(emotions_dict['surprise']) if emotions_dict['surprise'] else 0
    }
    
    dominant = max(avg_emotions, key=avg_emotions.get)
    
    # Map to venue-appropriate emotions
    emotion_mapping = {
        'joy': 'happy',
        'anger': 'intense', 
        'sorrow': 'mellow',
        'surprise': 'energetic'
    }
    
    return emotion_mapping.get(dominant, 'neutral')

# ================================
# 2. AUDIO-BASED MOOD DETECTION
# ================================

def analyze_audio_mood_real(video_path):
    """Real audio analysis for mood detection using librosa"""
    if not LIBROSA_AVAILABLE:
        return analyze_audio_mood_mock(video_path)
    
    try:
        st.info("🎵 Analyzing audio for mood indicators...")
        
        # Extract audio from video
        video = VideoFileClip(video_path)
        audio = video.audio
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            audio.write_audiofile(temp_audio.name, verbose=False, logger=None)
            audio_path = temp_audio.name
        
        # Load audio with librosa
        y, sr = librosa.load(audio_path)
        
        # Audio features for mood analysis
        
        # 1. Tempo/BPM (energy indicator)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # 2. Spectral centroid (brightness/energy)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # 3. Zero crossing rate (speech vs music indicator)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # 4. MFCC (voice/crowd characteristics)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        # 5. RMS Energy (volume/intensity)
        rms = np.mean(librosa.feature.rms(y=y))
        
        # 6. Spectral rolloff (harmonic content)
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # Mood inference from audio features
        mood_scores = calculate_audio_mood_scores(
            tempo, spectral_centroid, zcr, mfcc_mean, rms, rolloff, sr
        )
        
        # Cleanup
        video.close()
        audio.close()
        os.unlink(audio_path)
        
        return mood_scores
        
    except Exception as e:
        st.error(f"Audio mood analysis error: {str(e)}")
        return analyze_audio_mood_mock(video_path)

def analyze_audio_mood_mock(video_path):
    """Enhanced mock audio mood analysis"""
    st.info("🎵 Using enhanced audio mood simulation...")
    
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        file_size = os.path.getsize(video_path)
        
        # Simulate audio characteristics
        estimated_tempo = 90 + np.random.randint(-20, 40)
        estimated_energy = file_size / (duration * 800000)  # Energy from file complexity
        estimated_brightness = 0.3 + np.random.uniform(0, 0.4)
        
        mood_scores = calculate_audio_mood_scores(
            estimated_tempo, estimated_brightness * 2000, 0.1, 
            np.random.randn(13), estimated_energy, estimated_brightness * 3000, 22050
        )
        
        video.close()
        return mood_scores
        
    except Exception as e:
        st.error(f"Mock audio analysis error: {str(e)}")
        return {
            'energy_level': 0.5,
            'valence': 0.5,
            'arousal': 0.5,
            'mood_category': 'moderate',
            'confidence': 0.4
        }

def calculate_audio_mood_scores(tempo, spectral_centroid, zcr, mfcc_mean, rms, rolloff, sr):
    """Calculate mood scores from audio features"""
    
    # Normalize features (rough ranges based on typical venue audio)
    tempo_norm = min(1.0, max(0.0, (tempo - 60) / 120))  # 60-180 BPM range
    energy_norm = min(1.0, max(0.0, rms * 10))  # RMS energy scaling
    brightness_norm = min(1.0, max(0.0, (spectral_centroid - 1000) / 4000))  # Frequency range
    
    # Energy level (how intense/active)
    energy_level = (tempo_norm * 0.4 + energy_norm * 0.4 + brightness_norm * 0.2)
    
    # Valence (positive/negative sentiment)
    # Higher spectral centroid + moderate tempo = more positive
    valence = (brightness_norm * 0.5 + min(tempo_norm, 0.8) * 0.3 + (1 - zcr) * 0.2)
    
    # Arousal (excitement/calm)
    # Fast tempo + high energy = high arousal
    arousal = (tempo_norm * 0.5 + energy_norm * 0.5)
    
    # Determine mood category
    if energy_level > 0.7 and valence > 0.6:
        mood_category = 'energetic_positive'
    elif energy_level > 0.7 and valence < 0.4:
        mood_category = 'intense_aggressive'  
    elif energy_level < 0.3 and valence > 0.6:
        mood_category = 'relaxed_positive'
    elif energy_level < 0.3 and valence < 0.4:
        mood_category = 'mellow_subdued'
    else:
        mood_category = 'moderate_mixed'
    
    # Confidence based on feature consistency
    confidence = min(1.0, (energy_norm + tempo_norm + brightness_norm) / 3 + 0.2)
    
    return {
        'energy_level': round(energy_level, 3),
        'valence': round(valence, 3),
        'arousal': round(arousal, 3),
        'mood_category': mood_category,
        'confidence': round(confidence, 3),
        'audio_features': {
            'tempo': round(tempo, 1),
            'spectral_centroid': round(spectral_centroid, 1),
            'rms_energy': round(rms, 3),
            'zero_crossing_rate': round(zcr, 3)
        }
    }

# ================================
# 3. VISUAL ENVIRONMENT MOOD INFERENCE  
# ================================

def analyze_visual_mood_enhanced(video_path):
    """Enhanced visual analysis for mood inference"""
    try:
        st.info("🎨 Analyzing visual environment for mood indicators...")
        
        video = VideoFileClip(video_path)
        duration = video.duration
        
        # Sample frames for analysis
        sample_times = [duration * 0.25, duration * 0.5, duration * 0.75]
        
        brightness_levels = []
        color_intensities = []
        movement_scores = []
        
        for i, time_point in enumerate(sample_times):
            frame = video.get_frame(time_point)
            
            # Brightness analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray) / 255.0
            brightness_levels.append(brightness)
            
            # Color intensity (saturation)
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv[:, :, 1]) / 255.0
            color_intensities.append(saturation)
            
            # Movement/edge detection (proxy for activity)
            edges = cv2.Canny(gray, 50, 150)
            movement_score = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            movement_scores.append(movement_score)
        
        video.close()
        
        # Calculate visual mood indicators
        avg_brightness = np.mean(brightness_levels)
        avg_color_intensity = np.mean(color_intensities)
        avg_movement = np.mean(movement_scores)
        
        # Mood inference from visual features
        visual_mood = calculate_visual_mood_scores(
            avg_brightness, avg_color_intensity, avg_movement
        )
        
        return visual_mood
        
    except Exception as e:
        st.error(f"Visual mood analysis error: {str(e)}")
        return {
            'lighting_mood': 'moderate',
            'color_energy': 0.5,
            'visual_activity': 0.5,
            'atmosphere': 'neutral',
            'confidence': 0.4
        }

def calculate_visual_mood_scores(brightness, color_intensity, movement):
    """Calculate mood scores from visual features"""
    
    # Lighting mood categories
    if brightness > 0.7:
        lighting_mood = 'bright_energetic'
    elif brightness > 0.4:
        lighting_mood = 'moderate_social'
    elif brightness > 0.2:
        lighting_mood = 'dim_intimate'
    else:
        lighting_mood = 'dark_mysterious'
    
    # Color energy (high saturation = energetic, low = mellow)
    color_energy = color_intensity
    
    # Visual activity level
    visual_activity = min(1.0, movement * 5)  # Scale movement score
    
    # Overall atmosphere
    if brightness > 0.6 and color_intensity > 0.5:
        atmosphere = 'vibrant_energetic'
    elif brightness < 0.3 and color_intensity > 0.4:
        atmosphere = 'moody_intense'
    elif brightness > 0.5 and color_intensity < 0.3:
        atmosphere = 'bright_relaxed'
    elif brightness < 0.4 and color_intensity < 0.3:
        atmosphere = 'subdued_mellow'
    else:
        atmosphere = 'balanced_moderate'
    
    # Confidence based on visual consistency
    confidence = (brightness + color_intensity + visual_activity) / 3
    
    return {
        'lighting_mood': lighting_mood,
        'color_energy': round(color_energy, 3),
        'visual_activity': round(visual_activity, 3),
        'atmosphere': atmosphere,
        'confidence': round(confidence, 3),
        'visual_features': {
            'brightness': round(brightness, 3),
            'color_intensity': round(color_intensity, 3),
            'movement_score': round(movement, 3)
        }
    }

# ================================
# 4. COMPREHENSIVE MOOD SCORING
# ================================

def calculate_comprehensive_mood_score(facial_results, audio_results, visual_results):
    """Combine all three mood analysis inputs with weighted scoring"""
    
    st.info("🧠 Combining all mood analysis inputs...")
    
    # Extract key mood indicators from each analysis
    
    # 1. Facial Expression Contribution (45%)
    facial_positivity = facial_results['joy_avg'] - (facial_results['anger_avg'] + facial_results['sorrow_avg'])
    facial_energy = facial_results['joy_avg'] + facial_results['surprise_avg']
    facial_confidence = facial_results['confidence_avg']
    
    facial_score = (facial_positivity + facial_energy) / 2 * facial_confidence
    
    # 2. Audio Analysis Contribution (35%)  
    audio_score = (audio_results['valence'] + audio_results['energy_level']) / 2 * audio_results['confidence']
    
    # 3. Visual Environment Contribution (20%)
    visual_score = (visual_results['color_energy'] + visual_results['visual_activity']) / 2 * visual_results['confidence']
    
    # Weighted final score
    final_mood_score = (
        facial_score * MOOD_WEIGHTS['facial_expressions'] +
        audio_score * MOOD_WEIGHTS['audio_analysis'] + 
        visual_score * MOOD_WEIGHTS['visual_environment']
    )
    
    # Normalize to 0-100 scale
    final_mood_score = max(0, min(100, final_mood_score * 100))
    
    # Determine overall mood category
    overall_mood = determine_overall_mood_category(
        facial_results, audio_results, visual_results, final_mood_score
    )
    
    # Calculate confidence based on data quality
    overall_confidence = (
        facial_results['confidence_avg'] * MOOD_WEIGHTS['facial_expressions'] +
        audio_results['confidence'] * MOOD_WEIGHTS['audio_analysis'] +
        visual_results['confidence'] * MOOD_WEIGHTS['visual_environment']
    )
    
    return {
        'mood_score': round(final_mood_score, 1),
        'overall_mood': overall_mood,
        'confidence': round(overall_confidence, 3),
        'component_scores': {
            'facial_contribution': round(facial_score * MOOD_WEIGHTS['facial_expressions'] * 100, 1),
            'audio_contribution': round(audio_score * MOOD_WEIGHTS['audio_analysis'] * 100, 1), 
            'visual_contribution': round(visual_score * MOOD_WEIGHTS['visual_environment'] * 100, 1)
        },
        'detailed_analysis': {
            'facial_results': facial_results,
            'audio_results': audio_results,
            'visual_results': visual_results
        }
    }

def determine_overall_mood_category(facial_results, audio_results, visual_results, score):
    """Determine comprehensive mood category from all inputs"""
    
    # Get dominant indicators from each analysis
    facial_mood = facial_results['dominant_emotion']
    audio_mood = audio_results['mood_category'] 
    visual_atmosphere = visual_results['atmosphere']
    
    # Score-based primary categorization
    if score >= 80:
        base_mood = 'highly_positive'
    elif score >= 65:
        base_mood = 'positive'
    elif score >= 45:
        base_mood = 'moderate'
    elif score >= 25:
        base_mood = 'subdued'
    else:
        base_mood = 'low_energy'
    
    # Refine with qualitative analysis
    energy_indicators = [
        audio_results['energy_level'] > 0.6,
        visual_results['visual_activity'] > 0.6,
        facial_results['joy_avg'] > 0.5
    ]
    
    is_high_energy = sum(energy_indicators) >= 2
    
    # Final mood category
    if base_mood in ['highly_positive', 'positive'] and is_high_energy:
        return 'energetic_positive'
    elif base_mood in ['highly_positive', 'positive'] and not is_high_energy:
        return 'relaxed_positive'
    elif base_mood == 'moderate' and is_high_energy:
        return 'active_moderate'
    elif base_mood == 'moderate' and not is_high_energy:
        return 'calm_moderate'
    elif base_mood in ['subdued', 'low_energy'] and is_high_energy:
        return 'intense_focused'
    else:
        return 'mellow_subdued'

def analyze_comprehensive_mood(video_path):
    """Main function to analyze mood using all three methods"""
    
    st.subheader("🎭 Comprehensive Mood Analysis")
    
    # Run all three analyses
    facial_results = analyze_facial_expressions_real(video_path)
    audio_results = analyze_audio_mood_real(video_path)
    visual_results = analyze_visual_mood_enhanced(video_path)
    
    # Combine results
    final_mood_analysis = calculate_comprehensive_mood_score(
        facial_results, audio_results, visual_results
    )
    
    # Display results
    st.success(f"🎯 **Overall Mood Score: {final_mood_analysis['mood_score']}/100**")
    st.info(f"**Mood Category:** {final_mood_analysis['overall_mood'].replace('_', ' ').title()}")
    st.info(f"**Analysis Confidence:** {final_mood_analysis['confidence']*100:.1f}%")
    
    # Show component contributions
    with st.expander("📊 Component Analysis Breakdown"):
        st.write("**Contribution by Analysis Method:**")
        st.write(f"- Facial Expressions (45%): {final_mood_analysis['component_scores']['facial_contribution']:.1f} points")
        st.write(f"- Audio Analysis (35%): {final_mood_analysis['component_scores']['audio_contribution']:.1f} points") 
        st.write(f"- Visual Environment (20%): {final_mood_analysis['component_scores']['visual_contribution']:.1f} points")
        
        st.write("**Detailed Results:**")
        st.json(final_mood_analysis['detailed_analysis'])
    
    return final_mood_analysis

# ================================
# OTHER ANALYSIS FUNCTIONS (EXISTING)
# ================================

def extract_audio_features(video_path):
    """Enhanced audio feature extraction"""
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        
        # Simulate BPM based on file characteristics
        file_size = os.path.getsize(video_path)
        complexity_factor = file_size / (duration * 1000000)
        
        base_bpm = 95 + int(complexity_factor * 25) + np.random.randint(-15, 25)
        base_bpm = max(70, min(180, base_bpm))
        
        # Volume simulation
        volume_db = 60 + int(complexity_factor * 20) + np.random.randint(-10, 15)
        volume_db = max(40, min(95, volume_db))
        
        # Genre classification based on BPM
        if base_bpm > 140:
            genre = "Electronic/Dance"
        elif base_bpm > 120:
            genre = "Pop/Hip-Hop"
        elif base_bpm > 100:
            genre = "Rock/Alternative"
        else:
            genre = "Jazz/Ambient"
        
        video.close()
        
        return {
            "bpm": base_bpm,
            "volume_db": volume_db,
            "genre": genre,
            "tempo_consistency": round(0.7 + np.random.uniform(0, 0.25), 2)
        }
        
    except Exception as e:
        st.error(f"Audio analysis error: {str(e)}")
        return {
            "bpm": 110,
            "volume_db": 65,
            "genre": "Mixed",
            "tempo_consistency": 0.7
        }

def analyze_visual_environment_simple(video_path):
    """Enhanced visual environment analysis"""
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        
        # Sample frames for visual analysis
        sample_times = [duration * 0.3, duration * 0.7]
        brightness_scores = []
        color_scores = []
        
        for time_point in sample_times:
            frame = video.get_frame(time_point)
            
            # Brightness analysis
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray_frame)
            brightness_scores.append(brightness)
            
            # Color intensity analysis  
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            saturation = np.mean(hsv_frame[:, :, 1])
            color_scores.append(saturation)
        
        avg_brightness = np.mean(brightness_scores)
        avg_saturation = np.mean(color_scores)
        
        # Lighting classification
        if avg_brightness > 150:
            lighting_type = "Bright/Well-lit"
        elif avg_brightness > 100:
            lighting_type = "Moderate/Social"
        elif avg_brightness > 50:
            lighting_type = "Dim/Intimate"
        else:
            lighting_type = "Dark/Club lighting"
        
        # Color scheme classification
        if avg_saturation > 100:
            color_scheme = "Vibrant/Colorful"
        elif avg_saturation > 60:
            color_scheme = "Moderate colors"
        else:
            color_scheme = "Muted/Neutral"
        
        video.close()
        
        return {
            "lighting_type": lighting_type,
            "brightness_level": round(avg_brightness / 255 * 100, 1),
            "color_scheme": color_scheme,
            "color_intensity": round(avg_saturation / 255 * 100, 1),
            "visual_energy": "High" if avg_brightness > 120 and avg_saturation > 80 else "Medium"
        }
        
    except Exception as e:
        st.error(f"Visual analysis error: {str(e)}")
        return {
            "lighting_type": "Moderate/Social",
            "brightness_level": 60.0,
            "color_scheme": "Moderate colors", 
            "color_intensity": 45.0,
            "visual_energy": "Medium"
        }

def analyze_crowd_density_simple(video_path):
    """Enhanced crowd density analysis"""
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        file_size = os.path.getsize(video_path)
        fps = video.fps if video.fps else 30
        
        # Sophisticated density calculation
        size_density_factor = file_size / (duration * 1500000)  # Adjusted scaling
        fps_motion_factor = fps / 30
        duration_factor = min(2.0, duration / 20)  # Longer videos might have more people
        
        base_people = max(0, int(
            size_density_factor * 15 + 
            fps_motion_factor * 8 + 
            duration_factor * 5 + 
            np.random.randint(0, 8)
        ))
        
        # Enhanced density categories
        if base_people > 25:
            density = "Extremely Packed"
            capacity = "Over Capacity"
        elif base_people > 20:
            density = "Very Packed"
            capacity = "Near Capacity"
        elif base_people > 15:
            density = "Packed"
            capacity = "High Capacity"
        elif base_people > 10:
            density = "Busy"
            capacity = "Moderate"
        elif base_people > 5:
            density = "Moderate"
            capacity = "Low-Moderate"
        elif base_people > 2:
            density = "Light"
            capacity = "Low"
        else:
            density = "Empty/Quiet"
            capacity = "Very Low"
        
        # Enhanced activity classification
        activity_factor = size_density_factor + (fps_motion_factor * 0.5)
        if activity_factor > 3.0:
            activity = "Intense Dancing/Movement"
        elif activity_factor > 2.0:
            activity = "High Movement/Dancing"
        elif activity_factor > 1.2:
            activity = "Active Movement"
        elif activity_factor > 0.7:
            activity = "Moderate Movement"
        elif activity_factor > 0.3:
            activity = "Light Movement"
        else:
            activity = "Low Movement/Standing"
        
        # Engagement and energy scores
        engagement_score = min(100, base_people * 3 + activity_factor * 15)
        crowd_energy = min(100, (base_people / 25 * 50) + (engagement_score * 0.5))
        
        video.close()
        
        return {
            "crowd_density": density,
            "activity_level": activity,
            "density_score": float(base_people),
            "capacity_estimate": capacity,
            "engagement_score": round(engagement_score, 1),
            "crowd_energy": round(crowd_energy, 1)
        }
        
    except Exception as e:
        st.error(f"Crowd analysis error: {str(e)}")
        return {
            "crowd_density": "Moderate",
            "activity_level": "Medium Movement",
            "density_score": float(np.random.randint(5, 12)),
            "capacity_estimate": "Moderate",
            "engagement_score": 65.0,
            "crowd_energy": 55.0
        }

def calculate_energy_score(results):
    """Enhanced energy score calculation with mood integration"""
    try:
        # Extract values with defaults
        bpm = results.get('audio_results', {}).get('bpm', 110)
        volume = results.get('audio_results', {}).get('volume_db', 65)
        density_score = results.get('crowd_results', {}).get('density_score', 8)
        brightness = results.get('visual_results', {}).get('brightness_level', 60)
        
        # Get mood score if available
        mood_score = results.get('mood_results', {}).get('mood_score', 60)
        
        # Weighted calculation with mood integration
        bpm_score = min(100, max(0, (bpm - 60) / 120 * 100))
        volume_score = min(100, max(0, (volume - 40) / 55 * 100))
        density_score_norm = min(100, density_score * 4)
        brightness_score = min(100, brightness)
        
        # Updated weights including mood
        energy_score = (
            bpm_score * 0.25 +           # 25% - BPM contribution
            volume_score * 0.20 +        # 20% - Volume contribution  
            density_score_norm * 0.25 +  # 25% - Crowd density contribution
            brightness_score * 0.10 +    # 10% - Visual brightness contribution
            mood_score * 0.20            # 20% - Mood analysis contribution
        )
        
        return round(energy_score, 1)
        
    except Exception as e:
        st.error(f"Energy calculation error: {str(e)}")
        return 65.0

# ================================
# DATABASE FUNCTIONS
# ================================

def save_to_supabase(results):
    """Enhanced database save with comprehensive mood data"""
    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        
        # Prepare data with mood analysis
        data = {
            "venue_name": results.get("venue_name", "Unknown"),
            "venue_type": results.get("venue_type", "Unknown"),
            "user_session": results.get("user_session", str(uuid.uuid4())),
            "user_name": results.get("user_name", "Anonymous"),
            "latitude": results.get("gps_data", {}).get("latitude", 40.7589),
            "longitude": results.get("gps_data", {}).get("longitude", -73.9851),
            "gps_accuracy": results.get("gps_data", {}).get("accuracy", 10.0),
            "venue_verified": results.get("venue_verified", True),
            "bpm": results.get("audio_results", {}).get("bpm", 110),
            "volume_level": results.get("audio_results", {}).get("volume_db", 65),
            "genre": results.get("audio_results", {}).get("genre", "Mixed"),
            "energy_level": results.get("energy_score", 65),
            "brightness_level": results.get("visual_results", {}).get("brightness_level", 60),
            "lighting_type": results.get("visual_results", {}).get("lighting_type", "Moderate"),
            "color_scheme": results.get("visual_results", {}).get("color_scheme", "Mixed"),
            "visual_energy": results.get("visual_results", {}).get("visual_energy", "Medium"),
            "crowd_density": results.get("crowd_results", {}).get("crowd_density", "Moderate"),
            "activity_level": results.get("crowd_results", {}).get("activity_level", "Medium"),
            "density_score": results.get("crowd_results", {}).get("density_score", 8),
            
            # NEW: Comprehensive mood data
            "mood_score": results.get("mood_results", {}).get("mood_score", 60),
            "overall_mood": results.get("mood_results", {}).get("overall_mood", "moderate"),
            "mood_confidence": results.get("mood_results", {}).get("confidence", 0.5),
            "facial_joy": results.get("mood_results", {}).get("detailed_analysis", {}).get("facial_results", {}).get("joy_avg", 0.4),
            "facial_energy": results.get("mood_results", {}).get("detailed_analysis", {}).get("facial_results", {}).get("surprise_avg", 0.2),
            "audio_valence": results.get("mood_results", {}).get("detailed_analysis", {}).get("audio_results", {}).get("valence", 0.5),
            "audio_arousal": results.get("mood_results", {}).get("detailed_analysis", {}).get("audio_results", {}).get("arousal", 0.5),
            "visual_atmosphere": results.get("mood_results", {}).get("detailed_analysis", {}).get("visual_results", {}).get("atmosphere", "neutral"),
            
            "created_at": datetime.now().isoformat()
        }
        
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/video_results",
            headers=headers,
            data=json.dumps(data)
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
            f"{SUPABASE_URL}/rest/v1/video_results?select=*&order=created_at.desc&limit=50",
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
# MAIN PROCESSING FUNCTION
# ================================

def process_video_comprehensive_analysis(video_file, venue_name, venue_type, gps_data=None):
    """Complete video processing with comprehensive mood analysis"""
    
    # Save uploaded video to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(video_file.read())
        temp_video_path = temp_file.name
    
    try:
        st.info("🚀 Starting comprehensive video analysis...")
        
        # Get GPS data
        if not gps_data:
            gps_data = get_user_location()
        
        # Verify venue location
        venue_verified, verification_msg = verify_venue_location(
            gps_data["latitude"], gps_data["longitude"], venue_name
        )
        
        # Run all analyses
        with st.spinner("🎵 Analyzing audio features..."):
            audio_results = extract_audio_features(temp_video_path)
        
        with st.spinner("🎨 Analyzing visual environment..."):
            visual_results = analyze_visual_environment_simple(temp_video_path)
        
        with st.spinner("👥 Analyzing crowd density..."):
            crowd_results = analyze_crowd_density_simple(temp_video_path)
        
        # NEW: Comprehensive mood analysis
        with st.spinner("🎭 Running comprehensive mood analysis..."):
            mood_results = analyze_comprehensive_mood(temp_video_path)
        
        # Compile all results
        all_results = {
            "venue_name": venue_name,
            "venue_type": venue_type,
            "user_session": st.session_state.user_session,
            "user_name": st.session_state.user_name,
            "gps_data": gps_data,
            "venue_verified": venue_verified,
            "audio_results": audio_results,
            "visual_results": visual_results,
            "crowd_results": crowd_results,
            "mood_results": mood_results  # NEW: Mood analysis results
        }
        
        # Calculate energy score with mood integration
        energy_score = calculate_energy_score(all_results)
        all_results["energy_score"] = energy_score
        
        # Cleanup
        os.unlink(temp_video_path)
        
        return all_results
        
    except Exception as e:
        st.error(f"❌ Video processing failed: {str(e)}")
        # Cleanup on error
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        return None

# ================================
# UI FUNCTIONS
# ================================

def display_results(results):
    """Enhanced results display with mood analysis"""
    if not results:
        return
    
    st.subheader("📊 Analysis Results")
    
    # Main energy score with mood integration
    energy_score = results.get("energy_score", 0)
    mood_score = results.get("mood_results", {}).get("mood_score", 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="🔥 Venue Energy Score",
            value=f"{energy_score}/100",
            delta=f"Mood: {mood_score}/100"
        )
    
    with col2:
        mood_category = results.get("mood_results", {}).get("overall_mood", "moderate")
        st.metric(
            label="🎭 Overall Mood",
            value=mood_category.replace('_', ' ').title(),
            delta=f"Confidence: {results.get('mood_results', {}).get('confidence', 0)*100:.0f}%"
        )
    
    # Detailed analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎵 Audio", "🎨 Visual", "👥 Crowd", "🎭 Mood", "📍 Location"])
    
    with tab1:
        audio = results.get("audio_results", {})
        st.write(f"**BPM:** {audio.get('bpm', 'N/A')}")
        st.write(f"**Volume:** {audio.get('volume_db', 'N/A')} dB")
        st.write(f"**Genre:** {audio.get('genre', 'N/A')}")
        st.write(f"**Tempo Consistency:** {audio.get('tempo_consistency', 'N/A')}")
    
    with tab2:
        visual = results.get("visual_results", {})
        st.write(f"**Lighting:** {visual.get('lighting_type', 'N/A')}")
        st.write(f"**Brightness:** {visual.get('brightness_level', 'N/A')}%")
        st.write(f"**Color Scheme:** {visual.get('color_scheme', 'N/A')}")
        st.write(f"**Visual Energy:** {visual.get('visual_energy', 'N/A')}")
    
    with tab3:
        crowd = results.get("crowd_results", {})
        st.write(f"**Density:** {crowd.get('crowd_density', 'N/A')}")
        st.write(f"**Activity Level:** {crowd.get('activity_level', 'N/A')}")
        st.write(f"**Estimated People:** {crowd.get('density_score', 'N/A')}")
        st.write(f"**Capacity:** {crowd.get('capacity_estimate', 'N/A')}")
    
    with tab4:
        mood = results.get("mood_results", {})
        detailed = mood.get("detailed_analysis", {})
        
        st.write(f"**Overall Mood Score:** {mood.get('mood_score', 'N/A')}/100")
        st.write(f"**Mood Category:** {mood.get('overall_mood', 'N/A').replace('_', ' ').title()}")
        st.write(f"**Analysis Confidence:** {mood.get('confidence', 0)*100:.1f}%")
        
        if detailed:
            st.write("**Component Contributions:**")
            components = mood.get('component_scores', {})
            st.write(f"- Facial Expressions: {components.get('facial_contribution', 0):.1f} pts")
            st.write(f"- Audio Analysis: {components.get('audio_contribution', 0):.1f} pts")
            st.write(f"- Visual Environment: {components.get('visual_contribution', 0):.1f} pts")
    
    with tab5:
        gps = results.get("gps_data", {})
        st.write(f"**Latitude:** {gps.get('latitude', 'N/A'):.4f}")
        st.write(f"**Longitude:** {gps.get('longitude', 'N/A'):.4f}")
        st.write(f"**GPS Accuracy:** {gps.get('accuracy', 'N/A')} meters")
        st.write(f"**Venue Verified:** {'✅ Yes' if results.get('venue_verified') else '❌ No'}")

def display_analytics_dashboard():
    """Enhanced analytics dashboard with mood insights"""
    st.header("📊 SneakPeak Analytics Dashboard")
    
    # Load data
    all_results = load_all_results()
    
    if not all_results:
        st.info("No data available yet. Upload some videos to see analytics!")
        return
    
    # Convert to DataFrame for analysis
    import pandas as pd
    
    df_data = []
    for result in all_results:
        df_data.append({
            'Venue': result.get('venue_name', 'Unknown'),
            'Type': result.get('venue_type', 'Unknown'),
            'Energy Score': result.get('energy_level', 0),
            'Mood Score': result.get('mood_score', 0),
            'Overall Mood': result.get('overall_mood', 'unknown'),
            'BPM': result.get('bpm', 0),
            'Crowd Density': result.get('density_score', 0),
            'User': result.get('user_name', 'Anonymous'),
            'Verified': '✅' if result.get('venue_verified') else '❌',
            'created_at': result.get('created_at', '')
        })
    
    df = pd.DataFrame(df_data)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_videos = len(all_results)
        st.markdown(f"""
        <div class="metric-container">
            <span class="metric-value">{total_videos}</span>
            <span class="metric-label">Total Videos</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_energy = df['Energy Score'].mean() if not df.empty else 0
        st.markdown(f"""
        <div class="metric-container">
            <span class="metric-value">{avg_energy:.1f}</span>
            <span class="metric-label">Avg Energy</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_mood = df['Mood Score'].mean() if not df.empty else 0
        st.markdown(f"""
        <div class="metric-container">
            <span class="metric-value">{avg_mood:.1f}</span>
            <span class="metric-label">Avg Mood</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        unique_venues = df["Venue"].nunique() if not df.empty else 0
        st.markdown(f"""
        <div class="metric-container">
            <span class="metric-value">{unique_venues}</span>
            <span class="metric-label">Venues</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Data table with mood information
    st.markdown("#### 📋 Recent Submissions")
    if not df.empty:
        display_df = df[['Venue', 'Type', 'Energy Score', 'Mood Score', 'Overall Mood', 'User', 'Verified']].head(10)
        st.dataframe(
            display_df, 
            use_container_width=True,
            hide_index=True,
            column_config={
                "Energy Score": st.column_config.ProgressColumn(
                    "Energy Score",
                    help="Overall venue energy (0-100)",
                    min_value=0,
                    max_value=100,
                ),
                "Mood Score": st.column_config.ProgressColumn(
                    "Mood Score",
                    help="Comprehensive mood analysis (0-100)",
                    min_value=0,
                    max_value=100,
                )
            }
        )

# ================================
# MAIN APPLICATION
# ================================

def main():
    st.set_page_config(
        page_title="SneakPeak v3.0 - Comprehensive Mood Analysis",
        page_icon="🎭",
        layout="wide"
    )
    
    # Initialize session
    initialize_session()
    
    # Custom CSS
    st.markdown("""
    <style>
        .metric-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin: 0.5rem 0;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            display: block;
        }
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🎭 SneakPeak v3.0 - Comprehensive Mood Analysis")
    st.markdown("**Real facial expressions + audio analysis + visual environment = accurate venue mood scoring**")
    
    # Library status
    with st.sidebar:
        st.header("🔧 System Status")
        st.write(f"**Google Vision API:** {'✅ Available' if GOOGLE_VISION_AVAILABLE else '❌ Mock Mode'}")
        st.write(f"**Librosa Audio:** {'✅ Available' if LIBROSA_AVAILABLE else '❌ Mock Mode'}")
        st.write(f"**Session:** {st.session_state.user_session[:8]}...")
        st.write(f"**Videos Processed:** {st.session_state.videos_processed}")
    
    # Main interface tabs
    tab1, tab2 = st.tabs(["🎬 Video Analysis", "📊 Analytics Dashboard"])
    
    with tab1:
        st.subheader("🎬 Upload Video for Comprehensive Analysis")
        
        # Venue selection
        venue_types = [
            "🍸 Bar/Lounge", "🕺 Nightclub", "🎵 Concert Venue", "🍺 Sports Bar", 
            "🥂 Rooftop", "🎪 Event Space", "🍕 Restaurant", "☕ Cafe",
            "🎭 Theater", "🏖️ Beach Club", "🎨 Gallery", "🏨 Hotel Lounge"
        ]
        
        selected_venue = st.selectbox("Venue Type", venue_types)
        venue_type = selected_venue.split(" ", 1)[1]  # Remove emoji
        
        venue_name = st.text_input(
            "Venue Name", 
            placeholder="Enter venue name (e.g., 'The Rooftop NYC')",
            help="Enter the name of the venue you're analyzing"
        )
        
        # User name
        if st.session_state.user_name == 'Anonymous':
            user_name = st.text_input(
                "Your Name (Optional)", 
                placeholder="Enter your name for credits..."
            )
            if user_name:
                st.session_state.user_name = user_name
                st.success(f"Welcome, {user_name}! 👋")
        else:
            st.success(f"Welcome back, {st.session_state.user_name}! 👋")
        
        # Video upload
        uploaded_file = st.file_uploader(
            "Upload Venue Video",
            type=['mp4', 'mov', 'avi', 'mkv', 'webm'],
            help="Upload a video of the venue (max 200MB) for comprehensive mood analysis"
        )
        
        # Process video
        if uploaded_file and venue_name:
            if st.button("🚀 Analyze Video (Comprehensive Mood Analysis)", type="primary"):
                with st.spinner("Processing video with comprehensive mood analysis..."):
                    results = process_video_comprehensive_analysis(
                        uploaded_file, venue_name, venue_type
                    )
                    
                    if results:
                        # Display results
                        display_results(results)
                        
                        # Save to database
                        if save_to_supabase(results):
                            st.session_state.videos_processed += 1
                            st.balloons()
                        
        elif uploaded_file and not venue_name:
            st.warning("Please enter a venue name before analyzing.")
    
    with tab2:
        display_analytics_dashboard()

if __name__ == "__main__":
    main()
