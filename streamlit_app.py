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
import streamlit.components.v1 as components
from supabase import create_client, Client

# Supabase configuration
SUPABASE_URL = "https://tmmheslzkqiveylrnpal.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRtbWhlc2x6a3FpdmV5bHJucGFsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQzMzI5MjAsImV4cCI6MjA2OTkwODkyMH0.U-10R707xIs6rH-Vd5lBgh2INylFu6zn_EyoJYx_zpI"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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
if 'user' not in st.session_state:
    st.session_state.user = None
if 'user_location' not in st.session_state:
    st.session_state.user_location = None

def get_location_component():
    """
    A Streamlit component that uses JavaScript to get the user's location
    and returns it to the Streamlit app.
    """
    # JavaScript to get geolocation
    js_code = """
    <script>
    function getLocation() {
      if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
          (position) => {
            const data = {
              latitude: position.coords.latitude,
              longitude: position.coords.longitude,
              accuracy: position.coords.accuracy,
              error: null
            };
            window.parent.postMessage({
              type: 'streamlit:setComponentValue',
              componentId: 'user_location',
              value: data
            }, '*');
          },
          (error) => {
            const data = {
              latitude: null,
              longitude: null,
              accuracy: null,
              error: error.message
            };
            window.parent.postMessage({
              type: 'streamlit:setComponentValue',
              componentId: 'user_location',
              value: data
            }, '*');
          }
        );
      } else { 
        const data = {
          latitude: null,
          longitude: null,
          accuracy: null,
          error: "Geolocation is not supported by this browser."
        };
        window.parent.postMessage({
          type: 'streamlit:setComponentValue',
          componentId: 'user_location',
          value: data
        }, '*');
      }
    }
    
    // Call the function when the component loads
    getLocation();
    </script>
    """
    # This is a dummy call; the actual communication is handled by postMessage
    components.html(js_code, height=0, width=0, key='user_location')

def upload_video_to_supabase(uploaded_file, video_id):
    """Upload video file to Supabase Storage"""
    try:
        # Create unique filename
        file_extension = uploaded_file.name.split('.')[-1]
        filename = f"{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_extension}"
        
        # Upload to Supabase Storage
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "video/mp4" # Or match uploaded_file.type
        }
        
        # Use a more direct upload method by sending the file content in the body
        response = requests.post(
            f"{SUPABASE_URL}/storage/v1/object/videos/{filename}",
            headers=headers,
            data=uploaded_file.getvalue()
        )
        
        if response.status_code == 200:
            # Return the public URL
            video_url = f"{SUPABASE_URL}/storage/v1/object/public/videos/{filename}"
            return video_url, filename
        else:
            st.error(f"Video upload failed: {response.status_code}")
            if response.text:
                st.error(f"Error details: {response.text}")
            return None, None
            
    except Exception as e:
        st.error(f"Error uploading video: {str(e)}")
        return None, None

def verify_venue_location(latitude, longitude, venue_name):
    """Simple venue verification - in production this would check against venue database"""
    # For demo purposes, just return True if coordinates are reasonable
    if latitude and longitude:
        # Check if coordinates are in NYC area (rough bounds)
        if 40.4774 <= latitude <= 40.9176 and -74.2591 <= longitude <= -73.7004:
            return True
    return False

def save_user_rating(venue_id, user_id, rating, venue_name, venue_type):
    """Save a user's rating of a venue"""
    try:
        rating_data = {
            "venue_id": str(venue_id),
            "user_id": str(user_id)[:20],
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

def save_to_supabase(results, uploaded_file=None):
    """Save analysis results to Supabase database, and upload video if provided."""
    try:
        # Generate unique ID for this entry
        video_id = str(uuid.uuid4())
        
        # Upload video if provided
        video_url = None
        video_filename = None
        if uploaded_file:
            video_url, video_filename = upload_video_to_supabase(uploaded_file, video_id)
        
        # Get GPS data from results
        gps_data = results.get("gps_data", {})
        
        # Get current user ID
        user_id = st.session_state.user.id if st.session_state.user else None
        
        if not user_id:
            st.error("❌ Cannot save results. You must be logged in.")
            return False, None

        # Prepare data with proper type casting and validation
        db_data = {
            "id": video_id, # Add explicit ID
            "user_id": user_id, # New column for the user's ID
            "venue_name": str(results["venue_name"])[:100],  # Limit length
            "venue_type": str(results["venue_type"])[:50],
            
            # Video storage fields
            "video_url": video_url,
            "video_filename": video_filename,
            "video_stored": video_url is not None,

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
            return True, video_id
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
            return False, None
    except Exception as e:
        st.error(f"❌ Database error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False, None

def load_user_results(user_id):
    """Load videos uploaded by the logged-in user."""
    if not user_id:
        return []
    
    try:
        data = supabase.from_("video_results").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        st.write(f"🔍 Debug: Fetched {len(data.data)} videos for user {user_id}.")
        return data.data
    except Exception as e:
        st.error(f"❌ Database error during data load: {str(e)}")
        return []

def load_video_by_id(video_id):
    """Load a single video result by its ID from the Supabase database."""
    try:
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }
        
        # Make the GET request to fetch a single record
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/video_results?id=eq.{video_id}",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            if data:
                return data[0] # Return the first (and only) result
            else:
                return None
        else:
            st.error(f"❌ Failed to load video with ID {video_id}. Status code: {response.status_code}")
            st.error(f"Error details: {response.text}")
            return None
    except Exception as e:
        st.error(f"❌ Database error during video load: {str(e)}")
        return None

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
        return 50.0 # Default fallback

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
            volume_level = min(100, (file_size / 1000000) * 20 + 20) # Simulate volume
        except FileNotFoundError:
            volume_level = 50.0
        genres = ["Electronic/Dance", "Hip Hop", "Pop", "Rock", "R&B"]
        genre = np.random.choice(genres)
        energy_levels = ["Low", "Medium", "High"]
        energy_level = "High" if bpm > 110 and volume_level > 60 else "Medium"
        
        os.unlink(temp_audio.name)
        
        return {
            "bpm": int(bpm),
            "volume_level": volume_level,
            "genre": genre,
            "energy_level": energy_level
        }
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return {
            "bpm": 0, "volume_level": 0.0, "genre": "Unknown", "energy_level": "Unknown"
        }

def analyze_visual_features(video_path):
    """Simulate visual feature analysis"""
    lighting_types = ["Dark/Club Lighting", "Bright/Bar Lighting", "Mixed Indoor"]
    color_schemes = ["Purple-dominant", "Blue/Red", "White/Yellow", "Mixed"]
    visual_energies = ["Low", "Medium", "High"]
    
    return {
        "brightness_level": np.random.uniform(30, 90),
        "lighting_type": np.random.choice(lighting_types),
        "color_scheme": np.random.choice(color_schemes),
        "visual_energy": np.random.choice(visual_energies)
    }
    
def analyze_crowd_features(video_path):
    """Simulate crowd feature analysis"""
    densities = ["Empty", "Sparse", "Moderate", "Busy", "Packed"]
    activities = ["Still/Seated", "Social/Standing", "High Movement/Dancing"]
    
    crowd_density = np.random.choice(densities, p=[0.1, 0.2, 0.3, 0.3, 0.1])
    activity_level = np.random.choice(activities, p=[0.2, 0.4, 0.4])
    
    density_scores = {"Empty": 1, "Sparse": 5, "Moderate": 10, "Busy": 15, "Packed": 20}
    density_score = density_scores.get(crowd_density, 0)
    
    return {
        "crowd_density": crowd_density,
        "activity_level": activity_level,
        "density_score": float(density_score)
    }

def analyze_mood_recognition(video_path):
    """Simulate mood recognition and overall vibe"""
    moods = ["Excited", "Happy", "Energetic", "Social", "Festive", "Calm"]
    dominant_mood = np.random.choice(moods, p=[0.3, 0.25, 0.2, 0.15, 0.05, 0.05])
    confidence = np.random.uniform(0.6, 0.95)
    
    mood_breakdown = {mood: np.random.uniform(0.01, 0.3) for mood in moods}
    total = sum(mood_breakdown.values())
    for mood in mood_breakdown:
        mood_breakdown[mood] /= total
        
    vibe = "Positive" if dominant_mood in ["Excited", "Happy", "Energetic"] else "Mixed"
    
    return {
        "dominant_mood": dominant_mood,
        "confidence": float(confidence),
        "mood_breakdown": mood_breakdown,
        "overall_vibe": vibe
    }

def display_results(results):
    """Display the analysis results in a structured format."""
    st.subheader(f"📊 Analysis Results for {results['venue_name']}")
    
    # Use columns for better mobile layout
    col1, col2 = st.columns(2)
    
    # Display main metrics
    with col1:
        st.markdown(f'<div class="metric-card"><h4>Overall Vibe</h4><p>{results["mood_recognition"]["overall_vibe"]}</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h4>Energy Score</h4><p>{results["energy_score"]:.2f}/100</p></div>', unsafe_allow_html=True)
    
    # Use expanders for detailed information
    with st.expander("🔊 Audio Environment"):
        st.markdown(f"**BPM:** {results['audio_environment']['bpm']} BPM")
        st.markdown(f"**Volume Level:** {results['audio_environment']['volume_level']:.2f}%")
        st.markdown(f"**Genre:** {results['audio_environment']['genre']}")
        st.markdown(f"**Energy Level:** {results['audio_environment']['energy_level']}")
        
    with st.expander("💡 Visual Environment"):
        st.markdown(f"**Brightness:** {results['visual_environment']['brightness_level']:.2f}/255")
        st.markdown(f"**Lighting Type:** {results['visual_environment']['lighting_type']}")
        st.markdown(f"**Color Scheme:** {results['visual_environment']['color_scheme']}")
        st.markdown(f"**Visual Energy:** {results['visual_environment']['visual_energy']}")
        
    with st.expander("🕺 Crowd & Mood"):
        st.markdown(f"**Crowd Density:** {results['crowd_density']['crowd_density']}")
        st.markdown(f"**Activity Level:** {results['crowd_density']['activity_level']}")
        st.markdown(f"**Dominant Mood:** {results['mood_recognition']['dominant_mood']} (Confidence: {results['mood_recognition']['confidence']:.2f})")
        
        # Mood breakdown chart
        mood_df = pd.DataFrame(
            list(results['mood_recognition']['mood_breakdown'].items()),
            columns=['Mood', 'Confidence']
        ).sort_values(by='Confidence', ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(x='Confidence', y='Mood', data=mood_df, ax=ax, palette='viridis')
        ax.set_title("Mood Breakdown")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("")
        st.pyplot(fig)

def display_all_results_page():
    """Display a page with all results from the database, including stored videos."""
    st.subheader("Your Uploaded Videos")
    
    if st.session_state.user:
        user_videos = load_user_results(st.session_state.user.id)
        if user_videos:
            st.write(f"Showing {len(user_videos)} videos uploaded by you.")
            
            # Add a search bar
            search_term = st.text_input("Search your videos by venue name...", "")
            
            filtered_videos = [v for v in user_videos if search_term.lower() in v.get('venue_name', '').lower()]
            
            if not filtered_videos:
                st.info("No videos match your search criteria.")
            
            # Display each video result
            for video_data in filtered_videos:
                # Add a check to ensure video data is complete before displaying
                if 'id' in video_data and video_data['id']:
                    with st.expander(f"**{video_data['venue_name']}** ({video_data['venue_type']}) - {video_data['created_at'][:10]}"):
                        
                        # Display the actual stored video if a URL exists
                        video_url = video_data.get('video_url')
                        if video_url:
                            st.video(video_url)
                        else:
                            st.info("No video file was stored for this entry.")
                        
                        # Display key metrics
                        col_m1, col_m2 = st.columns(2)
                        col_m1.metric("Overall Vibe", video_data.get('overall_vibe', 'N/A'))
                        col_m2.metric("Energy Score", f"{video_data.get('energy_score', 0):.2f}/100")
                        
                        # Add a rating slider and a button to rate the video
                        rating = st.slider(f"Rate this video (1-5):", 1, 5, 3, key=f"slider_{video_data['id']}")
                        if st.button(f"Submit Rating for {video_data['venue_name']}", key=f"button_{video_data['id']}"):
                            if save_user_rating(video_data['id'], st.session_state.user.id, rating, video_data['venue_name'], video_data['venue_type']):
                                st.success("Your rating has been submitted!")
                            else:
                                st.error("There was an error submitting your rating.")
                                
                        # Display detailed data (optional)
                        st.json(video_data)
                else:
                    st.error("❌ A video record was found but is missing a unique ID. Skipping display.")
                    
        else:
            st.info("You have not uploaded any videos yet. Upload one from the 'Upload & Analyze' page!")
    else:
        st.warning("Please log in to view your uploaded videos.")

def handle_login(email, password):
    try:
        user = supabase.auth.sign_in_with_password({"email": email, "password": password}).user
        st.session_state.user = user
        st.success("Logged in successfully!")
    except Exception as e:
        st.error(f"Login failed: {e}")

def handle_signup(email, password):
    try:
        user = supabase.auth.sign_up({"email": email, "password": password}).user
        st.session_state.user = user
        st.success("Signed up and logged in successfully! Check your email to confirm.")
    except Exception as e:
        st.error(f"Sign up failed: {e}")

def handle_logout():
    supabase.auth.sign_out()
    st.session_state.user = None
    st.success("Logged out successfully!")

def main():
    st.markdown('<div class="main-header"><h1>SneakPeak Video Scorer</h1><p>A tool for real-time venue intelligence</p></div>', unsafe_allow_html=True)

    # Use a radio button to switch between pages
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload & Analyze", "View My Videos"])
    
    st.sidebar.header("User Account")
    if st.session_state.user:
        st.sidebar.success(f"Logged in as {st.session_state.user.email}")
        if st.sidebar.button("Log Out"):
            handle_logout()
            st.rerun()
    else:
        with st.sidebar.form("auth_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            
            login_button = st.form_submit_button("Log In")
            signup_button = st.form_submit_button("Sign Up")
            
            if login_button:
                handle_login(email, password)
                st.rerun()
            if signup_button:
                handle_signup(email, password)
                st.rerun()
        st.sidebar.text("You are currently not logged in.")

    if page == "Upload & Analyze":
        st.header("Upload a Video")
        if not st.session_state.user:
            st.warning("Please log in to upload and analyze a video.")
        else:
            st.info(f"You are logged in as {st.session_state.user.email}.")
            
            # Form for venue details and analysis
            with st.form("analysis_form"):
                st.subheader("Enter Venue Details")
                venue_name = st.text_input("Venue Name", "Demo Nightclub", key="venue_name_input")
                venue_type = st.selectbox("Venue Type", ["Club", "Bar", "Lounge", "Concert Hall"], key="venue_type_input")
                
                # Button to get GPS data
                st.subheader("GPS Location")
                if st.form_submit_button("Get Current Location"):
                    # Call the JS component to get location
                    get_location_component()
                    # Rerun the app to check for updated session state
                    st.rerun()

                # Display the fetched location or a message
                if st.session_state.user_location:
                    if st.session_state.user_location.get('error'):
                        st.error(f"Error getting location: {st.session_state.user_location['error']}")
                        latitude, longitude, accuracy = None, None, None
                    else:
                        latitude = st.session_state.user_location['latitude']
                        longitude = st.session_state.user_location['longitude']
                        accuracy = st.session_state.user_location['accuracy']
                        st.success("✅ Location fetched successfully! The data will be saved with the video.")
                else:
                    st.info("Click 'Get Current Location' to fetch GPS coordinates.")
                    latitude, longitude, accuracy = None, None, None

                # File uploader and analysis button inside the form
                uploaded_file = st.file_uploader("Choose a video file...", type=['mp4', 'mov', 'avi'])
                submitted = st.form_submit_button("Start Analysis")
                
                if submitted:
                    if uploaded_file:
                        if latitude is None or longitude is None:
                            st.error("Please get your GPS location before starting the analysis.")
                        elif not st.session_state.user:
                            st.error("Please log in to upload a video.")
                        else:
                            with st.spinner("Analyzing video..."):
                                # Simulate video processing
                                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                                tfile.write(uploaded_file.getvalue())
                                temp_path = tfile.name
                                tfile.close()

                                # Simulate analysis functions
                                audio_features = extract_audio_features(temp_path)
                                visual_features = analyze_visual_features(temp_path)
                                crowd_features = analyze_crowd_features(temp_path)
                                mood_features = analyze_mood_recognition(temp_path)
                                
                                # Verify location
                                is_verified = verify_venue_location(latitude, longitude, venue_name)

                                # Construct results dictionary
                                results = {
                                    "venue_name": venue_name,
                                    "venue_type": venue_type,
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "gps_data": {
                                        "latitude": latitude,
                                        "longitude": longitude,
                                        "accuracy": accuracy,
                                        "venue_verified": is_verified
                                    },
                                    "audio_environment": audio_features,
                                    "visual_environment": visual_features,
                                    "crowd_density": crowd_features,
                                    "mood_recognition": mood_features
                                }
                                
                                # Calculate energy score
                                results["energy_score"] = calculate_energy_score(results)
                                
                                # Save results and video
                                success, video_id = save_to_supabase(results, uploaded_file)
                                if success:
                                    st.success(f"Video saved with ID: {video_id}")
                                    saved_video_data = load_video_by_id(video_id)
                                    if saved_video_data:
                                        st.session_state.processed_videos.append(saved_video_data)
                                        display_results(saved_video_data)
                                    
                                # Clean up temp file
                                os.unlink(temp_path)
                    else:
                        st.error("Please upload a video file to proceed with the analysis.")
    
    elif page == "View My Videos":
        display_all_results_page()

if __name__ == "__main__":
    main()
