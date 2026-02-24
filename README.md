# SneakPeak

Real-time venue intelligence platform. Upload a short video of any venue, get an AI-generated energy score, and discover what's happening nearby.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sneakpeak.streamlit.app)

## Features

- **Video analysis** — upload 5-60s venue clips for automated scoring
- **Energy score** — weighted composite of audio (BPM, volume), visual (brightness, motion), crowd density, and mood
- **Privacy-first** — faces detected via Google Vision / Azure / OpenCV and blurred before storage
- **Real audio analysis** — BPM and volume extraction via Librosa (simulated fallback)
- **Discover page** — search, filter by type, sort by energy/recency, nearby with GPS
- **Authentication** — Supabase email/password auth, uploads gated behind login
- **Community ratings** — rate venues on vibe accuracy, leave comments
- **Auto-refresh** — live polling mode on the Discover page
- **Mobile-first** — responsive CSS, touch-friendly controls

## Architecture

```
streamlit_app.py          Main Streamlit app (UI + orchestration)
config/
  settings.py             Central config loader (.env / Streamlit secrets)
  database_schema.sql     Reference SQL for Supabase tables
utils/
  api_clients.py          Face detection (Google Vision -> Azure -> OpenCV)
  video_processing.py     Frame extraction, blur, motion, mood, thumbnails
  audio_analysis.py       Librosa BPM/volume + simulated fallback
  database.py             Supabase REST helpers (CRUD, storage, search)
  auth.py                 Supabase GoTrue auth (signup, login, logout)
tests/
  test_privacy_pipeline.py
```

**Backend:** Supabase (PostgreSQL + Storage + Auth)
**Face detection:** Google Cloud Vision API -> Azure Face API -> OpenCV Haar (3-tier fallback)
**Audio:** Librosa (BPM, RMS volume, spectral centroid)

## Setup

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USER/SneakPeak000.git
cd SneakPeak000
pip install -r requirements.txt
```

### 2. Configure environment

Copy `.env.example` or create `.env` with:

```
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_KEY=eyJ...
GOOGLE_CLOUD_API_KEY=AIza...
AZURE_FACE_API_KEY=          # optional
AZURE_FACE_ENDPOINT=         # optional
MAX_UPLOAD_SIZE_MB=100
VIDEO_MIN_DURATION=5
VIDEO_MAX_DURATION=60
```

### 3. Supabase setup

- Create a Supabase project at [supabase.com](https://supabase.com)
- Run `config/database_schema.sql` in the SQL Editor to create tables
- Create a Storage bucket called `venue-media` (public, 50MB limit)

### 4. Run locally

```bash
streamlit run streamlit_app.py
```

### 5. Deploy to Streamlit Cloud

1. Push to GitHub
2. Connect repo at [share.streamlit.io](https://share.streamlit.io)
3. Add secrets in the dashboard (see `.streamlit/secrets.toml.example`)
4. Deploy

## API Keys

| Service | Purpose | Required? |
|---------|---------|-----------|
| Supabase | Database, storage, auth | Yes |
| Google Cloud Vision | Face detection (primary) | Recommended |
| Azure Face API | Face detection (fallback) | Optional |
| Librosa (local) | Audio analysis | Bundled |
| OpenCV (local) | Face detection (offline fallback) | Bundled |

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```
