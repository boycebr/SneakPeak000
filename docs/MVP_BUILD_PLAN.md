# SneakPeak MVP Build Plan

**Last Updated:** 2026-02-23
**Goal:** Ship a working mobile-first web app where users upload short venue videos, get an AI-generated "energy score," and browse nearby venues — all with full face-blurring privacy.

---

## Current State Summary

### What Exists
- **`streamlit_app.py`** (957 lines) — monolith containing the full prototype:
  - 3-tab UI: Discover, Upload, Profile
  - Video upload → frame extraction → face detection (Google Vision) → visual analysis → crowd estimation → energy score
  - Supabase REST API integration (save results, fetch venues)
  - Mobile-optimized CSS, Plotly donut chart visualization
- **`requirements.txt`** — all major dependencies declared
- **`.devcontainer/`** — GitHub Codespaces ready (Python 3.11)
- **`.gitignore`** — properly excludes `.env` and `.streamlit/secrets.toml`

### What Works (with correct API keys)
- Video metadata extraction (MoviePy + OpenCV fallback)
- Frame extraction from uploaded videos
- Face detection via Google Cloud Vision API
- Face blurring via Gaussian blur on detected regions
- Visual analysis (brightness, saturation, lighting classification)
- Crowd density estimation from face counts
- Energy score calculation (weighted composite)
- Save/fetch results to/from Supabase
- Plotly energy donut chart

### What's Missing for MVP
| Gap | Severity | Notes |
|-----|----------|-------|
| API keys hardcoded in source | **CRITICAL** | Must load from `.env` / secrets |
| No Supabase database schema | **CRITICAL** | Tables may not exist yet |
| Audio analysis is fully faked | **HIGH** | Random numbers seeded on file path |
| No user authentication | **HIGH** | README promises Supabase auth |
| Discover filters are non-functional | **MEDIUM** | Buttons set state but query ignores it |
| No device geolocation | **MEDIUM** | Hardcoded NYC coords |
| No video storage (Supabase Storage) | **MEDIUM** | Only metadata saved, no video/thumbnail |
| Azure Face API not integrated | **LOW** | Stub only; Google Vision works |
| AWS Rekognition not integrated | **LOW** | Stub only |
| No tests | **LOW** | Acceptable for MVP |
| InsightFace dependency unused | **LOW** | Dead dependency in requirements |

---

## Phased Build Plan

### Phase 0: Foundation (Sequential — do first)
> These are prerequisites for everything else.

| Task | Description | Files |
|------|-------------|-------|
| 0.1 | Run `database_schema.sql` in Supabase SQL Editor | `config/database_schema.sql` |
| 0.2 | Refactor `streamlit_app.py` to import from `config/settings.py` instead of hardcoded keys | `streamlit_app.py`, `config/settings.py` |
| 0.3 | Verify Supabase connection with new credentials from `.env` | manual test |

**Estimated effort:** 30 minutes
**Must complete before:** Phases 1-3

---

### Phase 1: Privacy Pipeline (HIGHEST PRIORITY)
> Face detection + blurring is the core trust feature. Users won't upload if faces aren't protected.

| Task | Description | Parallelizable? |
|------|-------------|-----------------|
| 1.1 | Extract face detection logic into `utils/api_clients.py` | Yes (Agent A) |
| 1.2 | Extract blurring logic into `utils/video_processing.py` | Yes (Agent A) |
| 1.3 | Add Azure Face API as fallback in `utils/api_clients.py` | Yes (Agent B) |
| 1.4 | Build unified `detect_faces()` that tries Google → Azure → AWS in order | After 1.1 + 1.3 |
| 1.5 | Add face-blur preview in Upload results (show blurred thumbnail) | After 1.2 |

**Parallel streams:**
- **Agent A:** Google Vision extraction + blur pipeline (1.1, 1.2)
- **Agent B:** Azure Face API integration (1.3)
- **Sequential:** Unified detector (1.4) then preview UI (1.5)

---

### Phase 2: Video Upload + Storage (HIGH PRIORITY)
> Complete the upload pipeline: validate → process → store → display.

| Task | Description | Parallelizable? |
|------|-------------|-----------------|
| 2.1 | Add video duration/size validation (5s min, 60s max, 100MB max) | Yes (Agent A) |
| 2.2 | Set up Supabase Storage bucket for videos + thumbnails | Yes (Agent B) |
| 2.3 | Upload processed video + blurred thumbnail to Supabase Storage | After 2.2 |
| 2.4 | Save storage URLs alongside analysis results in `video_results` | After 2.3 |
| 2.5 | Display thumbnails on Discover page venue cards | After 2.4 |

**Parallel streams:**
- **Agent A:** Validation logic (2.1)
- **Agent B:** Supabase Storage setup (2.2)
- **Sequential:** Upload flow (2.3 → 2.4 → 2.5)

---

### Phase 3: Energy Score + Analysis Engine (HIGH PRIORITY)
> Make the scoring real and useful.

| Task | Description | Parallelizable? |
|------|-------------|-----------------|
| 3.1 | Replace simulated audio with real BPM/volume extraction (Librosa) | Yes (Agent A) |
| 3.2 | Improve visual analysis — add motion detection for activity level | Yes (Agent B) |
| 3.3 | Add mood estimation from face expressions (joy/sorrow from Vision API) | Yes (Agent B) |
| 3.4 | Recalibrate energy score weights with real data | After 3.1 + 3.2 + 3.3 |
| 3.5 | Add energy score history / trending for venues | After 3.4 |

**Parallel streams:**
- **Agent A:** Real audio analysis (3.1)
- **Agent B:** Visual + mood improvements (3.2, 3.3)
- **Sequential:** Score recalibration (3.4) then trending (3.5)

**Note on audio:** If Librosa integration proves too heavy for Streamlit Cloud, fall back to keeping simulated audio but clearly label it as "estimated" in the UI.

---

### Phase 4: Authentication + User Profiles (MEDIUM PRIORITY)
> Not blocking for a demo, but needed before any public launch.

| Task | Description | Parallelizable? |
|------|-------------|-----------------|
| 4.1 | Add Supabase Auth (email/password signup + login) | Yes (Agent A) |
| 4.2 | Create user profile page with submission history | Yes (Agent B) |
| 4.3 | Gate video upload behind auth (browsing stays public) | After 4.1 |
| 4.4 | Track per-user stats (videos submitted, reputation) | After 4.1 + 4.2 |

---

### Phase 5: Discover Page + UX Polish (MEDIUM PRIORITY)
> Make the browse experience actually useful.

| Task | Description | Parallelizable? |
|------|-------------|-----------------|
| 5.1 | Wire up Discover filters (Hot Now = high energy, Chill = low, Recent = time sort) | Yes (Agent A) |
| 5.2 | Add browser geolocation API via Streamlit components | Yes (Agent B) |
| 5.3 | Add venue search / name filtering | Yes (Agent A) |
| 5.4 | Add auto-refresh / "live" indicator for recent uploads | After 5.1 |

---

### Phase 6: Deployment + Hardening (BEFORE LAUNCH)

| Task | Description |
|------|-------------|
| 6.1 | Set up `.streamlit/secrets.toml` for Streamlit Cloud deployment |
| 6.2 | Clean up requirements.txt (remove unused deps like insightface) |
| 6.3 | Add rate limiting on uploads (prevent abuse) |
| 6.4 | Add basic error monitoring / logging |
| 6.5 | Update README with actual deploy link and setup instructions |
| 6.6 | Write basic tests for core functions |

---

## Dependency Graph

```
Phase 0 (Foundation)
  ├── Phase 1 (Privacy) ──────┐
  ├── Phase 2 (Upload+Storage) ├── Phase 5 (Discover UX)
  ├── Phase 3 (Energy Scoring) ┘
  └── Phase 4 (Auth) ──────────── Phase 6 (Deploy)
```

Phases 1, 2, 3, and 4 can all run in **parallel** after Phase 0 completes.
Phase 5 depends on 1-3 being done. Phase 6 is final.

---

## Agent Team Parallelization Map

For maximum speed with agent teams, here's how to split work across concurrent agents:

| Agent | Phase 0 | Phase 1 | Phase 2 | Phase 3 |
|-------|---------|---------|---------|---------|
| **Agent A** | Refactor settings import | Google Vision + blur pipeline | Video validation | Real audio (Librosa) |
| **Agent B** | Run SQL schema | Azure Face fallback | Supabase Storage | Visual + mood analysis |
| **Agent C** | — | — | — | Score recalibration |

After Phase 0 completes (~30 min), Agents A and B can work independently on Phases 1-3 simultaneously, with Agent C joining for integration tasks.

---

## MVP Definition of Done

The MVP is shippable when:
- [ ] User can upload a video (5-60 seconds)
- [ ] Faces are detected and blurred before any display/storage
- [ ] Energy score is calculated and shown with breakdown
- [ ] Results are saved to Supabase and visible on Discover page
- [ ] API keys are loaded from environment (not hardcoded)
- [ ] Database schema exists and works
- [ ] App runs on Streamlit Cloud or equivalent
