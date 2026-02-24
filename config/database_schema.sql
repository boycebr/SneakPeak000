-- SneakPeak MVP Database Schema
-- Run this in Supabase SQL Editor to set up the required tables.

-- ============================================
-- 1. video_results — core analysis data
-- ============================================
CREATE TABLE IF NOT EXISTS video_results (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now(),

    -- Venue info
    venue_name TEXT NOT NULL,
    venue_type TEXT NOT NULL DEFAULT 'Other',
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,

    -- Energy score (composite)
    energy_score NUMERIC(5,1) DEFAULT 0,

    -- Audio analysis
    bpm INTEGER,
    volume_level NUMERIC(5,1),
    genre TEXT,

    -- Visual analysis
    brightness_level NUMERIC(5,1),
    lighting_type TEXT,
    visual_energy TEXT,

    -- Crowd analysis
    crowd_density TEXT,
    density_score NUMERIC(5,1),
    estimated_people INTEGER DEFAULT 0,

    -- Privacy
    face_count INTEGER DEFAULT 0,
    privacy_protected BOOLEAN DEFAULT TRUE,

    -- Processing
    processing_complete BOOLEAN DEFAULT FALSE,
    session_id TEXT,
    video_duration NUMERIC(8,2) DEFAULT 0,

    -- Video storage (for future use)
    video_url TEXT,
    thumbnail_url TEXT
);

-- Index for Discover page queries
CREATE INDEX IF NOT EXISTS idx_video_results_created_at ON video_results(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_video_results_energy ON video_results(energy_score DESC);
CREATE INDEX IF NOT EXISTS idx_video_results_venue_type ON video_results(venue_type);

-- Enable Row Level Security
ALTER TABLE video_results ENABLE ROW LEVEL SECURITY;

-- Public read access (anyone can browse venues)
CREATE POLICY "Public read access" ON video_results
    FOR SELECT USING (true);

-- Authenticated insert (any logged-in user can submit)
CREATE POLICY "Authenticated insert" ON video_results
    FOR INSERT WITH CHECK (true);

-- ============================================
-- 2. user_profiles — optional, for auth
-- ============================================
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID REFERENCES auth.users(id) PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now(),
    display_name TEXT,
    avatar_url TEXT,
    videos_submitted INTEGER DEFAULT 0,
    reputation_score INTEGER DEFAULT 0
);

ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read all profiles" ON user_profiles
    FOR SELECT USING (true);

CREATE POLICY "Users can update own profile" ON user_profiles
    FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile" ON user_profiles
    FOR INSERT WITH CHECK (auth.uid() = id);
