-- ============================================
-- SneakPeak DB/Storage setup (idempotent)
-- - Ensures UUID user_id columns
-- - Enables RLS + own-row policies
-- - Adds unique constraint for ratings
-- - Allows authenticated uploads to 'videos' bucket
-- ============================================

-- A) Ensure video_results.user_id is UUID
DO $body$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name   = 'video_results'
      AND column_name  = 'user_id'
      AND data_type   <> 'uuid'
  ) THEN
    ALTER TABLE public.video_results
      ALTER COLUMN user_id TYPE uuid USING user_id::uuid;
  END IF;
END
$body$;

-- B) Ensure user_ratings.user_id exists and is UUID
ALTER TABLE public.user_ratings
  ADD COLUMN IF NOT EXISTS user_id uuid;

DO $body$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name   = 'user_ratings'
      AND column_name  = 'user_id'
      AND data_type   <> 'uuid'
  ) THEN
    ALTER TABLE public.user_ratings
      ALTER COLUMN user_id TYPE uuid USING user_id::uuid;
  END IF;
END
$body$;

-- C) Optional: set timestamp defaults if the columns exist
DO $body$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='public' AND table_name='user_ratings' AND column_name='created_at'
  ) THEN
    ALTER TABLE public.user_ratings ALTER COLUMN created_at SET DEFAULT now();
  END IF;

  IF EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema='public' AND table_name='user_ratings' AND column_name='rated_at'
  ) THEN
    ALTER TABLE public.user_ratings ALTER COLUMN rated_at SET DEFAULT now();
  END IF;
END
$body$;

-- D) Enable Row Level Security (safe if already enabled)
ALTER TABLE public.video_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_ratings ENABLE ROW LEVEL SECURITY;

-- E) Policies for video_results (select/insert own rows)
DO $body$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname='public' AND tablename='video_results' AND policyname='select own video results'
  ) THEN
    EXECUTE 'CREATE POLICY "select own video results"
             ON public.video_results
             FOR SELECT TO authenticated
             USING (user_id = auth.uid())';
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname='public' AND tablename='video_results' AND policyname='insert own video results'
  ) THEN
    EXECUTE 'CREATE POLICY "insert own video results"
             ON public.video_results
             FOR INSERT TO authenticated
             WITH CHECK (user_id = auth.uid())';
  END IF;
END
$body$;

-- F) Policies for user_ratings (select/insert/update/delete own rows)
DO $body$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname='public' AND tablename='user_ratings' AND policyname='select own ratings'
  ) THEN
    EXECUTE 'CREATE POLICY "select own ratings"
             ON public.user_ratings
             FOR SELECT TO authenticated
             USING (user_id = auth.uid())';
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname='public' AND tablename='user_ratings' AND policyname='insert own ratings'
  ) THEN
    EXECUTE 'CREATE POLICY "insert own ratings"
             ON public.user_ratings
             FOR INSERT TO authenticated
             WITH CHECK (user_id = auth.uid())';
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname='public' AND tablename='user_ratings' AND policyname='update own ratings'
  ) THEN
    EXECUTE 'CREATE POLICY "update own ratings"
             ON public.user_ratings
             FOR UPDATE TO authenticated
             USING (user_id = auth.uid())
             WITH CHECK (user_id = auth.uid())';
  END IF;

  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname='public' AND tablename='user_ratings' AND policyname='delete own ratings'
  ) THEN
    EXECUTE 'CREATE POLICY "delete own ratings"
             ON public.user_ratings
             FOR DELETE TO authenticated
             USING (user_id = auth.uid())';
  END IF;
END
$body$;

-- G) Optional: one rating per user per venue
DO $body$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint
    WHERE conname = 'uniq_user_venue_rating'
  ) THEN
    ALTER TABLE public.user_ratings
      ADD CONSTRAINT uniq_user_venue_rating UNIQUE (user_id, venue_id);
  END IF;
END
$body$;

-- H) Storage: allow authenticated uploads to the 'videos' bucket
DO $body$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_policies
    WHERE schemaname='storage'
      AND tablename='objects'
      AND policyname='Allow authenticated uploads to videos'
  ) THEN
    EXECUTE 'CREATE POLICY "Allow authenticated uploads to videos"
             ON storage.objects
             FOR INSERT
             TO authenticated
             WITH CHECK (bucket_id = ''videos'')';
  END IF;
END
$body$;

-- I) Verification (read-only)
-- Shows user_id types and active policies; safe to keep.
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_name IN ('video_results','user_ratings')
  AND column_name = 'user_id'
ORDER BY table_name;

SELECT schemaname, tablename, policyname, cmd, roles
FROM pg_policies
WHERE (schemaname='public' AND tablename IN ('video_results','user_ratings'))
   OR (schemaname='storage' AND tablename='objects')
ORDER BY schemaname, tablename, policyname;
