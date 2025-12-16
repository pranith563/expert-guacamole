-- 001_create_modelhub.sql
-- Creates: database + schema + modelhub.models table for indexing ModelHub artifacts.

\set ON_ERROR_STOP on

-- ============
-- Change these
-- ============
\set DB_NAME 'agentic_suite'
\set DB_OWNER 'postgres'
-- ============

-- Create DB (run while connected to postgres db)
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = :'DB_NAME') THEN
    EXECUTE format('CREATE DATABASE %I OWNER %I', :'DB_NAME', :'DB_OWNER');
  END IF;
END $$;

\connect :DB_NAME

-- UUID generator
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS modelhub;

-- Main table: one row per artifact file (pb/onnx/pkl/etc.)
CREATE TABLE IF NOT EXISTS modelhub.models (
  model_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- official = from local mirror; overlay = user workspace copy
  source             TEXT NOT NULL CHECK (source IN ('official','overlay')),

  -- Keep NOT NULL to support a clean UNIQUE constraint (official uses empty string)
  owner_user_id      TEXT NOT NULL DEFAULT '',

  project            TEXT NOT NULL,
  version            TEXT NOT NULL,

  -- Derived from directory path under <project>/<version>/...
  variant_path       TEXT NOT NULL DEFAULT '',

  -- Filename info (supports v1p1.pb, v1p1a8w8.pb in same folder)
  artifact_name      TEXT NOT NULL,
  artifact_stem      TEXT NOT NULL,

  -- Store path relative to official root (recommended) or absolute if you prefer
  file_path          TEXT NOT NULL,

  file_type          TEXT NOT NULL,   -- pb/pkl/onnx/tflite/dlc/...
  framework          TEXT NULL,       -- tensorflow/pytorch/onnx/...

  size_bytes         BIGINT NOT NULL,
  sha256             TEXT NOT NULL,

  base_model_id      UUID NULL,       -- overlay clone provenance (optional)
  meta               JSONB NOT NULL DEFAULT '{}'::jsonb,

  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Uniqueness: one artifact per (source, owner, project, version, variant_path, artifact_name)
ALTER TABLE modelhub.models
  DROP CONSTRAINT IF EXISTS uq_modelhub_models_identity;

ALTER TABLE modelhub.models
  ADD CONSTRAINT uq_modelhub_models_identity
  UNIQUE (source, owner_user_id, project, version, variant_path, artifact_name);

-- Useful indexes for browsing/filtering
CREATE INDEX IF NOT EXISTS ix_modelhub_models_project_version
ON modelhub.models (project, version);

CREATE INDEX IF NOT EXISTS ix_modelhub_models_source_owner
ON modelhub.models (source, owner_user_id);

CREATE INDEX IF NOT EXISTS ix_modelhub_models_sha
ON modelhub.models (sha256);

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION modelhub.set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_modelhub_models_updated_at ON modelhub.models;

CREATE TRIGGER trg_modelhub_models_updated_at
BEFORE UPDATE ON modelhub.models
FOR EACH ROW EXECUTE FUNCTION modelhub.set_updated_at();

SELECT source, count(*) FROM modelhub.models GROUP BY source;

SELECT project, count(*) 
FROM modelhub.models
WHERE source='official'
GROUP BY project
ORDER BY count(*) DESC
LIMIT 20;

SELECT *
FROM modelhub.models
WHERE project='<some_project>' AND version='<some_version>'
ORDER BY variant_path, artifact_name
LIMIT 50;
