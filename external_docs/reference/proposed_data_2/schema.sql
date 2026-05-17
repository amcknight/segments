-- proposed_data_2: SpinLab-side counter-proposal to reference/proposed_data
--
-- Each row in `attempts` is ONE ENDING EVENT (death OR survival) of a single
-- segment attempt-instance. `time_ms` is wall-clock milliseconds from the
-- start of that attempt-instance to the event itself, with no death-penalty
-- time baked in.
--
-- Scope: cold-starts-only. Every row represents an attempt-instance that
-- began from the segment's cold-fill state (no prior in-attempt deaths).
-- Warm-start / per-respawn events would need an additional column to flag
-- start condition; deliberately deferred to keep the fixture lean.
--
-- Changes from proposed_data v1:
--   + created_at INTEGER NOT NULL DEFAULT (unixepoch())
--     unix-seconds at the moment the event was recorded. Required for any
--     time-aware model behavior (Kalman drift between attempts, fatigue,
--     session boundaries derived from inter-event gaps, etc.). Cannot be
--     recovered after the fact, so recording it now is cheap insurance even
--     if the current fit ignores it.
--
-- Intentionally NOT added (defer until the PGM asks):
--   - attempt_group_id (would group events from one warm-start attempt)
--   - start_kind ('segment_start' | 'respawn')
--   - power_state (small | mushroom | etc.)
--   - notes / session_id / segment_version
--   - 'aborted' or 'reset' as an outcome — those rows are dropped at export.

CREATE TABLE IF NOT EXISTS attempts (
    id         INTEGER PRIMARY KEY,
    game       TEXT NOT NULL,
    segment    TEXT NOT NULL,
    outcome    TEXT NOT NULL CHECK (outcome IN ('survived', 'died')),
    time_ms    INTEGER NOT NULL,
    created_at INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX IF NOT EXISTS ix_attempts_seg ON attempts (game, segment, id);
