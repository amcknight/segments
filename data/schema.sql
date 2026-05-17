CREATE TABLE IF NOT EXISTS attempts (
    id      INTEGER PRIMARY KEY,
    game    TEXT NOT NULL,
    segment TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('survived', 'died')),
    time_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_attempts_seg ON attempts (game, segment, id);
