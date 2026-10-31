"""Forward-only schema migrations. Index N migrates user_version N to N + 1.

Never edit a released migration; append a new one. Transcript text stays in
integrity-checked meeting artifacts; the database holds metadata, processing
state, segment references, speaker mappings and validated extracted evidence.
"""

MIGRATIONS: tuple[tuple[str, ...], ...] = (
    (
        """CREATE TABLE meetings (
            id TEXT PRIMARY KEY CHECK (length(id) = 36),
            audio_sha256 TEXT NOT NULL CHECK (length(audio_sha256) = 64),
            attachment_kind TEXT CHECK (attachment_kind IN ('.txt', '.srt', '.json')),
            attachment_sha256 TEXT CHECK (length(attachment_sha256) = 64),
            duration REAL NOT NULL CHECK (duration > 0),
            container TEXT NOT NULL,
            imported_at TEXT NOT NULL,
            title TEXT CHECK (length(title) BETWEEN 1 AND 200),
            meeting_date TEXT
                CHECK (meeting_date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
            CHECK ((attachment_kind IS NULL) = (attachment_sha256 IS NULL))
        ) STRICT""",
        """CREATE UNIQUE INDEX meetings_identity
            ON meetings(audio_sha256, ifnull(attachment_sha256, ''))""",
        """CREATE TABLE jobs (
            id INTEGER PRIMARY KEY,
            meeting_id TEXT NOT NULL REFERENCES meetings(id),
            kind TEXT NOT NULL CHECK (kind IN ('full', 'notes')),
            mode TEXT NOT NULL CHECK (mode IN ('transcribe', 'align', 'compare')),
            force INTEGER NOT NULL DEFAULT 0 CHECK (force IN (0, 1)),
            state TEXT NOT NULL
                CHECK (state IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
            stage TEXT,
            attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
            max_attempts INTEGER NOT NULL CHECK (max_attempts >= 1),
            cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK (cancel_requested IN (0, 1)),
            worker TEXT,
            error_stage TEXT,
            error_class TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT
        ) STRICT""",
        """CREATE UNIQUE INDEX jobs_one_active
            ON jobs(meeting_id) WHERE state IN ('queued', 'running')""",
        "CREATE INDEX jobs_by_state ON jobs(state, id)",
        """CREATE TABLE stage_runs (
            id INTEGER PRIMARY KEY,
            job_id INTEGER NOT NULL REFERENCES jobs(id),
            stage TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN
                ('running', 'succeeded', 'skipped', 'failed', 'cancelled', 'interrupted')),
            started_at TEXT NOT NULL,
            finished_at TEXT,
            seconds REAL,
            error_class TEXT,
            detail TEXT
        ) STRICT""",
        "CREATE INDEX stage_runs_by_job ON stage_runs(job_id, id)",
        """CREATE TABLE checkpoints (
            meeting_id TEXT NOT NULL REFERENCES meetings(id),
            stage TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            artifacts TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (meeting_id, stage)
        ) STRICT""",
        """CREATE TABLE segments (
            meeting_id TEXT NOT NULL REFERENCES meetings(id),
            segment_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            start_s REAL NOT NULL,
            end_s REAL NOT NULL,
            speaker TEXT NOT NULL,
            overlap INTEGER NOT NULL CHECK (overlap IN (0, 1)),
            PRIMARY KEY (meeting_id, segment_id)
        ) STRICT""",
        """CREATE TABLE extraction_chunks (
            meeting_id TEXT NOT NULL REFERENCES meetings(id),
            cache_key TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (meeting_id, cache_key)
        ) STRICT""",
        """CREATE TABLE items (
            meeting_id TEXT NOT NULL REFERENCES meetings(id),
            item_id TEXT NOT NULL,
            category TEXT NOT NULL,
            statement TEXT,
            basis TEXT NOT NULL CHECK (basis IN ('explicit', 'inferred')),
            owner TEXT,
            deadline TEXT,
            disputed INTEGER NOT NULL CHECK (disputed IN (0, 1)),
            PRIMARY KEY (meeting_id, item_id)
        ) STRICT""",
        """CREATE TABLE item_evidence (
            meeting_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            segment_id TEXT NOT NULL,
            quote TEXT NOT NULL,
            PRIMARY KEY (meeting_id, item_id, position),
            FOREIGN KEY (meeting_id, item_id)
                REFERENCES items(meeting_id, item_id) ON DELETE CASCADE
        ) STRICT""",
        # Mappings/corrections are bound to one attributed transcript: a re-diarized
        # recording can renumber speakers, so stale entries are reported, never applied.
        """CREATE TABLE speaker_names (
            meeting_id TEXT NOT NULL REFERENCES meetings(id),
            transcript_sha256 TEXT NOT NULL,
            label TEXT NOT NULL CHECK (label GLOB 'Speaker [1-9]*'),
            name TEXT NOT NULL CHECK (length(name) BETWEEN 1 AND 100),
            confirmed_at TEXT NOT NULL,
            PRIMARY KEY (meeting_id, transcript_sha256, label)
        ) STRICT""",
        """CREATE TABLE speaker_corrections (
            meeting_id TEXT NOT NULL REFERENCES meetings(id),
            transcript_sha256 TEXT NOT NULL,
            segment_id TEXT NOT NULL,
            speaker TEXT NOT NULL,
            corrected_at TEXT NOT NULL,
            PRIMARY KEY (meeting_id, transcript_sha256, segment_id)
        ) STRICT""",
    ),
    # 2: every speaker label that occurs in an attributed transcript, including labels that
    # appear only on words inside mixed ("Uncertain") segments, so users can name them.
    (
        """CREATE TABLE speaker_labels (
            meeting_id TEXT NOT NULL REFERENCES meetings(id),
            label TEXT NOT NULL CHECK (label GLOB 'Speaker [1-9]*'),
            PRIMARY KEY (meeting_id, label)
        ) STRICT""",
    ),
    # 3: Phase 4 search. The FTS tables hold transcript text and extracted quotes so the
    # database is now as sensitive as transcript.json. Embeddings are optional and local.
    (
        """CREATE VIRTUAL TABLE segment_search USING fts5(
            text, meeting_id UNINDEXED, segment_id UNINDEXED,
            tokenize = 'unicode61 remove_diacritics 2'
        )""",
        """CREATE VIRTUAL TABLE item_search USING fts5(
            text, meeting_id UNINDEXED, item_id UNINDEXED, category UNINDEXED,
            tokenize = 'unicode61 remove_diacritics 2'
        )""",
        """CREATE TABLE segment_embeddings (
            meeting_id TEXT NOT NULL REFERENCES meetings(id),
            segment_id TEXT NOT NULL,
            model TEXT NOT NULL,
            dims INTEGER NOT NULL CHECK (dims > 0),
            vector BLOB NOT NULL CHECK (length(vector) = dims * 4),
            PRIMARY KEY (meeting_id, segment_id, model)
        ) STRICT""",
        "CREATE INDEX segment_embeddings_by_model ON segment_embeddings(model)",
    ),
)
