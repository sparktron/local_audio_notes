"""Reproducible local micro-benchmarks on synthetic data (no recordings, no models).

    .venv/bin/python scripts/benchmark_local.py --dir /path/on/the/data/volume

Measures SHA-256 full hashing versus the stat-keyed hash cache, FTS5 indexing and query
latency, and embedding similarity search. Synthetic text and random vectors only: these
numbers say nothing about transcription, diarization or LLM speed or quality.
"""

import argparse
import json
import os
import platform
import random
import sqlite3
import statistics
import tempfile
import time
from array import array
from pathlib import Path

from local_audio_notes.search.index import _similarities, fts_query
from local_audio_notes.storage.db import migrate
from local_audio_notes.storage.files import HashCache, sha256

WORDS = (  # noqa: SIM905
    "migration review storage format decision action deadline friday budget risk blocker "
    "question owner schedule release driver upgrade consent cloud local model transcript"
).split()


def timed(function, repeat=1):
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - start)
    return result, statistics.median(samples)


def hashing(directory: Path, mebibytes: int) -> dict:
    path = directory / "synthetic.bin"
    block = os.urandom(1024**2)
    with path.open("wb") as handle:
        for _ in range(mebibytes):
            handle.write(block)
    cache = HashCache(directory / "cache.json")
    _, full = timed(lambda: sha256(path))
    _, first = timed(lambda: cache.sha256(path))
    _, cached = timed(lambda: cache.sha256(path), repeat=5)
    path.unlink()
    return {
        "file_mib": mebibytes,
        "full_sha256_seconds": round(full, 3),
        "full_mib_per_second": round(mebibytes / full),
        "cache_miss_seconds": round(first, 3),
        "cache_hit_seconds": round(cached, 6),
        "note": "page cache is warm after writing; cold reads from disk are slower",
    }


def full_text(directory: Path, segments: int) -> dict:
    rng = random.Random(7)
    connection = sqlite3.connect(directory / "bench.sqlite3", isolation_level=None)
    migrate(connection)
    rows = [
        (" ".join(rng.choice(WORDS) for _ in range(14)), f"m{i // 1000:05d}", f"s{i:06d}")
        for i in range(segments)
    ]

    def build():
        connection.execute("BEGIN")
        connection.executemany(
            "INSERT INTO segment_search (text, meeting_id, segment_id) VALUES (?, ?, ?)", rows
        )
        connection.execute("COMMIT")

    _, build_seconds = timed(build)
    queries = ["migration friday", "driver upgrade blocker", "consent", "release schedule owner"]

    def run():
        for query in queries:
            connection.execute(
                "SELECT segment_id, bm25(segment_search) FROM segment_search "
                "WHERE segment_search MATCH ? ORDER BY bm25(segment_search) LIMIT 50",
                (fts_query(query),),
            ).fetchall()

    _, query_seconds = timed(run, repeat=5)
    connection.close()
    return {
        "segments": segments,
        "index_build_seconds": round(build_seconds, 2),
        "median_query_ms": round(query_seconds / len(queries) * 1000, 2),
        "sqlite": sqlite3.sqlite_version,
    }


def similarity(vectors: int, dims: int) -> dict:
    rng = random.Random(11)

    def unit():
        values = [rng.gauss(0, 1) for _ in range(dims)]
        norm = sum(v * v for v in values) ** 0.5
        return array("f", (v / norm for v in values))

    blobs = [unit().tobytes() for _ in range(vectors)]
    query = unit()
    _, seconds = timed(lambda: _similarities(query, blobs), repeat=3)
    try:
        import numpy  # noqa: F401

        backend = "numpy"
    except ImportError:
        backend = "pure-python"
    return {
        "vectors": vectors,
        "dims": dims,
        "backend": backend,
        "median_seconds": round(seconds, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, help="Scratch directory (default: system temp)")
    parser.add_argument("--hash-mib", type=int, default=1024)
    parser.add_argument("--segments", type=int, default=200_000)
    parser.add_argument("--vectors", type=int, default=100_000)
    parser.add_argument("--dims", type=int, default=768)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(dir=args.dir) as scratch:
        directory = Path(scratch)
        result = {
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "hashing": hashing(directory, args.hash_mib),
            "full_text_search": full_text(directory, args.segments),
            "semantic_similarity": similarity(args.vectors, args.dims),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
