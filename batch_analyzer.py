"""
batch_analyzer.py — Batch YouTube Ad Analysis Pipeline

Searches for 20 kitchen/home-remodeling ads, analyzes each with Claude Vision,
and writes results to ads_cache.json. Supports incremental runs (only new videos
are analyzed on subsequent invocations).

Usage:
    python batch_analyzer.py           # real Claude (~$0.20 total)
    python batch_analyzer.py --mock    # dummy labels, free
    python batch_analyzer.py --reset   # delete cache and re-run all
"""

# =============================================================================
# Imports + constants
# =============================================================================
import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Import the single-video pipeline as a module (no code duplication)
from analyze_one_ad import (
    PYTHON_EXE,
    FRAME_INTERVAL_SEC,
    MAX_FRAMES,
    INPUT_PRICE_PER_MTOK,
    OUTPUT_PRICE_PER_MTOK,
    MockUsage,
    check_dependencies,
    download_video,
    extract_frames,
    get_transcript,
    build_claude_payload,
    call_claude,
)

# ---------------------------------------------------------------------------
# Search configuration
# ---------------------------------------------------------------------------
QUERIES = [
    # Kitchen-specific (primary)
    "kitchen remodel before after advertisement",
    "kitchen renovation commercial ad",
    "kitchen makeover tv commercial",
    "kitchen cabinet remodel ad commercial",
    "kitchen redesign advertisement",
    "new kitchen installation commercial",
    # Home remodeling fallbacks (used once kitchen queries are exhausted)
    "bathroom renovation before after ad",
    "bathroom remodel tv commercial",
    "home remodeling commercial advertisement",
    "home renovation tv commercial",
    "basement remodel commercial ad",
    "room addition remodel commercial",
    "house renovation before after advertisement",
    "home improvement contractor commercial",
    "flooring renovation commercial ad",
]
TARGET = 20
VIDEOS_PER_QUERY = 6
MIN_VIEWS = 1_000
MAX_DURATION = 180
MIN_DURATION = 15

CACHE_PATH = Path(__file__).parent / "ads_cache.json"


# =============================================================================
# SECTION 1 — search_videos(query, n) -> list[dict]
# =============================================================================
def search_videos(query: str, n: int = VIDEOS_PER_QUERY) -> list[dict]:
    """Search YouTube for up to n videos; filter by duration and views."""
    cmd = [
        PYTHON_EXE, "-m", "yt_dlp",
        f"ytsearch{n}:{query}",
        "--print", (
            "%(id)s\t%(title)s\t%(view_count)s\t%(like_count)s"
            "\t%(comment_count)s\t%(uploader)s\t%(upload_date)s"
            "\t%(webpage_url)s\t%(duration)s"
        ),
        "--no-download",
        "--no-playlist",
        "--quiet",
        "--ignore-errors",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        print(f"  [WARN] Search timed out for query: {query!r}")
        return []

    videos = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        try:
            v = {
                "id": parts[0].strip(),
                "title": parts[1].strip(),
                "view_count": int(parts[2]) if parts[2] not in ("NA", "None", "") else 0,
                "like_count": int(parts[3]) if parts[3] not in ("NA", "None", "") else 0,
                "comment_count": int(parts[4]) if parts[4] not in ("NA", "None", "") else 0,
                "uploader": parts[5].strip(),
                "upload_date": parts[6].strip(),
                "webpage_url": parts[7].strip(),
                "duration": int(parts[8]) if parts[8] not in ("NA", "None", "") else 0,
            }
            if MIN_DURATION <= v["duration"] <= MAX_DURATION and v["view_count"] >= MIN_VIEWS:
                videos.append(v)
        except (ValueError, IndexError):
            continue

    return videos


# =============================================================================
# SECTION 2 — collect_candidates(queries, target) -> list[dict]
# =============================================================================
def collect_candidates(
    queries: list[str] = QUERIES,
    target: int = TARGET,
) -> list[dict]:
    """Loop queries, deduplicate by video_id, stop at target unique videos."""
    seen_ids: set[str] = set()
    candidates: list[dict] = []

    for query in queries:
        if len(candidates) >= target:
            break
        print(f"Searching: {query!r}")
        videos = search_videos(query)
        added = 0
        for v in videos:
            if v["id"] not in seen_ids:
                seen_ids.add(v["id"])
                candidates.append(v)
                added += 1
                if len(candidates) >= target:
                    break
        print(f"  -> {len(videos)} results, +{added} new  ({len(candidates)} total)")

    return candidates[:target]


# =============================================================================
# SECTION 3 — Mock label generation (deterministic per video_id)
# =============================================================================
_TONES = ["upbeat", "serious", "inspirational", "urgent", "calm"]
_EMOTIONAL_SETS = [
    ["aspiration", "pride"],
    ["trust", "family"],
    ["fear", "aspiration"],
    ["humor", "family"],
    ["pride"],
    ["aspiration", "trust"],
    ["family", "pride"],
]
_THEMES = [
    "before_after_transformation", "lifestyle", "testimonial",
    "product_demo", "problem_solution", "price_offer",
]
_HOOK_TYPES = [
    "visual_transformation", "pain_point", "question",
    "shocking_stat", "offer", "celebrity",
]
_NARRATOR = [
    "voiceover", "on_screen_talent", "customer_testimonial",
    "text_only", "mixed",
]
_PACING = ["fast_cuts", "medium", "slow_cinematic"]
_PALETTES = ["warm", "cool", "neutral", "high_contrast"]
_MUSIC = ["upbeat", "calm", "dramatic", "tense", "none"]
_SETTINGS = ["interior", "exterior", "studio", "mixed"]
_CTA_TYPES = ["phone_number", "website", "visit_store", "limited_time_offer"]
_REVEAL = ["early (<10s)", "mid (10-30s)", "late (>30s)", "never"]
_AUDIENCES = ["homeowners_general", "luxury", "budget_conscious", "diy", "families"]


def _mock_labels(video_id: str, duration: int) -> dict:
    """Generate deterministic varied mock labels seeded by video_id."""
    seed = int(hashlib.md5(video_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    has_cta = rng.random() > 0.3
    return {
        "tone": rng.choice(_TONES),
        "emotional_appeal": rng.choice(_EMOTIONAL_SETS),
        "theme": rng.choice(_THEMES),
        "product_reveal_timing": rng.choice(_REVEAL),
        "hook_type": rng.choice(_HOOK_TYPES),
        "narrator_type": rng.choice(_NARRATOR),
        "pacing": rng.choice(_PACING),
        "color_palette": rng.choice(_PALETTES),
        "music_mood": rng.choice(_MUSIC),
        "setting": rng.choice(_SETTINGS),
        "has_cta": has_cta,
        "cta_type": rng.choice(_CTA_TYPES) if has_cta else "none",
        "has_before_after": rng.random() > 0.4,
        "has_price_mention": rng.random() > 0.6,
        "ad_length_seconds": duration,
        "target_audience": rng.choice(_AUDIENCES),
    }


# =============================================================================
# SECTION 4 — analyze_video(video_meta, mock) -> dict
# =============================================================================
def analyze_video(video_meta: dict, mock: bool = False) -> dict:
    """Download + frame-extract + transcribe + label one video."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="yt_ad_"))
    t_start = time.time()

    try:
        video_path = download_video(video_meta["webpage_url"], tmp_dir)
        frames, timestamps = extract_frames(video_path, FRAME_INTERVAL_SEC, MAX_FRAMES)
        transcript = get_transcript(video_meta["id"])

        if mock:
            labels = _mock_labels(video_meta["id"], video_meta["duration"])
            usage = MockUsage()
        else:
            content = build_claude_payload(
                frames, timestamps, transcript, video_meta["duration"]
            )
            labels, usage = call_claude(content)
            labels["ad_length_seconds"] = video_meta["duration"]

        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        cost = (
            input_tokens / 1_000_000 * INPUT_PRICE_PER_MTOK
            + output_tokens / 1_000_000 * OUTPUT_PRICE_PER_MTOK
        )

        return {
            **video_meta,
            **labels,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
            "elapsed_sec": round(time.time() - t_start, 1),
            "error": None,
        }

    except Exception as exc:
        null_labels = {
            "tone": None, "emotional_appeal": None, "theme": None,
            "product_reveal_timing": None, "hook_type": None,
            "narrator_type": None, "pacing": None, "color_palette": None,
            "music_mood": None, "setting": None, "has_cta": None,
            "cta_type": None, "has_before_after": None, "has_price_mention": None,
            "ad_length_seconds": None, "target_audience": None,
        }
        return {
            **video_meta,
            **null_labels,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "elapsed_sec": round(time.time() - t_start, 1),
            "error": str(exc),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# SECTION 5 — Cache I/O
# =============================================================================
def load_cache(path: Path = CACHE_PATH) -> list[dict]:
    """Load existing cache; return empty list on missing or corrupt file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return []


def save_cache(results: list[dict], path: Path = CACHE_PATH) -> None:
    """Atomic write: write to .tmp then rename to avoid corrupt cache on crash."""
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


# =============================================================================
# SECTION 6 — run_batch(cache_path, mock, reset) -> list[dict]
# =============================================================================
def run_batch(
    cache_path: Path = CACHE_PATH,
    mock: bool = False,
    reset: bool = False,
) -> list[dict]:
    """
    Main batch runner.
    - reset: delete existing cache and re-analyze all videos
    - mock: use deterministic dummy labels (no API calls, free)
    Returns final list of all results (cached + newly analyzed).
    """
    if reset and cache_path.exists():
        cache_path.unlink()
        print("Cache deleted.")

    existing = load_cache(cache_path)
    cached_ids = {r["id"] for r in existing}
    results = list(existing)

    print(f"\nCollecting {TARGET} video candidates...")
    candidates = collect_candidates()
    new_candidates = [c for c in candidates if c["id"] not in cached_ids]

    already_done = len(cached_ids)
    print(
        f"\n{already_done} already cached, "
        f"{len(new_candidates)} new to analyze (target: {TARGET})\n"
    )

    if not new_candidates:
        print("Nothing new to analyze.")
        return results

    for i, video_meta in enumerate(new_candidates, start=1):
        global_n = already_done + i
        total_n = already_done + len(new_candidates)
        print(
            f"\n[{global_n}/{total_n}] Analyzing: "
            f"{video_meta['title'][:60]} ({video_meta['duration']}s)"
        )

        row = analyze_video(video_meta, mock=mock)
        if row.get("error"):
            print(f"  [ERROR] {row['error']}")
        else:
            print(
                f"  done in {row.get('elapsed_sec', 0)}s, "
                f"cost ${row.get('cost_usd', 0):.4f}"
            )

        results.append(row)
        save_cache(results, cache_path)

    total_cost = sum(r.get("cost_usd", 0) or 0 for r in results)
    print(f"\nBatch complete. {len(results)} videos cached.")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Cache: {cache_path}")
    return results


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch YouTube Ad Analysis")
    parser.add_argument(
        "--mock", action="store_true",
        help="Use deterministic dummy labels (no API calls, free)"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete cache and re-analyze all videos"
    )
    args = parser.parse_args()

    check_dependencies()
    run_batch(mock=args.mock, reset=args.reset)
