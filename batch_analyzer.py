"""
batch_analyzer.py — Batch YouTube Ad Analysis Pipeline

Searches for 200 home-remodeling ads across YouTube (primary) and Dailymotion
(fallback), analyzes each with Claude Vision, and writes results to
ads_cache.json. Supports incremental runs (only new videos are analyzed).

Usage:
    python batch_analyzer.py           # real Claude (~$2.00 total)
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
TARGET = 1000
VIDEOS_PER_QUERY = 15
MIN_VIEWS = 500
MAX_DURATION = 180
MIN_DURATION = 15

CACHE_PATH = Path(__file__).parent / "ads_cache.json"

# YouTube queries — ordered from most to least specific
YOUTUBE_QUERIES = [
    # Kitchen (primary focus)
    "kitchen remodel before after advertisement",
    "kitchen renovation commercial ad",
    "kitchen makeover tv commercial",
    "kitchen cabinet remodel ad commercial",
    "kitchen redesign advertisement",
    "kitchen countertop replacement commercial",
    "custom kitchen installation commercial",
    "kitchen remodeling company ad spot",
    "dream kitchen renovation commercial",
    "kitchen transformation before after ad",
    # Bathroom
    "bathroom renovation before after ad",
    "bathroom remodel tv commercial",
    "bathroom makeover commercial",
    "walk-in shower installation commercial",
    "bathtub replacement ad commercial",
    "bathroom tile renovation commercial",
    "master bath remodel commercial",
    "bathroom vanity replacement ad",
    # Basement & flooring
    "basement remodel commercial ad",
    "basement finishing renovation commercial",
    "hardwood floor installation commercial",
    "flooring renovation commercial ad",
    "carpet replacement home commercial",
    "tile floor installation ad",
    # Windows, doors, roofing, exterior
    "window replacement commercial advertisement",
    "door replacement home improvement ad",
    "roofing commercial advertisement",
    "siding replacement home commercial",
    "deck patio renovation commercial",
    "garage door replacement commercial",
    # HVAC, insulation, smart home
    "HVAC replacement home commercial ad",
    "home insulation renovation commercial",
    "smart home renovation commercial ad",
    # General home remodeling
    "home remodeling commercial advertisement",
    "home renovation tv commercial",
    "house renovation before after advertisement",
    "home improvement contractor commercial",
    "general contractor renovation commercial",
    "home makeover renovation ad",
    "whole home renovation commercial",
    "room addition remodel commercial",
    "home remodel before after tv spot",
    "house makeover commercial ad",
    # Big box & design brands
    "Home Depot renovation commercial",
    "Lowes home improvement commercial",
    "home design build remodel commercial",
    "kitchen bath showroom commercial",
    "home remodeling franchise commercial",
    # Interior & outdoor living
    "interior design renovation commercial",
    "open plan living renovation ad",
    "outdoor kitchen renovation commercial",
    "landscaping renovation commercial ad",
    "fence installation commercial ad",
    "pool installation commercial advertisement",
    # Specialty trades
    "plumbing renovation commercial ad",
    "electrical upgrade home commercial",
    "painting contractor commercial ad",
    "stucco siding renovation commercial",
    "attic insulation commercial ad",
    "solar panel home installation commercial",
    # Financing & services
    "home improvement loan commercial",
    "home warranty renovation commercial",
    "home staging renovation ad",
    # Home furnishing — living room
    "furniture store commercial advertisement",
    "sofa couch commercial ad",
    "living room furniture commercial",
    "sectional sofa commercial advertisement",
    "recliner furniture commercial ad",
    "Rooms To Go commercial",
    "Ashley Furniture commercial advertisement",
    "Wayfair furniture commercial ad",
    "IKEA home furnishing commercial",
    "Pottery Barn commercial advertisement",
    # Home furnishing — bedroom
    "bedroom furniture commercial advertisement",
    "mattress commercial advertisement",
    "memory foam mattress commercial ad",
    "bed frame headboard commercial",
    "Casper mattress commercial",
    "Purple mattress commercial ad",
    "Sleep Number commercial advertisement",
    # Home furnishing — dining & storage
    "dining room furniture commercial",
    "dining table commercial advertisement",
    "home storage furniture commercial",
    "bookshelf shelving unit commercial ad",
    "custom closet commercial advertisement",
    "closet organization system commercial",
    "garage storage organization commercial",
    # Home decor
    "home decor commercial advertisement",
    "interior decorating commercial ad",
    "area rug commercial advertisement",
    "curtains blinds window treatment commercial",
    "home lighting fixture commercial ad",
    "ceiling fan installation commercial ad",
    "wall art decor commercial advertisement",
    "home accessories decor commercial",
    # Kitchen appliances
    "kitchen appliance commercial advertisement",
    "refrigerator commercial advertisement",
    "dishwasher commercial ad",
    "range oven commercial advertisement",
    "microwave commercial ad",
    "kitchen renovation appliance commercial",
    # Bathroom fixtures & accessories
    "bathroom faucet commercial advertisement",
    "kitchen faucet commercial ad",
    "toilet replacement commercial advertisement",
    "bathroom accessories commercial ad",
    # Paint & wallpaper
    "interior paint commercial advertisement",
    "house painting commercial ad",
    "Sherwin Williams commercial",
    "Benjamin Moore paint commercial",
    "wallpaper home decor commercial",
    # Outdoor & patio furniture
    "patio furniture commercial advertisement",
    "outdoor furniture commercial ad",
    "backyard patio set commercial",
    "outdoor living space furniture commercial",
    # Smart home & tech
    "smart thermostat commercial advertisement",
    "home security system commercial ad",
    "smart lighting home commercial",
    "Ring doorbell commercial",
    "Nest thermostat commercial ad",
    # Plumbing & water
    "water heater replacement commercial ad",
    "water filtration system home commercial",
    "whole home water softener commercial",
    # Specialty home brands
    "West Elm commercial advertisement",
    "Restoration Hardware RH commercial",
    "Crate and Barrel commercial ad",
    "Williams Sonoma home commercial",
    "Pier 1 home decor commercial",
    "HomeGoods commercial advertisement",
    "TJMaxx home decor commercial",
    "Tuesday Morning home commercial",
    # Extra fill to reach 1000
    "CB2 furniture commercial advertisement",
    "Z Gallerie home decor commercial",
    "World Market home decor commercial ad",
    "Overstock furniture commercial advertisement",
    "laminate flooring installation commercial",
    "luxury vinyl plank flooring commercial",
    "kitchen island installation commercial",
    "home generator installation commercial ad",
]

# Dailymotion queries — used only after YouTube is exhausted
DAILYMOTION_QUERIES = [
    "kitchen remodel advertisement",
    "bathroom renovation commercial",
    "home remodeling ad",
    "house renovation commercial",
    "kitchen makeover ad",
    "home improvement commercial",
    "flooring renovation ad",
    "window replacement advertisement",
    "basement remodel commercial",
    "roofing commercial ad",
]


# =============================================================================
# SECTION 1 — search_videos(query, n, platform) -> list[dict]
# =============================================================================
def search_videos(
    query: str, n: int = VIDEOS_PER_QUERY, platform: str = "youtube"
) -> list[dict]:
    """Search for n videos on the given platform; apply duration/view filters."""
    if platform == "youtube":
        search_url = f"ytsearch{n}:{query}"
    elif platform == "dailymotion":
        search_url = f"https://www.dailymotion.com/search/{query.replace(' ', '%20')}/videos"
    else:
        return []

    cmd = [
        PYTHON_EXE, "-m", "yt_dlp",
        search_url,
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90,
                                creationflags=subprocess.CREATE_NO_WINDOW)
    except subprocess.TimeoutExpired:
        print(f"  [WARN] Search timed out: {platform} / {query!r}")
        return []

    videos = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        try:
            v = {
                "id":            parts[0].strip(),
                "title":         parts[1].strip(),
                "view_count":    int(parts[2]) if parts[2] not in ("NA", "None", "") else 0,
                "like_count":    int(parts[3]) if parts[3] not in ("NA", "None", "") else 0,
                "comment_count": int(parts[4]) if parts[4] not in ("NA", "None", "") else 0,
                "uploader":      parts[5].strip(),
                "upload_date":   parts[6].strip(),
                "webpage_url":   parts[7].strip(),
                "duration":      int(parts[8]) if parts[8] not in ("NA", "None", "") else 0,
                "platform":      platform,
            }
            if MIN_DURATION <= v["duration"] <= MAX_DURATION and v["view_count"] >= MIN_VIEWS:
                videos.append(v)
        except (ValueError, IndexError):
            continue

    return videos


# =============================================================================
# SECTION 2 — collect_candidates(target) -> list[dict]
# =============================================================================
SEARCH_WORKERS = 8  # parallel yt-dlp processes during search phase

def collect_candidates(target: int = TARGET) -> list[dict]:
    """
    Collect up to target unique videos.
    1. Try all YouTube queries in parallel (SEARCH_WORKERS threads).
    2. If still short, try Dailymotion queries as fallback.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    seen_ids: set[str] = set()
    candidates: list[dict] = []
    lock = threading.Lock()

    def _search_one(query, platform):
        return query, platform, search_videos(query, platform=platform)

    def _run_queries_parallel(queries, platform):
        with ThreadPoolExecutor(max_workers=SEARCH_WORKERS) as ex:
            futures = {ex.submit(_search_one, q, platform): q for q in queries}
            for fut in as_completed(futures):
                query, plat, videos = fut.result()
                with lock:
                    if len(candidates) >= target:
                        continue
                    added = 0
                    for v in videos:
                        if v["id"] not in seen_ids:
                            seen_ids.add(v["id"])
                            candidates.append(v)
                            added += 1
                            if len(candidates) >= target:
                                break
                    print(f"[{plat}] {query!r}")
                    print(f"  -> {len(videos)} results, +{added} new ({len(candidates)} total)")

    print(f"\n--- YouTube queries ({len(YOUTUBE_QUERIES)} queries, {SEARCH_WORKERS} parallel) ---")
    _run_queries_parallel(YOUTUBE_QUERIES, "youtube")

    if len(candidates) < target:
        print(f"\n--- Dailymotion fallback ({len(DAILYMOTION_QUERIES)} queries) ---")
        _run_queries_parallel(DAILYMOTION_QUERIES, "dailymotion")

    if len(candidates) < target:
        print(f"\n[WARN] Only found {len(candidates)} unique candidates (target: {target})")

    return candidates[:target]


# =============================================================================
# SECTION 3 — Mock label generation (deterministic per video_id)
# =============================================================================
_TONES = ["upbeat", "serious", "inspirational", "urgent", "calm"]
_EMOTIONAL_SETS = [
    ["aspiration", "pride"], ["trust", "family"], ["fear", "aspiration"],
    ["humor", "family"], ["pride"], ["aspiration", "trust"], ["family", "pride"],
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
    "voiceover", "on_screen_talent", "customer_testimonial", "text_only", "mixed",
]
_PACING    = ["fast_cuts", "medium", "slow_cinematic"]
_PALETTES  = ["warm", "cool", "neutral", "high_contrast"]
_MUSIC     = ["upbeat", "calm", "dramatic", "tense", "none"]
_SETTINGS  = ["interior", "exterior", "studio", "mixed"]
_CTA_TYPES = ["phone_number", "website", "visit_store", "limited_time_offer"]
_REVEAL    = ["early (<10s)", "mid (10-30s)", "late (>30s)", "never"]
_AUDIENCES = ["homeowners_general", "luxury", "budget_conscious", "diy", "families"]


def _mock_labels(video_id: str, duration: int) -> dict:
    seed = int(hashlib.md5(video_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    has_cta = rng.random() > 0.3
    return {
        "tone":                 rng.choice(_TONES),
        "emotional_appeal":     rng.choice(_EMOTIONAL_SETS),
        "theme":                rng.choice(_THEMES),
        "product_reveal_timing":rng.choice(_REVEAL),
        "hook_type":            rng.choice(_HOOK_TYPES),
        "narrator_type":        rng.choice(_NARRATOR),
        "pacing":               rng.choice(_PACING),
        "color_palette":        rng.choice(_PALETTES),
        "music_mood":           rng.choice(_MUSIC),
        "setting":              rng.choice(_SETTINGS),
        "has_cta":              has_cta,
        "cta_type":             rng.choice(_CTA_TYPES) if has_cta else "none",
        "has_before_after":     rng.random() > 0.4,
        "has_price_mention":    rng.random() > 0.6,
        "ad_length_seconds":    duration,
        "target_audience":      rng.choice(_AUDIENCES),
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

        input_tokens  = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        cost = (
            input_tokens  / 1_000_000 * INPUT_PRICE_PER_MTOK
            + output_tokens / 1_000_000 * OUTPUT_PRICE_PER_MTOK
        )

        return {
            **video_meta,
            **labels,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "cost_usd":      round(cost, 6),
            "elapsed_sec":   round(time.time() - t_start, 1),
            "error":         None,
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
            "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
            "elapsed_sec":  round(time.time() - t_start, 1),
            "error":        str(exc),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# SECTION 5 — Cache I/O
# =============================================================================
def load_cache(path: Path = CACHE_PATH) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return []


def save_cache(results: list[dict], path: Path = CACHE_PATH) -> None:
    """Atomic write: .tmp then rename."""
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


# =============================================================================
# SECTION 6 — run_batch
# =============================================================================
def run_batch(
    cache_path: Path = CACHE_PATH,
    mock: bool = False,
    reset: bool = False,
) -> list[dict]:
    if reset and cache_path.exists():
        cache_path.unlink()
        print("Cache deleted.")

    existing   = load_cache(cache_path)
    cached_ids = {r["id"] for r in existing}
    results    = list(existing)

    print(f"\nCollecting {TARGET} video candidates...")
    candidates     = collect_candidates()
    new_candidates = [c for c in candidates if c["id"] not in cached_ids]
    already_done   = len(cached_ids)

    print(
        f"\n{already_done} already cached, "
        f"{len(new_candidates)} new to analyze (target: {TARGET})\n"
    )

    if not new_candidates:
        print("Nothing new to analyze.")
        return results

    for i, video_meta in enumerate(new_candidates, start=1):
        # Stop once we have TARGET successful analyses
        successful = sum(1 for r in results if not r.get("error"))
        if successful >= TARGET:
            print(f"\nReached {TARGET} successful analyses — stopping.")
            break

        global_n = already_done + i
        total_n  = already_done + len(new_candidates)
        print(
            f"\n[{global_n}/{total_n}] Analyzing: "
            f"{video_meta['title'][:60]} ({video_meta['duration']}s)"
            f" [{video_meta.get('platform', 'youtube')}]"
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
    parser = argparse.ArgumentParser(description="Batch Home Remodeling Ad Analysis")
    parser.add_argument("--mock",  action="store_true", help="Dummy labels, no API calls")
    parser.add_argument("--reset", action="store_true", help="Delete cache and re-run all")
    args = parser.parse_args()

    check_dependencies()
    run_batch(mock=args.mock, reset=args.reset)
