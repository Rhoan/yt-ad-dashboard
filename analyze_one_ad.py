"""
analyze_one_ad.py — Single YouTube Ad Analysis Pipeline (Test Run)

Downloads one home-remodeling ad from YouTube, extracts frames, fetches the
transcript, and sends everything to Claude Vision for structured labeling.
Prints all 16 label keys + token cost to the terminal.
"""

# =============================================================================
# SECTION 0 — Imports + constants
# =============================================================================
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the same directory as this script
load_dotenv(Path(__file__).parent / ".env")

PYTHON_EXE = r"C:\Users\rohan\anaconda3\python.exe"

MODEL = "claude-sonnet-4-6"
FRAME_INTERVAL_SEC = 5
MAX_FRAMES = 20
FRAME_WIDTH = 512
FRAME_JPEG_QUALITY = 60
SEARCH_QUERY = "home remodeling kitchen bathroom renovation before after advertisement"
INPUT_PRICE_PER_MTOK = 3.00
OUTPUT_PRICE_PER_MTOK = 15.00

SYSTEM_PROMPT = (
    "You are an expert video-ad analyst specializing in home remodeling commercials. "
    "Analyze the provided frames and transcript, then respond with ONLY a valid JSON "
    "object matching the schema given in the user message. No markdown fences, no "
    "commentary — raw JSON only."
)

JSON_SCHEMA = """{
  "tone": "upbeat | serious | inspirational | urgent | calm",
  "emotional_appeal": ["<value1>", "<value2>"],
  "theme": "before_after_transformation | lifestyle | testimonial | product_demo | problem_solution | price_offer",
  "product_reveal_timing": "early (<10s) | mid (10-30s) | late (>30s) | never",
  "hook_type": "visual_transformation | pain_point | question | shocking_stat | offer | celebrity",
  "narrator_type": "voiceover | on_screen_talent | customer_testimonial | text_only | mixed",
  "pacing": "fast_cuts | medium | slow_cinematic",
  "color_palette": "warm | cool | neutral | high_contrast",
  "music_mood": "upbeat | calm | dramatic | tense | none",
  "setting": "interior | exterior | studio | mixed",
  "has_cta": true,
  "cta_type": "phone_number | website | visit_store | limited_time_offer | none",
  "has_before_after": true,
  "has_price_mention": true,
  "ad_length_seconds": 0,
  "target_audience": "homeowners_general | luxury | budget_conscious | diy | families"
}"""

LABEL_DESCRIPTIONS = {
    "tone": "Overall tone",
    "emotional_appeal": "Emotional appeals (array)",
    "theme": "Primary theme",
    "product_reveal_timing": "When product first revealed",
    "hook_type": "Opening hook type",
    "narrator_type": "Narrator / presenter type",
    "pacing": "Edit pacing",
    "color_palette": "Dominant color palette",
    "music_mood": "Background music mood",
    "setting": "Primary filming setting",
    "has_cta": "Has call-to-action?",
    "cta_type": "CTA type",
    "has_before_after": "Has before/after shots?",
    "has_price_mention": "Mentions price/cost?",
    "ad_length_seconds": "Ad length (seconds)",
    "target_audience": "Target audience",
}


# =============================================================================
# SECTION 1 — check_dependencies()
# =============================================================================
def check_dependencies() -> None:
    missing = []
    try:
        import cv2  # noqa: F401
    except ImportError:
        missing.append("opencv-python")
    try:
        import youtube_transcript_api  # noqa: F401
    except ImportError:
        missing.append("youtube-transcript-api")
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        missing.append("Pillow")

    if missing:
        print("Missing dependencies:", ", ".join(missing))
        print("Run:")
        print(
            "  C:\\Users\\rohan\\anaconda3\\python.exe -m pip install "
            + " ".join(missing)
        )
        sys.exit(1)


# =============================================================================
# SECTION 2 — find_ad_video(query) → dict
# =============================================================================
def find_ad_video(query: str) -> dict:
    print(f"Searching YouTube for: {query!r}")

    yt_dlp_path = shutil.which("yt-dlp") or "/c/Users/rohan/anaconda3/Scripts/yt-dlp.exe"

    cmd = [
        PYTHON_EXE, "-m", "yt_dlp",
        f"ytsearch5:{query}",
        "--print", "%(id)s\t%(title)s\t%(view_count)s\t%(like_count)s"
                   "\t%(comment_count)s\t%(uploader)s\t%(upload_date)s"
                   "\t%(webpage_url)s\t%(duration)s",
        "--no-download",
        "--no-playlist",
        "--quiet",
        "--ignore-errors",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    # yt-dlp exits non-zero when any individual result is unavailable;
    # only fail hard if stdout is completely empty (no results at all)
    if not result.stdout.strip():
        raise RuntimeError(f"yt-dlp search returned no results:\n{result.stderr}")

    candidates = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 9:
            continue
        try:
            video = {
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
            candidates.append(video)
        except (ValueError, IndexError):
            continue

    if not candidates:
        raise RuntimeError("No search results returned by yt-dlp.")

    # Filter: duration ≤ 180s AND views ≥ 50,000
    filtered = [
        v for v in candidates
        if v["duration"] <= 180 and v["view_count"] >= 50_000
    ]

    chosen = filtered[0] if filtered else candidates[0]
    print(
        f"Found: {chosen['title']} "
        f"({chosen['view_count']:,} views, {chosen['duration']}s)"
    )
    return chosen


# =============================================================================
# SECTION 3 — download_video(url, out_dir) → Path
# =============================================================================
def download_video(url: str, out_dir: Path) -> Path:
    print(f"Downloading at 360p to {out_dir} ...")

    cmd = [
        PYTHON_EXE, "-m", "yt_dlp",
        url,
        "-f", "best[height<=360]/best",
        "-o", str(out_dir / "%(id)s.%(ext)s"),
        "--quiet",
        "--no-playlist",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed:\n{result.stderr}")

    mp4_files = list(out_dir.glob("*.mp4"))
    if not mp4_files:
        # Try any video format as fallback
        all_videos = list(out_dir.glob("*.*"))
        if not all_videos:
            raise RuntimeError("Download completed but no video file found.")
        return all_videos[0]

    print(f"Downloaded: {mp4_files[0].name}")
    return mp4_files[0]


# =============================================================================
# SECTION 4 — extract_frames(path, interval_sec, max_frames)
#              → (list[bytes], list[float])
# =============================================================================
def extract_frames(
    path: Path, interval_sec: int, max_frames: int
) -> tuple[list[bytes], list[float]]:
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    print(f"Extracting frames (video: {duration_sec:.1f}s, {fps:.1f} fps)...")

    # Build candidate timestamps
    timestamps = []
    t = 0.0
    while t < duration_sec:
        timestamps.append(t)
        t += interval_sec
    if not timestamps:
        timestamps = [0.0]

    # Subsample if over cap
    if len(timestamps) > max_frames:
        step = len(timestamps) / max_frames
        timestamps = [timestamps[int(i * step)] for i in range(max_frames)]

    jpeg_list: list[bytes] = []
    kept_timestamps: list[float] = []

    for ts in timestamps:
        frame_idx = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # Resize to FRAME_WIDTH maintaining aspect ratio
        orig_w, orig_h = img.size
        new_h = int(orig_h * FRAME_WIDTH / orig_w)
        img = img.resize((FRAME_WIDTH, new_h), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=FRAME_JPEG_QUALITY)
        jpeg_list.append(buf.getvalue())
        kept_timestamps.append(ts)

    cap.release()
    print(f"Extracted {len(jpeg_list)} frames")
    return jpeg_list, kept_timestamps


# =============================================================================
# SECTION 5 — get_transcript(video_id) → str
# =============================================================================
def get_transcript(video_id: str) -> str:
    print("Fetching transcript...")
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        # v1.2.4+: instance-based API with fetch()
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=["en", "en-US"])
        text = " ".join(snippet.text for snippet in fetched)
        print("Transcript: OK")
        return text
    except Exception as exc:
        print(f"Transcript: unavailable ({type(exc).__name__}: {exc})")
        return "[No transcript available — analysis based on visual frames only]"


# =============================================================================
# SECTION 6 — build_claude_payload(frames, timestamps, transcript, duration)
#              → list[dict]
# =============================================================================
def build_claude_payload(
    frames: list[bytes],
    timestamps: list[float],
    transcript: str,
    duration: int,
) -> list[dict]:
    content = []

    # Opening text block
    content.append(
        {
            "type": "text",
            "text": (
                f"You are analyzing a home remodeling advertisement.\n"
                f"Video duration: {duration} seconds.\n\n"
                f"TRANSCRIPT:\n{transcript}\n\n"
                f"Below are {len(frames)} evenly-spaced frames from the video. "
                "Each frame is preceded by its timestamp label."
            ),
        }
    )

    # Interleaved timestamp labels + image blocks
    for i, (jpeg_bytes, ts) in enumerate(zip(frames, timestamps)):
        content.append(
            {
                "type": "text",
                "text": f"[Frame {i + 1} — {ts:.1f}s]",
            }
        )
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(jpeg_bytes).decode("ascii"),
                },
            }
        )

    # Closing instruction + schema
    content.append(
        {
            "type": "text",
            "text": (
                "Based on the transcript and all frames above, return ONLY a JSON "
                "object matching this schema exactly (no markdown fences, no extra keys):\n\n"
                f"{JSON_SCHEMA}\n\n"
                "Rules:\n"
                "- emotional_appeal must be an array of 1–3 strings from: "
                "aspiration, fear, trust, humor, family, pride\n"
                "- ad_length_seconds must equal the video duration provided above "
                f"({duration}), not a value you infer from frames\n"
                "- For boolean fields use JSON true/false (not strings)\n"
                "- Pick the single best match for each enum field"
            ),
        }
    )

    return content


# =============================================================================
# SECTION 7 — call_claude(content) → (dict, usage)
# =============================================================================
def call_claude(content: list[dict]) -> tuple[dict, object]:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not found. Set it as an environment variable "
            "or add ANTHROPIC_API_KEY=sk-ant-... to a .env file in this directory."
        )

    client = anthropic.Anthropic(api_key=api_key)

    print(f"Sending {sum(1 for c in content if c['type'] == 'image')} frames "
          f"+ transcript to {MODEL}...")

    for attempt in range(5):
        try:
            message = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            break
        except Exception as exc:
            if "429" in str(exc) and attempt < 4:
                wait = 60 * (attempt + 1)
                print(f"  [rate limit] waiting {wait}s (attempt {attempt+1}/5)...")
                time.sleep(wait)
            else:
                raise

    raw = message.content[0].text
    # Strip markdown fences (same pattern as claude_agent.py)
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    labels = json.loads(raw)
    return labels, message.usage


# =============================================================================
# SECTION 8 — print_results_table(labels, video_meta)
# =============================================================================
def print_results_table(labels: dict, video_meta: dict) -> None:
    SEP = "=" * 70
    COL1 = 30

    print(f"\n{SEP}")
    print("  AD ANALYSIS RESULTS")
    print(SEP)

    # Metadata rows
    meta_rows = [
        ("video_title", video_meta.get("title", "N/A")),
        ("channel", video_meta.get("uploader", "N/A")),
        ("upload_date", video_meta.get("upload_date", "N/A")),
        ("views", f"{video_meta.get('view_count', 0):,}"),
        ("likes", f"{video_meta.get('like_count', 0):,}"),
        ("comments", f"{video_meta.get('comment_count', 0):,}"),
        ("url", video_meta.get("webpage_url", "N/A")),
    ]
    for key, val in meta_rows:
        print(f"  {key.ljust(COL1)}{val}")

    print(f"  {'-' * (COL1 + 30)}")

    # Label rows
    for key in LABEL_DESCRIPTIONS:
        val = labels.get(key, "N/A")
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val)
        elif isinstance(val, bool):
            val = str(val).lower()
        else:
            val = str(val)
        print(f"  {key.ljust(COL1)}{val}")

    print(SEP)


# =============================================================================
# SECTION 9 — print_cost_summary(usage, elapsed_sec)
# =============================================================================
def print_cost_summary(usage: object, elapsed_sec: float) -> None:
    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)

    input_cost = input_tokens / 1_000_000 * INPUT_PRICE_PER_MTOK
    output_cost = output_tokens / 1_000_000 * OUTPUT_PRICE_PER_MTOK
    total_cost = input_cost + output_cost

    print("\n--- Token Usage & Cost ---")
    print(f"  Input tokens:   {input_tokens:>10,}")
    print(f"  Output tokens:  {output_tokens:>10,}")
    print(f"  Input cost:     ${input_cost:>10.4f}")
    print(f"  Output cost:    ${output_cost:>10.4f}")
    print(f"  Total cost:     ${total_cost:>10.4f}")
    print(f"  Elapsed time:   {elapsed_sec:>9.1f}s")
    print("--------------------------\n")


# =============================================================================
# SECTION 10 — main()
# =============================================================================
MOCK_LABELS = {
    "tone": "inspirational",
    "emotional_appeal": ["aspiration", "pride"],
    "theme": "before_after_transformation",
    "product_reveal_timing": "early (<10s)",
    "hook_type": "visual_transformation",
    "narrator_type": "voiceover",
    "pacing": "medium",
    "color_palette": "warm",
    "music_mood": "upbeat",
    "setting": "interior",
    "has_cta": True,
    "cta_type": "website",
    "has_before_after": True,
    "has_price_mention": False,
    "ad_length_seconds": 0,  # overwritten from metadata
    "target_audience": "homeowners_general",
}

class MockUsage:
    input_tokens = 0
    output_tokens = 0


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="YouTube Ad Analysis Pipeline")
    parser.add_argument(
        "--mock", action="store_true",
        help="Skip Claude API call and use dummy labels (for pipeline testing)"
    )
    args = parser.parse_args()

    check_dependencies()

    tmp_dir = Path(tempfile.mkdtemp(prefix="yt_ad_"))
    t_start = time.time()

    try:
        # Step 2: Find video
        video_meta = find_ad_video(SEARCH_QUERY)

        # Step 3: Download
        video_path = download_video(video_meta["webpage_url"], tmp_dir)

        # Step 4: Extract frames
        frames, timestamps = extract_frames(video_path, FRAME_INTERVAL_SEC, MAX_FRAMES)

        # Step 5: Transcript
        transcript = get_transcript(video_meta["id"])

        if args.mock:
            print(f"\n[MOCK MODE] Skipping Claude API call — using dummy labels.")
            labels = dict(MOCK_LABELS)
            usage = MockUsage()
        else:
            # Step 6: Build payload
            content = build_claude_payload(
                frames, timestamps, transcript, video_meta["duration"]
            )
            # Step 7: Call Claude
            labels, usage = call_claude(content)

        # Inject ad_length_seconds from metadata (not Claude-inferred)
        labels["ad_length_seconds"] = video_meta["duration"]

        elapsed = time.time() - t_start

        # Step 8: Print results
        print_results_table(labels, video_meta)

        # Step 9: Print cost
        print_cost_summary(usage, elapsed)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as exc:
        print(f"\nError: {exc}")
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
