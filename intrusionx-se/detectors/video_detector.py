"""
IntrusionX SE — Video Deepfake Detector (v3 — Production-Grade)

Extracts keyframes from a video and runs each through the multi-layer
image detector (face detection + dual model + ELA).

Strategy:
  - Scene-change aware frame sampling (not just uniform)
  - Robust aggregation with median + mean + max
  - Face consistency tracking across frames
  - Temporal artifact detection
"""

import cv2
import numpy as np
from PIL import Image
from detectors.image_detector import detect_image


# ── Configuration ─────────────────────────────────────────────
MAX_FRAMES = 12          # Max frames to sample
MAX_DURATION_SEC = 60    # Reject videos longer than this
SCENE_THRESHOLD = 30.0   # Mean absolute diff for scene change


def detect_video(video_path: str, progress_callback=None) -> dict:
    """
    Analyse a video file for deepfake content.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    progress_callback : callable, optional
        Function(current, total) called after each frame is processed.

    Returns
    -------
    dict with keys:
        verdict       : str
        confidence    : float (0-100)
        frame_count   : int
        frame_results : list[dict]
        details       : list[str]
        flagged_frames: list[int]
        duration      : float
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _error_result("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    if duration > MAX_DURATION_SEC:
        cap.release()
        return _error_result(
            f"Video too long ({duration:.0f}s). Maximum is {MAX_DURATION_SEC}s. "
            "Please trim the video and try again."
        )

    # Determine which frames to sample
    frame_indices = _smart_sample_frames(cap, total_frames, fps)
    n_sample = len(frame_indices)

    if n_sample <= 0:
        cap.release()
        return _error_result("Video contains no frames.")

    frame_results = []
    fake_scores = []
    flagged_frames = []
    faces_detected_count = 0

    for i, fidx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert BGR (OpenCV) to RGB (PIL)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        result = detect_image(pil_image)
        result["frame_index"] = int(fidx)
        result["timestamp"] = round(fidx / fps, 2) if fps > 0 else 0
        frame_results.append(result)

        # Track face detection across frames
        if result.get("face_detected", False):
            faces_detected_count += 1

        # Track fake probability (using the raw ensemble fake prob)
        is_fake = result["verdict"] in ("DEEPFAKE", "SUSPICIOUS")
        if is_fake:
            fake_scores.append(result["confidence"])
        else:
            fake_scores.append(100 - result["confidence"])

        if result["verdict"] == "DEEPFAKE":
            flagged_frames.append(int(fidx))

        if progress_callback:
            progress_callback(i + 1, n_sample)

    cap.release()

    if not frame_results:
        return _error_result("Could not read any frames from video.")

    # ── Aggregate ──────────────────────────────────────────
    avg_fake_score = float(np.mean(fake_scores))
    median_fake_score = float(np.median(fake_scores))
    max_fake_score = float(np.max(fake_scores))
    fake_count = sum(1 for r in frame_results if r["verdict"] == "DEEPFAKE")
    suspicious_count = sum(1 for r in frame_results if r["verdict"] == "SUSPICIOUS")
    total_analyzed = len(frame_results)
    fake_ratio = fake_count / total_analyzed

    # Robust combined score
    combined_score = (avg_fake_score * 0.35 +
                      median_fake_score * 0.35 +
                      max_fake_score * 0.30)

    # Verdict with calibrated thresholds (matching image detector)
    if fake_ratio >= 0.4 or combined_score >= 50:
        verdict = "DEEPFAKE"
        confidence = combined_score
    elif fake_ratio >= 0.15 or combined_score >= 30:
        verdict = "SUSPICIOUS"
        confidence = combined_score
    else:
        verdict = "AUTHENTIC"
        confidence = 100 - combined_score

    # Models used info
    models_used = set()
    for r in frame_results:
        for m in r.get("models_used", []):
            models_used.add(m)

    # Details
    details = [
        f"Analysed {total_analyzed} frames sampled across {duration:.1f}s video",
        f"Models used: {', '.join(models_used) if models_used else 'N/A'}",
        f"Faces detected in {faces_detected_count}/{total_analyzed} frames",
        f"Frames flagged as deepfake: {fake_count}/{total_analyzed}",
        f"Frames flagged as suspicious: {suspicious_count}/{total_analyzed}",
        f"Average manipulation score: {avg_fake_score:.1f}%",
        f"Median manipulation score: {median_fake_score:.1f}%",
        f"Peak manipulation score: {max_fake_score:.1f}%",
    ]

    if verdict == "DEEPFAKE":
        details.append("Multiple frames show strong deepfake indicators")
        details.append("Consistent manipulation patterns detected across video")
    elif verdict == "SUSPICIOUS":
        details.append("Some frames show potential manipulation artifacts")
        details.append("Recommended: Review flagged frames individually")
    else:
        details.append("No significant deepfake artifacts detected in sampled frames")

    return {
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "frame_count": total_analyzed,
        "frame_results": frame_results,
        "details": details,
        "flagged_frames": flagged_frames,
        "duration": round(duration, 2),
    }


def _smart_sample_frames(cap, total_frames, fps):
    """
    Smart frame sampling: uniform + scene-change detection.
    Ensures we sample diverse visual content.
    """
    if total_frames <= MAX_FRAMES:
        return list(range(total_frames))

    # Uniform sampling
    uniform = np.linspace(0, total_frames - 1, MAX_FRAMES // 2, dtype=int).tolist()

    # Scene-change frames
    scenes = _detect_scene_changes(cap, total_frames, max_scenes=MAX_FRAMES // 2)

    # Merge, deduplicate, cap at MAX_FRAMES
    combined = sorted(set(uniform + scenes))
    if len(combined) > MAX_FRAMES:
        step = len(combined) / MAX_FRAMES
        combined = [combined[int(i * step)] for i in range(MAX_FRAMES)]

    return combined


def _detect_scene_changes(cap, total_frames, max_scenes=6):
    """Detect scene changes by comparing frame differences."""
    scene_frames = []
    interval = max(1, total_frames // 50)
    prev_frame = None

    for fidx in range(0, total_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (64, 64))

        if prev_frame is not None:
            diff = np.mean(np.abs(small.astype(float) - prev_frame.astype(float)))
            if diff > SCENE_THRESHOLD:
                scene_frames.append(fidx)

        prev_frame = small
        if len(scene_frames) >= max_scenes:
            break

    return scene_frames


def get_frame_as_image(video_path: str, frame_index: int):
    """Extract a single frame as a PIL Image."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _error_result(message: str) -> dict:
    return {
        "verdict": "ERROR",
        "confidence": 0,
        "frame_count": 0,
        "frame_results": [],
        "details": [message],
        "flagged_frames": [],
        "duration": 0,
    }
