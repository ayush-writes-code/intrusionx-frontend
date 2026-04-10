"""
IntrusionX SE — Image Deepfake Detector (v3 — Production-Grade)

Multi-layer detection architecture:

  Layer 1: FACE DETECTION
    → OpenCV Haar cascade to detect and crop face regions
    → If face found: run face-specific deepfake model on cropped face
    → If no face: run AI-generated image detector on full image

  Layer 2: DUAL-MODEL ENSEMBLE
    Model A: dima806/deepfake_vs_real_image_detection
             ViT fine-tuned on 140K real/fake face images. 15K+ downloads.
             Labels: {0: "Real", 1: "Fake"}
    Model B: umm-maybe/AI-image-detector
             Swin Transformer trained on diverse AI images (SD, MJ, DALL-E).
             Labels: {0: "artificial", 1: "human"}
             NOTE: "artificial" = AI-generated = FAKE,  "human" = REAL

  Layer 3: ERROR LEVEL ANALYSIS (ELA)
    → Lightweight statistical forensics used as tiebreaker

  ENSEMBLE STRATEGY:
    → Use MAX(fake_prob) when models disagree (catches more fakes)
    → Use AVERAGE when models agree (higher confidence)
    → Lower thresholds: 50% for DEEPFAKE, 30% for SUSPICIOUS
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageChops, ImageEnhance
import io

# ── Model configs ─────────────────────────────────────────────
MODEL_A = "dima806/deepfake_vs_real_image_detection"
# ViT, {0: "Real", 1: "Fake"}

MODEL_B = "umm-maybe/AI-image-detector"
# Swin, {0: "artificial", 1: "human"}
# INVERTED: class 0 ("artificial") = FAKE

# ── Face detection ────────────────────────────────────────────
_face_cascade = None
FACE_PADDING = 0.3  # 30% padding around detected face

# ── Lazy-loaded models ────────────────────────────────────────
_model_a_processor = None
_model_a = None
_model_b_processor = None
_model_b = None


def _get_hf_token():
    """Get HuggingFace token from environment."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


# ══════════════════════════════════════════════════════════════
#  FACE DETECTION (OpenCV Haar Cascade)
# ══════════════════════════════════════════════════════════════

def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        if _face_cascade.empty():
            print("[FaceDetect] WARNING: Could not load Haar cascade.")
            _face_cascade = None
    return _face_cascade


def detect_faces(image: Image.Image) -> list:
    """
    Detect face regions in a PIL image.
    Returns list of (x, y, w, h) tuples with padding applied.
    """
    cascade = _get_face_cascade()
    if cascade is None:
        return []

    # Convert to grayscale numpy array
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Detect faces at multiple scales
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        return []

    # Add padding
    h_img, w_img = gray.shape[:2]
    padded_faces = []
    for (x, y, w, h) in faces:
        pad_w = int(w * FACE_PADDING)
        pad_h = int(h * FACE_PADDING)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w_img, x + w + pad_w)
        y2 = min(h_img, y + h + pad_h)
        padded_faces.append((x1, y1, x2 - x1, y2 - y1))

    return padded_faces


def crop_face(image: Image.Image, face_box: tuple) -> Image.Image:
    """Crop a face region from a PIL image."""
    x, y, w, h = face_box
    return image.crop((x, y, x + w, y + h))


# ══════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════

def _load_model_a():
    """Load dima806/deepfake_vs_real_image_detection (ViT)."""
    global _model_a_processor, _model_a
    if _model_a is None:
        token = _get_hf_token()
        print(f"[ImageDetector] Loading Model A: {MODEL_A} ...")
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            _model_a_processor = AutoImageProcessor.from_pretrained(MODEL_A, token=token)
            _model_a = AutoModelForImageClassification.from_pretrained(MODEL_A, token=token)
            _model_a.eval()
            print(f"[ImageDetector] Model A loaded ✓  Labels: {_model_a.config.id2label}")
        except Exception as e:
            print(f"[ImageDetector] Model A FAILED: {e}")
            _model_a = None
    return _model_a_processor, _model_a


def _load_model_b():
    """Load umm-maybe/AI-image-detector (Swin)."""
    global _model_b_processor, _model_b
    if _model_b is None:
        token = _get_hf_token()
        print(f"[ImageDetector] Loading Model B: {MODEL_B} ...")
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            _model_b_processor = AutoImageProcessor.from_pretrained(MODEL_B, token=token)
            _model_b = AutoModelForImageClassification.from_pretrained(MODEL_B, token=token)
            _model_b.eval()
            print(f"[ImageDetector] Model B loaded ✓  Labels: {_model_b.config.id2label}")
        except Exception as e:
            print(f"[ImageDetector] Model B FAILED: {e}")
            _model_b = None
    return _model_b_processor, _model_b


# ══════════════════════════════════════════════════════════════
#  SINGLE-MODEL INFERENCE
# ══════════════════════════════════════════════════════════════

def _infer(processor, model, image: Image.Image, model_name: str) -> dict:
    """
    Run inference on a single model.
    Returns normalized result with a consistent 'fake_probability' field (0–100).
    """
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    # Build per-class prob dict
    prob_dict = {}
    for idx, p in enumerate(probs):
        label = model.config.id2label.get(idx, f"class_{idx}")
        prob_dict[label] = round(p.item() * 100, 2)

    predicted_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label.get(predicted_idx, "unknown")
    confidence = probs[predicted_idx].item() * 100

    # ── CRITICAL: Normalize fake_probability correctly per model ──
    fake_probability = _extract_fake_prob(prob_dict, model_name)

    return {
        "model": model_name,
        "label": predicted_label,
        "confidence": confidence,
        "fake_probability": fake_probability,
        "probs": prob_dict,
    }


def _extract_fake_prob(prob_dict: dict, model_name: str) -> float:
    """
    Extract the fake probability correctly based on known label mappings.
    This is the CRITICAL function — wrong logic here = wrong verdicts.
    """
    if "AI-image-detector" in model_name:
        # umm-maybe model: {0: "artificial", 1: "human"}
        # "artificial" = AI-generated = FAKE
        return prob_dict.get("artificial", prob_dict.get("class_0", 50.0))

    elif "deepfake_vs_real" in model_name:
        # dima806 model: {0: "Real", 1: "Fake"}
        return prob_dict.get("Fake", prob_dict.get("class_1", 50.0))

    # ── Generic fallback ──
    for label, prob in prob_dict.items():
        label_l = label.lower()
        if any(kw in label_l for kw in ("fake", "deepfake", "artificial", "synthetic", "ai")):
            return prob

    for label, prob in prob_dict.items():
        label_l = label.lower()
        if any(kw in label_l for kw in ("real", "human", "authentic", "realism")):
            return 100.0 - prob

    return 50.0


# ══════════════════════════════════════════════════════════════
#  ERROR LEVEL ANALYSIS (ELA) — Lightweight Forensics
# ══════════════════════════════════════════════════════════════

def _compute_ela_score(image: Image.Image) -> float:
    """
    Compute an ELA-based manipulation score (0–100).
    Higher = more likely manipulated.
    """
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Re-save at lower quality
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        resaved = Image.open(buf)

        # Compute difference
        diff = ImageChops.difference(image, resaved)
        diff_array = np.array(diff, dtype=np.float32)

        # Statistics
        mean_diff = np.mean(diff_array)
        std_diff = np.std(diff_array)
        max_diff = np.max(diff_array)

        # Normalize to 0–100
        # Images with high uniform error levels tend to be AI-generated
        # Images with localized high error = likely tampered
        score = min(100, (mean_diff * 2.0) + (std_diff * 0.5))

        return round(score, 2)
    except Exception:
        return 0.0


# ══════════════════════════════════════════════════════════════
#  MAIN DETECTION PIPELINE
# ══════════════════════════════════════════════════════════════

def detect_image(image: Image.Image) -> dict:
    """
    Full detection pipeline:
    1. Face detection → crop face region (if found)
    2. Run Model A (face deepfake) on face or full image
    3. Run Model B (AI-image detector) on full image
    4. Compute ELA score
    5. Ensemble → final verdict

    Returns
    -------
    dict with keys:
        verdict     : str   — "DEEPFAKE" | "AUTHENTIC" | "SUSPICIOUS"
        confidence  : float — 0-100
        label       : str   — raw model label
        probs       : dict  — probability breakdown
        details     : list[str] — human-readable analysis
        models_used : list[str] — model names
        face_detected: bool
        ela_score   : float
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    models_used = []
    model_results = []

    # ── Step 1: Face Detection ────────────────────────────
    faces = detect_faces(image)
    face_detected = len(faces) > 0
    analysis_image = image  # Default: full image for Model A

    if face_detected:
        # Use the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        face_crop = crop_face(image, largest_face)
        # Ensure face crop is reasonable size
        if face_crop.size[0] >= 40 and face_crop.size[1] >= 40:
            analysis_image = face_crop

    # ── Step 2: Model A — Face deepfake detector ─────────
    proc_a, mod_a = _load_model_a()
    if mod_a is not None:
        try:
            r_a = _infer(proc_a, mod_a, analysis_image, MODEL_A)
            model_results.append(r_a)
            models_used.append("deepfake_vs_real (ViT)")
        except Exception as e:
            print(f"[ImageDetector] Model A inference error: {e}")

    # ── Step 3: Model B — AI-image detector (always full image) ──
    proc_b, mod_b = _load_model_b()
    if mod_b is not None:
        try:
            r_b = _infer(proc_b, mod_b, image, MODEL_B)
            model_results.append(r_b)
            models_used.append("AI-image-detector (Swin)")
        except Exception as e:
            print(f"[ImageDetector] Model B inference error: {e}")

    # ── Step 4: ELA Score ─────────────────────────────────
    ela_score = _compute_ela_score(image)

    # ── Step 5: Error case ────────────────────────────────
    if not model_results:
        return {
            "verdict": "ERROR",
            "confidence": 0,
            "label": "error",
            "probs": {},
            "details": ["No models could be loaded. Check connection & HF_TOKEN."],
            "models_used": [],
            "face_detected": face_detected,
            "ela_score": ela_score,
        }

    # ── Step 6: Ensemble ──────────────────────────────────
    return _ensemble(model_results, models_used, face_detected, ela_score)


# ══════════════════════════════════════════════════════════════
#  ENSEMBLE LOGIC
# ══════════════════════════════════════════════════════════════

def _ensemble(results: list, models_used: list, face_detected: bool, ela_score: float) -> dict:
    """
    Ensemble strategy:
    - If ONE model says fake: use MAX(fake_prob)  — catches more fakes
    - If BOTH agree: use AVERAGE — higher confidence
    - ELA acts as a tiebreaker / confidence booster
    """
    fake_probs = [r["fake_probability"] for r in results]

    if len(results) == 2:
        r_a, r_b = results[0], results[1]

        # Agreement check (both above or below 50%)
        a_says_fake = r_a["fake_probability"] >= 50
        b_says_fake = r_b["fake_probability"] >= 50
        models_agree = a_says_fake == b_says_fake

        if models_agree:
            # Average when agreeing
            ensemble_fake = (r_a["fake_probability"] + r_b["fake_probability"]) / 2
            # Agreement bonus
            if a_says_fake:
                ensemble_fake = min(100, ensemble_fake + 3)
            else:
                ensemble_fake = max(0, ensemble_fake - 3)
        else:
            # DISAGREEMENT: use MAX to catch fakes (prefer safety)
            ensemble_fake = max(r_a["fake_probability"], r_b["fake_probability"])
            # Slight reduction since only one model flagged it
            ensemble_fake = ensemble_fake * 0.85

    else:
        # Single model
        ensemble_fake = results[0]["fake_probability"]

    # ── ELA adjustment (small influence, max ±8 points) ──
    if ela_score > 30:
        # High ELA = more manipulation artifacts
        ela_boost = min(8, (ela_score - 30) * 0.2)
        ensemble_fake = min(100, ensemble_fake + ela_boost)
    elif ela_score < 5:
        # Very low ELA with AI images (too clean = suspicious)
        if ensemble_fake > 40:
            ensemble_fake = min(100, ensemble_fake + 3)

    # ── Classify ──────────────────────────────────────────
    verdict, confidence = _classify(ensemble_fake)

    # ── Build combined probs dict ─────────────────────────
    combined_probs = {}
    for i, r in enumerate(results):
        suffix = f" ({models_used[i].split('(')[0].strip()})" if len(results) > 1 else ""
        for k, v in r["probs"].items():
            combined_probs[f"{k}{suffix}"] = v
    combined_probs["ELA Score"] = ela_score

    # ── Details ───────────────────────────────────────────
    details = _build_details(results, models_used, face_detected, ela_score,
                             verdict, confidence, ensemble_fake)

    return {
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "label": results[0]["label"],
        "probs": combined_probs,
        "details": details,
        "models_used": models_used,
        "face_detected": face_detected,
        "ela_score": ela_score,
    }


def _classify(fake_probability: float) -> tuple:
    """
    Convert fake probability (0–100) to verdict + confidence.

    Thresholds (calibrated):
      ≥ 50% fake → DEEPFAKE
      ≥ 30% fake → SUSPICIOUS
      < 30% fake → AUTHENTIC
    """
    if fake_probability >= 50:
        return "DEEPFAKE", fake_probability
    elif fake_probability >= 30:
        return "SUSPICIOUS", max(fake_probability, 100 - fake_probability)
    else:
        return "AUTHENTIC", 100 - fake_probability


def _build_details(results, models_used, face_detected, ela_score,
                   verdict, confidence, ensemble_fake):
    """Build detailed analysis text."""
    details = []

    # Header
    n_models = len(results)
    details.append(f"🔬 **Multi-Layer Analysis** ({n_models} model{'s' if n_models > 1 else ''} + ELA)")

    # Face detection
    if face_detected:
        details.append("👤 Face detected — analysed cropped face region for Model A")
    else:
        details.append("🖼️ No face detected — analysing full image")

    # Per-model results
    for i, r in enumerate(results):
        name = models_used[i] if i < len(models_used) else r["model"]
        fake_p = r["fake_probability"]
        pred = "FAKE" if fake_p >= 50 else "REAL"
        icon = "🔴" if fake_p >= 50 else "🟢"
        details.append(f"{icon} {name}: {pred} (fake prob: {fake_p:.1f}%)")

    # ELA
    if ela_score > 30:
        details.append(f"📊 ELA Score: {ela_score:.1f} — elevated error levels detected")
    elif ela_score > 15:
        details.append(f"📊 ELA Score: {ela_score:.1f} — moderate error levels")
    else:
        details.append(f"📊 ELA Score: {ela_score:.1f} — low error levels")

    # Agreement
    if len(results) == 2:
        fake_probs = [r["fake_probability"] for r in results]
        if (fake_probs[0] >= 50) == (fake_probs[1] >= 50):
            details.append("✅ Both models **agree** — high reliability")
        else:
            details.append("⚠️ Models **disagree** — using cautious ensemble (max strategy)")

    # Verdict details
    details.append("")  # Separator
    if verdict == "DEEPFAKE":
        details.append(f"🔴 **DEEPFAKE detected** with {confidence:.1f}% confidence")
        details.append("AI-generated or manipulated content indicators found")
        if confidence > 85:
            details.append("Very high confidence — strong synthetic generation markers")
        details.append("Cross-verify with metadata analysis below")
    elif verdict == "SUSPICIOUS":
        details.append(f"🟡 **SUSPICIOUS** — inconclusive at {confidence:.1f}%")
        details.append("Some indicators present but not definitive")
        details.append("Manual review recommended")
    else:
        details.append(f"🟢 **AUTHENTIC** with {confidence:.1f}% confidence")
        details.append("No significant manipulation artifacts detected")

    return details
