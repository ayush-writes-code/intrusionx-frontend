"""
IntrusionX SE — Audio Deepfake Detector (v2 — Upgraded)

Uses a high-accuracy Wav2Vec2-XLSR model for deepfake audio detection:
  Primary:  garystafford/wav2vec2-deepfake-voice-detector
            (97.9% accuracy, trained on ElevenLabs / Amazon Polly / Kokoro / Hume AI etc.)
  Fallback: Spectral analysis heuristics (if model unavailable)

Labels:
  Class 0 → Real (human speech)
  Class 1 → Fake (AI-generated)
"""

import os
import numpy as np

try:
    import librosa
    import soundfile as sf
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

# ── Model config ──────────────────────────────────────────────
HF_AUDIO_MODEL = "garystafford/wav2vec2-deepfake-voice-detector"
# Class 0 = real, Class 1 = fake

_audio_model = None
_feature_extractor = None
_pipeline_loaded = False


def _get_hf_token():
    """Get HuggingFace token from environment (optional, for private/gated models)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _load_model():
    """Lazy-load the Wav2Vec2 audio classification model."""
    global _audio_model, _feature_extractor, _pipeline_loaded
    if _pipeline_loaded:
        return _audio_model, _feature_extractor
    _pipeline_loaded = True
    try:
        import torch
        from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
        token = _get_hf_token()
        print(f"[AudioDetector] Loading model: {HF_AUDIO_MODEL} ...")
        _feature_extractor = AutoFeatureExtractor.from_pretrained(HF_AUDIO_MODEL, token=token)
        _audio_model = AutoModelForAudioClassification.from_pretrained(HF_AUDIO_MODEL, token=token)
        _audio_model.eval()
        print("[AudioDetector] Wav2Vec2-XLSR model loaded ✓")
    except Exception as e:
        print(f"[AudioDetector] Model loading failed ({e}). Using spectral fallback.")
        _audio_model = None
        _feature_extractor = None
    return _audio_model, _feature_extractor


def detect_audio(audio_path: str) -> dict:
    """
    Analyse an audio file for deepfake characteristics.

    Returns
    -------
    dict with keys:
        verdict    : str — "DEEPFAKE" | "AUTHENTIC" | "SUSPICIOUS"
        confidence : float (0-100)
        details    : list[str]
        method     : str — "wav2vec2_xlsr" or "spectral_analysis"
        features   : dict — extracted audio features
    """
    if not LIBROSA_OK:
        return _error("librosa not installed. Cannot process audio.")

    if not os.path.isfile(audio_path):
        return _error("Audio file not found.")

    # Load audio
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True, duration=30)
    except Exception as e:
        return _error(f"Could not load audio: {e}")

    if len(y) < sr * 0.5:
        return _error("Audio too short (minimum 0.5 seconds).")

    # ── Attempt Wav2Vec2-XLSR model first ────────────────────
    model, extractor = _load_model()
    if model is not None and extractor is not None:
        return _detect_with_wav2vec2(model, extractor, y, sr)

    # ── Fallback: spectral analysis ───────────────────────────
    return _detect_spectral(y, sr)


def _detect_with_wav2vec2(model, extractor, y, sr):
    """Use the Wav2Vec2-XLSR deepfake voice detector model."""
    try:
        import torch

        # Process audio with feature extractor
        inputs = extractor(y, sampling_rate=16000, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]

        # Class 0 = Real, Class 1 = Fake
        prob_real = probs[0].item() * 100
        prob_fake = probs[1].item() * 100

        prob_dict = {
            model.config.id2label.get(0, "real"): round(prob_real, 2),
            model.config.id2label.get(1, "fake"): round(prob_fake, 2),
        }

        # Determine verdict
        if prob_fake >= 70:
            verdict = "DEEPFAKE"
            confidence = prob_fake
        elif prob_fake >= 40:
            verdict = "SUSPICIOUS"
            confidence = max(prob_fake, prob_real)
        else:
            verdict = "AUTHENTIC"
            confidence = prob_real

        # Also extract spectral features for additional details
        features = _extract_features(y, sr)

        details = [
            f"🔬 **Wav2Vec2-XLSR Analysis** (garystafford model)",
            f"Real probability: {prob_real:.1f}%",
            f"Fake probability: {prob_fake:.1f}%",
            f"Spectral centroid mean: {features['spectral_centroid_mean']:.1f} Hz",
            f"Zero-crossing rate: {features['zcr_mean']:.4f}",
            f"RMS energy std: {features['rms_std']:.4f}",
        ]

        if verdict == "DEEPFAKE":
            details.append("Audio exhibits strong characteristics of synthetic generation")
            details.append("Potential TTS / voice cloning artifacts detected (ElevenLabs, Polly, etc.)")
        elif verdict == "AUTHENTIC":
            details.append("Audio exhibits natural human speech patterns")
            details.append("No synthetic generation markers detected")
        else:
            details.append("Inconclusive — some anomalies detected, manual review recommended")

        return {
            "verdict": verdict,
            "confidence": round(confidence, 2),
            "details": details,
            "method": "wav2vec2_xlsr",
            "features": features,
            "probs": prob_dict,
        }

    except Exception as e:
        print(f"[AudioDetector] Wav2Vec2 inference failed: {e}, falling back to spectral.")
        return _detect_spectral(y, sr)


def _detect_spectral(y, sr):
    """
    Lightweight spectral analysis fallback.
    Uses statistical features that tend to differ between real and synthetic audio.
    """
    features = _extract_features(y, sr)

    # ── Heuristic scoring ──────────────────────────────────
    # These thresholds are tuned for common TTS artifacts:
    # - Synthetic audio often has lower spectral variation
    # - TTS audio tends to have unnaturally consistent energy
    # - Real speech has more varied zero-crossing rates

    score = 50.0  # Start neutral

    # Spectral flatness: synthetic audio is often "smoother"
    if features["spectral_flatness_mean"] > 0.15:
        score += 10
    elif features["spectral_flatness_mean"] < 0.02:
        score += 8

    # RMS energy consistency: TTS has unnaturally stable energy
    if features["rms_std"] < 0.02:
        score += 12  # Too consistent → likely synthetic
    elif features["rms_std"] > 0.08:
        score -= 10  # Natural variation → likely real

    # Zero-crossing rate variance
    if features["zcr_std"] < 0.01:
        score += 10  # Too uniform
    elif features["zcr_std"] > 0.05:
        score -= 8

    # Spectral centroid variance
    if features["spectral_centroid_std"] < 200:
        score += 8
    elif features["spectral_centroid_std"] > 800:
        score -= 8

    # MFCC variance (lower in synthetic speech)
    if features["mfcc_var_mean"] < 50:
        score += 10

    # Clamp to 0-100
    score = max(0, min(100, score))

    if score >= 65:
        verdict = "DEEPFAKE"
        confidence = score
    elif score >= 45:
        verdict = "SUSPICIOUS"
        confidence = score
    else:
        verdict = "AUTHENTIC"
        confidence = 100 - score

    details = [
        f"Analysis method: Spectral Feature Analysis (fallback)",
        f"Spectral centroid: {features['spectral_centroid_mean']:.1f} Hz (std: {features['spectral_centroid_std']:.1f})",
        f"Spectral flatness: {features['spectral_flatness_mean']:.4f}",
        f"Zero-crossing rate: {features['zcr_mean']:.4f} (std: {features['zcr_std']:.4f})",
        f"RMS energy std: {features['rms_std']:.4f}",
        f"MFCC variance: {features['mfcc_var_mean']:.2f}",
    ]

    if verdict == "DEEPFAKE":
        details.append("Audio shows signs of synthetic generation (low spectral variation)")
        details.append("Energy patterns are unnaturally consistent")
    elif verdict == "AUTHENTIC":
        details.append("Audio exhibits natural speech variability")
        details.append("No obvious synthetic markers detected")
    else:
        details.append("Some features are borderline — manual review recommended")

    return {
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "details": details,
        "method": "spectral_analysis",
        "features": features,
        "probs": {},
    }


def _extract_features(y, sr) -> dict:
    """Extract statistical audio features used for analysis."""
    # Spectral centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    # Spectral flatness
    flat = librosa.feature.spectral_flatness(y=y)[0]
    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_var = np.var(mfccs, axis=1)

    return {
        "spectral_centroid_mean": float(np.mean(cent)),
        "spectral_centroid_std": float(np.std(cent)),
        "spectral_flatness_mean": float(np.mean(flat)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "mfcc_var_mean": float(np.mean(mfcc_var)),
        "duration": float(len(y) / sr),
    }


def _error(message: str) -> dict:
    return {
        "verdict": "ERROR",
        "confidence": 0,
        "details": [message],
        "method": "none",
        "features": {},
        "probs": {},
    }
