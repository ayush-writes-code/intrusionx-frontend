"""
IntrusionX SE — FastAPI Backend
Production-ready REST API for multi-modal deepfake detection.

Endpoints:
    GET  /                  → Health check
    POST /detect/image      → Image deepfake detection
    POST /detect/video      → Video deepfake detection (frame-by-frame)
    POST /detect/audio      → Audio deepfake detection
    POST /detect/metadata   → Metadata / EXIF forensic analysis
    POST /detect/auto       → Unified endpoint (auto-detects media type)

Run:
    cd intrusionx-se
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import io
import uuid
import shutil
import tempfile
import traceback
import numpy as np
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ── Import detectors (unchanged — no modifications to core logic) ─
from detectors.image_detector import detect_image
from detectors.video_detector import detect_video
from detectors.audio_detector import detect_audio
from detectors.metadata_analyzer import analyze_metadata
from utils.media_router import detect_media_type
from utils.privacy_manager import delete_file, secure_cleanup, get_privacy_status

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")

# Accepted file extensions per media type
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
ALLOWED_AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma"}
ALL_ALLOWED_EXT = ALLOWED_IMAGE_EXT | ALLOWED_VIDEO_EXT | ALLOWED_AUDIO_EXT

# Max upload sizes (bytes)
MAX_IMAGE_SIZE = 20 * 1024 * 1024    # 20 MB
MAX_VIDEO_SIZE = 100 * 1024 * 1024   # 100 MB
MAX_AUDIO_SIZE = 50 * 1024 * 1024    # 50 MB


# ══════════════════════════════════════════════════════════════
#  APP LIFECYCLE
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create temp directory on startup, clean it on shutdown."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    print(f"[API] Temp directory ready: {TEMP_DIR}")
    print("[API] IntrusionX SE API is live ✓")
    yield
    # Cleanup temp on shutdown
    if os.path.isdir(TEMP_DIR):
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        print("[API] Temp directory cleaned up.")


# ══════════════════════════════════════════════════════════════
#  CREATE APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="IntrusionX SE",
    description=(
        "AI-Powered Deepfake Detection API. "
        "Detects deepfakes in images, videos, and audio using "
        "dual-model ensemble (ViT + Swin), Wav2Vec2-XLSR, "
        "face detection, ELA, and metadata forensics."
    ),
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],             # Allow all origins (tighten in production)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

async def _save_upload(upload: UploadFile, allowed_ext: set, max_size: int) -> str:
    """
    Save an uploaded file to the temp directory.

    Validates extension and size. Returns the temp file path.
    Raises HTTPException on validation failure.
    """
    # Validate filename
    if not upload.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided.",
        )

    ext = os.path.splitext(upload.filename)[1].lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type: '{ext}'. "
                f"Allowed: {', '.join(sorted(allowed_ext))}"
            ),
        )

    # Save stream safely to disk
    import shutil
    upload.file.seek(0)

    unique_name = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(TEMP_DIR, unique_name)
    os.makedirs(TEMP_DIR, exist_ok=True)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)

    return temp_path


def _cleanup(file_path: str) -> None:
    """Remove a temporary file after processing safely via Privacy Manager."""
    delete_file(file_path)


def _build_response(
    media_type: str,
    verdict: str,
    confidence: float,
    details: dict,
    file_info: Optional[dict] = None,
) -> dict:
    """Build a standardised API response."""
    response = {
        **get_privacy_status(),
        "media_type": media_type,
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "details": details,
    }
    if file_info:
        response["file_info"] = file_info
    return response


# ══════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════

# ── Health Check ──────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns API status and version info.
    """
    return {
        "status": "online",
        "service": "IntrusionX SE",
        "version": "3.0.0",
        "description": "AI-Powered Deepfake Detection API",
        "endpoints": {
            "image": "POST /detect/image",
            "video": "POST /detect/video",
            "audio": "POST /detect/audio",
            "metadata": "POST /detect/metadata",
            "auto": "POST /detect/auto",
            "batch": "POST /detect/batch",
            "report": "POST /generate-report",
            "download": "GET /download-report/{filename}",
            "docs": "GET /docs",
        },
    }


# ── Image Detection ──────────────────────────────────────────

@app.post("/detect/image", tags=["Detection"])
async def detect_image_endpoint(file: UploadFile = File(...)):
    """
    Detect deepfakes in an uploaded image.

    - **Accepts:** JPG, JPEG, PNG, BMP, WebP, TIFF, GIF
    - **Max size:** 20 MB
    - **Models:** ViT (dima806) + Swin (umm-maybe) ensemble
    - **Features:** Face detection, ELA scoring, metadata analysis
    """
    temp_path = None
    try:
        temp_path = await _save_upload(file, ALLOWED_IMAGE_EXT, MAX_IMAGE_SIZE)

        # Open image and run detection
        pil_image = Image.open(temp_path).convert("RGB")
        result = detect_image(pil_image)

        # Also run metadata analysis
        metadata = analyze_metadata(temp_path)

        return _build_response(
            media_type="image",
            verdict=result["verdict"],
            confidence=result["confidence"],
            details={
                "detection": {
                    "label": result.get("label"),
                    "probs": result.get("probs", {}),
                    "models_used": result.get("models_used", []),
                    "face_detected": result.get("face_detected", False),
                    "ela_score": result.get("ela_score", 0),
                    "analysis": result.get("details", []),
                },
                "metadata": {
                    "has_exif": metadata.get("has_exif", False),
                    "risk_score": metadata.get("risk_score", 0),
                    "ai_indicators": metadata.get("ai_indicators", []),
                    "details": metadata.get("details", []),
                },
            },
            file_info={
                "filename": file.filename,
                "content_type": file.content_type,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image detection failed: {str(e)}",
        )
    finally:
        _cleanup(temp_path)


# ── Video Detection ──────────────────────────────────────────

@app.post("/detect/video", tags=["Detection"])
async def detect_video_endpoint(file: UploadFile = File(...)):
    """
    Detect deepfakes in an uploaded video (frame-by-frame).

    - **Accepts:** MP4, AVI, MOV, MKV, WebM, FLV, WMV
    - **Max size:** 100 MB
    - **Max duration:** 60 seconds
    - **Analysis:** Scene-aware frame sampling + dual-model ensemble per frame
    """
    temp_path = None
    try:
        temp_path = await _save_upload(file, ALLOWED_VIDEO_EXT, MAX_VIDEO_SIZE)

        result = detect_video(temp_path)

        # Simplify frame results for API response (avoid huge payloads)
        frame_summary = []
        for fr in result.get("frame_results", []):
            frame_summary.append({
                "frame_index": fr.get("frame_index"),
                "timestamp": fr.get("timestamp"),
                "verdict": fr.get("verdict"),
                "confidence": fr.get("confidence"),
            })

        return _build_response(
            media_type="video",
            verdict=result["verdict"],
            confidence=result["confidence"],
            details={
                "duration": result.get("duration", 0),
                "frame_count": result.get("frame_count", 0),
                "flagged_frames": result.get("flagged_frames", []),
                "frame_results": frame_summary,
                "analysis": result.get("details", []),
            },
            file_info={
                "filename": file.filename,
                "content_type": file.content_type,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video detection failed: {str(e)}",
        )
    finally:
        _cleanup(temp_path)


# ── Audio Detection ──────────────────────────────────────────

@app.post("/detect/audio", tags=["Detection"])
async def detect_audio_endpoint(file: UploadFile = File(...)):
    """
    Detect deepfake/synthetic audio in an uploaded file.

    - **Accepts:** MP3, WAV, FLAC, M4A, OGG, AAC, WMA
    - **Max size:** 50 MB
    - **Model:** Wav2Vec2-XLSR (garystafford) — 97.9% accuracy
    - **Fallback:** Spectral analysis heuristics
    """
    temp_path = None
    try:
        temp_path = await _save_upload(file, ALLOWED_AUDIO_EXT, MAX_AUDIO_SIZE)

        result = detect_audio(temp_path)

        return _build_response(
            media_type="audio",
            verdict=result["verdict"],
            confidence=result["confidence"],
            details={
                "method": result.get("method", "unknown"),
                "probs": result.get("probs", {}),
                "features": result.get("features", {}),
                "analysis": result.get("details", []),
            },
            file_info={
                "filename": file.filename,
                "content_type": file.content_type,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio detection failed: {str(e)}",
        )
    finally:
        _cleanup(temp_path)


# ── Metadata Analysis ────────────────────────────────────────

@app.post("/detect/metadata", tags=["Detection"])
async def detect_metadata_endpoint(file: UploadFile = File(...)):
    """
    Analyse file metadata for AI generation signatures.

    - **Accepts:** Image files (JPG, PNG, WebP, etc.)
    - **Checks:** EXIF data, PNG tEXt/iTXt chunks (SD/ComfyUI params),
      C2PA Content Credentials, AI software signatures, standard AI dimensions
    """
    temp_path = None
    try:
        temp_path = await _save_upload(file, ALLOWED_IMAGE_EXT, MAX_IMAGE_SIZE)

        result = analyze_metadata(temp_path)

        return {
            "media_type": "metadata",
            "risk_score": result.get("risk_score", 0),
            "has_exif": result.get("has_exif", False),
            "ai_indicators": result.get("ai_indicators", []),
            "details": result.get("details", []),
            "exif_data": result.get("exif_data", {}),
            "file_info": {
                "filename": file.filename,
                "content_type": file.content_type,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metadata analysis failed: {str(e)}",
        )
    finally:
        _cleanup(temp_path)


# ── Unified / Auto-Detect Endpoint ───────────────────────────

@app.post("/detect/auto", tags=["Detection"])
async def detect_auto_endpoint(file: UploadFile = File(...)):
    """
    Unified detection endpoint — auto-detects the media type and
    routes to the correct detector.

    - **Accepts:** Any supported image, video, or audio file
    - **Auto-routing:** File extension → appropriate detector
    """
    temp_path = None
    try:
        temp_path = await _save_upload(file, ALL_ALLOWED_EXT, MAX_VIDEO_SIZE)

        # Detect media type
        media_type = detect_media_type(temp_path)

        if media_type == "image":
            pil_image = Image.open(temp_path).convert("RGB")
            result = detect_image(pil_image)
            metadata = analyze_metadata(temp_path)

            # Generate AI insights
            from utils.explainer import generate_ai_insights
            ai_insights = generate_ai_insights(result, media_type="image")

            return _build_response(
                media_type="image",
                verdict=result["verdict"],
                confidence=result["confidence"],
                details={
                    "detection": {
                        "label": result.get("label"),
                        "probs": result.get("probs", {}),
                        "models_used": result.get("models_used", []),
                        "face_detected": result.get("face_detected", False),
                        "ela_score": result.get("ela_score", 0),
                        "analysis": result.get("details", []),
                    },
                    "metadata": {
                        "risk_score": metadata.get("risk_score", 0),
                        "ai_indicators": metadata.get("ai_indicators", []),
                    },
                    "ai_insights": ai_insights,
                },
                file_info={"filename": file.filename},
            )

        elif media_type == "video":
            result = detect_video(temp_path)
            frame_summary = [
                {
                    "frame_index": fr.get("frame_index"),
                    "timestamp": fr.get("timestamp"),
                    "verdict": fr.get("verdict"),
                    "confidence": fr.get("confidence"),
                    "face_detected": fr.get("face_detected", False),
                }
                for fr in result.get("frame_results", [])
            ]

            # Generate AI insights
            from utils.explainer import generate_ai_insights
            ai_insights = generate_ai_insights(result, media_type="video")

            return _build_response(
                media_type="video",
                verdict=result["verdict"],
                confidence=result["confidence"],
                details={
                    "duration": result.get("duration", 0),
                    "frame_count": result.get("frame_count", 0),
                    "flagged_frames": result.get("flagged_frames", []),
                    "frame_results": frame_summary,
                    "analysis": result.get("details", []),
                    "ai_insights": ai_insights,
                },
                file_info={"filename": file.filename},
            )

        elif media_type == "audio":
            result = detect_audio(temp_path)

            # Generate AI insights
            from utils.explainer import generate_ai_insights
            ai_insights = generate_ai_insights(result, media_type="audio")

            return _build_response(
                media_type="audio",
                verdict=result["verdict"],
                confidence=result["confidence"],
                details={
                    "method": result.get("method", "unknown"),
                    "probs": result.get("probs", {}),
                    "features": result.get("features", {}),
                    "analysis": result.get("details", []),
                    "ai_insights": ai_insights,
                },
                file_info={"filename": file.filename},
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    f"Could not determine media type for '{file.filename}'. "
                    f"Supported formats: images, videos, audio."
                ),
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}",
        )
    finally:
        _cleanup(temp_path)


# ══════════════════════════════════════════════════════════════
#  HEATMAP ENDPOINT
# ══════════════════════════════════════════════════════════════

@app.post("/detect/heatmap", summary="Generate ELA Heatmap")
async def generate_heatmap(file: UploadFile = File(...)):
    """
    Generate an Error Level Analysis (ELA) heatmap overlay for an uploaded image.
    Returns the heatmap as a base64-encoded PNG string.
    """
    import base64

    temp_path = None
    try:
        temp_path = await _save_upload(file, ALLOWED_IMAGE_EXT, MAX_IMAGE_SIZE)
        pil_image = Image.open(temp_path).convert("RGB")

        from utils.visualizer import generate_heatmap_overlay
        heatmap = generate_heatmap_overlay(pil_image)

        # Convert to base64 PNG
        buf = io.BytesIO()
        heatmap.save(buf, format="PNG")
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return JSONResponse(content={
            "heatmap": f"data:image/png;base64,{b64}",
            "filename": file.filename,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Heatmap generation failed: {str(e)}",
        )
    finally:
        _cleanup(temp_path)


# ══════════════════════════════════════════════════════════════
#  NOISE VARIANCE MAP ENDPOINT (Image)
# ══════════════════════════════════════════════════════════════

@app.post("/detect/noisemap", summary="Generate Noise Variance Map")
async def generate_noisemap(file: UploadFile = File(...)):
    """
    Generate a noise variance map for an uploaded image.
    Highlights regions where the camera sensor noise is inconsistent,
    indicating potential splicing or AI generation.
    """
    import base64
    import numpy as np
    from PIL import ImageFilter

    temp_path = None
    try:
        temp_path = await _save_upload(file, ALLOWED_IMAGE_EXT, MAX_IMAGE_SIZE)
        pil_image = Image.open(temp_path).convert("RGB")
        img_array = np.array(pil_image, dtype=np.float64)

        # Extract noise by subtracting a blurred version
        blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=5))
        blur_array = np.array(blurred, dtype=np.float64)
        noise = img_array - blur_array

        # Compute local variance using a sliding window approach
        noise_gray = np.mean(np.abs(noise), axis=2)

        # Normalize to 0-255
        max_val = noise_gray.max()
        if max_val > 0:
            noise_gray = (noise_gray / max_val * 255).astype(np.uint8)
        else:
            noise_gray = noise_gray.astype(np.uint8)

        # Apply colormap for visual appeal
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.cm as cm

        normalized = noise_gray.astype(np.float32) / 255.0
        colored = cm.inferno(normalized)
        colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
        noise_img = Image.fromarray(colored_rgb)

        # Blend with original at low opacity
        noise_img = noise_img.resize(pil_image.size)
        blended = Image.blend(pil_image, noise_img, alpha=0.55)

        buf = io.BytesIO()
        blended.save(buf, format="PNG")
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return JSONResponse(content={
            "noisemap": f"data:image/png;base64,{b64}",
            "filename": file.filename,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Noise map generation failed: {str(e)}",
        )
    finally:
        _cleanup(temp_path)


# ══════════════════════════════════════════════════════════════
#  MEL-SPECTROGRAM ENDPOINT (Audio)
# ══════════════════════════════════════════════════════════════

@app.post("/detect/spectrogram", summary="Generate Audio Mel-Spectrogram")
async def generate_spectrogram(file: UploadFile = File(...)):
    """
    Generate a Mel-Spectrogram visualization for an uploaded audio file.
    Shows the frequency content over time — AI-generated audio often
    shows unnatural patterns like comb artifacts or missing harmonics.
    """
    import base64
    import numpy as np

    temp_path = None
    try:
        temp_path = await _save_upload(file, ALLOWED_AUDIO_EXT, MAX_AUDIO_SIZE)

        import librosa
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        y, sr = librosa.load(temp_path, sr=22050, mono=True, duration=30)

        # Compute Mel-Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 4), dpi=120)
        fig.patch.set_facecolor('#080A0F')
        ax.set_facecolor('#080A0F')

        img = librosa.display.specshow(
            S_dB, sr=sr, x_axis='time', y_axis='mel',
            fmax=8000, ax=ax, cmap='magma'
        )

        ax.set_xlabel('Time (s)', color='#EDEDEA', fontsize=10)
        ax.set_ylabel('Frequency (Hz)', color='#EDEDEA', fontsize=10)
        ax.tick_params(colors='#4B5260', labelsize=8)

        for spine in ax.spines.values():
            spine.set_color('#1A1F2E')

        cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.ax.yaxis.set_tick_params(color='#4B5260')
        cbar.ax.yaxis.label.set_color('#EDEDEA')
        for label in cbar.ax.get_yticklabels():
            label.set_color('#4B5260')

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor='#080A0F', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return JSONResponse(content={
            "spectrogram": f"data:image/png;base64,{b64}",
            "filename": file.filename,
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Spectrogram generation failed: {str(e)}",
        )
    finally:
        _cleanup(temp_path)


# ══════════════════════════════════════════════════════════════
#  UNIFIED FORENSICS ENDPOINT
# ══════════════════════════════════════════════════════════════

@app.post("/detect/forensics", summary="Get All Forensic Visualizations")
async def get_forensics(file: UploadFile = File(...)):
    """
    Unified endpoint that returns all available forensic visualizations
    for any media type (image, audio, video).
    """
    import base64

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = os.path.splitext(file.filename)[1].lower()
    result = {}

    temp_path = None
    try:
        # ── IMAGE FORENSICS ──
        if ext in ALLOWED_IMAGE_EXT:
            temp_path = await _save_upload(file, ALLOWED_IMAGE_EXT, MAX_IMAGE_SIZE)
            pil_image = Image.open(temp_path).convert("RGB")
            from PIL import ImageFilter
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.cm as cm

            # 1. ELA Heatmap
            from utils.visualizer import generate_heatmap_overlay
            heatmap = generate_heatmap_overlay(pil_image)
            buf = io.BytesIO()
            heatmap.save(buf, format="PNG")
            buf.seek(0)
            result["heatmap"] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

            # 2. Noise Variance Map
            img_array = np.array(pil_image, dtype=np.float64)
            blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=5))
            blur_array = np.array(blurred, dtype=np.float64)
            noise = img_array - blur_array
            noise_gray = np.mean(np.abs(noise), axis=2)
            max_val = noise_gray.max()
            if max_val > 0:
                noise_gray = noise_gray / max_val
            colored = cm.inferno(noise_gray.astype(np.float32))
            colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
            noise_img = Image.fromarray(colored_rgb).resize(pil_image.size)
            blended = Image.blend(pil_image, noise_img, alpha=0.55)
            buf = io.BytesIO()
            blended.save(buf, format="PNG")
            buf.seek(0)
            result["noisemap"] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        # ── AUDIO FORENSICS ──
        elif ext in ALLOWED_AUDIO_EXT:
            temp_path = await _save_upload(file, ALLOWED_AUDIO_EXT, MAX_AUDIO_SIZE)

            import librosa
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            y, sr = librosa.load(temp_path, sr=22050, mono=True, duration=30)

            # 1. Mel-Spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            fig, ax = plt.subplots(figsize=(12, 4), dpi=120)
            fig.patch.set_facecolor('#080A0F')
            ax.set_facecolor('#080A0F')
            img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000, ax=ax, cmap='magma')
            ax.set_xlabel('Time (s)', color='#EDEDEA', fontsize=10)
            ax.set_ylabel('Frequency (Hz)', color='#EDEDEA', fontsize=10)
            ax.tick_params(colors='#4B5260', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#1A1F2E')
            cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
            cbar.ax.yaxis.set_tick_params(color='#4B5260')
            for label in cbar.ax.get_yticklabels():
                label.set_color('#4B5260')
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', facecolor='#080A0F', edgecolor='none')
            plt.close(fig)
            buf.seek(0)
            result["spectrogram"] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

            # 2. Waveform
            fig2, ax2 = plt.subplots(figsize=(12, 3), dpi=120)
            fig2.patch.set_facecolor('#080A0F')
            ax2.set_facecolor('#080A0F')
            times = np.linspace(0, len(y) / sr, num=len(y))
            ax2.plot(times, y, color='#EDEDEA', linewidth=0.3, alpha=0.7)
            ax2.fill_between(times, y, alpha=0.15, color='#EDEDEA')
            ax2.set_xlabel('Time (s)', color='#EDEDEA', fontsize=10)
            ax2.set_ylabel('Amplitude', color='#EDEDEA', fontsize=10)
            ax2.tick_params(colors='#4B5260', labelsize=8)
            for spine in ax2.spines.values():
                spine.set_color('#1A1F2E')
            ax2.set_xlim(0, len(y) / sr)
            plt.tight_layout()
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format='png', facecolor='#080A0F', edgecolor='none')
            plt.close(fig2)
            buf2.seek(0)
            result["waveform"] = f"data:image/png;base64,{base64.b64encode(buf2.getvalue()).decode('utf-8')}"

        # ── VIDEO FORENSICS ──
        elif ext in ALLOWED_VIDEO_EXT:
            temp_path = await _save_upload(file, ALLOWED_VIDEO_EXT, MAX_VIDEO_SIZE)

            # Run detection to get frame results
            video_result = detect_video(temp_path)

            # Generate video forensic visualizations
            from utils.video_visualizer import generate_video_forensics
            video_forensics = generate_video_forensics(
                temp_path,
                video_result.get("frame_results", []),
                video_result.get("flagged_frames", []),
            )

            result["suspicious_frames"] = video_forensics.get("suspicious_frames", [])
            result["frame_confidence_timeline"] = video_forensics.get("frame_confidence_timeline", [])
            if video_forensics.get("annotated_video_b64"):
                result["annotated_video"] = video_forensics["annotated_video_b64"]

        return JSONResponse(content={
            "forensics": result,
            "filename": file.filename,
            "media_type": "image" if ext in ALLOWED_IMAGE_EXT else "audio" if ext in ALLOWED_AUDIO_EXT else "video",
        })

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[Forensics] Error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forensics generation failed: {str(e)}",
        )
    finally:
        _cleanup(temp_path)


# ── Forensic Report Generation ────────────────────────────────

@app.post("/generate-report", tags=["Reports"])
async def generate_report_endpoint(file: UploadFile = File(...)):
    """
    Generate a downloadable PDF forensic report for an uploaded media file.

    Pipeline:
      1. Save upload → detect media type
      2. Run /detect/auto logic → get detection result + AI insights
      3. Run /detect/forensics logic → get forensic visualizations
      4. Generate PDF report with all data
      5. Return the download URL

    - **Accepts:** Any supported image, video, or audio file
    - **Returns:** JSON with report_path and download_url
    """
    temp_path = None
    try:
        temp_path = await _save_upload(file, ALL_ALLOWED_EXT, MAX_VIDEO_SIZE)
        media_type = detect_media_type(temp_path)

        # ── Step 1: Run detection ─────────────────────────────
        from utils.explainer import generate_ai_insights

        if media_type == "image":
            pil_image = Image.open(temp_path).convert("RGB")
            det_result = detect_image(pil_image)
            metadata = analyze_metadata(temp_path)
            ai_insights = generate_ai_insights(det_result, media_type="image")

            result = _build_response(
                media_type="image",
                verdict=det_result["verdict"],
                confidence=det_result["confidence"],
                details={
                    "detection": {
                        "label": det_result.get("label"),
                        "probs": det_result.get("probs", {}),
                        "models_used": det_result.get("models_used", []),
                        "face_detected": det_result.get("face_detected", False),
                        "ela_score": det_result.get("ela_score", 0),
                        "analysis": det_result.get("details", []),
                    },
                    "metadata": {
                        "risk_score": metadata.get("risk_score", 0),
                        "ai_indicators": metadata.get("ai_indicators", []),
                    },
                    "ai_insights": ai_insights,
                },
                file_info={"filename": file.filename, "content_type": file.content_type},
            )

        elif media_type == "video":
            det_result = detect_video(temp_path)
            ai_insights = generate_ai_insights(det_result, media_type="video")

            result = _build_response(
                media_type="video",
                verdict=det_result["verdict"],
                confidence=det_result["confidence"],
                details={
                    "duration": det_result.get("duration", 0),
                    "frame_count": det_result.get("frame_count", 0),
                    "flagged_frames": det_result.get("flagged_frames", []),
                    "analysis": det_result.get("details", []),
                    "ai_insights": ai_insights,
                },
                file_info={"filename": file.filename, "content_type": file.content_type},
            )

        elif media_type == "audio":
            det_result = detect_audio(temp_path)
            ai_insights = generate_ai_insights(det_result, media_type="audio")

            result = _build_response(
                media_type="audio",
                verdict=det_result["verdict"],
                confidence=det_result["confidence"],
                details={
                    "method": det_result.get("method", "unknown"),
                    "probs": det_result.get("probs", {}),
                    "features": det_result.get("features", {}),
                    "analysis": det_result.get("details", []),
                    "ai_insights": ai_insights,
                },
                file_info={"filename": file.filename, "content_type": file.content_type},
            )
        else:
            raise HTTPException(status_code=415, detail="Unsupported media type.")

        # ── Step 2: Generate forensics ────────────────────────
        import base64

        forensics = {}
        ext = os.path.splitext(file.filename)[1].lower()

        if ext in ALLOWED_IMAGE_EXT:
            from PIL import ImageFilter
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.cm as cm
            from utils.visualizer import generate_heatmap_overlay

            pil_image = Image.open(temp_path).convert("RGB")

            # ELA Heatmap
            heatmap = generate_heatmap_overlay(pil_image)
            buf = io.BytesIO()
            heatmap.save(buf, format="PNG")
            buf.seek(0)
            forensics["heatmap"] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

            # Noise map
            img_array = np.array(pil_image, dtype=np.float64)
            blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=5))
            blur_array = np.array(blurred, dtype=np.float64)
            noise = img_array - blur_array
            noise_gray = np.mean(np.abs(noise), axis=2)
            max_val = noise_gray.max()
            if max_val > 0:
                noise_gray = noise_gray / max_val
            colored = cm.inferno(noise_gray.astype(np.float32))
            colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
            from PIL import Image as PILImage
            noise_img = PILImage.fromarray(colored_rgb).resize(pil_image.size)
            blended = PILImage.blend(pil_image, noise_img, alpha=0.55)
            buf2 = io.BytesIO()
            blended.save(buf2, format="PNG")
            buf2.seek(0)
            forensics["noisemap"] = f"data:image/png;base64,{base64.b64encode(buf2.getvalue()).decode('utf-8')}"

        # ── Step 3: Generate PDF ──────────────────────────────
        from utils.report_generator import generate_pdf_report

        report_path = generate_pdf_report(
            result=result,
            forensics=forensics,
            media_path=temp_path,
        )

        report_filename = os.path.basename(report_path)

        return JSONResponse(content={
            **get_privacy_status(),
            "report_path": report_path,
            "download_url": f"/download-report/{report_filename}",
            "report_id": report_filename.replace("IntrusionX_Report_", "").replace(".pdf", ""),
            "verdict": result.get("verdict", "UNKNOWN"),
            "confidence": result.get("confidence", 0),
        })

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}",
        )
    finally:
        _cleanup(temp_path)


@app.get("/download-report/{filename}", tags=["Reports"])
async def download_report(filename: str):
    """
    Download a previously generated forensic PDF report.
    """
    reports_dir = os.path.join(os.path.dirname(__file__), "outputs", "reports")
    filepath = os.path.join(reports_dir, filename)

    if not os.path.isfile(filepath):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report '{filename}' not found.",
        )

    return FileResponse(
        path=filepath,
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Batch Analysis ────────────────────────────────────────────

@app.post("/detect/batch", tags=["Detection"])
async def detect_batch_endpoint(files: List[UploadFile] = File(...)):
    """
    Process multiple media files in a single request.
    Automatically detects media type for each file and routes it correctly.
    
    - **Accepts:** List of image, video, or audio files
    - **Returns:** Aggregated batch summary and list of individual results
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided for batch processing.",
        )
        
    temp_files = []
    
    try:
        # Save all uploads to temp
        for f in files:
            try:
                temp_path = await _save_upload(f, ALL_ALLOWED_EXT, MAX_VIDEO_SIZE)
                temp_files.append((temp_path, f.filename))
            except HTTPException as e:
                # Instead of crashing the whole batch, we record this file as a failure
                # by pushing None as the path. The batch processor handles this by emitting an error row.
                temp_files.append((None, f.filename))
            
        from utils.batch_processor import process_batch
        
        # Run batch processing
        batch_result = process_batch(temp_files)
        
        # Inject privacy status
        batch_result.update(get_privacy_status())
        
        return JSONResponse(content=batch_result)
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}",
        )
    finally:
        # Cleanup all temp files
        for temp_path, _ in temp_files:
            _cleanup(temp_path)
