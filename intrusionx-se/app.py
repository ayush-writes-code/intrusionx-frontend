"""
IntrusionX SE — AI-Powered Deepfake Detection System
Main Gradio Application

Detects deepfakes in images, videos, and audio using:
- Dual-model ensemble (ViT + Swin Transformer) for visual analysis
- Face detection + cropping for targeted deepfake analysis
- Wav2Vec 2.0 / Spectral Analysis for audio forensics
- EXIF / PNG Metadata + C2PA Content Credentials analysis
- Error Level Analysis (ELA) for statistical forensics

Features:
- 🎯 Unified Upload: Drop any file — auto-routed to the right detector
- 📷 Image Tab: Dedicated image analysis with ELA heatmap
- 🎬 Video Tab: Frame-by-frame analysis with progress tracking
- 🎙️ Audio Tab: Synthetic voice detection with spectral features
"""

import os
import json
import gradio as gr
from PIL import Image

from detectors.image_detector import detect_image
from detectors.video_detector import detect_video
from detectors.audio_detector import detect_audio
from detectors.metadata_analyzer import analyze_metadata
from utils.media_router import detect_media_type, route_detection
from utils.explainer import (
    explain_image_result,
    explain_video_result,
    explain_audio_result,
    explain_metadata_result,
)
from utils.visualizer import generate_heatmap_overlay, generate_confidence_gauge
from utils.preprocessing import validate_image, validate_video, validate_audio, get_file_info

# ── CSS ───────────────────────────────────────────────────────
CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "custom.css")
custom_css = ""
if os.path.isfile(CSS_PATH):
    with open(CSS_PATH) as f:
        custom_css = f.read()


# ══════════════════════════════════════════════════════════════
#  UNIFIED UPLOAD HANDLER
# ══════════════════════════════════════════════════════════════

def handle_unified(file_input, progress=gr.Progress()):
    """
    Unified handler: accepts any file, auto-detects type, routes to
    the correct detector, and returns formatted results.

    Returns: (result_markdown, gauge_image, details_json)
    """
    if file_input is None:
        return (
            "⚪ **Drop any image, video, or audio file to analyse.**",
            None,
            "",
        )

    file_path = file_input

    # ── Detect media type ─────────────────────────────────
    media_type = detect_media_type(file_path)

    # ── Progress callback for video ───────────────────────
    prog_cb = None
    if media_type == "video":
        def prog_cb(current, total):
            progress(current / total, desc=f"Analysing frame {current}/{total}...")

    # ── Route to detector ─────────────────────────────────
    result = route_detection(file_path, progress_callback=prog_cb)

    # ── Handle errors ─────────────────────────────────────
    if result["verdict"] == "ERROR":
        error_msg = result["details"].get("error", "Unknown error")
        return f"⚪ **Error:** {error_msg}", None, ""

    # ── Format the response ───────────────────────────────
    verdict = result["verdict"]
    confidence = result["confidence"]
    detection = result["details"].get("detection", {})

    # Build the explanation markdown
    explanation = _format_unified_result(result, media_type, detection)

    # Generate confidence gauge
    gauge = generate_confidence_gauge(confidence, verdict)

    # Build JSON details panel
    details_md = _format_details_panel(result, detection)

    return explanation, gauge, details_md


def _format_unified_result(result: dict, media_type: str, detection: dict) -> str:
    """Format the unified result into rich markdown."""
    verdict = result["verdict"]
    confidence = result["confidence"]
    file_info = result.get("file_info", {})

    icon = {"DEEPFAKE": "🔴", "SUSPICIOUS": "🟡", "AUTHENTIC": "🟢"}.get(verdict, "⚪")
    type_icon = {"image": "📷", "video": "🎬", "audio": "🎙️"}.get(media_type, "📁")
    type_label = media_type.upper()

    # ── Header ────────────────────────────────────────────
    md = f"## {icon} Verdict: **{verdict}**\n\n"
    md += f"### Confidence: {confidence:.1f}%\n\n"

    # Confidence bar
    filled = int(confidence / 5)
    bar = "█" * filled + "░" * (20 - filled)
    md += f"`{bar}` **{confidence:.1f}%**\n\n"

    # ── Media info ────────────────────────────────────────
    md += f"---\n\n"
    md += f"**{type_icon} Detected Media Type:** `{type_label}`\n\n"
    if file_info:
        md += f"**File:** `{file_info.get('filename', 'N/A')}` "
        md += f"({file_info.get('size_mb', 0)} MB)\n\n"

    # ── Detection details ─────────────────────────────────
    details = detection.get("details", [])
    if isinstance(details, list) and details:
        md += "### Analysis Details\n\n"
        for d in details:
            md += f"- {d}\n"
        md += "\n"

    # ── Model probabilities ───────────────────────────────
    probs = detection.get("probs", {})
    if probs:
        md += "### Model Probabilities\n\n"
        for label, prob in probs.items():
            if isinstance(prob, (int, float)):
                md += f"- **{label}**: {prob:.1f}%\n" if prob != int(prob) else f"- **{label}**: {prob}%\n"
            else:
                md += f"- **{label}**: {prob}\n"
        md += "\n"

    # ── Video-specific: frame summary ─────────────────────
    if media_type == "video":
        frame_count = detection.get("frame_count", 0)
        flagged = detection.get("flagged_frames", [])
        duration = detection.get("duration", 0)
        if frame_count:
            md += f"### Video Analysis\n\n"
            md += f"- **Duration:** {duration:.1f}s\n"
            md += f"- **Frames analysed:** {frame_count}\n"
            md += f"- **Frames flagged:** {len(flagged)}\n\n"

            # Frame table
            frame_results = detection.get("frame_results", [])
            if frame_results:
                md += "| Frame | Time | Verdict | Confidence |\n"
                md += "|-------|------|---------|------------|\n"
                for fr in frame_results:
                    fi = fr.get("frame_index", 0)
                    t = fr.get("timestamp", 0)
                    v = fr.get("verdict", "?")
                    c = fr.get("confidence", 0)
                    flag = " ⚠️" if v == "DEEPFAKE" else ""
                    md += f"| #{fi} | {t:.1f}s | {v}{flag} | {c:.1f}% |\n"
                md += "\n"

    # ── Audio-specific: feature table ─────────────────────
    if media_type == "audio":
        method = detection.get("method", "")
        if method:
            md += f"**Detection Method:** {method.replace('_', ' ').title()}\n\n"
        features = detection.get("features", {})
        if features:
            md += "### Audio Features\n\n"
            md += "| Feature | Value |\n"
            md += "|---------|-------|\n"
            for k, v in features.items():
                name = k.replace("_", " ").title()
                if isinstance(v, float):
                    md += f"| {name} | {v:.4f} |\n"
                else:
                    md += f"| {name} | {v} |\n"
            md += "\n"

    # ── Metadata (images only) ────────────────────────────
    metadata = result["details"].get("metadata", {})
    if metadata:
        risk = metadata.get("risk_score", 0)
        indicators = metadata.get("ai_indicators", [])
        meta_details = metadata.get("details", [])

        risk_icon = "🔴" if risk >= 50 else "🟡" if risk >= 25 else "🟢"
        risk_label = "HIGH" if risk >= 50 else "MODERATE" if risk >= 25 else "LOW"

        md += f"### Metadata Analysis\n\n"
        md += f"**{risk_icon} Metadata Risk:** {risk_label} ({risk:.0f}%)\n\n"

        if indicators:
            for ind in indicators:
                md += f"- ⚠️ {ind}\n"
            md += "\n"

        if meta_details:
            for d in meta_details:
                md += f"- {d}\n"
            md += "\n"

    return md


def _format_details_panel(result: dict, detection: dict) -> str:
    """Format a collapsible JSON details panel."""
    # Build a clean summary dict for display
    summary = {
        "media_type": result["media_type"],
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "file": result.get("file_info", {}).get("filename", "N/A"),
        "models_used": detection.get("models_used", []),
        "face_detected": detection.get("face_detected", None),
        "ela_score": detection.get("ela_score", None),
        "method": detection.get("method", None),
    }
    # Remove None values
    summary = {k: v for k, v in summary.items() if v is not None}

    md = "### Raw Detection Output\n\n"
    md += "```json\n"
    md += json.dumps(summary, indent=2, default=str)
    md += "\n```\n"

    return md


# ══════════════════════════════════════════════════════════════
#  DEDICATED TAB HANDLERS (unchanged from original)
# ══════════════════════════════════════════════════════════════

def handle_image(image_input):
    """Process an uploaded image through all detection layers."""
    if image_input is None:
        return (
            "⚪ **Please upload an image to analyse.**",
            None,
            None,
            "",
        )

    # If Gradio gives us a filepath string, open it
    if isinstance(image_input, str):
        pil_image = Image.open(image_input)
        file_path = image_input
    else:
        pil_image = image_input
        file_path = None

    # ── Run detectors ──────────────────────────────────────
    result = detect_image(pil_image)

    # ── Generate visuals ───────────────────────────────────
    heatmap = generate_heatmap_overlay(pil_image)
    gauge = generate_confidence_gauge(result["confidence"], result["verdict"])

    # ── Generate explanation ───────────────────────────────
    explanation = explain_image_result(result)

    # ── Metadata analysis ──────────────────────────────────
    meta_text = ""
    if file_path and os.path.isfile(file_path):
        meta = analyze_metadata(file_path)
        meta_text = explain_metadata_result(meta)

    return explanation, heatmap, gauge, meta_text


def handle_video(video_input, progress=gr.Progress()):
    """Process an uploaded video through frame-by-frame detection."""
    if video_input is None:
        return "⚪ **Please upload a video to analyse.**", None, ""

    video_path = video_input

    # Validate
    is_valid, msg = validate_video(video_path)
    if not is_valid:
        return f"⚪ **Error:** {msg}", None, ""

    # Progress callback for Gradio
    def prog_cb(current, total):
        progress(current / total, desc=f"Analysing frame {current}/{total}...")

    # Run detection
    result = detect_video(video_path, progress_callback=prog_cb)

    # Generate explanation
    explanation = explain_video_result(result)

    # Generate gauge
    gauge = None
    if result["verdict"] != "ERROR":
        gauge = generate_confidence_gauge(result["confidence"], result["verdict"])

    # Frame analysis summary
    frame_text = ""
    if result.get("frame_results"):
        frame_text = "### Flagged Frames\n\n"
        flagged = [f for f in result["frame_results"] if f["verdict"] == "DEEPFAKE"]
        if flagged:
            for f in flagged:
                frame_text += f"- **Frame #{f['frame_index']}** at {f['timestamp']:.1f}s — "
                frame_text += f"{f['confidence']:.1f}% confidence\n"
        else:
            frame_text += "No frames were flagged as deepfake.\n"

    return explanation, gauge, frame_text


def handle_audio(audio_input):
    """Process uploaded audio through deepfake detection."""
    if audio_input is None:
        return "⚪ **Please upload an audio file to analyse.**", None

    audio_path = audio_input

    # Validate
    is_valid, msg = validate_audio(audio_path)
    if not is_valid:
        return f"⚪ **Error:** {msg}", None

    # Run detection
    result = detect_audio(audio_path)

    # Generate explanation
    explanation = explain_audio_result(result)

    # Generate gauge
    gauge = None
    if result["verdict"] != "ERROR":
        gauge = generate_confidence_gauge(result["confidence"], result["verdict"])

    return explanation, gauge


# ══════════════════════════════════════════════════════════════
#  BUILD THE GRADIO UI
# ══════════════════════════════════════════════════════════════

HEADER_HTML = """
<div style="text-align: center; padding: 20px 0 10px 0;">
    <h1 class="app-title">IntrusionX SE</h1>
    <p class="app-subtitle">AI-Powered Deepfake Detection System</p>
    <p style="color: #555577; font-size: 0.85rem; margin-top: 8px;">
        Multi-modal analysis &bull; Vision Transformer &bull; Audio Forensics &bull; Metadata Verification
    </p>
</div>
"""

ABOUT_MD = """
## About IntrusionX SE

**IntrusionX SE** is a multi-modal deepfake detection system that analyses images, videos, and audio
to determine whether media content is authentic or synthetically generated.

### How It Works

| Modality | Technology | Description |
|----------|-----------|-------------|
| **Image** | ViT + Swin Ensemble | Dual-model detection with face cropping + ELA |
| **Video** | Frame Sampling + Ensemble | Scene-aware keyframe extraction + per-frame analysis |
| **Audio** | Wav2Vec2-XLSR | Fine-tuned for ElevenLabs, Polly, Kokoro, Hume AI detection |
| **Metadata** | EXIF + PNG Chunks + C2PA | Checks for AI software, generation params, content credentials |
| **Visual** | Error Level Analysis (ELA) | Highlights potentially manipulated regions via heatmap |

### Detection Layers

1. **Face Detection** — OpenCV Haar cascade for face cropping
2. **AI Model Ensemble** — Dual model analysis with MAX-strategy disagreement handling
3. **Error Level Analysis** — Statistical forensics integrated into scoring
4. **Metadata Forensics** — EXIF + PNG chunk + C2PA inspection
5. **Spectral Features** — Audio frequency and energy pattern analysis

### Confidence Levels

- 🟢 **AUTHENTIC** — High confidence the media is real
- 🟡 **SUSPICIOUS** — Inconclusive, manual review recommended
- 🔴 **DEEPFAKE** — High confidence the media is synthetically generated

### Limitations

- No detection system is 100% accurate
- Latest generation techniques may evade detection
- Results should be used as one factor in verification, not the sole determinant
- Video analysis is limited to 60-second clips

---

*Built with Vision Transformers, PyTorch, and Gradio*
*Powered by HuggingFace open-source models*
"""


def build_app():
    """Build and return the Gradio Blocks app."""

    with gr.Blocks(
        title="IntrusionX SE — Deepfake Detection",
        css=custom_css,
        theme=gr.themes.Base(
            primary_hue="cyan",
            secondary_hue="purple",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="#0f0f19",
            body_background_fill_dark="#0f0f19",
            block_background_fill="#12122a",
            block_background_fill_dark="#12122a",
            input_background_fill="#0e0e1e",
            input_background_fill_dark="#0e0e1e",
            button_primary_background_fill="linear-gradient(135deg, #00d2ff, #7b2ff7)",
            button_primary_background_fill_hover="linear-gradient(135deg, #00b8e6, #6a1ef0)",
            button_primary_text_color="white",
            border_color_primary="#222244",
            block_border_color="#222244",
        ),
    ) as app:

        # ── Header ────────────────────────────────────────
        gr.HTML(HEADER_HTML)

        # ── Main Tabs ─────────────────────────────────────
        with gr.Tabs() as tabs:

            # ┌─────────────────────────────────────────────┐
            # │  UNIFIED UPLOAD TAB (PRIMARY)               │
            # └─────────────────────────────────────────────┘
            with gr.Tab("🎯 Analyse Media", id="unified_tab"):
                gr.Markdown(
                    "### Drop any file — images, videos, or audio\n"
                    "*Auto-detects the media type and routes to the correct detector.*\n\n"
                    "**Supported:** JPG, PNG, WebP, BMP, MP4, AVI, MOV, MKV, WebM, MP3, WAV, FLAC, M4A, OGG"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        unified_input = gr.File(
                            label="Upload Media File",
                            file_types=[
                                ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif",
                                ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv",
                                ".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma",
                            ],
                            type="filepath",
                        )
                        unified_btn = gr.Button(
                            "🔍 Analyse Media",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=1):
                        unified_result = gr.Markdown(
                            label="Analysis Result",
                            value="*Upload any media file and click **Analyse Media** to start*",
                        )
                        unified_gauge = gr.Image(
                            label="Confidence Gauge",
                            height=200,
                            show_label=True,
                            interactive=False,
                        )

                with gr.Row():
                    unified_details = gr.Markdown(
                        label="Details",
                        value="",
                    )

                unified_btn.click(
                    fn=handle_unified,
                    inputs=[unified_input],
                    outputs=[unified_result, unified_gauge, unified_details],
                    show_progress="full",
                )

            # ┌─────────────────────────────────────────────┐
            # │  IMAGE TAB                                  │
            # └─────────────────────────────────────────────┘
            with gr.Tab("📷 Image", id="image_tab"):
                gr.Markdown("### Upload an image to check for deepfake manipulation")
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(
                            label="Upload Image",
                            type="pil",
                            height=350,
                            sources=["upload", "clipboard"],
                        )
                        img_btn = gr.Button(
                            "🔍 Analyse Image",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=1):
                        img_result = gr.Markdown(
                            label="Analysis Result",
                            value="*Upload an image and click Analyse to start*",
                        )
                        img_gauge = gr.Image(
                            label="Confidence Gauge",
                            height=200,
                            show_label=True,
                            interactive=False,
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        img_heatmap = gr.Image(
                            label="🗺️ ELA Heatmap (Manipulation Regions)",
                            height=350,
                            show_label=True,
                            interactive=False,
                        )
                    with gr.Column(scale=1):
                        img_meta = gr.Markdown(
                            label="Metadata Analysis",
                            value="*Metadata analysis will appear here after processing*",
                        )

                img_btn.click(
                    fn=handle_image,
                    inputs=[img_input],
                    outputs=[img_result, img_heatmap, img_gauge, img_meta],
                    show_progress="full",
                )

            # ┌─────────────────────────────────────────────┐
            # │  VIDEO TAB                                  │
            # └─────────────────────────────────────────────┘
            with gr.Tab("🎬 Video", id="video_tab"):
                gr.Markdown("### Upload a video to analyse frame-by-frame for deepfakes")
                gr.Markdown(
                    "*Supports MP4, AVI, MOV, MKV, WebM. Maximum 60 seconds.*",
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        vid_input = gr.Video(
                            label="Upload Video",
                            height=350,
                            sources=["upload"],
                        )
                        vid_btn = gr.Button(
                            "🔍 Analyse Video",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=1):
                        vid_result = gr.Markdown(
                            label="Analysis Result",
                            value="*Upload a video and click Analyse to start*",
                        )
                        vid_gauge = gr.Image(
                            label="Confidence Gauge",
                            height=200,
                            show_label=True,
                            interactive=False,
                        )

                with gr.Row():
                    vid_frames = gr.Markdown(
                        label="Frame Analysis",
                        value="",
                    )

                vid_btn.click(
                    fn=handle_video,
                    inputs=[vid_input],
                    outputs=[vid_result, vid_gauge, vid_frames],
                    show_progress="full",
                )

            # ┌─────────────────────────────────────────────┐
            # │  AUDIO TAB                                  │
            # └─────────────────────────────────────────────┘
            with gr.Tab("🎙️ Audio", id="audio_tab"):
                gr.Markdown("### Upload an audio file to check for synthetic voice generation")
                gr.Markdown(
                    "*Supports WAV, MP3, FLAC, OGG, M4A. Maximum 30 seconds analysed.*",
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        aud_input = gr.Audio(
                            label="Upload Audio",
                            type="filepath",
                            sources=["upload", "microphone"],
                        )
                        aud_btn = gr.Button(
                            "🔍 Analyse Audio",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=1):
                        aud_result = gr.Markdown(
                            label="Analysis Result",
                            value="*Upload audio and click Analyse to start*",
                        )
                        aud_gauge = gr.Image(
                            label="Confidence Gauge",
                            height=200,
                            show_label=True,
                            interactive=False,
                        )

                aud_btn.click(
                    fn=handle_audio,
                    inputs=[aud_input],
                    outputs=[aud_result, aud_gauge],
                    show_progress="full",
                )

            # ┌─────────────────────────────────────────────┐
            # │  ABOUT TAB                                  │
            # └─────────────────────────────────────────────┘
            with gr.Tab("📋 About", id="about_tab"):
                gr.Markdown(ABOUT_MD)

        # ── Footer ────────────────────────────────────────
        gr.HTML("""
        <div style="text-align: center; padding: 20px 0; color: #444466; font-size: 0.75rem;">
            IntrusionX SE &bull; Multi-Modal Deepfake Detection &bull; 
            Powered by Vision Transformers &amp; Wav2Vec 2.0 &bull;
            Built with Gradio &amp; HuggingFace
        </div>
        """)

    return app


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        show_error=True,
    )
