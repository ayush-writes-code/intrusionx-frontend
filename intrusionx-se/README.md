# 🛡️ IntrusionX SE — AI-Powered Deepfake Detection System

> *"Protecting Truth in the Age of AI"*

**IntrusionX SE** is a multi-modal deepfake detection system that analyses **images**, **videos**, and **audio** to determine whether media content is authentic or synthetically generated.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange?logo=gradio)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📷 **Image Detection** | Vision Transformer (ViT) binary classifier for real vs deepfake |
| 🎬 **Video Detection** | Frame-by-frame analysis with aggregate scoring |
| 🎙️ **Audio Detection** | Wav2Vec 2.0 / spectral analysis for synthetic voice detection |
| 🗺️ **ELA Heatmap** | Error Level Analysis highlighting manipulated regions |
| 📋 **Metadata Forensics** | EXIF analysis detecting AI generation markers |
| 📊 **Confidence Gauge** | Visual confidence meter with verdict classification |
| 💬 **Explanations** | Human-readable analysis of why content was flagged |

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/intrusionx-se.git
cd intrusionx-se
pip install -r requirements.txt
```

### 2. Run Locally

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

### 3. Deploy to HuggingFace Spaces

1. Create a new Space on [HuggingFace](https://huggingface.co/new-space)
2. Select **Gradio** as the SDK
3. Push this repo to the Space
4. Done! Free hosting with a public URL

## 🏗️ Architecture

```
User Upload → Preprocessor → Media Type Router
                                ├── Image → ViT Detector ────────────┐
                                ├── Video → Frame Extractor + ViT ───┤──→ Results Engine → Verdict
                                └── Audio → Wav2Vec / Spectral ──────┘
                                         + Metadata Analyzer ────────┘
```

## 📁 Project Structure

```
intrusionx-se/
├── app.py                  # Main Gradio application
├── detectors/
│   ├── image_detector.py   # ViT-based image detection
│   ├── video_detector.py   # Frame extraction + image model
│   ├── audio_detector.py   # Audio deepfake detection
│   └── metadata_analyzer.py # EXIF forensics
├── utils/
│   ├── explainer.py        # Human-readable explanations
│   ├── visualizer.py       # Heatmaps & confidence gauges
│   └── preprocessing.py    # File validation
├── assets/
│   └── custom.css          # Dark theme styling
├── requirements.txt
└── README.md
```

## 🧠 Tech Stack

| Component | Technology |
|-----------|-----------|
| UI | Gradio 4.0+ |
| Image Detection | Vision Transformer (ViT) via HuggingFace |
| Audio Detection | Wav2Vec 2.0 / Spectral Analysis |
| Video Processing | OpenCV |
| ML Framework | PyTorch + Transformers |
| Visualization | Matplotlib + ELA |
| Metadata | Pillow + exifread |

## ⚠️ Limitations

- No detection system is 100% accurate
- Latest generation techniques may evade detection
- Results should be used as one factor in verification
- Video analysis limited to 60-second clips
- Audio analysis limited to 30 seconds

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built for hackathon demonstration purposes. IntrusionX SE — Protecting Truth in the Age of AI.*
