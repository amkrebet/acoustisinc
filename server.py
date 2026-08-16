#!/usr/bin/env python3
"""
================================================================================
HI-RES AUDIO FORENSIC EXPLORER (WEB APPLICATION SERVER)
================================================================================
A real-time on-demand forensic laboratory web application for high-resolution
audio libraries.
- Dynamically browse filesystem folders and albums.
- Real-time on-demand 64-bit double precision DSP forensic analysis.
- Interactive spectrogram with zoom, pan, coordinates, and exact dB level cursor HUD.
- Interactive spectrum curves with Peak/RMS traces and Nyquist reference markers.
- Built-in audio streaming player for in-browser listening and visual verification.
================================================================================
"""

import os
import sys
import io
import time
import json
import base64
import argparse
import urllib.parse
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
import soundfile as sf
import numpy as np

# Core DSP functions
from analyser import analyze_audio_forensics, encode_spectrogram_and_lookup


# Default root directory for initial browsing
INITIAL_ROOT = os.getcwd()


def format_bytes(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def format_seconds(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:d}:{s:02d}"


def get_directory_contents(target_path):
    """
    Lists subdirectories and audio files in target_path with rich metadata.
    """
    if not target_path or not str(target_path).strip():
        target_path = INITIAL_ROOT
    if str(target_path).startswith("~"):
        target_path = os.path.expanduser(target_path)
    target_path = os.path.abspath(target_path)

    if not os.path.exists(target_path):
        return {"error": f"Path does not exist: {target_path}"}
    if not os.path.isdir(target_path):
        return {"error": f"Path is not a directory: {target_path}"}

    folders = []
    files = []

    try:
        entries = sorted(os.listdir(target_path), key=lambda s: s.lower())
    except Exception as e:
        return {"error": f"Cannot read directory: {str(e)}"}

    for name in entries:
        if name.startswith("."):
            continue
        full_path = os.path.join(target_path, name)

        try:
            if os.path.isdir(full_path):
                audio_count = 0
                try:
                    sub_entries = os.listdir(full_path)
                    audio_count = sum(1 for e in sub_entries if e.lower().endswith(('.flac', '.wav', '.aiff', '.m4a', '.mp3')) and not e.endswith('.WIP') and not e.startswith('.'))
                except Exception:
                    pass
                folders.append({
                    "name": name,
                    "path": full_path,
                    "audio_count": audio_count
                })

            elif os.path.isfile(full_path) and name.lower().endswith(('.flac', '.wav', '.aiff', '.m4a', '.mp3')) and not name.endswith('.WIP'):
                size_bytes = 0
                try:
                    size_bytes = os.path.getsize(full_path)
                except Exception:
                    pass

                file_meta = {
                    "name": name,
                    "path": full_path,
                    "size_bytes": size_bytes,
                    "size_str": format_bytes(size_bytes),
                    "samplerate": 0,
                    "channels": 0,
                    "duration_str": "--:--",
                    "is_hires": False
                }
                try:
                    info = sf.info(full_path)
                    file_meta["samplerate"] = info.samplerate
                    file_meta["channels"] = info.channels
                    file_meta["duration_s"] = info.duration
                    file_meta["duration_str"] = format_seconds(info.duration)
                    file_meta["is_hires"] = info.samplerate > 48000
                    file_meta["subtype"] = info.subtype
                except Exception:
                    pass

                files.append(file_meta)
        except Exception:
            continue

    # Build breadcrumbs
    parts = []
    accum = ""
    for segment in target_path.strip("/").split("/"):
        if not segment:
            continue
        accum += "/" + segment
        parts.append({"name": segment, "path": accum})

    parent_path = os.path.dirname(target_path) if target_path != "/" else "/"

    return {
        "current_path": target_path,
        "parent_path": parent_path,
        "breadcrumbs": parts,
        "folders": folders,
        "files": files
    }


def analyze_file_on_demand(filepath):
    """
    Performs on-demand DSP analysis with strict 64-bit double precision.
    """
    if not os.path.exists(filepath):
        return {"status": "error", "message": "File not found"}

    try:
        t0 = time.time()
        data, sr = sf.read(filepath, dtype='float64', start=0, stop=60*192000)
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        taper_len = min(int(sr * 0.05), len(data) // 10)
        if taper_len > 0:
            taper = np.sin(np.linspace(0, np.pi/2, taper_len))**2
            data[:taper_len] *= taper
            data[-taper_len:] *= taper[::-1]

        spec_db, freqs, peak_dbfs, rms_dbfs, assessment_text, dr_metrics = analyze_audio_forensics(data, sr)

        nyquist = sr / 2.0
        duration_s = len(data) / float(sr)

        # True, uncolored physical dBFS spectral traces
        display_peak = np.copy(peak_dbfs)
        display_rms = np.copy(rms_dbfs)

        step = max(1, len(freqs) // 2048)
        curve_freqs_khz = (freqs[::step] / 1000.0).round(3).tolist()
        curve_peak_db = display_peak[::step].round(2).tolist()
        curve_rms_db = display_rms[::step].round(2).tolist()

        webp_b64, lookup_b64, lookup_w, lookup_h = encode_spectrogram_and_lookup(spec_db)

        # Extract verdict and noise profile
        verdict = "ANALYZED"
        noise_profile = ""
        for line in assessment_text.splitlines():
            if line.startswith("ASSESSMENT:"):
                verdict = line.replace("ASSESSMENT:", "").strip()
            elif line.startswith("NOISE PROFILE:"):
                noise_profile = line.replace("NOISE PROFILE:", "").strip().strip("[]")

        return {
            "status": "ok",
            "filename": os.path.basename(filepath),
            "filepath": filepath,
            "sr": sr,
            "nyquist_khz": nyquist / 1000.0,
            "duration_s": duration_s,
            "verdict": verdict,
            "noise_profile": noise_profile,
            "dr_score": dr_metrics.get("dr_score", 0),
            "dr_val": dr_metrics.get("dr_val", 0.0),
            "crest_factor_db": dr_metrics.get("crest_factor_db", 0.0),
            "integrated_lufs": dr_metrics.get("integrated_lufs", -140.0),
            "lra_lu": dr_metrics.get("lra_lu", 0.0),
            "report_text": assessment_text,
            "webp_base64": webp_b64,
            "lookup_base64": lookup_b64,
            "lookup_w": lookup_w,
            "lookup_h": lookup_h,
            "curve_freqs_khz": curve_freqs_khz,
            "curve_peaks": curve_peak_db,
            "curve_rms": curve_rms_db,
            "analysis_time": round(time.time() - t0, 3)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hi-Res Audio Forensic Explorer</title>
    <style>
        :root {
            --bg: #0d1117;
            --surface: #161b22;
            --surface-hover: #1f242c;
            --border: #30363d;
            --text: #c9d1d9;
            --text-heading: #f0f6fc;
            --text-muted: #8b949e;
            --accent-cyan: #00e5ff;
            --accent-pink: #ff007f;
            --accent-green: #aeea00;
            --accent-yellow: #ffea00;
            --accent-red: #ff1744;
            --accent-blue: #58a6ff;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }
        header {
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            flex-shrink: 0;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 700;
            font-size: 1.1rem;
            color: var(--text-heading);
        }
        .brand-badge {
            background: rgba(0, 229, 255, 0.12);
            border: 1px solid rgba(0, 229, 255, 0.3);
            color: var(--accent-cyan);
            font-size: 0.75rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
        }
        .bookmarks {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .btn-bookmark {
            background: #21262d;
            border: 1px solid var(--border);
            color: var(--text);
            padding: 5px 10px;
            font-size: 0.8rem;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .btn-bookmark:hover {
            background: #30363d;
            color: var(--text-heading);
            border-color: var(--accent-blue);
        }
        .main-layout {
            display: flex;
            flex: 1;
            overflow: hidden;
        }
        /* Left Pane: Library Explorer */
        .sidebar {
            width: 420px;
            min-width: 340px;
            background: var(--surface);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .nav-bar {
            padding: 12px;
            border-bottom: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            gap: 8px;
            background: #11141a;
        }
        .breadcrumbs {
            display: flex;
            align-items: center;
            gap: 4px;
            flex-wrap: wrap;
            font-size: 0.85rem;
        }
        .crumb {
            color: var(--accent-blue);
            cursor: pointer;
            text-decoration: none;
            padding: 2px 4px;
            border-radius: 3px;
        }
        .crumb:hover {
            background: rgba(88, 166, 255, 0.15);
        }
        .search-box {
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 6px 10px;
            color: var(--text-heading);
            font-size: 0.85rem;
            width: 100%;
            outline: none;
        }
        .search-box:focus {
            border-color: var(--accent-blue);
        }
        .tree-list {
            flex: 1;
            overflow-y: auto;
            padding: 8px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .folder-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 10px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.85rem;
            user-select: none;
            transition: background 0.1s;
            overflow: hidden;
            min-width: 0;
            gap: 8px;
        }
        .folder-item:hover {
            background: var(--surface-hover);
        }
        .folder-name-container {
            flex: 1;
            min-width: 0;
            overflow: hidden;
            white-space: nowrap;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .folder-name {
            display: inline-block;
            white-space: nowrap;
            will-change: transform;
        }
        .file-item {
            display: flex;
            flex-direction: column;
            padding: 8px 10px;
            border-radius: 5px;
            cursor: pointer;
            background: #11141a;
            border: 1px solid #21262d;
            margin-bottom: 4px;
            transition: all 0.15s ease;
            overflow: hidden;
            min-width: 0;
        }
        .file-item:hover {
            background: var(--surface-hover);
            border-color: #38404a;
        }
        .file-item.active {
            border-color: var(--accent-cyan);
            background: rgba(0, 229, 255, 0.05);
        }
        .file-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            min-width: 0;
        }
        .file-title-container {
            flex: 1;
            min-width: 0;
            overflow: hidden;
            white-space: nowrap;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .file-title {
            display: inline-block;
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-heading);
            white-space: nowrap;
            will-change: transform;
        }
        .ticker-active {
            animation: ticker-scroll var(--ticker-duration, 4s) cubic-bezier(0.45, 0.05, 0.55, 0.95) infinite alternate;
        }
        @keyframes ticker-scroll {
            0%, 20% {
                transform: translateX(0);
            }
            80%, 100% {
                transform: translateX(var(--ticker-offset, 0px));
            }
        }
        .file-meta {
            display: flex;
            gap: 8px;
            align-items: center;
            margin-top: 4px;
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        .badge-hires {
            background: rgba(0, 229, 255, 0.15);
            border: 1px solid rgba(0, 229, 255, 0.4);
            color: var(--accent-cyan);
            padding: 1px 6px;
            border-radius: 3px;
            font-weight: 600;
        }
        .badge-cd {
            background: #21262d;
            border: 1px solid var(--border);
            color: #8b949e;
            padding: 1px 6px;
            border-radius: 3px;
        }
        /* Right Pane: Lab & Visualizer */
        .workspace {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: var(--text-muted);
            gap: 12px;
        }
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 12px;
        }
        .card-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-heading);
        }
        .toolbar {
            display: flex;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
        }
        .btn {
            background: #21262d;
            border: 1px solid var(--border);
            color: var(--text);
            padding: 4px 10px;
            font-size: 0.75rem;
            font-weight: 500;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s ease;
            user-select: none;
        }
        .btn:hover {
            background: #30363d;
            color: var(--text-heading);
            border-color: #8b949e;
        }
        .canvas-container {
            position: relative;
            width: 100%;
            height: 320px;
            background: #11141a;
            border-radius: 6px;
            overflow: hidden;
            touch-action: none;
        }
        canvas {
            display: block;
            width: 100%;
            height: 100%;
            cursor: crosshair;
        }
        .hud-overlay {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(13, 17, 23, 0.9);
            backdrop-filter: blur(6px);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 6px 12px;
            font-family: "JetBrains Mono", monospace;
            font-size: 0.75rem;
            pointer-events: none;
            display: flex;
            flex-direction: column;
            gap: 3px;
            z-index: 10;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }
        .verdict-banner {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            border-radius: 6px;
            background: rgba(174, 234, 0, 0.08);
            border: 1px solid rgba(174, 234, 0, 0.3);
            margin-bottom: 4px;
        }
        .verdict-fake {
            background: rgba(255, 23, 68, 0.08);
            border-color: rgba(255, 23, 68, 0.3);
        }
        .verdict-native {
            background: rgba(0, 229, 255, 0.08);
            border-color: rgba(0, 229, 255, 0.3);
        }
        .report-box {
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 14px;
            font-family: "JetBrains Mono", SFMono-Regular, Consolas, monospace;
            font-size: 0.82rem;
            line-height: 1.5;
            color: var(--accent-green);
            white-space: pre-wrap;
            position: relative;
        }
        .legend-bar {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 14px;
            font-size: 0.75rem;
            margin-top: 8px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .legend-dot {
            width: 8px;
            height: 8px;
            border-radius: 2px;
        }
        .spinner {
            border: 3px solid rgba(255,255,255,0.1);
            border-top: 3px solid var(--accent-cyan);
            border-radius: 50%;
            width: 32px;
            height: 32px;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        audio {
            width: 100%;
            height: 36px;
            outline: none;
            margin-top: 8px;
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <span>🔬 Hi-Res Audio Forensic Explorer</span>
            <span class="brand-badge">HTML5 Dynamic DSP</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; flex: 1; max-width: 680px;">
            <input type="text" id="pathBar" class="search-box" style="flex: 1; margin: 0; padding: 6px 12px; font-size: 0.85rem;" placeholder="Enter folder path (e.g. /path/to/music or ~)...">
            <button class="btn" onclick="navigateToPathBar()" style="padding: 6px 12px; font-weight: bold;">Go</button>
            <button class="btn" onclick="loadDirectory('.')" title="Initial Directory">📍 Start</button>
            <button class="btn" onclick="loadDirectory('~')" title="User Home">🏠 Home</button>
            <button class="btn" onclick="loadDirectory('/')" title="Root Filesystem">📂 Root</button>
        </div>
    </header>

    <div class="main-layout">
        <!-- Sidebar Navigation -->
        <aside class="sidebar">
            <div class="nav-bar">
                <div class="breadcrumbs" id="breadcrumbs"></div>
                <input type="text" class="search-box" id="searchBox" placeholder="Filter folders & tracks...">
            </div>
            <div class="tree-list" id="treeList">
                <div style="padding: 20px; text-align: center; color: #8b949e;">Loading library...</div>
            </div>
        </aside>

        <!-- Main Workspace -->
        <main class="workspace" id="workspace">
            <div class="empty-state" id="emptyState">
                <svg width="48" height="48" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"/>
                </svg>
                <p>Select any audio track from the left panel to execute real-time forensic analysis.</p>
            </div>

            <!-- Active Analysis Content (hidden until loaded) -->
            <div id="analysisContent" style="display: none; flex-direction: column; gap: 16px;">
                <!-- Track Overview Card -->
                <section class="card">
                    <div class="card-header" style="margin-bottom: 8px;">
                        <div>
                            <h2 id="trackTitle" style="font-size: 1.15rem; color: var(--text-heading); word-break: break-all;">Track Title</h2>
                            <div id="trackMeta" style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">--</div>
                        </div>
                        <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
                            <div id="verdictBadge" class="badge-hires" style="font-size: 0.82rem; padding: 4px 10px;">--</div>
                            <div id="drBadge" class="badge-hires" style="font-size: 0.82rem; padding: 4px 10px; display: none;">--</div>
                            <div id="noiseBadge" class="badge-cd" style="font-size: 0.82rem; padding: 4px 10px; display: none;">--</div>
                        </div>
                    </div>
                    <audio id="audioPlayer" controls></audio>
                </section>

                <!-- Panel 1: Spectrogram Heatmap -->
                <section class="card">
                    <div class="card-header">
                        <span class="card-title">Linear Spectrogram (Time vs Frequency)</span>
                        <div class="toolbar">
                            <span style="font-size: 0.75rem; color: #8b949e;">Zoom:</span>
                            <button class="btn" onclick="zoomSpec(1.35)">+ In</button>
                            <button class="btn" onclick="zoomSpec(1/1.35)">- Out</button>
                            <button class="btn" onclick="setSpecPreset('audible')">0-20 kHz</button>
                            <button class="btn" onclick="setSpecPreset('ultrasonic')">20 kHz+</button>
                            <button class="btn" onclick="resetSpecZoom()">&#x21BA; Reset</button>
                        </div>
                    </div>
                    <div class="canvas-container">
                        <canvas id="spectrogramCanvas"></canvas>
                        <div class="hud-overlay">
                            <div><span style="color: #8b949e;">Time :</span> <strong id="specHudTime">-- s</strong></div>
                            <div><span style="color: var(--accent-cyan);">Freq :</span> <strong id="specHudFreq">-- kHz</strong></div>
                            <div><span style="color: var(--accent-green);">Level:</span> <strong id="specHudDb">-- dBFS</strong></div>
                        </div>
                    </div>
                    <div style="font-size: 0.75rem; color: #8b949e; margin-top: 6px; display: flex; justify-content: space-between;">
                        <span>💡 Scroll wheel to zoom, click &amp; drag to pan.</span>
                        <span>0 dBFS &rarr; -165 dBFS</span>
                    </div>
                </section>

                <!-- Panel 2: Interactive Spectrum Curve -->
                <section class="card">
                    <div class="card-header">
                        <span class="card-title">Frequency Spectrum &amp; Noise Profile</span>
                        <div class="toolbar">
                            <span style="font-size: 0.75rem; color: #8b949e;">Zoom:</span>
                            <button class="btn" onclick="zoomCurve(1.35)">+ In</button>
                            <button class="btn" onclick="zoomCurve(1/1.35)">- Out</button>
                            <button class="btn" onclick="setCurvePreset('audible')">0-20 kHz</button>
                            <button class="btn" onclick="setCurvePreset('cutoff')">15-25 kHz</button>
                            <button class="btn" onclick="setCurvePreset('ultrasonic')">20 kHz+</button>
                            <button class="btn" onclick="resetCurveZoom()">&#x21BA; Reset</button>
                        </div>
                    </div>
                    <div class="canvas-container">
                        <canvas id="spectrumCanvas"></canvas>
                        <div class="hud-overlay">
                            <div><span style="color: #8b949e;">Freq:</span> <strong id="hudFreq">-- kHz</strong></div>
                            <div><span style="color: var(--accent-cyan);">Peak:</span> <strong id="hudPeak">-- dBFS</strong></div>
                            <div><span style="color: var(--accent-pink);">RMS :</span> <strong id="hudRMS">-- dBFS</strong></div>
                        </div>
                    </div>
                    <div class="legend-bar">
                        <div class="legend-item"><div class="legend-dot" style="background: var(--accent-cyan);"></div>Peak Hold</div>
                        <div class="legend-item"><div class="legend-dot" style="background: var(--accent-pink);"></div>RMS Noise Floor</div>
                        <div class="legend-item"><div class="legend-dot" style="background: var(--accent-red);"></div>20 kHz Limit</div>
                        <div class="legend-item"><div class="legend-dot" style="background: var(--accent-yellow);"></div>22.05 kHz CD</div>
                    </div>
                </section>

                <!-- Panel 3: Forensic Lab Report Card -->
                <section class="card">
                    <div class="card-header">
                        <span class="card-title">Forensic Assessment Lab Report</span>
                        <button class="btn" onclick="copyReport()">Copy Report</button>
                    </div>
                    <div class="report-box" id="reportText">--</div>
                </section>
            </div>
        </main>
    </div>

    <script>
        let currentPath = '';
        let currentAnalysis = null;
        let directoryData = null;

        function navigateToPathBar() {
            const p = document.getElementById('pathBar').value.trim();
            loadDirectory(p || '.');
        }

        document.getElementById('pathBar').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') navigateToPathBar();
        });

        // Search filtering
        document.getElementById('searchBox').addEventListener('input', (e) => {
            renderDirectory(e.target.value.toLowerCase());
        });

        async function loadDirectory(path) {
            const player = document.getElementById('audioPlayer');
            if (player && !player.paused) player.pause();

            currentPath = path || '';
            const searchBox = document.getElementById('searchBox');
            if (searchBox) searchBox.value = '';

            const treeList = document.getElementById('treeList');
            treeList.innerHTML = '<div style="padding: 20px; text-align: center; color: #8b949e;"><div class="spinner" style="margin: 0 auto 10px;"></div>Loading directory...</div>';

            try {
                const res = await fetch(`/api/browse?path=${encodeURIComponent(currentPath)}`);
                directoryData = await res.json();
                if (directoryData.error) {
                    throw new Error(directoryData.error);
                }
                currentPath = directoryData.current_path;
                document.getElementById('pathBar').value = currentPath;
                renderBreadcrumbs(directoryData.breadcrumbs, directoryData.parent_path);
                renderDirectory('');
            } catch (err) {
                treeList.innerHTML = `<div style="padding: 20px; color: var(--accent-red); text-align: center;">Failed to load directory:<br><small>${err.message}</small></div>`;
            }
        }

        function renderBreadcrumbs(crumbs, parentPath) {
            const el = document.getElementById('breadcrumbs');
            el.innerHTML = '';

            const upBtn = document.createElement('span');
            upBtn.className = 'crumb';
            upBtn.innerHTML = '&#x21E7; Up';
            upBtn.onclick = () => loadDirectory(parentPath);
            el.appendChild(upBtn);

            crumbs.forEach(c => {
                const span = document.createElement('span');
                span.innerHTML = ' / ';
                span.style.color = '#8b949e';
                el.appendChild(span);

                const a = document.createElement('span');
                a.className = 'crumb';
                a.textContent = c.name;
                a.onclick = () => loadDirectory(c.path);
                el.appendChild(a);
            });
        }

        function setupTicker(itemEl, textEl, containerEl) {
            itemEl.addEventListener('mouseenter', () => {
                const diff = textEl.scrollWidth - containerEl.clientWidth;
                if (diff > 4) {
                    const duration = Math.max(3.5, diff / 25);
                    textEl.style.setProperty('--ticker-offset', `-${diff + 6}px`);
                    textEl.style.setProperty('--ticker-duration', `${duration.toFixed(1)}s`);
                    textEl.classList.add('ticker-active');
                }
            });
            itemEl.addEventListener('mouseleave', () => {
                textEl.classList.remove('ticker-active');
                textEl.style.transform = '';
            });
        }

        function renderDirectory(filter) {
            if (!directoryData) return;
            const treeList = document.getElementById('treeList');
            treeList.innerHTML = '';

            // Folders
            const filteredFolders = directoryData.folders.filter(f => !filter || f.name.toLowerCase().includes(filter));
            filteredFolders.forEach(f => {
                const div = document.createElement('div');
                div.className = 'folder-item';
                div.title = f.name;
                div.innerHTML = `
                    <div class="folder-name-container">
                        <span style="flex-shrink: 0;">📁</span>
                        <span class="folder-name"><strong>${escapeHtml(f.name)}</strong></span>
                    </div>
                    <span style="font-size: 0.75rem; color: #8b949e; flex-shrink: 0;">${f.audio_count} tracks</span>
                `;
                const containerEl = div.querySelector('.folder-name-container');
                const nameEl = div.querySelector('.folder-name');
                setupTicker(div, nameEl, containerEl);
                div.onclick = () => loadDirectory(f.path);
                treeList.appendChild(div);
            });

            // Files
            const filteredFiles = directoryData.files.filter(f => !filter || f.name.toLowerCase().includes(filter));
            filteredFiles.forEach((f, idx) => {
                const div = document.createElement('div');
                div.className = 'file-item';
                div.id = `file-item-${idx}`;
                div.title = f.name;
                
                const badgeClass = f.is_hires ? 'badge-hires' : 'badge-cd';
                const badgeText = f.samplerate ? `${(f.samplerate/1000).toFixed(1)}k` : 'FLAC';

                div.innerHTML = `
                    <div class="file-top">
                        <div class="file-title-container">
                            <span style="flex-shrink: 0;">🎵</span>
                            <span class="file-title">${escapeHtml(f.name)}</span>
                        </div>
                        <span class="${badgeClass}" style="flex-shrink: 0;">${badgeText}</span>
                    </div>
                    <div class="file-meta">
                        <span>⏱ ${f.duration_str}</span>
                        <span>📦 ${f.size_str}</span>
                        ${f.subtype ? `<span>${escapeHtml(f.subtype)}</span>` : ''}
                    </div>
                `;
                const containerEl = div.querySelector('.file-title-container');
                const titleEl = div.querySelector('.file-title');
                setupTicker(div, titleEl, containerEl);
                div.onclick = () => analyzeTrack(f, div);
                treeList.appendChild(div);
            });

            if (filteredFolders.length === 0 && filteredFiles.length === 0) {
                treeList.innerHTML = '<div style="padding: 20px; text-align: center; color: #8b949e;">No audio files found.</div>';
            }
        }

        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        async function analyzeTrack(file, fileEl) {
            document.querySelectorAll('.file-item').forEach(el => el.classList.remove('active'));
            if (fileEl) fileEl.classList.add('active');

            const emptyState = document.getElementById('emptyState');
            const analysisContent = document.getElementById('analysisContent');

            emptyState.style.display = 'flex';
            emptyState.innerHTML = `<div class="spinner"></div><p style="margin-top: 12px;">Running 64-bit DSP Forensic Analysis on <strong>${escapeHtml(file.name)}</strong>...</p>`;
            analysisContent.style.display = 'none';

            try {
                const res = await fetch(`/api/analyze?path=${encodeURIComponent(file.path)}`);
                const data = await res.json();
                if (data.status !== 'ok') throw new Error(data.message);

                currentAnalysis = data;
                emptyState.style.display = 'none';
                analysisContent.style.display = 'flex';

                // Update UI metadata
                document.getElementById('trackTitle').textContent = data.filename;
                document.getElementById('trackMeta').textContent = `${data.sr.toLocaleString()} Hz | ${data.nyquist_khz.toFixed(1)} kHz Nyquist | ${data.duration_s.toFixed(1)}s sample | Analyzed in ${data.analysis_time}s`;
                
                const vBadge = document.getElementById('verdictBadge');
                vBadge.textContent = data.verdict;
                if (data.verdict.includes('FAKE') || data.verdict.includes('UPSAMPLED') || data.verdict.includes('ZERO-STUFFED')) {
                    vBadge.style.background = 'rgba(255, 23, 68, 0.15)';
                    vBadge.style.borderColor = 'rgba(255, 23, 68, 0.4)';
                    vBadge.style.color = '#ff1744';
                } else if (data.verdict.includes('NATIVE')) {
                    vBadge.style.background = 'rgba(0, 229, 255, 0.15)';
                    vBadge.style.borderColor = 'rgba(0, 229, 255, 0.4)';
                    vBadge.style.color = '#00e5ff';
                } else {
                    vBadge.style.background = '#21262d';
                    vBadge.style.borderColor = '#30363d';
                    vBadge.style.color = '#8b949e';
                }

                // Dynamic Range badge
                const drBadge = document.getElementById('drBadge');
                if (data.dr_score !== undefined && data.dr_score !== null) {
                    drBadge.style.display = 'inline-block';
                    drBadge.textContent = `DR${data.dr_score} (${data.dr_val} dB)`;
                    if (data.dr_score >= 14) {
                        drBadge.style.background = 'rgba(0, 229, 255, 0.15)';
                        drBadge.style.borderColor = 'rgba(0, 229, 255, 0.4)';
                        drBadge.style.color = '#00e5ff';
                    } else if (data.dr_score >= 10) {
                        drBadge.style.background = 'rgba(0, 230, 118, 0.15)';
                        drBadge.style.borderColor = 'rgba(0, 230, 118, 0.4)';
                        drBadge.style.color = '#00e676';
                    } else if (data.dr_score >= 7) {
                        drBadge.style.background = 'rgba(255, 214, 0, 0.15)';
                        drBadge.style.borderColor = 'rgba(255, 214, 0, 0.4)';
                        drBadge.style.color = '#ffd600';
                    } else {
                        drBadge.style.background = 'rgba(255, 23, 68, 0.15)';
                        drBadge.style.borderColor = 'rgba(255, 23, 68, 0.4)';
                        drBadge.style.color = '#ff1744';
                    }
                } else {
                    drBadge.style.display = 'none';
                }

                // Noise profile badge
                const nBadge = document.getElementById('noiseBadge');
                if (data.noise_profile) {
                    nBadge.style.display = 'inline-block';
                    nBadge.textContent = data.noise_profile;
                    if (data.noise_profile.includes('PSYCHOACOUSTIC')) {
                        nBadge.style.background = 'rgba(174, 234, 0, 0.15)';
                        nBadge.style.borderColor = 'rgba(174, 234, 0, 0.4)';
                        nBadge.style.color = '#aeea00';
                    } else if (data.noise_profile.includes('FLAT')) {
                        nBadge.style.background = 'rgba(88, 166, 255, 0.15)';
                        nBadge.style.borderColor = 'rgba(88, 166, 255, 0.4)';
                        nBadge.style.color = '#58a6ff';
                    } else if (data.noise_profile.includes('DSD') || data.noise_profile.includes('HIGH')) {
                        nBadge.style.background = 'rgba(255, 140, 0, 0.15)';
                        nBadge.style.borderColor = 'rgba(255, 140, 0, 0.4)';
                        nBadge.style.color = '#ff9800';
                    } else {
                        nBadge.style.background = '#21262d';
                        nBadge.style.borderColor = '#30363d';
                        nBadge.style.color = '#8b949e';
                    }
                } else {
                    nBadge.style.display = 'none';
                }

                // Audio stream player
                const player = document.getElementById('audioPlayer');
                player.src = `/api/stream?path=${encodeURIComponent(file.path)}`;

                // Report text
                document.getElementById('reportText').textContent = data.report_text;

                // Initialize Canvas Plots
                initSpectrogram(data);
                initSpectrumCurve(data);

            } catch (err) {
                emptyState.innerHTML = `<p style="color: var(--accent-red);">Analysis Error: ${err.message}</p>`;
            }
        }


        // ==========================================
        // SPECTROGRAM CANVAS WITH ZOOM & PAN
        // ==========================================
        let specTMin = 0, specTMax = 60, specFMin = 0, specFMax = 88.2;
        let specImg = new Image(), rawLookup = null, lookupW = 0, lookupH = 0;
        let specIsDragging = false, specDragStartX = 0, specDragStartY = 0;
        let specInitTMin = 0, specInitTMax = 0, specInitFMin = 0, specInitFMax = 0;
        let specMouseX = -1, specMouseY = -1;

        const specCanvas = document.getElementById('spectrogramCanvas');
        const sCtx = specCanvas.getContext('2d');
        const specHudTime = document.getElementById('specHudTime');
        const specHudFreq = document.getElementById('specHudFreq');
        const specHudDb = document.getElementById('specHudDb');

        function initSpectrogram(data) {
            specTMin = 0.0; specTMax = data.duration_s;
            specFMin = 0.0; specFMax = data.nyquist_khz;
            lookupW = data.lookup_w; lookupH = data.lookup_h;
            rawLookup = Uint8Array.from(atob(data.lookup_base64), c => c.charCodeAt(0));

            specImg = new Image();
            specImg.src = "data:image/webp;base64," + data.webp_base64;
            specImg.onload = () => resizeSpecCanvas();
        }

        function getSpectrogramDb(t, fKhz) {
            if (!rawLookup || !currentAnalysis) return -165.0;
            const x = Math.max(0, Math.min(lookupW - 1, Math.floor((t / currentAnalysis.duration_s) * lookupW)));
            const y = Math.max(0, Math.min(lookupH - 1, Math.floor((1.0 - (fKhz / currentAnalysis.nyquist_khz)) * lookupH)));
            const u8 = rawLookup[y * lookupW + x];
            return (u8 / 255.0) * 165.0 - 165.0;
        }

        function resetSpecZoom() {
            if (!currentAnalysis) return;
            specTMin = 0.0; specTMax = currentAnalysis.duration_s;
            specFMin = 0.0; specFMax = currentAnalysis.nyquist_khz;
            drawSpectrogram(specCanvas.getBoundingClientRect().width, specCanvas.getBoundingClientRect().height);
        }

        function setSpecPreset(type) {
            if (!currentAnalysis) return;
            if (type === 'audible') {
                specFMin = 0.0; specFMax = Math.min(20.0, currentAnalysis.nyquist_khz);
            } else if (type === 'ultrasonic') {
                specFMin = Math.min(20.0, currentAnalysis.nyquist_khz); specFMax = currentAnalysis.nyquist_khz;
            }
            drawSpectrogram(specCanvas.getBoundingClientRect().width, specCanvas.getBoundingClientRect().height);
        }

        function zoomSpec(factor, centerTRatio = 0.5, centerFRatio = 0.5) {
            if (!currentAnalysis) return;
            const curTW = specTMax - specTMin;
            const newTW = Math.max(0.5, Math.min(currentAnalysis.duration_s, curTW / factor));
            const centerT = specTMin + curTW * centerTRatio;
            specTMin = Math.max(0, centerT - newTW * centerTRatio);
            specTMax = Math.min(currentAnalysis.duration_s, specTMin + newTW);
            if (specTMax - specTMin < newTW) specTMin = Math.max(0, specTMax - newTW);

            const curFW = specFMax - specFMin;
            const newFW = Math.max(1.0, Math.min(currentAnalysis.nyquist_khz, curFW / factor));
            const centerF = specFMin + curFW * centerFRatio;
            specFMin = Math.max(0, centerF - newFW * centerFRatio);
            specFMax = Math.min(currentAnalysis.nyquist_khz, specFMin + newFW);
            if (specFMax - specFMin < newFW) specFMin = Math.max(0, specFMax - newFW);

            drawSpectrogram(specCanvas.getBoundingClientRect().width, specCanvas.getBoundingClientRect().height);
        }

        function resizeSpecCanvas() {
            const rect = specCanvas.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            specCanvas.width = rect.width * dpr;
            specCanvas.height = rect.height * dpr;
            sCtx.scale(dpr, dpr);
            drawSpectrogram(rect.width, rect.height);
        }

        function drawSpectrogram(w, h) {
            if (!currentAnalysis) return;
            sCtx.clearRect(0, 0, w, h);
            const padL = 55, padR = 70, padT = 15, padB = 35;
            const plotW = w - padL - padR;
            const plotH = h - padT - padB;

            if (plotW <= 0 || plotH <= 0) return;

            // Draw Heatmap Bitmap with Zoom/Sub-Rect
            if (specImg.complete && specImg.naturalWidth > 0) {
                const sx = (specTMin / currentAnalysis.duration_s) * specImg.naturalWidth;
                const sw = ((specTMax - specTMin) / currentAnalysis.duration_s) * specImg.naturalWidth;
                const sy = (1.0 - (specFMax / currentAnalysis.nyquist_khz)) * specImg.naturalHeight;
                const sh = ((specFMax - specFMin) / currentAnalysis.nyquist_khz) * specImg.naturalHeight;

                sCtx.save();
                sCtx.beginPath();
                sCtx.rect(padL, padT, plotW, plotH);
                sCtx.clip();
                sCtx.drawImage(specImg, sx, sy, sw, sh, padL, padT, plotW, plotH);
                sCtx.restore();
            }

            sCtx.strokeStyle = "#30363d";
            sCtx.lineWidth = 1;
            sCtx.strokeRect(padL, padT, plotW, plotH);

            // Y-Axis Ticks (Frequency kHz)
            sCtx.fillStyle = "#8b949e";
            sCtx.font = "10px -apple-system, sans-serif";
            sCtx.textAlign = "right";
            sCtx.textBaseline = "middle";

            const fRange = specFMax - specFMin;
            const fStep = fRange > 40 ? 20 : (fRange > 20 ? 10 : (fRange > 8 ? 5 : 2));
            const firstF = Math.ceil(specFMin / fStep) * fStep;

            for (let f = firstF; f <= specFMax; f += fStep) {
                const y = padT + (1.0 - (f - specFMin) / fRange) * plotH;
                sCtx.strokeStyle = "#1f242c";
                sCtx.beginPath();
                sCtx.moveTo(padL - 4, y);
                sCtx.lineTo(padL, y);
                sCtx.stroke();
                sCtx.fillText(f.toFixed(fStep < 1 ? 1 : 0) + "k", padL - 6, y);
            }

            // X-Axis Ticks (Time seconds)
            sCtx.textAlign = "center";
            sCtx.textBaseline = "top";
            const tRange = specTMax - specTMin;
            const tStep = tRange > 40 ? 10 : (tRange > 15 ? 5 : (tRange > 5 ? 2 : 1));
            const firstT = Math.ceil(specTMin / tStep) * tStep;

            for (let t = firstT; t <= specTMax; t += tStep) {
                const x = padL + ((t - specTMin) / tRange) * plotW;
                sCtx.strokeStyle = "#1f242c";
                sCtx.beginPath();
                sCtx.moveTo(x, padT + plotH);
                sCtx.lineTo(x, padT + plotH + 4);
                sCtx.stroke();
                sCtx.fillText(t.toFixed(tStep < 1 ? 1 : 0) + "s", x, padT + plotH + 6);
            }

            // Colorbar Scale
            const barX = padL + plotW + 12;
            const barW = 10;
            const barH = plotH;
            const grad = sCtx.createLinearGradient(0, padT, 0, padT + barH);
            grad.addColorStop(0.00, "#fcffa4");
            grad.addColorStop(0.25, "#f98e09");
            grad.addColorStop(0.50, "#bc3754");
            grad.addColorStop(0.75, "#57106e");
            grad.addColorStop(1.00, "#000004");

            sCtx.fillStyle = grad;
            sCtx.fillRect(barX, padT, barW, barH);
            sCtx.strokeStyle = "#30363d";
            sCtx.strokeRect(barX, padT, barW, barH);

            sCtx.textAlign = "left";
            sCtx.textBaseline = "middle";
            sCtx.fillStyle = "#8b949e";
            sCtx.font = "9px -apple-system, sans-serif";
            const dbTicks = [0, -40, -80, -120, -165];
            for (let d of dbTicks) {
                const y = padT + (d / -165.0) * barH;
                sCtx.fillText(d === 0 ? "0" : d + "", barX + barW + 4, y);
            }

            // Hover Crosshair
            if (specMouseX >= padL && specMouseX <= padL + plotW && specMouseY >= padT && specMouseY <= padT + plotH) {
                sCtx.strokeStyle = "rgba(255, 255, 255, 0.4)";
                sCtx.setLineDash([2, 2]);
                sCtx.beginPath();
                sCtx.moveTo(specMouseX, padT);
                sCtx.lineTo(specMouseX, padT + plotH);
                sCtx.moveTo(padL, specMouseY);
                sCtx.lineTo(padL + plotW, specMouseY);
                sCtx.stroke();
                sCtx.setLineDash([]);

                const curT = specTMin + ((specMouseX - padL) / plotW) * tRange;
                const curF = specFMin + (1.0 - (specMouseY - padT) / plotH) * fRange;
                const curDb = getSpectrogramDb(curT, curF);

                specHudTime.textContent = curT.toFixed(2) + " s";
                specHudFreq.textContent = curF.toFixed(2) + " kHz";
                specHudDb.textContent = curDb.toFixed(1) + " dBFS";
            }
        }

        specCanvas.addEventListener('mousedown', (e) => {
            specIsDragging = true;
            specDragStartX = e.clientX; specDragStartY = e.clientY;
            specInitTMin = specTMin; specInitTMax = specTMax;
            specInitFMin = specFMin; specInitFMax = specFMax;
        });

        window.addEventListener('mouseup', () => { specIsDragging = false; });

        specCanvas.addEventListener('mousemove', (e) => {
            const rect = specCanvas.getBoundingClientRect();
            specMouseX = e.clientX - rect.left;
            specMouseY = e.clientY - rect.top;

            if (specIsDragging && currentAnalysis) {
                const padL = 55, padR = 70, padT = 15, padB = 35;
                const plotW = rect.width - padL - padR;
                const plotH = rect.height - padT - padB;
                const dx = e.clientX - specDragStartX;
                const dy = e.clientY - specDragStartY;

                const curTW = specInitTMax - specInitTMin;
                const curFW = specInitFMax - specInitFMin;

                const dt = -(dx / plotW) * curTW;
                const df = (dy / plotH) * curFW;

                specTMin = Math.max(0, Math.min(currentAnalysis.duration_s - curTW, specInitTMin + dt));
                specTMax = specTMin + curTW;

                specFMin = Math.max(0, Math.min(currentAnalysis.nyquist_khz - curFW, specInitFMin + df));
                specFMax = specFMin + curFW;
            }

            drawSpectrogram(rect.width, rect.height);
        });

        specCanvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const rect = specCanvas.getBoundingClientRect();
            const padL = 55, padR = 70, padT = 15, padB = 35;
            const plotW = rect.width - padL - padR;
            const plotH = rect.height - padT - padB;
            const mX = e.clientX - rect.left;
            const mY = e.clientY - rect.top;

            if (mX >= padL && mX <= padL + plotW && mY >= padT && mY <= padT + plotH) {
                const tRatio = (mX - padL) / plotW;
                const fRatio = 1.0 - (mY - padT) / plotH;
                const factor = e.deltaY < 0 ? 1.25 : (1.0 / 1.25);
                zoomSpec(factor, tRatio, fRatio);
            }
        }, { passive: false });

        specCanvas.addEventListener('mouseleave', () => {
            specMouseX = -1; specMouseY = -1;
            const rect = specCanvas.getBoundingClientRect();
            drawSpectrogram(rect.width, rect.height);
            specHudTime.textContent = "-- s";
            specHudFreq.textContent = "-- kHz";
            specHudDb.textContent = "-- dBFS";
        });


        // ==========================================
        // SPECTRUM CURVE CANVAS WITH ZOOM & PAN
        // ==========================================
        let curveFMin = 0.0, curveFMax = 88.2;
        let curveDbMin = -175.0, curveDbMax = 0.0;
        let curveIsDragging = false, curveDragStartX = 0, curveDragStartY = 0;
        let curveInitFMin = 0, curveInitFMax = 0, curveInitDbMin = 0, curveInitDbMax = 0;

        const curveCanvas = document.getElementById('spectrumCanvas');
        const cCtx = curveCanvas.getContext('2d');
        const hudFreq = document.getElementById('hudFreq');
        const hudPeak = document.getElementById('hudPeak');
        const hudRMS = document.getElementById('hudRMS');
        let curveMouseX = -1;

        function initSpectrumCurve(data) {
            curveFMin = 0.0; curveFMax = data.nyquist_khz;
            curveDbMin = -175.0; curveDbMax = 0.0;
            resizeCurveCanvas();
        }

        function resetCurveZoom() {
            if (!currentAnalysis) return;
            curveFMin = 0.0; curveFMax = currentAnalysis.nyquist_khz;
            curveDbMin = -175.0; curveDbMax = 0.0;
            drawCurve(curveCanvas.getBoundingClientRect().width, curveCanvas.getBoundingClientRect().height);
        }

        function setCurvePreset(type) {
            if (!currentAnalysis) return;
            if (type === 'audible') {
                curveFMin = 0.0; curveFMax = Math.min(20.0, currentAnalysis.nyquist_khz);
                curveDbMin = -120.0; curveDbMax = 0.0;
            } else if (type === 'cutoff') {
                curveFMin = 15.0; curveFMax = Math.min(25.0, currentAnalysis.nyquist_khz);
                curveDbMin = -175.0; curveDbMax = -40.0;
            } else if (type === 'ultrasonic') {
                curveFMin = Math.min(20.0, currentAnalysis.nyquist_khz); curveFMax = currentAnalysis.nyquist_khz;
                curveDbMin = -175.0; curveDbMax = -120.0;
            }
            drawCurve(curveCanvas.getBoundingClientRect().width, curveCanvas.getBoundingClientRect().height);
        }

        function zoomCurve(factor, centerFRatio = 0.5, centerDbRatio = 0.5) {
            if (!currentAnalysis) return;
            const curFW = curveFMax - curveFMin;
            const newFW = Math.max(1.0, Math.min(currentAnalysis.nyquist_khz, curFW / factor));
            const centerF = curveFMin + curFW * centerFRatio;
            curveFMin = Math.max(0, centerF - newFW * centerFRatio);
            curveFMax = Math.min(currentAnalysis.nyquist_khz, curveFMin + newFW);
            if (curveFMax - curveFMin < newFW) curveFMin = Math.max(0, curveFMax - newFW);

            const curDbW = curveDbMax - curveDbMin;
            const newDbW = Math.max(10.0, Math.min(175.0, curDbW / factor));
            const centerDb = curveDbMin + curDbW * centerDbRatio;
            curveDbMin = Math.max(-175.0, centerDb - newDbW * centerDbRatio);
            curveDbMax = Math.min(0.0, curveDbMin + newDbW);
            if (curveDbMax - curveDbMin < newDbW) curveDbMin = Math.max(-175.0, curveDbMax - newDbW);

            drawCurve(curveCanvas.getBoundingClientRect().width, curveCanvas.getBoundingClientRect().height);
        }

        function resizeCurveCanvas() {
            const rect = curveCanvas.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            curveCanvas.width = rect.width * dpr;
            curveCanvas.height = rect.height * dpr;
            cCtx.scale(dpr, dpr);
            drawCurve(rect.width, rect.height);
        }

        function freqToX(fKhz, width, padL, padR) {
            const plotW = width - padL - padR;
            return padL + ((fKhz - curveFMin) / (curveFMax - curveFMin)) * plotW;
        }

        function dbToY(db, height, padT, padB) {
            const plotH = height - padT - padB;
            const clamped = Math.max(curveDbMin, Math.min(curveDbMax, db));
            return padT + (1.0 - (clamped - curveDbMin) / (curveDbMax - curveDbMin)) * plotH;
        }

        function drawCurve(w, h) {
            if (!currentAnalysis) return;
            cCtx.clearRect(0, 0, w, h);
            const padL = 55, padR = 25, padT = 15, padB = 35;
            const plotW = w - padL - padR;
            const plotH = h - padT - padB;

            if (plotW <= 0 || plotH <= 0) return;

            cCtx.fillStyle = "#11141a";
            cCtx.fillRect(padL, padT, plotW, plotH);

            // Horizontal Grid Lines
            const dbRange = curveDbMax - curveDbMin;
            const dbStep = dbRange > 100 ? 20 : (dbRange > 40 ? 10 : 5);
            const firstDb = Math.ceil(curveDbMin / dbStep) * dbStep;

            cCtx.strokeStyle = "#1f242c";
            cCtx.lineWidth = 1;
            cCtx.fillStyle = "#6e7681";
            cCtx.font = "10px -apple-system, sans-serif";
            cCtx.textAlign = "right";
            cCtx.textBaseline = "middle";

            for (let db = firstDb; db <= curveDbMax; db += dbStep) {
                const y = dbToY(db, h, padT, padB);
                cCtx.beginPath();
                cCtx.moveTo(padL, y);
                cCtx.lineTo(padL + plotW, y);
                cCtx.stroke();
                cCtx.fillText(db.toFixed(0) + " dB", padL - 6, y);
            }

            // Vertical Frequency Grid Lines
            cCtx.textAlign = "center";
            cCtx.textBaseline = "top";
            const fRange = curveFMax - curveFMin;
            const fStep = fRange > 40 ? 20 : (fRange > 20 ? 10 : (fRange > 8 ? 5 : (fRange > 3 ? 1 : 0.5)));
            const firstF = Math.ceil(curveFMin / fStep) * fStep;

            for (let f = firstF; f <= curveFMax; f += fStep) {
                const x = freqToX(f, w, padL, padR);
                cCtx.beginPath();
                cCtx.moveTo(x, padT);
                cCtx.lineTo(x, padT + plotH);
                cCtx.stroke();
                cCtx.fillText(f.toFixed(fStep < 1 ? 1 : 0) + "k", x, padT + plotH + 6);
            }

            // Clip curves inside plot
            cCtx.save();
            cCtx.beginPath();
            cCtx.rect(padL, padT, plotW, plotH);
            cCtx.clip();

            // Reference Marker: 20 kHz
            if (curveFMin <= 20.0 && curveFMax >= 20.0) {
                const x20 = freqToX(20.0, w, padL, padR);
                cCtx.strokeStyle = "#ff1744";
                cCtx.setLineDash([3, 3]);
                cCtx.beginPath();
                cCtx.moveTo(x20, padT);
                cCtx.lineTo(x20, padT + plotH);
                cCtx.stroke();
            }

            // Reference Marker: 22.05 kHz
            if (curveFMin <= 22.05 && curveFMax >= 22.05) {
                const x22 = freqToX(22.05, w, padL, padR);
                cCtx.strokeStyle = "#ffea00";
                cCtx.setLineDash([4, 4]);
                cCtx.beginPath();
                cCtx.moveTo(x22, padT);
                cCtx.lineTo(x22, padT + plotH);
                cCtx.stroke();
            }
            cCtx.setLineDash([]);

            // Draw RMS Curve (Magenta)
            const fArr = currentAnalysis.curve_freqs_khz;
            const rArr = currentAnalysis.curve_rms;
            const pArr = currentAnalysis.curve_peaks;

            cCtx.strokeStyle = "#ff007f";
            cCtx.lineWidth = 1.2;
            cCtx.beginPath();
            let firstPoint = true;
            for (let i = 0; i < fArr.length; i++) {
                if (fArr[i] >= curveFMin - 1.0 && fArr[i] <= curveFMax + 1.0) {
                    const x = freqToX(fArr[i], w, padL, padR);
                    const y = dbToY(rArr[i], h, padT, padB);
                    if (firstPoint) { cCtx.moveTo(x, y); firstPoint = false; }
                    else cCtx.lineTo(x, y);
                }
            }
            cCtx.stroke();

            // Draw Peak Curve (Cyan)
            cCtx.strokeStyle = "#00e5ff";
            cCtx.lineWidth = 1.2;
            cCtx.beginPath();
            firstPoint = true;
            for (let i = 0; i < fArr.length; i++) {
                if (fArr[i] >= curveFMin - 1.0 && fArr[i] <= curveFMax + 1.0) {
                    const x = freqToX(fArr[i], w, padL, padR);
                    const y = dbToY(pArr[i], h, padT, padB);
                    if (firstPoint) { cCtx.moveTo(x, y); firstPoint = false; }
                    else cCtx.lineTo(x, y);
                }
            }
            cCtx.stroke();
            cCtx.restore();

            // Hover Crosshair
            if (curveMouseX >= padL && curveMouseX <= padL + plotW) {
                const ratio = (curveMouseX - padL) / plotW;
                const targetFreq = curveFMin + ratio * (curveFMax - curveFMin);
                let closestIdx = 0, minDiff = Infinity;
                for (let i = 0; i < fArr.length; i++) {
                    const d = Math.abs(fArr[i] - targetFreq);
                    if (d < minDiff) { minDiff = d; closestIdx = i; }
                }

                const curX = freqToX(fArr[closestIdx], w, padL, padR);
                const curYPeak = dbToY(pArr[closestIdx], h, padT, padB);
                const curYRMS = dbToY(rArr[closestIdx], h, padT, padB);

                cCtx.strokeStyle = "rgba(255, 255, 255, 0.4)";
                cCtx.setLineDash([2, 2]);
                cCtx.beginPath();
                cCtx.moveTo(curX, padT);
                cCtx.lineTo(curX, padT + plotH);
                cCtx.stroke();
                cCtx.setLineDash([]);

                cCtx.fillStyle = "#00e5ff";
                cCtx.beginPath();
                cCtx.arc(curX, curYPeak, 3.5, 0, Math.PI * 2);
                cCtx.fill();

                cCtx.fillStyle = "#ff007f";
                cCtx.beginPath();
                cCtx.arc(curX, curYRMS, 3.5, 0, Math.PI * 2);
                cCtx.fill();

                hudFreq.textContent = `${fArr[closestIdx].toFixed(2)} kHz (${(fArr[closestIdx]*1000).toFixed(0)} Hz)`;
                hudPeak.textContent = `${pArr[closestIdx].toFixed(1)} dBFS`;
                hudRMS.textContent = `${rArr[closestIdx].toFixed(1)} dBFS`;
            }
        }

        curveCanvas.addEventListener('mousedown', (e) => {
            curveIsDragging = true;
            curveDragStartX = e.clientX; curveDragStartY = e.clientY;
            curveInitFMin = curveFMin; curveInitFMax = curveFMax;
            curveInitDbMin = curveDbMin; curveInitDbMax = curveDbMax;
        });

        window.addEventListener('mouseup', () => { curveIsDragging = false; });

        curveCanvas.addEventListener('mousemove', (e) => {
            const rect = curveCanvas.getBoundingClientRect();
            curveMouseX = e.clientX - rect.left;

            if (curveIsDragging && currentAnalysis) {
                const padL = 55, padR = 25, padT = 15, padB = 35;
                const plotW = rect.width - padL - padR;
                const plotH = rect.height - padT - padB;
                const dx = e.clientX - curveDragStartX;
                const dy = e.clientY - curveDragStartY;

                const curFW = curveInitFMax - curveInitFMin;
                const curDbW = curveInitDbMax - curveInitDbMin;

                const df = -(dx / plotW) * curFW;
                const dDb = (dy / plotH) * curDbW;

                curveFMin = Math.max(0, Math.min(currentAnalysis.nyquist_khz - curFW, curveInitFMin + df));
                curveFMax = curveFMin + curFW;

                curveDbMin = Math.max(-175.0, Math.min(0.0 - curDbW, curveInitDbMin + dDb));
                curveDbMax = curveDbMin + curDbW;
            }

            drawCurve(rect.width, rect.height);
        });

        curveCanvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const rect = curveCanvas.getBoundingClientRect();
            const padL = 55, padR = 25, padT = 15, padB = 35;
            const plotW = rect.width - padL - padR;
            const plotH = rect.height - padT - padB;
            const mX = e.clientX - rect.left;
            const mY = e.clientY - rect.top;

            if (mX >= padL && mX <= padL + plotW && mY >= padT && mY <= padT + plotH) {
                const fRatio = (mX - padL) / plotW;
                const dbRatio = 1.0 - (mY - padT) / plotH;
                const factor = e.deltaY < 0 ? 1.25 : (1.0 / 1.25);
                zoomCurve(factor, fRatio, dbRatio);
            }
        }, { passive: false });

        curveCanvas.addEventListener('mouseleave', () => {
            curveMouseX = -1;
            const rect = curveCanvas.getBoundingClientRect();
            drawCurve(rect.width, rect.height);
            hudFreq.textContent = "-- kHz";
            hudPeak.textContent = "-- dBFS";
            hudRMS.textContent = "-- dBFS";
        });

        window.addEventListener('resize', () => {
            resizeSpecCanvas();
            resizeCurveCanvas();
        });

        function copyReport() {
            if (!currentAnalysis) return;
            navigator.clipboard.writeText(currentAnalysis.report_text).then(() => {
                alert('Forensic Report copied to clipboard!');
            });
        }

        // Initialize on load
        loadDirectory(currentPath);
    </script>
</body>
</html>"""


class ForensicWebHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Clean logging
        sys.stderr.write(f"[{self.log_date_time_string()}] {self.command} {self.path}\n")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif path == "/api/browse":
            target = params.get("path", [""])[0]
            data = get_directory_contents(target)
            if data is None:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Directory not found"}).encode("utf-8"))
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode("utf-8"))

        elif path == "/api/analyze":
            target = params.get("path", [""])[0]
            result = analyze_file_on_demand(target)
            self.send_response(200 if result.get("status") == "ok" else 400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

        elif path == "/api/stream":
            target = params.get("path", [""])[0]
            if not os.path.exists(target) or not os.path.isfile(target):
                self.send_response(404)
                self.end_headers()
                return

            file_size = os.path.getsize(target)
            range_header = self.headers.get('Range', None)

            mime = "audio/flac"
            if target.lower().endswith(".wav"): mime = "audio/wav"
            elif target.lower().endswith(".mp3"): mime = "audio/mpeg"

            if range_header:
                # Byte-range request for seamless audio streaming/seeking
                range_match = range_header.strip().lower()
                if range_match.startswith("bytes="):
                    parts = range_match[6:].split("-")
                    start = int(parts[0]) if parts[0] else 0
                    end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
                    end = min(end, file_size - 1)
                    length = end - start + 1

                    self.send_response(206)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                    self.send_header("Content-Length", str(length))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()

                    with open(target, 'rb') as f:
                        f.seek(start)
                        remaining = length
                        while remaining > 0:
                            chunk_size = min(65536, remaining)
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            try:
                                self.wfile.write(chunk)
                                remaining -= len(chunk)
                            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                                break
                    return

            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(target, 'rb') as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        break
        else:
            self.send_response(404)
            self.end_headers()


def main():
    global INITIAL_ROOT
    parser = argparse.ArgumentParser(description="Hi-Res Audio Forensic Explorer Web Server")
    parser.add_argument("--root", default=os.getcwd(), help="Initial directory to browse (default: current working directory)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP server port (default: 8765)")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP server host (default: 0.0.0.0)")
    args = parser.parse_args()

    INITIAL_ROOT = os.path.abspath(os.path.expanduser(args.root))
    if not os.path.exists(INITIAL_ROOT):
        INITIAL_ROOT = os.getcwd()

    server_address = (args.host, args.port)
    httpd = ThreadingHTTPServer(server_address, ForensicWebHandler)
    print(f"\n=======================================================")
    print(f"🔬 HI-RES AUDIO FORENSIC EXPLORER (WEB APPLICATION)")
    print(f"=======================================================")
    print(f"Initial Root: {INITIAL_ROOT}")
    print(f"Server URL  : http://localhost:{args.port}")
    print(f"Network URL : http://{args.host}:{args.port}")
    print(f"Mode        : Multi-Threaded Concurrent I/O")
    print(f"Precision   : 64-bit Double Precision (Strict float64)")
    print(f"Status      : Live & Ready for On-Demand Analysis")
    print(f"=======================================================\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == "__main__":
    main()
