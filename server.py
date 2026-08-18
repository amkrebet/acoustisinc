#!/usr/bin/env python3
"""
================================================================================
HI-RES AUDIO FORENSIC EXPLORER (WEB APPLICATION SERVER)
================================================================================
A real-time on-demand forensic laboratory web application for high-resolution
audio libraries.
- GPU-Accelerated 64-bit double precision DSP with automatic CPU fallback.
- Whole-folder background pre-processing with progressive real-time UI badge streaming.
- Immediate badge population upon on-demand track selection.
- Automatic cache invalidation on folder entry to dynamically recompute graphs and recommendations.
- Action recommendation engine suggesting optimal least-damaging DSP workflows.
- Dual hypothesis evaluation (Primary recommendation + Alternative possibility).
- Interactive spectrogram with WebGL/Canvas zoom, pan, crosshair reticle, and exact dB level cursor HUD.
- Interactive spectrum curves with Peak/RMS traces, crosshair reticle, dynamic HUD tracking, and context-aware Nyquist markers.
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
import threading
import queue
import html
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
import soundfile as sf
import numpy as np

# Core DSP functions
from analyser import analyze_audio_forensics, encode_spectrogram_and_lookup
from provenance_engine import load_audio_resilient, probe_audio_info_resilient
from gpu_analyser import gpu_engine, analyze_audio_forensics_accelerated


# Default root directory for initial browsing
INITIAL_ROOT = os.getcwd()
ACTIVE_RULES_PATH = None

# Global in-memory caches
ANALYSIS_CACHE = {}
BADGE_CACHE = {}
CACHE_LOCK = threading.Lock()


def format_bytes(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def format_seconds(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:d}:{s:02d}"


def clear_folder_cache(folder_path):
    """
    Clears analysis and badge caches for all files within a folder.
    Ensures that entering a folder always recomputes fresh graphs and recommendations.
    """
    if not folder_path:
        return
    folder_path = os.path.abspath(folder_path)
    with CACHE_LOCK:
        to_del_analysis = [k for k in ANALYSIS_CACHE if k.startswith(folder_path)]
        for k in to_del_analysis:
            del ANALYSIS_CACHE[k]
        to_del_badges = [k for k in BADGE_CACHE if k.startswith(folder_path)]
        for k in to_del_badges:
            del BADGE_CACHE[k]


def extract_badge_data(res):
    prov = res.get("provenance", {})
    primary = prov.get("primary", {})
    bitdepth = prov.get("bitdepth", {})
    rec = prov.get("recommendation", {})
    alt = prov.get("alternative", {})
    
    verdict = primary.get("label") or res.get("verdict", "Native Master")
    badge_class = primary.get("badge_class", "badge-provenance-native")
    
    # Shorten verdict for compact badge display
    short_verdict = verdict
    if "Upsampled from" in verdict:
        short_verdict = verdict.replace("Master", "").strip()
    elif "MQA Studio" in verdict:
        short_verdict = "MQA Studio"
    elif "MQA Authenticated" in verdict:
        short_verdict = "MQA Master"
    elif "Native" in verdict:
        short_verdict = verdict.replace("Master", "").strip()
    elif "Leaky" in verdict:
        short_verdict = "Leaky SRC"
    elif "Fake Hi-Res" in verdict:
        short_verdict = "Fake Hi-Res"

    return {
        "filepath": res.get("filepath"),
        "filename": res.get("filename"),
        "sr": res.get("sr"),
        "sr_str": f"{res.get('sr', 44100)/1000:.1f}k",
        "dr_score": res.get("dr_score", 0),
        "crest_factor_db": res.get("crest_factor_db", 0.0),
        "verdict": verdict,
        "short_verdict": short_verdict,
        "badge_class": badge_class,
        "confidence": primary.get("confidence", "High"),
        "effective_bits": bitdepth.get("effective_bits", 16),
        "container_bits": bitdepth.get("container_bits", 16),
        "is_zero_padded": bitdepth.get("is_zero_padded", False),
        "trailing_zero_bits": bitdepth.get("trailing_zero_bits", 0),
        "is_mqa": bool("MQA" in verdict or (prov.get("mqa") or {}).get("is_mqa")),
        "recommendation": rec,
        "alternative": alt
    }


class FolderScanManager:
    """
    Manages asynchronous background folder analysis and real-time SSE badge streaming.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.active_scan_dir = None
        self.listeners = {}  # dir -> list of queue.Queue
        self.active_thread = None

    def add_listener(self, folder_dir, fresh=True):
        if fresh:
            clear_folder_cache(folder_dir)

        q = queue.Queue()
        with self.lock:
            if folder_dir not in self.listeners:
                self.listeners[folder_dir] = []
            self.listeners[folder_dir].append(q)

            # Re-spawn scan worker for fresh re-analysis upon entering folder
            if self.active_thread is None or not self.active_thread.is_alive() or self.active_scan_dir != folder_dir or fresh:
                self.active_scan_dir = folder_dir
                t = threading.Thread(target=self._scan_worker, args=(folder_dir,), daemon=True)
                self.active_thread = t
                t.start()
        return q

    def remove_listener(self, folder_dir, q):
        with self.lock:
            if folder_dir in self.listeners and q in self.listeners[folder_dir]:
                self.listeners[folder_dir].remove(q)

    def _broadcast(self, folder_dir, event_type, data):
        with self.lock:
            queues = list(self.listeners.get(folder_dir, []))
        for q in queues:
            try:
                q.put_nowait((event_type, data))
            except Exception:
                pass

    def _scan_worker(self, folder_dir):
        if not os.path.exists(folder_dir) or not os.path.isdir(folder_dir):
            return

        try:
            files = [
                os.path.join(folder_dir, f) for f in sorted(os.listdir(folder_dir))
                if f.lower().endswith(('.flac', '.wav', '.aiff', '.m4a', '.mp3'))
                and not f.endswith('.WIP') and not f.startswith('.')
            ]
        except Exception:
            files = []

        # Progressively scan tracks freshly using latest rules & DSP engine
        for f in files:
            with self.lock:
                if self.active_scan_dir != folder_dir:
                    break  # User moved to another folder

            try:
                res = analyze_file_on_demand(f, force_fresh=True)
                if res.get("status") == "ok":
                    with CACHE_LOCK:
                        ANALYSIS_CACHE[f] = res
                        badge_data = extract_badge_data(res)
                        BADGE_CACHE[f] = badge_data
                    self._broadcast(folder_dir, "track_badge", badge_data)
            except Exception:
                pass

        self._broadcast(folder_dir, "scan_complete", {"folder": folder_dir, "count": len(files)})


folder_scan_mgr = FolderScanManager()


def get_directory_contents(target_path, fresh=True):
    """
    Lists subdirectories and audio files in target_path with rich metadata.
    """
    if not target_path or not str(target_path).strip():
        target_path = INITIAL_ROOT
    if str(target_path).startswith("~"):
        target_path = os.path.expanduser(target_path)
    target_path = os.path.abspath(target_path)

    if fresh:
        clear_folder_cache(target_path)

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
                    info = probe_audio_info_resilient(full_path)
                    file_meta["samplerate"] = info['samplerate']
                    file_meta["channels"] = info['channels']
                    file_meta["duration_s"] = info['duration']
                    file_meta["duration_str"] = format_seconds(info['duration'])
                    file_meta["is_hires"] = info['samplerate'] > 48000
                    file_meta["subtype"] = info['subtype']
                except Exception:
                    pass

                with CACHE_LOCK:
                    if full_path in BADGE_CACHE:
                        file_meta["badge"] = BADGE_CACHE[full_path]

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
        "files": files,
        "gpu_enabled": gpu_engine.enabled,
        "gpu_device": gpu_engine.device_name
    }


def analyze_file_on_demand(filepath, rules_path=None, force_fresh=True):
    """
    Performs on-demand DSP analysis with strict 64-bit double precision.
    When force_fresh is True, recomputes spectrograms, curves, and recommendations dynamically.
    """
    if not os.path.exists(filepath):
        return {"status": "error", "message": "File not found"}

    if not force_fresh:
        with CACHE_LOCK:
            if filepath in ANALYSIS_CACHE:
                return ANALYSIS_CACHE[filepath]

    try:
        t0 = time.time()
        # Fast direct decoding of 25 seconds of audio for instant responsiveness
        try:
            data, sr = sf.read(filepath, frames=int(192000 * 25), dtype='float64', always_2d=True)
        except Exception:
            data, sr = load_audio_resilient(filepath, dtype='float64', frames=int(192000 * 25))

        if data.ndim > 1:
            data = np.mean(data, axis=1)

        taper_len = min(int(sr * 0.05), len(data) // 10)
        if taper_len > 0:
            taper = np.sin(np.linspace(0, np.pi/2, taper_len))**2
            data[:taper_len] *= taper
            data[-taper_len:] *= taper[::-1]

        # Automatically uses GPU if available, multi-threaded CPU fallback otherwise
        spec_db, freqs, peak_dbfs, rms_dbfs, assessment_text, dr_metrics, provenance_info = analyze_audio_forensics(
            data, sr, rules_path=rules_path or ACTIVE_RULES_PATH, filepath=filepath
        )

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
        verdict = provenance_info.get("label", "ANALYZED")
        noise_profile = ""
        for line in assessment_text.splitlines():
            if line.startswith("NOISE PROFILE:"):
                noise_profile = line.replace("NOISE PROFILE:", "").strip().strip("[]")

        result = {
            "status": "ok",
            "filename": os.path.basename(filepath),
            "filepath": filepath,
            "sr": sr,
            "nyquist_khz": nyquist / 1000.0,
            "duration_s": duration_s,
            "verdict": verdict,
            "provenance": provenance_info,
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
            "analysis_time": round(time.time() - t0, 3),
            "gpu_enabled": gpu_engine.enabled,
            "gpu_device": gpu_engine.device_name
        }

        badge_data = extract_badge_data(result)
        result["badge"] = badge_data

        with CACHE_LOCK:
            ANALYSIS_CACHE[filepath] = result
            BADGE_CACHE[filepath] = badge_data
        return result
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
        .gpu-badge {
            background: rgba(0, 229, 255, 0.15);
            border: 1px solid rgba(0, 229, 255, 0.45);
            color: var(--accent-cyan);
            font-size: 0.75rem;
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 4px;
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }
        .gpu-badge.cpu-mode {
            background: rgba(139, 148, 158, 0.15);
            border-color: rgba(139, 148, 158, 0.4);
            color: var(--text-muted);
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
            width: 440px;
            min-width: 360px;
            background: var(--surface);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            height: 100%;
        }
        .nav-bar {
            padding: 12px;
            border-bottom: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            gap: 8px;
            background: #11141a;
            flex-shrink: 0;
        }
        .nav-top-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
        }
        .breadcrumbs {
            display: flex;
            align-items: center;
            gap: 4px;
            flex-wrap: wrap;
            font-size: 0.85rem;
            flex: 1;
            min-width: 0;
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
            flex: 1 1 0;
            min-height: 0;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 8px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .tree-list::-webkit-scrollbar {
            width: 6px;
        }
        .tree-list::-webkit-scrollbar-track {
            background: #0d1117;
        }
        .tree-list::-webkit-scrollbar-thumb {
            background: #30363d;
            border-radius: 3px;
        }
        .tree-list::-webkit-scrollbar-thumb:hover {
            background: #58a6ff;
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
            flex-shrink: 0;
            min-height: 38px;
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
            flex-shrink: 0;
            min-height: 52px;
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
        .file-badges {
            display: flex;
            align-items: center;
            gap: 4px;
            flex-wrap: wrap;
            flex-shrink: 0;
        }
        .badge-tag {
            font-size: 0.70rem;
            font-weight: 600;
            padding: 1px 6px;
            border-radius: 3px;
            white-space: nowrap;
            animation: badge-fade-in 0.25s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        @keyframes badge-fade-in {
            from { opacity: 0; transform: scale(0.92); }
            to { opacity: 1; transform: scale(1); }
        }
        .badge-tag.badge-dr {
            background: rgba(255, 234, 0, 0.12);
            border: 1px solid rgba(255, 234, 0, 0.4);
            color: var(--accent-yellow);
        }
        .badge-tag.badge-bits {
            background: rgba(88, 166, 255, 0.12);
            border: 1px solid rgba(88, 166, 255, 0.4);
            color: var(--accent-blue);
        }
        .badge-tag.badge-bits-padded {
            background: rgba(255, 112, 67, 0.18);
            border: 1px solid rgba(255, 112, 67, 0.45);
            color: #ff7043;
        }
        .badge-tag.badge-sr {
            background: rgba(0, 229, 255, 0.12);
            border: 1px solid rgba(0, 229, 255, 0.35);
            color: var(--accent-cyan);
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
        /* Provenance Banners & Badges */
        .provenance-tag {
            font-size: 0.82rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
            display: inline-block;
        }
        .badge-provenance-native {
            background: rgba(0, 229, 255, 0.12);
            border: 1px solid rgba(0, 229, 255, 0.4);
            color: var(--accent-cyan);
        }
        .badge-provenance-leaky {
            background: rgba(255, 171, 0, 0.12);
            border: 1px solid rgba(255, 171, 0, 0.4);
            color: #ffab00;
        }
        .badge-provenance-upsampled {
            background: rgba(255, 23, 68, 0.12);
            border: 1px solid rgba(255, 23, 68, 0.4);
            color: var(--accent-red);
        }
        .badge-provenance-fake {
            background: rgba(255, 23, 68, 0.2);
            border: 1px solid var(--accent-red);
            color: var(--accent-red);
        }
        .badge-provenance-unclear {
            background: rgba(139, 148, 158, 0.12);
            border: 1px solid rgba(139, 148, 158, 0.4);
            color: var(--text-muted);
        }
        .badge-provenance-mqa {
            background: rgba(186, 104, 200, 0.16);
            border: 1px solid rgba(186, 104, 200, 0.45);
            color: #ce93d8;
        }
        .confidence-pill {
            font-size: 0.72rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 10px;
            background: #21262d;
            border: 1px solid #30363d;
            display: inline-block;
        }
        .conf-high {
            color: var(--accent-green);
            border-color: rgba(174, 234, 0, 0.4);
            background: rgba(174, 234, 0, 0.08);
        }
        .conf-mod {
            color: var(--accent-yellow);
            border-color: rgba(255, 234, 0, 0.4);
            background: rgba(255, 234, 0, 0.08);
        }
        .conf-low {
            color: #8b949e;
            border-color: rgba(139, 148, 158, 0.4);
        }
        /* Action Recommendation Box */
        .action-recommendation-box {
            margin-top: 12px;
            padding: 10px 14px;
            background: rgba(22, 27, 34, 0.95);
            border: 1px solid #30363d;
            border-left: 4px solid var(--accent-yellow);
            border-radius: 6px;
            display: flex;
            flex-direction: column;
            gap: 6px;
            font-size: 0.83rem;
            animation: badge-fade-in 0.2s ease;
        }
        .action-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            flex-wrap: wrap;
        }
        .action-title {
            font-weight: 600;
            color: var(--text-heading);
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .risk-pill {
            font-size: 0.70rem;
            font-weight: 700;
            padding: 1px 6px;
            border-radius: 3px;
            text-transform: uppercase;
        }
        .risk-zero {
            background: rgba(174, 234, 0, 0.15);
            border: 1px solid rgba(174, 234, 0, 0.4);
            color: var(--accent-green);
        }
        .risk-minimal {
            background: rgba(88, 166, 255, 0.15);
            border: 1px solid rgba(88, 166, 255, 0.4);
            color: var(--accent-blue);
        }
        .risk-low {
            background: rgba(255, 234, 0, 0.15);
            border: 1px solid rgba(255, 234, 0, 0.4);
            color: var(--accent-yellow);
        }
        .action-details {
            color: var(--text);
            line-height: 1.45;
            font-size: 0.81rem;
        }
        .action-code {
            background: #0d1117;
            border: 1px solid #21262d;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: ui-monospace, SFMono-Regular, "SF Mono", monospace;
            font-size: 0.75rem;
            color: var(--accent-cyan);
            display: inline-block;
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
            text-align: center;
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
            margin-bottom: 12px;
            flex-wrap: wrap;
            gap: 8px;
        }
        .card-title {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-heading);
        }
        .canvas-container {
            position: relative;
            background: #000;
            border-radius: 6px;
            overflow: hidden;
            border: 1px solid var(--border);
            height: 340px;
            cursor: crosshair;
        }
        canvas {
            display: block;
            width: 100%;
            height: 100%;
        }
        .hud-overlay {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(13, 17, 23, 0.88);
            backdrop-filter: blur(4px);
            border: 1px solid var(--border);
            padding: 6px 12px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.8rem;
            pointer-events: none;
            display: flex;
            gap: 12px;
            z-index: 10;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }
        .hud-item span {
            color: var(--text-heading);
            font-weight: 600;
        }
        .report-box {
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 12px;
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
            font-size: 0.82rem;
            white-space: pre-wrap;
            line-height: 1.45;
            color: var(--text);
            max-height: 220px;
            overflow-y: auto;
        }
        .audio-player-container {
            display: flex;
            align-items: center;
            gap: 16px;
            background: #11141a;
            padding: 10px 16px;
            border-radius: 6px;
            border: 1px solid var(--border);
        }
        audio {
            flex: 1;
            height: 36px;
            outline: none;
        }
        .spinner {
            border: 3px solid rgba(255,255,255,0.1);
            border-top: 3px solid var(--accent-cyan);
            border-radius: 50%;
            width: 24px;
            height: 24px;
            animation: spin 0.8s linear infinite;
            display: inline-block;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <span>🔬 AcoustiSinc Forensic Lab</span>
            <span class="brand-badge">Strict 64-Bit Float</span>
            <span id="gpuStatusBadge" class="gpu-badge">⚡ GPU Initializing...</span>
        </div>
        <div class="bookmarks">
            <button class="btn-bookmark" onclick="loadDirectory('/mnt/PrimaryFS/FLAC_music/music', true)">📁 Music Library</button>
            <button class="btn-bookmark" onclick="loadDirectory('/mnt/PrimaryFS/FLAC_music/music/Qobuz Downloads', true)">🎧 Qobuz Releases</button>
            <button class="btn-bookmark" onclick="loadDirectory('/mnt/PrimaryFS/1xxK_min/music', true)">⚡ Upsampled Output</button>
            <button class="btn-bookmark" onclick="reloadCurrentDirectory(true)" title="Force fresh re-analysis on current folder">🔄 Re-Analyze Folder</button>
        </div>
    </header>

    <div class="main-layout">
        <!-- Sidebar File Browser -->
        <aside class="sidebar">
            <div class="nav-bar">
                <div class="nav-top-row">
                    <div class="breadcrumbs" id="breadcrumbs">
                        <span class="crumb" onclick="loadDirectory('.', true)">&#x21E7; Root</span>
                    </div>
                    <button class="btn-bookmark" style="padding: 2px 6px; font-size: 0.72rem;" onclick="reloadCurrentDirectory(true)" title="Re-Analyze Folder">🔄 Refresh</button>
                </div>
                <input type="text" class="search-box" id="pathBar" placeholder="Path..." value="" onchange="navigateToPathBar()" />
                <input type="text" class="search-box" id="searchBox" placeholder="Filter albums & tracks..." />
            </div>
            <div class="tree-list" id="treeList">
                <div style="padding: 20px; text-align: center; color: #8b949e;">
                    <div class="spinner" style="margin: 0 auto 10px;"></div>
                    Connecting to audio library...
                </div>
            </div>
        </aside>

        <!-- Right Visual Workspace -->
        <main class="workspace" id="workspace">
            <div class="empty-state" id="emptyState">
                <svg width="48" height="48" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="margin-bottom: 12px; opacity: 0.5;">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"></path>
                </svg>
                <h3>Select a Track for 64-bit Forensic DSP Analysis</h3>
                <p style="font-size: 0.85rem; margin-top: 4px;">Choose an audio file from the library sidebar to inspect its spectral provenance and true dynamic range.</p>
            </div>

            <div id="analysisContent" style="display: none; flex-direction: column; gap: 16px;">
                <!-- Audio Player Banner -->
                <div class="audio-player-container">
                    <audio id="audioPlayer" controls preload="metadata"></audio>
                </div>

                <!-- Provenance Assessment Banner -->
                <div class="card" id="provenanceCard" style="border-left: 4px solid var(--accent-cyan);">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 12px;">
                        <div style="flex: 1; min-width: 0;">
                            <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                                <span class="provenance-tag badge-provenance-native" id="provTag">NATIVE MASTER</span>
                                <span class="confidence-pill conf-high" id="provConf">HIGH CONFIDENCE</span>
                                <span class="brand-badge" id="bitDepthBadge" style="display: none;">24-BIT PCM</span>
                                <span class="brand-badge" id="mqaBadge" style="display: none; background: rgba(186, 104, 200, 0.18); border-color: rgba(186, 104, 200, 0.45); color: #ce93d8;">MQA</span>
                            </div>
                            <h2 id="trackTitle" style="font-size: 1.15rem; color: var(--text-heading); margin-top: 6px;">Track Name</h2>
                            <p id="provDetails" style="font-size: 0.85rem; color: var(--text); margin-top: 4px;">Assessment details...</p>

                            <!-- Alternative Hypothesis Row -->
                            <div id="altProvContainer" style="display: none; margin-top: 6px; font-size: 0.80rem; color: var(--text-muted); align-items: center; gap: 8px; flex-wrap: wrap;">
                                <span style="font-weight: 600; color: #8b949e;">Alternative Hypothesis:</span>
                                <span class="provenance-tag badge-provenance-upsampled" id="altProvTag" style="font-size: 0.72rem; padding: 1px 6px;">--</span>
                                <span class="confidence-pill conf-mod" id="altProvConf" style="font-size: 0.68rem; padding: 1px 6px;">--</span>
                                <span id="altProvDetails" style="font-size: 0.76rem; color: var(--text-muted);">--</span>
                            </div>

                            <!-- Visual Morphology Metrics Row -->
                            <div id="visMetricsRow" style="display: none; margin-top: 6px; font-size: 0.76rem; color: var(--text-muted); align-items: center; gap: 6px; flex-wrap: wrap;">
                                <span class="badge-tag" id="visKneeBadge" style="background: rgba(0, 229, 255, 0.08); border: 1px solid rgba(0, 229, 255, 0.3); color: var(--accent-cyan);" title="Inflection point where steep slope transitions to flat floor">📐 Knee: --</span>
                                <span class="badge-tag" id="visCorrBadge" style="background: rgba(174, 234, 0, 0.08); border: 1px solid rgba(174, 234, 0, 0.3); color: var(--accent-green);" title="Cross-band correlation between audible rhythm and ultrasonic harmonics">🎵 Coherence: --</span>
                                <span class="badge-tag" id="visPurityBadge" style="background: rgba(255, 112, 67, 0.08); border: 1px solid rgba(255, 112, 67, 0.3); color: #ff7043;" title="Stopband noise cleanliness, transient crest, and spur analysis">📻 Noise: --</span>
                                <span class="badge-tag" id="visVarBadge" style="background: rgba(88, 166, 255, 0.08); border: 1px solid rgba(88, 166, 255, 0.3); color: var(--accent-blue);" title="Temporal variance across time slices">📊 Dynamics: --</span>
                            </div>

                            <!-- Recommended DSP Course of Action Box -->
                            <div class="action-recommendation-box" id="actionRecBox" style="display: none;">
                                <div class="action-header">
                                    <div class="action-title">
                                        <span id="actionTitlePrefix">💡 Recommended DSP Action:</span>
                                        <span id="actionName" style="color: var(--accent-cyan); font-weight: 600;">--</span>
                                    </div>
                                    <span class="risk-pill risk-minimal" id="actionRisk">MINIMAL RISK</span>
                                </div>
                                <div class="action-details" id="actionDesc">--</div>
                                <div id="actionCodeContainer" style="display: none; margin-top: 2px;">
                                    <span style="font-size: 0.72rem; color: var(--text-muted); margin-right: 6px;">DSP Flag:</span>
                                    <code class="action-code" id="actionCode">--</code>
                                </div>
                            </div>
                        </div>
                        <div style="text-align: right; flex-shrink: 0;">
                            <div style="font-size: 1.25rem; font-weight: 700; color: var(--accent-yellow);" id="drScoreValue">DR14</div>
                            <div style="font-size: 0.72rem; color: var(--text-muted);" id="crestFactorValue">Crest: 14.5 dB</div>
                        </div>
                    </div>
                </div>

                <!-- Spectrogram Card -->
                <div class="card">
                    <div class="card-header">
                        <div>
                            <span class="card-title">Interactive Spectrogram (16,384-point Blackman-Harris STFT)</span>
                            <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 2px;" id="trackMeta">-- Hz | -- kHz Nyquist</div>
                        </div>
                        <div style="display: flex; gap: 8px;">
                            <button class="btn-bookmark" onclick="resetSpecZoom()">Reset View</button>
                        </div>
                    </div>
                    <div class="canvas-container" id="specContainer">
                        <canvas id="spectrogramCanvas"></canvas>
                        <div class="hud-overlay">
                            <div class="hud-item">T: <span id="specHudTime">-- s</span></div>
                            <div class="hud-item">F: <span id="specHudFreq">-- kHz</span></div>
                            <div class="hud-item">Mag: <span id="specHudDb">-- dBFS</span></div>
                        </div>
                    </div>
                </div>

                <!-- FFT Spectrum Curves Card -->
                <div class="card">
                    <div class="card-header">
                        <div>
                            <span class="card-title">Spectral Energy Distribution (Peak-Hold & RMS vs Frequency)</span>
                            <div id="curveLegend" style="font-size: 0.75rem; color: var(--text-muted); margin-top: 3px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                                <span style="color: var(--accent-cyan); font-weight: 600;">— Peak Hold</span>
                                <span style="color: var(--accent-pink); font-weight: 600;">— RMS Power</span>
                                <span style="color: #ffab00; font-weight: 600;">-- Container Nyquist</span>
                                <span id="legendProjectedCutoff" style="color: #00e676; font-weight: 600; display: none;">-- 🎯 Projected Filter</span>
                            </div>
                        </div>
                        <div style="display: flex; gap: 8px;">
                            <button class="btn-bookmark" onclick="setCurvePreset('audible')">0–20 kHz</button>
                            <button class="btn-bookmark" onclick="setCurvePreset('cutoff')">15–25 kHz</button>
                            <button class="btn-bookmark" onclick="setCurvePreset('ultrasonic')">Ultrasonic</button>
                            <button class="btn-bookmark" onclick="resetCurveZoom()">Full Band</button>
                        </div>
                    </div>
                    <div class="canvas-container" id="curveContainer">
                        <canvas id="spectrumCanvas"></canvas>
                        <div class="hud-overlay">
                            <div class="hud-item">Freq: <span id="hudFreq">-- kHz</span></div>
                            <div class="hud-item">Peak: <span id="hudPeak">-- dBFS</span></div>
                            <div class="hud-item">RMS: <span id="hudRMS">-- dBFS</span></div>
                            <div class="hud-item" id="hudProjItem" style="display: none;">Proj: <span id="hudProj" style="color: #00e676; font-weight: 600;">-- dBFS</span></div>
                        </div>
                    </div>
                </div>

                <!-- Lab Assessment Box -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Forensic Laboratory Assessment</span>
                        <button class="btn-bookmark" onclick="copyReport()">📋 Copy Report</button>
                    </div>
                    <div class="report-box" id="reportText">Running analysis...</div>
                </div>
            </div>
        </main>
    </div>

    <script>
        let currentPath = '';
        let directoryData = null;
        let currentAnalysis = null;
        let folderEventSource = null;
        let badgeCache = {};

        // Fetch GPU Status on startup
        fetch('/api/browse?path=' + encodeURIComponent(currentPath))
            .then(res => res.json())
            .then(data => {
                const b = document.getElementById('gpuStatusBadge');
                if (data.gpu_enabled) {
                    b.innerHTML = '⚡ GPU: ' + escapeHtml(data.gpu_device);
                    b.className = 'gpu-badge';
                } else {
                    b.innerHTML = '💻 CPU Multi-Thread';
                    b.className = 'gpu-badge cpu-mode';
                }
            })
            .catch(() => {});

        function navigateToPathBar() {
            const p = document.getElementById('pathBar').value.trim();
            loadDirectory(p || '.', true);
        }

        function reloadCurrentDirectory(fresh = true) {
            loadDirectory(currentPath, fresh);
        }

        document.getElementById('pathBar').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') navigateToPathBar();
        });

        document.getElementById('searchBox').addEventListener('input', (e) => {
            renderDirectory(e.target.value.toLowerCase());
        });

        function startFolderStream(folderPath, fresh = true) {
            if (folderEventSource) {
                folderEventSource.close();
                folderEventSource = null;
            }

            folderEventSource = new EventSource(`/api/folder_stream?path=${encodeURIComponent(folderPath)}&fresh=${fresh ? 1 : 0}`);
            folderEventSource.addEventListener('track_badge', (e) => {
                try {
                    const badge = JSON.parse(e.data);
                    badgeCache[badge.filepath] = badge;
                    updateFileBadgeDOM(badge);
                } catch (err) {}
            });
        }

        function updateFileBadgeDOM(badge) {
            if (!badge || !badge.filepath) return;
            const fileItem = document.querySelector(`.file-item[data-filepath="${CSS.escape(badge.filepath)}"]`);
            if (!fileItem) return;

            const badgeContainer = fileItem.querySelector('.file-badges');
            if (!badgeContainer) return;

            let bitsBadgeHtml = '';
            if (badge.is_zero_padded) {
                bitsBadgeHtml = `<span class="badge-tag badge-bits-padded" title="${badge.trailing_zero_bits} LSBs padded (${badge.container_bits}b container)">${badge.effective_bits}b (Pad)</span>`;
            } else if (badge.container_bits >= 24) {
                bitsBadgeHtml = `<span class="badge-tag badge-bits" title="True ${badge.effective_bits}-bit PCM">${badge.effective_bits}b</span>`;
            }

            let provClass = badge.badge_class || 'badge-provenance-native';
            let provTitle = escapeHtml(badge.verdict);
            if (badge.recommendation && badge.recommendation.action) {
                const conf = (badge.confidence || '').toLowerCase();
                if (conf !== 'low') {
                    const isPot = (conf === 'moderate' || conf === 'medium' || badge.recommendation.is_potential);
                    const labelPrefix = isPot ? '💡 Potential Action:' : '💡 Recommended Action:';
                    provTitle += `&#10;${labelPrefix} ${escapeHtml(badge.recommendation.action)}`;
                }
            }
            if (badge.alternative && badge.alternative.label) {
                provTitle += `&#10;🥈 Alternative: ${escapeHtml(badge.alternative.label)}`;
            }
            let provBadgeHtml = `<span class="badge-tag ${provClass}" title="${provTitle}">${escapeHtml(badge.short_verdict)}</span>`;

            let drBadgeHtml = badge.dr_score ? `<span class="badge-tag badge-dr" title="TT DR Score (DR${badge.dr_score})">DR${badge.dr_score}</span>` : '';

            badgeContainer.innerHTML = `
                <span class="badge-tag badge-sr">${badge.sr_str}</span>
                ${bitsBadgeHtml}
                ${drBadgeHtml}
                ${provBadgeHtml}
            `;
        }

        async function loadDirectory(path, fresh = true) {
            const player = document.getElementById('audioPlayer');
            if (player && !player.paused) player.pause();

            currentPath = path || '';
            const searchBox = document.getElementById('searchBox');
            if (searchBox) searchBox.value = '';

            // Invalidate frontend badge cache and workspace view on folder transition
            badgeCache = {};
            currentAnalysis = null;
            document.getElementById('analysisContent').style.display = 'none';
            document.getElementById('emptyState').style.display = 'flex';

            const treeList = document.getElementById('treeList');
            treeList.innerHTML = '<div style="padding: 20px; text-align: center; color: #8b949e;"><div class="spinner" style="margin: 0 auto 10px;"></div>Analyzing folder with 64-bit DSP...</div>';

            try {
                const res = await fetch(`/api/browse?path=${encodeURIComponent(currentPath)}&fresh=${fresh ? 1 : 0}`);
                directoryData = await res.json();
                if (directoryData.error) throw new Error(directoryData.error);

                currentPath = directoryData.current_path;
                document.getElementById('pathBar').value = currentPath;

                const gpuB = document.getElementById('gpuStatusBadge');
                if (gpuB && directoryData.gpu_enabled) {
                    gpuB.innerHTML = '⚡ GPU: ' + escapeHtml(directoryData.gpu_device || 'Active');
                    gpuB.className = 'gpu-badge';
                }

                renderBreadcrumbs(directoryData.breadcrumbs, directoryData.parent_path);
                renderDirectory('');

                // Connect SSE stream for progressive fresh folder badges
                startFolderStream(currentPath, fresh);
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
            upBtn.onclick = () => loadDirectory(parentPath, true);
            el.appendChild(upBtn);

            crumbs.forEach(c => {
                const span = document.createElement('span');
                span.innerHTML = ' / ';
                span.style.color = '#8b949e';
                el.appendChild(span);

                const a = document.createElement('span');
                a.className = 'crumb';
                a.textContent = c.name;
                a.onclick = () => loadDirectory(c.path, true);
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
                div.onclick = () => loadDirectory(f.path, true);
                treeList.appendChild(div);
            });

            // Files
            const filteredFiles = directoryData.files.filter(f => !filter || f.name.toLowerCase().includes(filter));
            filteredFiles.forEach((f, idx) => {
                const div = document.createElement('div');
                div.className = 'file-item';
                div.id = `file-item-${idx}`;
                div.setAttribute('data-filepath', f.path);
                div.title = f.name;

                const srText = f.samplerate ? `${(f.samplerate/1000).toFixed(1)}k` : 'FLAC';
                const cachedBadge = badgeCache[f.path] || f.badge;

                div.innerHTML = `
                    <div class="file-top">
                        <div class="file-title-container">
                            <span style="flex-shrink: 0;">🎵</span>
                            <span class="file-title">${escapeHtml(f.name)}</span>
                        </div>
                        <div class="file-badges">
                            <span class="badge-tag badge-sr">${srText}</span>
                        </div>
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

                if (cachedBadge) {
                    badgeCache[f.path] = cachedBadge;
                    updateFileBadgeDOM(cachedBadge);
                }
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
                const res = await fetch(`/api/analyze?path=${encodeURIComponent(file.path)}&fresh=1`);
                const data = await res.json();
                if (data.status !== 'ok') throw new Error(data.message);

                currentAnalysis = data;
                emptyState.style.display = 'none';
                analysisContent.style.display = 'flex';

                // Synchronously update the sidebar badges immediately
                if (data.badge) {
                    badgeCache[data.filepath] = data.badge;
                    updateFileBadgeDOM(data.badge);
                }

                // Update UI metadata
                document.getElementById('trackTitle').textContent = data.filename;
                document.getElementById('trackMeta').textContent = `${data.sr.toLocaleString()} Hz | ${data.nyquist_khz.toFixed(1)} kHz Nyquist | ${data.duration_s.toFixed(1)}s sample | ${data.analysis_time}s (${data.gpu_enabled ? 'GPU' : 'CPU'})`;
                
                // MQA Badge
                const mqaBadge = document.getElementById('mqaBadge');
                if (data.provenance && data.provenance.label && data.provenance.label.includes('MQA')) {
                    mqaBadge.style.display = 'inline-block';
                    mqaBadge.textContent = data.provenance.label.includes('Studio') ? 'MQA STUDIO' : 'MQA';
                } else {
                    mqaBadge.style.display = 'none';
                }

                // Bit Depth Badge
                const bdBadge = document.getElementById('bitDepthBadge');
                if (data.provenance && data.provenance.bitdepth) {
                    const bd = data.provenance.bitdepth;
                    bdBadge.style.display = 'inline-block';
                    if (bd.is_zero_padded) {
                        bdBadge.textContent = `${bd.effective_bits}B PADDED (${bd.container_bits}B)`;
                        bdBadge.style.background = 'rgba(255, 112, 67, 0.18)';
                        bdBadge.style.borderColor = 'rgba(255, 112, 67, 0.45)';
                        bdBadge.style.color = '#ff7043';
                    } else if (bd.container_bits >= 24) {
                        bdBadge.textContent = `${bd.effective_bits}-BIT PCM`;
                        bdBadge.style.background = 'rgba(0, 229, 255, 0.12)';
                        bdBadge.style.borderColor = 'rgba(0, 229, 255, 0.35)';
                        bdBadge.style.color = '#00e5ff';
                    } else {
                        bdBadge.textContent = '16-BIT PCM';
                        bdBadge.style.background = 'rgba(139, 148, 158, 0.12)';
                        bdBadge.style.borderColor = 'rgba(139, 148, 158, 0.35)';
                        bdBadge.style.color = 'var(--text-muted)';
                    }
                } else {
                    bdBadge.style.display = 'none';
                }

                // Provenance Banner
                const provTag = document.getElementById('provTag');
                const provConf = document.getElementById('provConf');
                const provDetails = document.getElementById('provDetails');
                const provCard = document.getElementById('provenanceCard');

                if (data.provenance && data.provenance.primary) {
                    const p = data.provenance.primary;
                    provTag.textContent = p.label.toUpperCase();
                    provTag.className = `provenance-tag ${p.badge_class || 'badge-provenance-native'}`;
                    
                    provConf.textContent = `${p.confidence.toUpperCase()} CONFIDENCE (${Math.round(p.score * 100)}%)`;
                    provConf.className = `confidence-pill ${p.confidence.toLowerCase() === 'high' ? 'conf-high' : p.confidence.toLowerCase() === 'moderate' ? 'conf-mod' : 'conf-low'}`;

                    provDetails.textContent = p.details;

                    if (p.badge_class === 'badge-provenance-upsampled' || p.badge_class === 'badge-provenance-fake') {
                        provCard.style.borderLeftColor = 'var(--accent-red)';
                    } else if (p.badge_class === 'badge-provenance-leaky') {
                        provCard.style.borderLeftColor = '#ffab00';
                    } else if (p.badge_class === 'badge-provenance-mqa') {
                        provCard.style.borderLeftColor = '#ba68c8';
                    } else {
                        provCard.style.borderLeftColor = 'var(--accent-cyan)';
                    }
                } else {
                    provTag.textContent = data.verdict;
                    provTag.className = 'provenance-tag badge-provenance-native';
                    provConf.textContent = 'ANALYZED';
                    provDetails.textContent = data.noise_profile || 'Standard spectral profile.';
                    provCard.style.borderLeftColor = 'var(--accent-cyan)';
                }

                // Alternative Hypothesis
                const altContainer = document.getElementById('altProvContainer');
                if (data.provenance && data.provenance.alternative) {
                    const alt = data.provenance.alternative;
                    altContainer.style.display = 'flex';
                    const altTag = document.getElementById('altProvTag');
                    const altConf = document.getElementById('altProvConf');
                    const altDet = document.getElementById('altProvDetails');
                    
                    altTag.textContent = alt.label.toUpperCase();
                    altTag.className = `provenance-tag ${alt.badge_class || 'badge-provenance-upsampled'}`;
                    altConf.textContent = `${alt.confidence.toUpperCase()} CONFIDENCE (${Math.round((alt.score || 0.65) * 100)}%)`;
                    altConf.className = `confidence-pill ${alt.confidence.toLowerCase() === 'high' ? 'conf-high' : alt.confidence.toLowerCase() === 'moderate' ? 'conf-mod' : 'conf-low'}`;
                    altDet.textContent = alt.details || '';
                } else {
                    altContainer.style.display = 'none';
                }

                // Visual Morphology Indicators
                const visRow = document.getElementById('visMetricsRow');
                if (data.provenance && data.provenance.visual_morphology) {
                    const v = data.provenance.visual_morphology;
                    visRow.style.display = 'flex';
                    
                    const kneeEl = document.getElementById('visKneeBadge');
                    if (v.detected_knees && v.detected_knees.length > 0) {
                        const kneeTexts = v.detected_knees.slice(0, 2).map(k => `${k.freq_khz}k`);
                        kneeEl.textContent = `📐 Knee${v.detected_knees.length > 1 ? 's' : ''}: ${kneeTexts.join(', ')}`;
                        kneeEl.title = v.detected_knees.map(k => `Knee at ${k.freq_khz} kHz: drop ${k.drop_db} dB down to shelf ${k.level_dbfs} dBFS (slope: ${k.pre_slope_db_per_khz} dB/kHz)`).join('\\n');
                    } else {
                        kneeEl.textContent = `📐 Smooth Rolloff (No Knee)`;
                    }

                    const corrEl = document.getElementById('visCorrBadge');
                    const r = v.rhythmic_coherence || 0.0;
                    if (r >= 0.45) {
                        corrEl.textContent = `🎵 Dynamic Harmonics (r = ${r > 0 ? '+' : ''}${r.toFixed(2)})`;
                        corrEl.style.color = 'var(--accent-green)';
                        corrEl.style.borderColor = 'rgba(174, 234, 0, 0.4)';
                    } else if (v.is_stationary_ultrasonic) {
                        corrEl.textContent = `📻 Static Dither/Noise (r = ${r > 0 ? '+' : ''}${r.toFixed(2)})`;
                        corrEl.style.color = '#ff7043';
                        corrEl.style.borderColor = 'rgba(255, 112, 67, 0.4)';
                    } else {
                        corrEl.textContent = `📊 Coherence (r = ${r > 0 ? '+' : ''}${r.toFixed(2)})`;
                        corrEl.style.color = 'var(--text-muted)';
                        corrEl.style.borderColor = 'var(--border)';
                    }

                    const purityEl = document.getElementById('visPurityBadge');
                    if (v.stopband_purity) {
                        const pur = v.stopband_purity;
                        if (!pur.has_stopband) {
                            purityEl.textContent = `✨ Passband: Authentic Harmonics (Peak ${pur.max_peak_dbfs} dBFS)`;
                            purityEl.style.color = 'var(--accent-cyan)';
                            purityEl.style.borderColor = 'rgba(0, 229, 255, 0.4)';
                            purityEl.title = pur.description;
                        } else if (pur.is_messy) {
                            purityEl.textContent = `📻 Stopband: Messy Hash (Spurs: ${pur.max_peak_dbfs} dBFS, +${pur.crest_db} dB)`;
                            purityEl.style.color = '#ff7043';
                            purityEl.style.borderColor = 'rgba(255, 112, 67, 0.4)';
                            purityEl.title = pur.description;
                        } else {
                            purityEl.textContent = `✨ Stopband: Clean Dither (${pur.median_rms_dbfs} dBFS)`;
                            purityEl.style.color = 'var(--accent-green)';
                            purityEl.style.borderColor = 'rgba(174, 234, 0, 0.4)';
                            purityEl.title = pur.description;
                        }
                    }

                    const varEl = document.getElementById('visVarBadge');
                    varEl.textContent = `📊 Var: Aud ${v.audible_temporal_variance.toFixed(0)} dB² | Ultra ${v.ultrasonic_temporal_variance.toFixed(0)} dB²`;
                } else {
                    visRow.style.display = 'none';
                }

                // Recommended / Potential Course of Action Box
                const recBox = document.getElementById('actionRecBox');
                if (data.provenance && data.provenance.recommendation && data.provenance.primary) {
                    const conf = (data.provenance.primary.confidence || '').toLowerCase();
                    const score = data.provenance.primary.score !== undefined ? data.provenance.primary.score : 0.8;
                    
                    if (conf === 'low' || score < 0.50) {
                        recBox.style.display = 'none';
                    } else {
                        const r = data.provenance.recommendation;
                        recBox.style.display = 'flex';
                        
                        const isPotential = (conf === 'moderate' || conf === 'medium' || score < 0.85 || r.is_potential);
                        const prefixEl = document.getElementById('actionTitlePrefix');
                        if (prefixEl) {
                            prefixEl.textContent = isPotential ? '💡 Potential DSP Action:' : '💡 Recommended DSP Action:';
                        }
                        
                        document.getElementById('actionName').textContent = r.action;
                        
                        const riskPill = document.getElementById('actionRisk');
                        riskPill.textContent = (r.risk_level || 'MINIMAL RISK').toUpperCase();
                        riskPill.className = `risk-pill ${r.risk_class || 'risk-minimal'}`;
                        
                        document.getElementById('actionDesc').textContent = r.details;
                        
                        const codeCont = document.getElementById('actionCodeContainer');
                        const codeEl = document.getElementById('actionCode');
                        if (r.dsp_params) {
                            codeCont.style.display = 'block';
                            codeEl.textContent = r.dsp_params;
                        } else {
                            codeCont.style.display = 'none';
                        }
                    }
                } else {
                    recBox.style.display = 'none';
                }

                // Dynamic Range
                document.getElementById('drScoreValue').textContent = `DR${data.dr_score}`;
                document.getElementById('crestFactorValue').textContent = `Crest: ${data.crest_factor_db.toFixed(1)} dB | LRA: ${data.lra_lu.toFixed(1)} LU`;

                // Report text
                document.getElementById('reportText').textContent = data.report_text;

                // Load audio in player
                const player = document.getElementById('audioPlayer');
                player.src = `/api/stream?path=${encodeURIComponent(file.path)}`;

                // Dynamic Curve Legend
                updateCurveLegend(data);

                // Initialize Canvases
                initSpectrogram(data);
                initSpectrumCurve(data);
            } catch (err) {
                emptyState.style.display = 'flex';
                emptyState.innerHTML = `<div style="color: var(--accent-red); text-align: center;">Error running DSP analysis:<br><small>${err.message}</small></div>`;
            }
        }

        function updateCurveLegend(data) {
            const legendEl = document.getElementById('curveLegend');
            if (!legendEl || !data) return;
            let html = `
                <span style="color: var(--accent-cyan); font-weight: 600;">— Peak Hold</span>
                <span style="color: var(--accent-pink); font-weight: 600;">— RMS Power</span>
            `;
            if (data.nyquist_khz > 20.0) {
                html += `<span style="color: #ff1744; font-weight: 600;">-- 20 kHz Hearing Limit</span>`;
            }
            const prov = data.provenance || {};
            if (prov.suspected_nyquist_hz && prov.suspected_nyquist_hz < (data.sr / 2.0) - 200) {
                const sNyqF = (prov.suspected_nyquist_hz / 1000.0).toFixed(1);
                const sBaseF = ((prov.suspected_base_sr_hz || (prov.suspected_nyquist_hz * 2)) / 1000.0).toFixed(1);
                html += `<span style="color: #ffea00; font-weight: 600;">-- ${sNyqF} kHz (${sBaseF}k Lineage)</span>`;
            }
            html += `<span style="color: #ffab00; font-weight: 600;">-- ${data.nyquist_khz.toFixed(1)} kHz Container Nyquist</span>`;
            legendEl.innerHTML = html;
        }

        // ==========================================
        // SPECTROGRAM CANVAS WITH ZOOM, PAN & HUD
        // ==========================================
        let specImg = null;
        let specLookup = null;
        let lookupW = 256;
        let lookupH = 128;
        let specTMin = 0.0, specTMax = 45.0;
        let specFMin = 0.0, specFMax = 88.2;
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
            lookupW = data.lookup_w || 256;
            lookupH = data.lookup_h || 128;

            specImg = new Image();
            specImg.onload = () => {
                resizeSpecCanvas();
            };
            const b64 = data.webp_base64 || '';
            specImg.src = b64.startsWith('data:') ? b64 : ('data:image/webp;base64,' + b64);

            if (data.lookup_base64) {
                const raw = atob(data.lookup_base64);
                specLookup = new Uint8Array(raw.length);
                for (let i = 0; i < raw.length; i++) specLookup[i] = raw.charCodeAt(i);
            } else {
                specLookup = null;
            }
        }

        function getSpectrogramDb(t, fKhz) {
            if (!currentAnalysis || !specLookup) return -165.0;
            const dur = currentAnalysis.duration_s;
            const nyq = currentAnalysis.nyquist_khz;
            if (t < 0 || t > dur || fKhz < 0 || fKhz > nyq) return -165.0;
            const x = Math.max(0, Math.min(lookupW - 1, Math.floor((t / dur) * lookupW)));
            const y = Math.max(0, Math.min(lookupH - 1, Math.floor((1.0 - (fKhz / nyq)) * lookupH)));
            const u8 = specLookup[y * lookupW + x];
            return (u8 / 255.0) * 165.0 - 165.0;
        }

        function resetSpecZoom() {
            if (!currentAnalysis) return;
            specTMin = 0.0; specTMax = currentAnalysis.duration_s;
            specFMin = 0.0; specFMax = currentAnalysis.nyquist_khz;
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
            if (!currentAnalysis || !specImg || !specImg.complete) return;
            sCtx.clearRect(0, 0, w, h);

            const padL = 55, padR = 70, padT = 15, padB = 35;
            const plotW = w - padL - padR;
            const plotH = h - padT - padB;

            if (plotW <= 0 || plotH <= 0) return;

            sCtx.save();
            sCtx.beginPath();
            sCtx.rect(padL, padT, plotW, plotH);
            sCtx.clip();

            const sx = (specTMin / currentAnalysis.duration_s) * specImg.naturalWidth;
            const sw = ((specTMax - specTMin) / currentAnalysis.duration_s) * specImg.naturalWidth;
            const sy = (1.0 - (specFMax / currentAnalysis.nyquist_khz)) * specImg.naturalHeight;
            const sh = ((specFMax - specFMin) / currentAnalysis.nyquist_khz) * specImg.naturalHeight;

            sCtx.imageSmoothingEnabled = true;
            sCtx.drawImage(specImg, sx, sy, sw, sh, padL, padT, plotW, plotH);

            // Interactive Crosshair & HUD on hover
            if (specMouseX >= padL && specMouseX <= padL + plotW && specMouseY >= padT && specMouseY <= padT + plotH) {
                sCtx.strokeStyle = "rgba(255, 255, 255, 0.45)";
                sCtx.lineWidth = 1;
                sCtx.setLineDash([3, 3]);
                sCtx.beginPath();
                sCtx.moveTo(specMouseX, padT);
                sCtx.lineTo(specMouseX, padT + plotH);
                sCtx.moveTo(padL, specMouseY);
                sCtx.lineTo(padL + plotW, specMouseY);
                sCtx.stroke();
                sCtx.setLineDash([]);

                const curT = specTMin + ((specMouseX - padL) / plotW) * (specTMax - specTMin);
                const curF = specFMin + (1.0 - (specMouseY - padT) / plotH) * (specFMax - specFMin);
                const curDb = getSpectrogramDb(curT, curF);

                specHudTime.textContent = curT.toFixed(2) + " s";
                specHudFreq.textContent = curF.toFixed(2) + " kHz (" + (curF * 1000).toFixed(0) + " Hz)";
                specHudDb.textContent = curDb.toFixed(1) + " dBFS";
            }

            sCtx.restore();

            // Axes & Labels
            sCtx.strokeStyle = '#30363d';
            sCtx.lineWidth = 1;
            sCtx.strokeRect(padL, padT, plotW, plotH);

            sCtx.fillStyle = '#8b949e';
            sCtx.font = '10px monospace';
            sCtx.textAlign = 'right';
            sCtx.textBaseline = 'middle';

            const fStep = (specFMax - specFMin) > 40 ? 20 : (specFMax - specFMin) > 15 ? 5 : 2;
            for (let f = Math.ceil(specFMin / fStep) * fStep; f <= specFMax; f += fStep) {
                const y = padT + (1.0 - (f - specFMin) / (specFMax - specFMin)) * plotH;
                sCtx.fillText(`${f}k`, padL - 6, y);
                sCtx.strokeStyle = 'rgba(255,255,255,0.06)';
                sCtx.beginPath(); sCtx.moveTo(padL, y); sCtx.lineTo(padL + plotW, y); sCtx.stroke();
            }

            sCtx.textAlign = 'center';
            sCtx.textBaseline = 'top';
            const tStep = (specTMax - specTMin) > 20 ? 10 : (specTMax - specTMin) > 8 ? 2 : 0.5;
            for (let t = Math.ceil(specTMin / tStep) * tStep; t <= specTMax; t += tStep) {
                const x = padL + ((t - specTMin) / (specTMax - specTMin)) * plotW;
                sCtx.fillText(`${t.toFixed(1)}s`, x, padT + plotH + 6);
                sCtx.strokeStyle = 'rgba(255,255,255,0.06)';
                sCtx.beginPath(); sCtx.moveTo(x, padT); sCtx.lineTo(x, padT + plotH); sCtx.stroke();
            }

            // Colorbar on right
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
            sCtx.font = "9px monospace";
            for (let d of [0, -40, -80, -120, -165]) {
                const y = padT + (d / -165.0) * barH;
                sCtx.fillText(d === 0 ? "0dB" : `${d}dB`, barX + barW + 4, y);
            }
        }

        specCanvas.addEventListener('mousedown', (e) => {
            specIsDragging = true;
            specDragStartX = e.clientX;
            specDragStartY = e.clientY;
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
        // SPECTRUM CURVE CANVAS WITH ZOOM, PAN & HUD
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
            curveFMin = 0.0; curveFMax = data.nyquist_khz * 1.04;
            curveDbMin = -175.0; curveDbMax = 0.0;
            resizeCurveCanvas();
        }

        function resetCurveZoom() {
            if (!currentAnalysis) return;
            curveFMin = 0.0; curveFMax = currentAnalysis.nyquist_khz * 1.04;
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
                curveFMin = Math.min(20.0, currentAnalysis.nyquist_khz); curveFMax = currentAnalysis.nyquist_khz * 1.04;
                curveDbMin = -175.0; curveDbMax = -120.0;
            }
            drawCurve(curveCanvas.getBoundingClientRect().width, curveCanvas.getBoundingClientRect().height);
        }

        function zoomCurve(factor, centerFRatio = 0.5, centerDbRatio = 0.5) {
            if (!currentAnalysis) return;
            const curFW = curveFMax - curveFMin;
            const newFW = Math.max(1.0, Math.min(currentAnalysis.nyquist_khz * 1.04, curFW / factor));
            const centerF = curveFMin + curFW * centerFRatio;
            curveFMin = Math.max(0, centerF - newFW * centerFRatio);
            curveFMax = Math.min(currentAnalysis.nyquist_khz * 1.04, curveFMin + newFW);
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
            const padL = 55, padR = 25, padT = 20, padB = 35;
            const plotW = w - padL - padR;
            const plotH = h - padT - padB;

            if (plotW <= 0 || plotH <= 0) return;

            cCtx.fillStyle = "#11141a";
            cCtx.fillRect(padL, padT, plotW, plotH);

            const freqs = currentAnalysis.curve_freqs_khz;
            const peaks = currentAnalysis.curve_peaks;
            const rms = currentAnalysis.curve_rms;

            cCtx.save();
            cCtx.beginPath();
            cCtx.rect(padL, padT, plotW, plotH);
            cCtx.clip();

            // 1. File Container Nyquist Limit
            const nyqF = currentAnalysis.nyquist_khz;
            if (curveFMin <= nyqF && curveFMax >= nyqF) {
                const xNyq = freqToX(nyqF, w, padL, padR);
                cCtx.strokeStyle = "rgba(255, 171, 0, 0.95)";
                cCtx.lineWidth = 1.5;
                cCtx.setLineDash([5, 3]);
                cCtx.beginPath(); cCtx.moveTo(xNyq, padT); cCtx.lineTo(xNyq, padT + plotH); cCtx.stroke();
                cCtx.setLineDash([]);
                cCtx.fillStyle = "#ffab00";
                cCtx.font = "bold 9px monospace";
                cCtx.textAlign = "right";
                cCtx.fillText(`${nyqF.toFixed(1)}k Nyquist`, Math.min(xNyq - 4, padL + plotW - 4), padT + 8);
            }

            // 2. 20 kHz Audible Hearing Limit (shown only when container Nyquist > 20 kHz)
            if (currentAnalysis.nyquist_khz > 20.0 && curveFMin <= 20.0 && curveFMax >= 20.0) {
                const x20 = freqToX(20.0, w, padL, padR);
                cCtx.strokeStyle = "rgba(255, 23, 68, 0.85)";
                cCtx.lineWidth = 1.5;
                cCtx.setLineDash([4, 4]);
                cCtx.beginPath(); cCtx.moveTo(x20, padT); cCtx.lineTo(x20, padT + plotH); cCtx.stroke();
                cCtx.setLineDash([]);
                cCtx.fillStyle = "#ff1744";
                cCtx.font = "9px monospace";
                cCtx.textAlign = "center";
                cCtx.fillText("20k Hearing", x20, padT + 8);
            }

            // 3. Suspected Native Lineage Nyquist (shown ONLY if estimated source differs from container)
            const prov = currentAnalysis.provenance || {};
            if (prov.suspected_nyquist_hz && prov.suspected_nyquist_hz < (currentAnalysis.sr / 2.0) - 200) {
                const sNyqF = prov.suspected_nyquist_hz / 1000.0;
                const sBaseF = (prov.suspected_base_sr_hz || (prov.suspected_nyquist_hz * 2)) / 1000.0;
                if (curveFMin <= sNyqF && curveFMax >= sNyqF) {
                    const xsNyq = freqToX(sNyqF, w, padL, padR);
                    cCtx.strokeStyle = "rgba(255, 234, 0, 0.95)";
                    cCtx.lineWidth = 1.5;
                    cCtx.setLineDash([4, 3]);
                    cCtx.beginPath(); cCtx.moveTo(xsNyq, padT); cCtx.lineTo(xsNyq, padT + plotH); cCtx.stroke();
                    cCtx.setLineDash([]);
                    cCtx.fillStyle = "#ffea00";
                    cCtx.font = "bold 9px monospace";
                    cCtx.textAlign = "center";
                    cCtx.fillText(`${sNyqF.toFixed(1)}k (${sBaseF.toFixed(1)}k Source)`, xsNyq, padT + 20);
                }
            }

            // Peak curve (Cyan)
            cCtx.strokeStyle = "rgba(0, 229, 255, 0.9)";
            cCtx.lineWidth = 1.5;
            cCtx.beginPath();
            let first = true;
            for (let i = 0; i < freqs.length; i++) {
                if (freqs[i] < curveFMin || freqs[i] > curveFMax) continue;
                const x = freqToX(freqs[i], w, padL, padR);
                const y = dbToY(peaks[i], h, padT, padB);
                if (first) { cCtx.moveTo(x, y); first = false; }
                else { cCtx.lineTo(x, y); }
            }
            cCtx.stroke();

            // RMS curve (Magenta)
            cCtx.strokeStyle = "rgba(255, 0, 127, 0.9)";
            cCtx.lineWidth = 1.5;
            cCtx.beginPath();
            first = true;
            for (let i = 0; i < freqs.length; i++) {
                if (freqs[i] < curveFMin || freqs[i] > curveFMax) continue;
                const x = freqToX(freqs[i], w, padL, padR);
                const y = dbToY(rms[i], h, padT, padB);
                if (first) { cCtx.moveTo(x, y); first = false; }
                else { cCtx.lineTo(x, y); }
            }
            cCtx.stroke();

            // 4. Projected Filter Cutoff Response (Dashed Neon Green - plotted ONLY between Cutoff Start and Container Nyquist)
            let recCutoffKhz = null;
            const rec = currentAnalysis.provenance ? currentAnalysis.provenance.recommendation : null;
            if (rec) {
                if (rec.filter_cutoff_khz) {
                    recCutoffKhz = rec.filter_cutoff_khz;
                } else if (rec.dsp_params) {
                    const m = rec.dsp_params.match(/--(?:cutoff|apodize)\\s+(\\d+)/);
                    if (m) recCutoffKhz = parseFloat(m[1]) / 1000.0;
                }
            }

            function calcProjectedLevel(f, origRmsDb, cutoffF) {
                if (f < cutoffF) return origRmsDb;
                const deltaF = f - cutoffF;
                const wTrans = Math.min(1.2, Math.max(0.5, 0.03 * cutoffF));
                let attDb = 0;
                if (deltaF <= wTrans) {
                    const ratio = deltaF / wTrans;
                    attDb = 120.0 * Math.pow((1.0 - Math.cos(Math.PI * ratio)) / 2.0, 1.5);
                } else {
                    attDb = 120.0 + 20.0 * ((deltaF - wTrans) / wTrans);
                }
                return Math.max(-170.0, origRmsDb - attDb);
            }

            const legProj = document.getElementById('legendProjectedCutoff');
            if (recCutoffKhz && recCutoffKhz < currentAnalysis.nyquist_khz) {
                const cutoffLabel = (recCutoffKhz % 1 === 0) ? `${recCutoffKhz.toFixed(0)}k` : `${recCutoffKhz.toFixed(2).replace(/0$/, '')}k`;
                if (legProj) {
                    legProj.style.display = 'inline';
                    legProj.textContent = `-- 🎯 Projected Filter (@ ${cutoffLabel})`;
                }

                // Vertical Cutoff Marker Line
                if (curveFMin <= recCutoffKhz && curveFMax >= recCutoffKhz) {
                    const xCut = freqToX(recCutoffKhz, w, padL, padR);
                    cCtx.strokeStyle = "rgba(0, 230, 118, 0.85)";
                    cCtx.lineWidth = 1.2;
                    cCtx.setLineDash([3, 3]);
                    cCtx.beginPath(); cCtx.moveTo(xCut, padT); cCtx.lineTo(xCut, padT + plotH); cCtx.stroke();
                    cCtx.setLineDash([]);
                    cCtx.fillStyle = "#00e676";
                    cCtx.font = "bold 9px monospace";
                    cCtx.textAlign = "left";
                    cCtx.fillText(`Cutoff @ ${cutoffLabel}`, xCut + 4, padT + plotH - 8);
                }

                // Projected Filter Rolloff Curve (ONLY from Cutoff to Nyquist, applied to ACTUAL measured signal)
                cCtx.strokeStyle = "#00e676";
                cCtx.lineWidth = 2.0;
                cCtx.setLineDash([5, 3]);
                cCtx.beginPath();
                let pFirst = true;
                const nyq = currentAnalysis.nyquist_khz;
                for (let i = 0; i < freqs.length; i++) {
                    const f = freqs[i];
                    if (f < recCutoffKhz || f > nyq) continue;
                    if (f < curveFMin || f > curveFMax) continue;
                    
                    const projDb = calcProjectedLevel(f, rms[i], recCutoffKhz);
                    const x = freqToX(f, w, padL, padR);
                    const y = dbToY(projDb, h, padT, padB);
                    if (pFirst) { cCtx.moveTo(x, y); pFirst = false; }
                    else { cCtx.lineTo(x, y); }
                }
                cCtx.stroke();
                cCtx.setLineDash([]);
            } else {
                if (legProj) legProj.style.display = 'none';
            }

            // Interactive Hover Reticle & HUD
            if (curveMouseX >= padL && curveMouseX <= padL + plotW) {
                const ratio = (curveMouseX - padL) / plotW;
                const targetFreq = curveFMin + ratio * (curveFMax - curveFMin);
                let closestIdx = 0, minDiff = Infinity;
                for (let i = 0; i < freqs.length; i++) {
                    const d = Math.abs(freqs[i] - targetFreq);
                    if (d < minDiff) { minDiff = d; closestIdx = i; }
                }

                const curX = freqToX(freqs[closestIdx], w, padL, padR);
                const curYPeak = dbToY(peaks[closestIdx], h, padT, padB);
                const curYRMS = dbToY(rms[closestIdx], h, padT, padB);

                cCtx.strokeStyle = "rgba(255, 255, 255, 0.45)";
                cCtx.lineWidth = 1;
                cCtx.setLineDash([2, 2]);
                cCtx.beginPath();
                cCtx.moveTo(curX, padT);
                cCtx.lineTo(curX, padT + plotH);
                cCtx.stroke();
                cCtx.setLineDash([]);

                // Peak point indicator (Cyan)
                cCtx.fillStyle = "#00e5ff";
                cCtx.beginPath();
                cCtx.arc(curX, curYPeak, 4, 0, Math.PI * 2);
                cCtx.fill();
                cCtx.strokeStyle = "#fff";
                cCtx.lineWidth = 1;
                cCtx.stroke();

                // RMS point indicator (Pink)
                cCtx.fillStyle = "#ff007f";
                cCtx.beginPath();
                cCtx.arc(curX, curYRMS, 4, 0, Math.PI * 2);
                cCtx.fill();
                cCtx.strokeStyle = "#fff";
                cCtx.lineWidth = 1;
                cCtx.stroke();

                // Projected point indicator (Neon Green) if active in cutoff zone
                const hudProjItem = document.getElementById('hudProjItem');
                const hudProj = document.getElementById('hudProj');
                if (recCutoffKhz && freqs[closestIdx] >= recCutoffKhz && freqs[closestIdx] <= currentAnalysis.nyquist_khz) {
                    const projDb = calcProjectedLevel(freqs[closestIdx], rms[closestIdx], recCutoffKhz);
                    
                    if (hudProjItem && hudProj) {
                        hudProjItem.style.display = 'block';
                        hudProj.textContent = `${projDb.toFixed(1)} dBFS`;
                    }

                    const curYProj = dbToY(projDb, h, padT, padB);
                    cCtx.fillStyle = "#00e676";
                    cCtx.beginPath();
                    cCtx.arc(curX, curYProj, 4, 0, Math.PI * 2);
                    cCtx.fill();
                    cCtx.strokeStyle = "#fff";
                    cCtx.lineWidth = 1;
                    cCtx.stroke();
                } else {
                    if (hudProjItem) hudProjItem.style.display = 'none';
                }

                // Update HUD Text
                hudFreq.textContent = `${freqs[closestIdx].toFixed(2)} kHz (${(freqs[closestIdx] * 1000).toFixed(0)} Hz)`;
                hudPeak.textContent = `${peaks[closestIdx].toFixed(1)} dBFS`;
                hudRMS.textContent = `${rms[closestIdx].toFixed(1)} dBFS`;
            } else {
                const hudProjItem = document.getElementById('hudProjItem');
                if (hudProjItem) hudProjItem.style.display = 'none';
            }

            cCtx.restore();

            // Axes & Labels
            cCtx.strokeStyle = '#30363d';
            cCtx.lineWidth = 1;
            cCtx.strokeRect(padL, padT, plotW, plotH);

            cCtx.fillStyle = '#8b949e';
            cCtx.font = '10px monospace';
            cCtx.textAlign = 'right';
            cCtx.textBaseline = 'middle';

            const dbStep = (curveDbMax - curveDbMin) > 80 ? 20 : 10;
            for (let db = Math.ceil(curveDbMin / dbStep) * dbStep; db <= curveDbMax; db += dbStep) {
                const y = dbToY(db, h, padT, padB);
                cCtx.fillText(`${db}dB`, padL - 6, y);
                cCtx.strokeStyle = 'rgba(255,255,255,0.06)';
                cCtx.beginPath(); cCtx.moveTo(padL, y); cCtx.lineTo(padL + plotW, y); cCtx.stroke();
            }

            cCtx.textAlign = 'center';
            cCtx.textBaseline = 'top';
            const fStep = (curveFMax - curveFMin) > 40 ? 20 : (curveFMax - curveFMin) > 15 ? 5 : 2;
            for (let f = Math.ceil(curveFMin / fStep) * fStep; f <= curveFMax; f += fStep) {
                const x = freqToX(f, w, padL, padR);
                cCtx.fillText(`${f}k`, x, padT + plotH + 6);
                cCtx.strokeStyle = 'rgba(255,255,255,0.06)';
                cCtx.beginPath(); cCtx.moveTo(x, padT); cCtx.lineTo(x, padT + plotH); cCtx.stroke();
            }
        }

        curveCanvas.addEventListener('mousedown', (e) => {
            curveIsDragging = true;
            curveDragStartX = e.clientX;
            curveDragStartY = e.clientY;
            curveInitFMin = curveFMin; curveInitFMax = curveFMax;
            curveInitDbMin = curveDbMin; curveInitDbMax = curveDbMax;
        });

        window.addEventListener('mouseup', () => { curveIsDragging = false; });

        curveCanvas.addEventListener('mousemove', (e) => {
            const rect = curveCanvas.getBoundingClientRect();
            curveMouseX = e.clientX - rect.left;

            if (curveIsDragging && currentAnalysis) {
                const padL = 55, padR = 25, padT = 20, padB = 35;
                const plotW = rect.width - padL - padR;
                const plotH = rect.height - padT - padB;
                const dx = e.clientX - curveDragStartX;
                const dy = e.clientY - curveDragStartY;

                const curFW = curveInitFMax - curveInitFMin;
                const curDbW = curveInitDbMax - curveInitDbMin;

                const df = -(dx / plotW) * curFW;
                const dDb = (dy / plotH) * curDbW;

                curveFMin = Math.max(0, Math.min(currentAnalysis.nyquist_khz * 1.04 - curFW, curveInitFMin + df));
                curveFMax = curveFMin + curFW;

                curveDbMin = Math.max(-175.0, Math.min(0.0 - curDbW, curveInitDbMin + dDb));
                curveDbMax = curveDbMin + curDbW;
            }

            drawCurve(rect.width, rect.height);
        });

        curveCanvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const rect = curveCanvas.getBoundingClientRect();
            const padL = 55, padR = 25, padT = 20, padB = 35;
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
        loadDirectory(currentPath, true);
    </script>
</body>
</html>"""


class ForensicWebHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        cmd = getattr(self, 'command', 'HTTP')
        pth = getattr(self, 'path', '')
        sys.stderr.write(f"[{self.log_date_time_string()}] {cmd} {pth}\n")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            gpu_text = f"⚡ GPU: {html.escape(gpu_engine.device_name)}" if gpu_engine.enabled else "💻 CPU Multi-Thread"
            gpu_cls = "gpu-badge" if gpu_engine.enabled else "gpu-badge cpu-mode"
            page_rendered = HTML_PAGE.replace(
                '<span id="gpuStatusBadge" class="gpu-badge">⚡ GPU Initializing...</span>',
                f'<span id="gpuStatusBadge" class="{gpu_cls}">{gpu_text}</span>'
            )
            self.wfile.write(page_rendered.encode("utf-8"))

        elif path == "/api/browse":
            target = params.get("path", [""])[0]
            fresh_param = params.get("fresh", ["1"])[0] == "1"
            data = get_directory_contents(target, fresh=fresh_param)
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

        elif path == "/api/folder_stream":
            target = params.get("path", [""])[0]
            fresh_param = params.get("fresh", ["1"])[0] == "1"
            if not target or not os.path.exists(target) or not os.path.isdir(target):
                self.send_response(404)
                self.end_headers()
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            q = folder_scan_mgr.add_listener(target, fresh=fresh_param)
            try:
                while True:
                    try:
                        event_type, data = q.get(timeout=25.0)
                        msg = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                        self.wfile.write(msg.encode("utf-8"))
                        self.wfile.flush()
                        if event_type == "scan_complete":
                            break
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            finally:
                folder_scan_mgr.remove_listener(target, q)

        elif path == "/api/analyze":
            target = params.get("path", [""])[0]
            fresh_param = params.get("fresh", ["1"])[0] == "1"
            result = analyze_file_on_demand(target, force_fresh=fresh_param)
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
    global INITIAL_ROOT, ACTIVE_RULES_PATH
    parser = argparse.ArgumentParser(description="Hi-Res Audio Forensic Explorer Web Server")
    parser.add_argument("--root", default=os.getcwd(), help="Initial directory to browse (default: current working directory)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP server port (default: 8765)")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP server host (default: 0.0.0.0)")
    parser.add_argument("--rules", default=None, help="Path to custom provenance rules configuration JSON")
    args = parser.parse_args()

    INITIAL_ROOT = os.path.abspath(os.path.expanduser(args.root))
    if not os.path.exists(INITIAL_ROOT):
        INITIAL_ROOT = os.getcwd()

    if args.rules:
        ACTIVE_RULES_PATH = os.path.abspath(os.path.expanduser(args.rules))

    server_address = (args.host, args.port)
    httpd = ThreadingHTTPServer(server_address, ForensicWebHandler)
    print(f"\n=======================================================")
    print(f"🔬 HI-RES AUDIO FORENSIC EXPLORER (WEB APPLICATION)")
    print(f"=======================================================")
    print(f"Initial Root: {INITIAL_ROOT}")
    print(f"Server URL  : http://localhost:{args.port}")
    print(f"Network URL : http://{args.host}:{args.port}")
    print(f"Mode        : Multi-Threaded Concurrent I/O")
    print(f"Engine      : {'⚡ GPU Acceleration (' + gpu_engine.device_name + ')' if gpu_engine.enabled else '💻 CPU Multi-Thread'}")
    print(f"Precision   : 64-bit Double Precision (Strict float64)")
    if ACTIVE_RULES_PATH:
        print(f"Rules Path  : {ACTIVE_RULES_PATH}")
    print(f"Cache Policy: Fresh Dynamic Re-Analysis upon Folder Entry")
    print(f"Status      : Live with Real-Time Progressive Badge Streaming")
    print(f"=======================================================\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == "__main__":
    main()
