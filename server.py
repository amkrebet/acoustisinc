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
import subprocess
import signal
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
import soundfile as sf
import numpy as np

# Core DSP functions
from analyser import analyze_audio_forensics, encode_spectrogram_and_lookup
from provenance_engine import load_audio_resilient, probe_audio_info_resilient
from gpu_analyser import gpu_engine, analyze_audio_forensics_accelerated
from upsampler import run_upsample_job, WebPromptController


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


def scan_track_badge_fast(filepath, rules_path=None):
    """
    Fast background scanner for sidebar badges (under 1.5s per track).
    Audits a 30s sample for provenance, bit depth, MQA, and DR score without full WebP rendering.
    """
    try:
        data, sr = load_audio_resilient(filepath, dtype='float64', frames=int(192000 * 30))
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        taper_len = min(int(sr * 0.05), len(data) // 10)
        if taper_len > 0:
            taper = np.sin(np.linspace(0, np.pi/2, taper_len))**2
            data[:taper_len] *= taper
            data[-taper_len:] *= taper[::-1]

        spec_db, freqs, peak_dbfs, rms_dbfs, assessment_text, dr_metrics, provenance_info = analyze_audio_forensics(
            data, sr, rules_path=rules_path or ACTIVE_RULES_PATH, filepath=filepath
        )

        return {
            "status": "ok",
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "sr": sr,
            "nyquist_khz": sr / 2000.0,
            "duration_s": len(data) / float(sr),
            "dr_score": dr_metrics.get("dr_score", 0),
            "crest_factor_db": dr_metrics.get("crest_factor_db", 0.0),
            "verdict": provenance_info.get("label", "ANALYZED"),
            "provenance": provenance_info
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


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

        # Progressively scan tracks with lightweight badge auditor
        for f in files:
            with self.lock:
                if self.active_scan_dir != folder_dir:
                    break  # User moved to another folder

            try:
                res = scan_track_badge_fast(f)
                if res.get("status") == "ok":
                    badge_data = extract_badge_data(res)
                    with CACHE_LOCK:
                        BADGE_CACHE[f] = badge_data
                    self._broadcast(folder_dir, "track_badge", badge_data)
            except Exception:
                pass

        self._broadcast(folder_dir, "scan_complete", {"folder": folder_dir, "count": len(files)})


folder_scan_mgr = FolderScanManager()


def find_album_summary_in_dir(target_path):
    """
    Locates an existing album analysis summary or report in the target directory or its mirror directory.
    """
    if not target_path or not os.path.exists(target_path):
        return None

    if os.path.isfile(target_path):
        target_path = os.path.dirname(target_path)

    candidates = [
        ("ALBUM_REPORT.html", "html"),
        ("album_report.html", "html"),
        ("ALBUM_REPORT.md", "markdown"),
        ("album_report.md", "markdown"),
        ("ALBUM_SUMMARY.md", "markdown"),
        ("album_summary.md", "markdown"),
        ("analysis_summary.md", "markdown"),
        ("ANALYSIS_SUMMARY.md", "markdown"),
        ("PROVENANCE_SUMMARY.md", "markdown"),
        ("provenance_summary.md", "markdown"),
        ("report.md", "markdown"),
        ("REPORT.md", "markdown"),
    ]

    dirs_to_check = [target_path]

    # Check paired mirror directory (e.g., source FLAC_music vs upsampled 1xxK_min output)
    if "/FLAC_music/music/" in target_path:
        paired = target_path.replace("/FLAC_music/music/", "/1xxK_min/music/")
        if os.path.exists(paired) and os.path.isdir(paired):
            dirs_to_check.append(paired)
    elif "/1xxK_min/music/" in target_path:
        paired = target_path.replace("/1xxK_min/music/", "/FLAC_music/music/")
        if os.path.exists(paired) and os.path.isdir(paired):
            dirs_to_check.append(paired)

    for d in dirs_to_check:
        for fname, ftype in candidates:
            fp = os.path.join(d, fname)
            if os.path.exists(fp) and os.path.isfile(fp):
                return {
                    "filename": fname,
                    "path": fp,
                    "type": ftype,
                    "size_bytes": os.path.getsize(fp),
                    "is_mirror": (d != target_path)
                }

    # Search for any album report / summary report file in directory
    for d in dirs_to_check:
        try:
            entries = sorted(os.listdir(d))
            for entry in entries:
                el = entry.lower()
                if el.endswith(('_report.html', 'album_report.html')):
                    fp = os.path.join(d, entry)
                    return {
                        "filename": entry,
                        "path": fp,
                        "type": "html",
                        "size_bytes": os.path.getsize(fp),
                        "is_mirror": (d != target_path)
                    }
                elif el.endswith(('_report.md', 'album_report.md', '_summary.md')):
                    fp = os.path.join(d, entry)
                    return {
                        "filename": entry,
                        "path": fp,
                        "type": "markdown",
                        "size_bytes": os.path.getsize(fp),
                        "is_mirror": (d != target_path)
                    }
        except Exception:
            pass

    return None


def derive_default_destination_dir(source_path):
    """
    Computes a smart default mirror destination directory for upsampling.
    e.g., /mnt/PrimaryFS/FLAC_music/music/Artist/Album -> /mnt/PrimaryFS/1xxK_min/music/Artist/Album
    """
    if not source_path:
        return ""
    src_abs = os.path.abspath(source_path)
    if os.path.isfile(src_abs):
        src_dir = os.path.dirname(src_abs)
    else:
        src_dir = src_abs

    if "/FLAC_music/music/" in src_dir:
        return src_dir.replace("/FLAC_music/music/", "/1xxK_min/music/")
    elif "/FLAC_music/" in src_dir:
        return src_dir.replace("/FLAC_music/", "/1xxK_min/")
    elif "/flac_music/" in src_dir.lower():
        idx = src_dir.lower().find("/flac_music/")
        return src_dir[:idx] + "/1xxK_min/" + src_dir[idx + len("/flac_music/"):]
    else:
        return src_dir.rstrip(os.sep) + "_upsampled_min"


class UpsampleJobManager:
    """
    Manages in-process asynchronous background upsampling sessions using shared
    AcoustiSinc DSP and reporting libraries with real-time stage tracking and direct thread synchronization.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.status = "idle"  # idle, running, waiting_for_input, completed, failed, cancelled
        self.job_id = None
        self.mode = "album"  # track, album
        self.src_path = None
        self.dst_dir = None
        self.current_track = ""
        self.track_index = 0
        self.total_tracks = 0
        self.progress_percent = 0.0
        self.stage = "Idle"
        self.log_lines = []  # list of dict: {"time": "HH:MM:SS", "text": "...", "idx": n}
        self.error_message = ""
        self.report_path = None
        self.prompt_data = None
        self.start_time = 0
        self.end_time = 0
        self.controller = None
        self.thread = None
        self._line_counter = 0

    def append_log(self, text):
        with self.lock:
            self._line_counter += 1
            t_str = time.strftime("%H:%M:%S")
            self.log_lines.append({"time": t_str, "text": text, "idx": self._line_counter})
            if len(self.log_lines) > 2500:
                self.log_lines.pop(0)

    def set_stage(self, stage, track_name="", track_idx=0, total_tracks=0, progress_pct=0.0):
        with self.lock:
            self.stage = stage
            if track_name:
                self.current_track = track_name
            if track_idx > 0:
                self.track_index = track_idx
            if total_tracks > 0:
                self.total_tracks = total_tracks
            if progress_pct > 0:
                self.progress_percent = round(progress_pct, 1)

    def set_waiting_prompt(self, prompt_data):
        with self.lock:
            self.status = "waiting_for_input"
            self.prompt_data = prompt_data
            self.current_track = prompt_data.get("track_file", self.current_track)
            self.track_index = prompt_data.get("track_idx", self.track_index)
            self.total_tracks = prompt_data.get("total_tracks", self.total_tracks)
            self.stage = f"Awaiting Input [{self.track_index}/{self.total_tracks}]"

    def is_cancelled(self):
        with self.lock:
            return self.status == "cancelled"

    def start_job(self, config):
        with self.lock:
            if self.status in ["running", "waiting_for_input"]:
                if self.thread and self.thread.is_alive():
                    return {"status": "error", "message": "An upsampling job is already in progress. Please wait for it to finish or click Abort."}
                else:
                    self.status = "idle"

            src_path = config.get("source_path", "").strip()
            if not src_path or not os.path.exists(src_path):
                return {"status": "error", "message": f"Source path not found: {src_path}"}

            is_file = os.path.isfile(src_path)
            self.mode = "track" if is_file else "album"
            self.src_path = os.path.abspath(src_path)

            dst_dir = config.get("dest_dir", "").strip()
            if not dst_dir:
                dst_dir = derive_default_destination_dir(self.src_path)
            self.dst_dir = os.path.abspath(dst_dir)

            try:
                os.makedirs(self.dst_dir, exist_ok=True)
            except Exception as e:
                return {"status": "error", "message": f"Failed to create target directory: {e}"}

            phase = config.get("phase", "min")
            if phase not in ["min", "linear"]: phase = "min"

            cutoff = config.get("cutoff_hz")
            cutoff_val = None
            if cutoff:
                try:
                    c_val = float(cutoff)
                    if c_val > 0: cutoff_val = c_val
                except (ValueError, TypeError):
                    pass

            dither = config.get("dither", "shibata")
            if dither not in ["shibata", "high_rate", "none"]: dither = "shibata"

            mqa = config.get("mqa", "adaptive")
            if mqa not in ["adaptive", "simple", "strip", "ignore"]: mqa = "adaptive"

            overwrite_str = str(config.get("overwrite", "on")).lower()
            overwrite_mode = "on" if overwrite_str in ["on", "true", "1"] else "off"

            interactive = config.get("interactive", True)
            controller_mode = "ask" if interactive else ("auto" if config.get("use_recommended") else "none")

            default_params = {
                "phase_mode": phase,
                "dither_mode": dither,
                "apodizing": bool(config.get("apodizing") or cutoff_val),
                "mqa_mode": mqa,
                "cutoff_hz": cutoff_val,
                "steep": bool(config.get("steep")),
                "overwrite_mode": overwrite_mode
            }

            if is_file:
                total_t = 1
            else:
                try:
                    total_t = len([f for f in os.listdir(self.src_path) if f.lower().endswith(('.flac', '.wav', '.aiff', '.m4a')) and not f.endswith('.WIP')])
                except Exception:
                    total_t = 0

            self.status = "running"
            self.job_id = f"job_{int(time.time())}"
            self.current_track = os.path.basename(self.src_path) if is_file else "Preparing Album Scan..."
            self.track_index = 0
            self.total_tracks = max(1, total_t)
            self.progress_percent = 2.0
            self.stage = "Initializing"
            self.log_lines = []
            self._line_counter = 0
            self.error_message = ""
            self.report_path = None
            self.prompt_data = None
            self.start_time = time.time()
            self.end_time = 0

            self.controller = WebPromptController(self, mode=controller_mode, overwrite_mode=overwrite_mode, default_params=default_params)

            self.thread = threading.Thread(
                target=self._run_job_thread,
                args=(self.src_path, self.dst_dir, default_params, overwrite_mode, self.controller, self.job_id),
                daemon=True
            )
            self.thread.start()

            return {
                "status": "ok",
                "job_id": self.job_id,
                "mode": self.mode,
                "src_path": self.src_path,
                "dst_dir": self.dst_dir,
                "total_tracks": self.total_tracks
            }

    def _run_job_thread(self, src_path, dst_dir, default_params, overwrite_mode, controller, job_id):
        try:
            res = run_upsample_job(
                source_path=src_path,
                target_dir=dst_dir,
                default_params=default_params,
                overwrite_mode=overwrite_mode,
                prompt_ctrl=controller
            )
            with self.lock:
                if self.job_id != job_id:
                    return
                self.end_time = time.time()
                self.prompt_data = None
                if self.status == "cancelled":
                    self.stage = "Cancelled"
                elif res.get("status") == "error":
                    self.status = "failed"
                    self.stage = "Failed"
                    self.error_message = res.get("message", "Processing error")
                else:
                    self.status = "completed"
                    self.progress_percent = 100.0
                    self.stage = "Finished Successfully"
                    if res.get("report_path"):
                        self.report_path = res["report_path"]
                    else:
                        alb_rep = os.path.join(self.dst_dir, "ALBUM_REPORT.html")
                        if os.path.exists(alb_rep):
                            self.report_path = alb_rep
        except Exception as e:
            with self.lock:
                if self.job_id == job_id:
                    self.status = "failed"
                    self.stage = "Failed"
                    self.error_message = str(e)
                    self.end_time = time.time()
                    self.append_log(f"[Fatal Exception]: {e}")

    def send_response(self, resp_dict):
        with self.lock:
            if self.status != "waiting_for_input" or not self.controller:
                return {"status": "error", "message": "No job is currently waiting for input."}
            self.status = "running"
            self.prompt_data = None
            self.stage = f"Resampling [{self.track_index}/{self.total_tracks}] (64-bit Sinc)"
            self.controller.receive_web_response(resp_dict)
            return {"status": "ok"}

    def cancel_job(self):
        with self.lock:
            self.status = "cancelled"
            self.stage = "Cancelling..."
            if self.controller:
                self.controller.receive_web_response({"choice": "q"})
            self.end_time = time.time()
            return {"status": "ok", "message": "Job cancellation requested."}

    def get_status(self, since_idx=0):
        with self.lock:
            filtered_logs = [l for l in self.log_lines if l["idx"] > since_idx]
            elapsed = (self.end_time if self.end_time else time.time()) - self.start_time if self.start_time else 0
            return {
                "status": self.status,
                "job_id": self.job_id,
                "mode": self.mode,
                "src_path": self.src_path,
                "dst_dir": self.dst_dir,
                "current_track": self.current_track,
                "track_index": self.track_index,
                "total_tracks": self.total_tracks,
                "progress_percent": self.progress_percent,
                "stage": self.stage,
                "error_message": self.error_message,
                "report_path": self.report_path,
                "report_url": f"/api/raw_summary?path={urllib.parse.quote(self.report_path)}" if self.report_path else None,
                "prompt_data": self.prompt_data,
                "elapsed_seconds": round(elapsed, 1),
                "logs": filtered_logs,
                "max_log_idx": self.log_lines[-1]["idx"] if self.log_lines else since_idx
            }


upsample_job_mgr = UpsampleJobManager()


def get_directory_contents(target_path, fresh=True):
    """
    Lists subdirectories and audio files in target_path with rich metadata.
    """
    if not target_path or not str(target_path).strip():
        target_path = INITIAL_ROOT
    if str(target_path).startswith("~"):
        target_path = os.path.expanduser(target_path)
    target_path = os.path.abspath(target_path)

    # If target_path is a file, use its parent directory
    if os.path.isfile(target_path):
        target_path = os.path.dirname(target_path)

    # If target_path does not exist, walk up to deepest existing ancestor directory
    while target_path and not os.path.exists(target_path) and target_path != "/":
        target_path = os.path.dirname(target_path)
    if not target_path or not os.path.exists(target_path):
        target_path = INITIAL_ROOT
    if os.path.isfile(target_path):
        target_path = os.path.dirname(target_path)

    if fresh:
        clear_folder_cache(target_path)

    folders = []
    files = []

    try:
        entries = sorted(os.listdir(target_path), key=lambda s: s.lower())
    except Exception as e:
        return {
            "error": f"Cannot read directory: {str(e)}",
            "current_path": target_path,
            "parent_path": os.path.dirname(target_path) if target_path != "/" else "/",
            "breadcrumbs": [{"name": "/", "path": "/"}],
            "folders": [],
            "dirs": [],
            "files": []
        }

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
    album_summary = find_album_summary_in_dir(target_path)

    return {
        "current_path": target_path,
        "parent_path": parent_path,
        "breadcrumbs": parts,
        "folders": folders,
        "dirs": folders,
        "files": files,
        "album_summary": album_summary,
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
        # Decode full audio track with strict 64-bit double precision
        try:
            data, sr = sf.read(filepath, dtype='float64', always_2d=True)
        except Exception:
            data, sr = load_audio_resilient(filepath, dtype='float64')

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

        # Normalised reference frequency limit for aligned visual scaling across 44.1k/48k originals and upsampled 176.4k/192k
        base_family = 44100 if (sr % 44100 == 0 or (sr % 11025 == 0 and sr % 48000 != 0)) else 48000
        target_4x_nyquist_khz = (base_family * 4.0 / 2.0) / 1000.0  # 88.2 kHz for 44.1k family, 96.0 kHz for 48k family
        normalised_max_khz = max(nyquist / 1000.0, target_4x_nyquist_khz)

        result = {
            "status": "ok",
            "filename": os.path.basename(filepath),
            "filepath": filepath,
            "sr": sr,
            "nyquist_khz": nyquist / 1000.0,
            "normalised_max_khz": normalised_max_khz,
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
        .modal-backdrop {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0, 0, 0, 0.78);
            backdrop-filter: blur(5px);
            z-index: 1000;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 24px;
        }
        .modal-content {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            width: 92vw;
            max-width: 1280px;
            height: 88vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            box-shadow: 0 16px 48px rgba(0,0,0,0.85);
            animation: badge-fade-in 0.2s ease;
        }
        .modal-header {
            padding: 12px 20px;
            background: #11141a;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            flex-shrink: 0;
        }
        .modal-body {
            flex: 1;
            overflow: auto;
            padding: 24px;
            background: var(--bg);
        }
        .modal-iframe {
            width: 100%;
            height: 100%;
            min-height: 72vh;
            border: none;
            background: #0a0c10;
            border-radius: 6px;
        }
        .markdown-rendered {
            color: var(--text);
            font-size: 0.88rem;
            line-height: 1.6;
        }
        .markdown-rendered table {
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0 24px;
            font-size: 0.84rem;
        }
        .markdown-rendered th, .markdown-rendered td {
            border: 1px solid #30363d;
            padding: 8px 12px;
            text-align: left;
        }
        .markdown-rendered th {
            background: #161b22;
            color: var(--text-heading);
            font-weight: 600;
        }
        .markdown-rendered tr:nth-child(even) {
            background: rgba(255, 255, 255, 0.02);
        }
        .markdown-rendered h1, .markdown-rendered h2, .markdown-rendered h3 {
            color: var(--text-heading);
            margin: 18px 0 10px;
        }
        .markdown-rendered h1 { font-size: 1.4rem; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
        .markdown-rendered h2 { font-size: 1.15rem; }
        .markdown-rendered h3 { font-size: 0.95rem; }
        .markdown-rendered hr {
            border: 0;
            border-top: 1px solid #30363d;
            margin: 20px 0;
        }
        .markdown-rendered code {
            background: #161b22;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            color: var(--accent-cyan);
            border: 1px solid #30363d;
        }
        .markdown-rendered li {
            margin-bottom: 4px;
        }

        /* Upsample Studio Modal & Spacious Form Styles */
        .modal-upsample-dialog {
            max-width: 900px;
            width: 94vw;
            height: auto;
            max-height: 94vh;
            display: flex;
            flex-direction: column;
            border-radius: 12px;
            border: 1px solid #30363d;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.85), 0 0 1px 1px rgba(255, 255, 255, 0.06);
            background: #0d1117;
        }
        .modal-section-card {
            background: rgba(22, 27, 34, 0.75);
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .modal-section-header {
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--accent-cyan);
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 4px;
        }
        .form-grid-3 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
        }
        .form-grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .form-group.full-width {
            grid-column: 1 / -1;
        }
        .form-label {
            font-size: 0.80rem;
            font-weight: 600;
            color: var(--text-heading);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .form-label-sub {
            font-size: 0.70rem;
            font-weight: normal;
            color: var(--text-muted);
        }
        .form-path-row {
            display: flex;
            align-items: center;
            gap: 8px;
            width: 100%;
        }
        .form-input, .form-select {
            background: #090d13;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 8px 12px;
            color: var(--text-heading);
            font-size: 0.85rem;
            outline: none;
            transition: all 0.15s ease;
            width: 100%;
            height: 40px;
            box-sizing: border-box;
        }
        .form-input:focus, .form-select:focus {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 0 2px rgba(0, 229, 255, 0.15);
        }
        .form-input[readonly] {
            color: #8b949e;
            background: #11151c;
            border-color: #21262d;
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
            font-size: 0.80rem;
        }
        .btn-browse {
            padding: 0 16px;
            height: 40px;
            background: #21262d;
            border: 1px solid var(--border);
            color: var(--accent-cyan);
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            white-space: nowrap;
            font-size: 0.82rem;
            transition: all 0.15s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            flex-shrink: 0;
        }
        .btn-browse:hover {
            border-color: var(--accent-cyan);
            background: rgba(0, 229, 255, 0.15);
        }
        .form-checkbox-card {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 9px 12px;
            background: rgba(13, 17, 23, 0.7);
            border: 1px solid #30363d;
            border-radius: 6px;
            cursor: pointer;
            user-select: none;
            transition: all 0.15s ease;
            height: 40px;
            box-sizing: border-box;
        }
        .form-checkbox-card:hover {
            border-color: var(--accent-cyan);
            background: rgba(0, 229, 255, 0.05);
        }
        .preset-pill-group {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 6px;
        }
        .preset-pill {
            background: #21262d;
            border: 1px solid var(--border);
            color: var(--text);
            padding: 5px 12px;
            font-size: 0.76rem;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .preset-pill:hover {
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
            background: rgba(0, 229, 255, 0.08);
        }
        .preset-pill.active {
            background: rgba(0, 229, 255, 0.15);
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
            font-weight: 600;
        }
        .cutoff-chip-group {
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
            margin-top: 4px;
        }
        .cutoff-chip {
            background: #161b22;
            border: 1px solid #30363d;
            color: var(--text-muted);
            padding: 3px 8px;
            font-size: 0.72rem;
            font-weight: 500;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.15s ease;
            user-select: none;
        }
        .cutoff-chip:hover {
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
            background: rgba(0, 229, 255, 0.08);
        }
        .cutoff-chip.active {
            background: rgba(0, 229, 255, 0.15);
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
            font-weight: 600;
        }

        /* Tile-Oriented DSP Studio Grid & Control Styles */
        .dsp-tile-card {
            background: #1c2128;
            border: 1px solid #373e47;
            border-radius: 10px;
            padding: 14px 16px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45);
            transition: all 0.2s ease;
        }
        .dsp-tile-card:hover {
            border-color: rgba(0, 229, 255, 0.45);
            box-shadow: 0 6px 22px rgba(0, 0, 0, 0.6), 0 0 1px rgba(0, 229, 255, 0.25);
        }
        .dsp-tile-header {
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: var(--accent-cyan);
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid rgba(48, 54, 61, 0.6);
            padding-bottom: 6px;
        }
        .tile-segmented-grid {
            display: grid;
            gap: 6px;
        }
        .tile-segmented-grid.cols-2 {
            grid-template-columns: 1fr 1fr;
        }
        .tile-segmented-grid.cols-3 {
            grid-template-columns: 1fr 1fr 1fr;
        }
        .tile-segmented-grid.cols-4 {
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        }
        .tile-option-btn {
            background: #11161f;
            border: 1px solid #333a45;
            border-radius: 6px;
            padding: 8px 10px;
            cursor: pointer;
            display: flex;
            flex-direction: column;
            gap: 3px;
            transition: all 0.15s ease;
            user-select: none;
            text-align: left;
            position: relative;
        }
        .tile-option-btn:hover {
            border-color: rgba(0, 229, 255, 0.6);
            background: rgba(0, 229, 255, 0.07);
        }
        .tile-option-btn.active {
            border-color: var(--accent-cyan);
            background: rgba(0, 229, 255, 0.15);
            box-shadow: 0 0 10px rgba(0, 229, 255, 0.25), inset 0 0 0 1px var(--accent-cyan);
        }
        .tile-option-btn.active::after {
            content: "✓";
            position: absolute;
            top: 6px;
            right: 8px;
            font-size: 0.72rem;
            color: var(--accent-cyan);
            font-weight: 900;
        }
        .tile-option-title {
            font-size: 0.80rem;
            font-weight: 700;
            color: var(--text-heading);
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .tile-option-btn.active .tile-option-title {
            color: var(--accent-cyan);
        }
        .tile-option-sub {
            font-size: 0.68rem;
            color: var(--text-muted);
            line-height: 1.25;
        }
        .phase-ab-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }
        .phase-card-option {
            background: #11161f;
            border: 1px solid #333a45;
            border-radius: 6px;
            padding: 8px 10px;
            cursor: pointer;
            display: flex;
            flex-direction: column;
            gap: 4px;
            transition: all 0.15s ease;
            user-select: none;
            position: relative;
        }
        .phase-card-option:hover {
            border-color: var(--accent-cyan);
            background: rgba(0, 229, 255, 0.07);
        }
        .phase-card-option.active {
            border-color: var(--accent-cyan);
            background: rgba(0, 229, 255, 0.15);
            box-shadow: 0 0 10px rgba(0, 229, 255, 0.25), inset 0 0 0 1px var(--accent-cyan);
        }
        .phase-card-option.active::after {
            content: "✓";
            position: absolute;
            top: 6px;
            right: 8px;
            font-size: 0.72rem;
            color: var(--accent-cyan);
            font-weight: 900;
        }
        .phase-card-title {
            font-size: 0.80rem;
            font-weight: 700;
            color: var(--text-heading);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .phase-card-sub {
            font-size: 0.68rem;
            color: var(--text-muted);
            line-height: 1.3;
        }
        .cutoff-slider-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-top: 2px;
        }
        .cutoff-slider {
            flex: 1;
            -webkit-appearance: none;
            appearance: none;
            height: 6px;
            border-radius: 3px;
            background: #21262d;
            outline: none;
            cursor: pointer;
        }
        .cutoff-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--accent-cyan);
            cursor: pointer;
            box-shadow: 0 0 8px rgba(0, 229, 255, 0.6);
            transition: transform 0.1s ease;
        }
        .cutoff-slider::-webkit-slider-thumb:hover {
            transform: scale(1.2);
        }
        .cutoff-slider::-moz-range-thumb {
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--accent-cyan);
            cursor: pointer;
            box-shadow: 0 0 8px rgba(0, 229, 255, 0.6);
        }

        /* Interactive Decision Prompt & UI Styles */
        .prompt-audit-card {
            background: #1c2128;
            border: 1px solid #373e47;
            border-left: 5px solid var(--accent-cyan);
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            font-size: 0.84rem;
            animation: badge-fade-in 0.2s ease;
        }
        .decision-btn-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 14px;
        }
        .btn-decision {
            padding: 11px 14px;
            font-size: 0.84rem;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 4px;
            transition: all 0.15s;
            outline: none;
            text-align: center;
        }
        .btn-decision-sub {
            font-size: 0.70rem;
            font-weight: normal;
            opacity: 0.85;
        }
        .btn-decision-primary {
            background: rgba(0, 229, 255, 0.18);
            border: 1px solid var(--accent-cyan);
            color: var(--accent-cyan);
        }
        .btn-decision-primary:hover {
            background: rgba(0, 229, 255, 0.32);
        }
        .btn-decision-auto {
            background: rgba(0, 230, 118, 0.18);
            border: 1px solid #00e676;
            color: #00e676;
        }
        .btn-decision-auto:hover {
            background: rgba(0, 230, 118, 0.32);
        }
        .btn-decision-lock {
            background: rgba(186, 104, 200, 0.18);
            border: 1px solid #ce93d8;
            color: #ce93d8;
        }
        .btn-decision-lock:hover {
            background: rgba(186, 104, 200, 0.32);
        }
        .btn-decision-skip {
            background: rgba(255, 171, 0, 0.15);
            border: 1px solid #ffab00;
            color: #ffab00;
        }
        .btn-decision-skip:hover {
            background: rgba(255, 171, 0, 0.28);
        }
        .btn-decision-abort {
            background: rgba(255, 23, 68, 0.15);
            border: 1px solid #ff5252;
            color: #ff5252;
        }
        .btn-decision-abort:hover {
            background: rgba(255, 23, 68, 0.28);
        }
        .zoom-btn-group {
            display: inline-flex;
            align-items: center;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            overflow: hidden;
        }
        .zoom-axis-label {
            font-size: 0.70rem;
            font-weight: 700;
            color: var(--text-muted);
            padding: 3px 6px;
            background: rgba(255, 255, 255, 0.03);
            border-right: 1px solid #30363d;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        .btn-zoom {
            background: transparent;
            border: none;
            color: var(--text);
            padding: 3px 8px;
            font-size: 0.85rem;
            font-weight: 700;
            cursor: pointer;
            line-height: 1;
            transition: all 0.12s;
        }
        .btn-zoom:hover {
            background: rgba(0, 229, 255, 0.2);
            color: var(--accent-cyan);
        }
        .btn-zoom:active {
            background: rgba(0, 229, 255, 0.4);
        }
        .folder-picker-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 9px 12px;
            border-bottom: 1px solid #21262d;
            cursor: pointer;
            transition: background 0.15s;
            border-radius: 4px;
        }
        .folder-picker-item:hover {
            background: rgba(0, 229, 255, 0.1);
        }
        @keyframes pulse-glow {
            0% { opacity: 0.6; transform: scale(0.92); }
            50% { opacity: 1; transform: scale(1.1); }
            100% { opacity: 0.6; transform: scale(0.92); }
        }
        .drawer-terminal-line {
            white-space: pre-wrap;
            word-break: break-all;
            line-height: 1.35;
        }
        .drawer-terminal-line.highlight {
            color: var(--accent-cyan);
            font-weight: 600;
        }
        .drawer-terminal-line.error {
            color: #ff5252;
            font-weight: 600;
        }

        /* AcoustiSinc Header Status & Hover Dropdown Telemetry */
        .header-upsample-container {
            position: relative;
            display: inline-flex;
            align-items: center;
        }
        .header-upsample-widget {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: #11141a;
            border: 1px solid rgba(0, 229, 255, 0.35);
            border-radius: 6px;
            padding: 3px 10px;
            cursor: pointer;
            transition: all 0.2s ease;
            user-select: none;
        }
        .header-upsample-widget:hover,
        .header-upsample-container:hover .header-upsample-widget,
        .header-upsample-container.is-open .header-upsample-widget {
            border-color: var(--accent-cyan);
            background: #161b22;
            box-shadow: 0 0 12px rgba(0, 229, 255, 0.25);
        }
        .upsample-status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-cyan);
            box-shadow: 0 0 8px var(--accent-cyan);
            flex-shrink: 0;
        }
        .upsample-status-title {
            font-size: 0.76rem;
            font-weight: 600;
            color: var(--text-heading);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            line-height: 1.2;
        }
        .upsample-mini-bar {
            width: 50px;
            height: 5px;
            background: #0d1117;
            border-radius: 3px;
            overflow: hidden;
            border: 1px solid #30363d;
            flex-shrink: 0;
        }
        .upsample-mini-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #00e5ff, #00e676);
            transition: width 0.3s ease;
        }
        .upsample-popover-card {
            position: absolute;
            top: calc(100% + 6px);
            right: 0;
            width: 380px;
            max-width: 90vw;
            background: rgba(13, 17, 23, 0.97);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(0, 229, 255, 0.35);
            border-radius: 8px;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.85), 0 0 1px rgba(0, 229, 255, 0.5);
            z-index: 10000;
            display: none;
            flex-direction: column;
            overflow: hidden;
            animation: popover-fade 0.18s ease-out;
        }
        @keyframes popover-fade {
            from { opacity: 0; transform: translateY(-4px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .header-upsample-container:hover .upsample-popover-card,
        .header-upsample-container.is-open .upsample-popover-card {
            display: flex;
        }
        .popover-header {
            padding: 10px 14px;
            background: #161b22;
            border-bottom: 1px solid #30363d;
        }
        .popover-progress-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 6px;
        }
        .popover-progress-bar {
            flex: 1;
            height: 6px;
            background: #0d1117;
            border-radius: 3px;
            overflow: hidden;
            border: 1px solid #30363d;
        }
        .popover-progress-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #00e5ff, #00e676);
            transition: width 0.3s ease;
        }
        .popover-body {
            padding: 8px 12px;
            max-height: 240px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 4px;
            background: #0a0c10;
        }
        .popover-stream-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.70rem;
            font-weight: 700;
            color: var(--text-muted);
            letter-spacing: 0.5px;
            padding: 2px 2px 4px 2px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            margin-bottom: 4px;
        }
        .popover-events-list {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .telemetry-event {
            display: flex;
            align-items: flex-start;
            gap: 8px;
            padding: 5px 8px;
            border-radius: 4px;
            background: rgba(22, 27, 34, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.04);
            font-size: 0.74rem;
            line-height: 1.35;
        }
        .telemetry-event.event-gpu {
            border-left: 3px solid var(--accent-cyan);
            background: rgba(0, 229, 255, 0.04);
        }
        .telemetry-event.event-dither {
            border-left: 3px solid #ce93d8;
            background: rgba(206, 147, 216, 0.04);
        }
        .telemetry-event.event-io {
            border-left: 3px solid #80cbc4;
            background: rgba(128, 203, 196, 0.04);
        }
        .telemetry-event.event-headroom {
            border-left: 3px solid var(--accent-yellow);
            background: rgba(255, 234, 0, 0.04);
        }
        .telemetry-event.event-report {
            border-left: 3px solid #00e676;
            background: rgba(0, 230, 118, 0.04);
        }
        .telemetry-event.event-error {
            border-left: 3px solid #ff5252;
            background: rgba(255, 82, 82, 0.08);
        }
        .telemetry-time {
            font-family: ui-monospace, SFMono-Regular, monospace;
            font-size: 0.68rem;
            color: var(--text-muted);
            flex-shrink: 0;
            margin-top: 1px;
        }
        .telemetry-tag {
            font-size: 0.65rem;
            font-weight: 700;
            padding: 1px 5px;
            border-radius: 3px;
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-heading);
            flex-shrink: 0;
        }
        .telemetry-text {
            color: #c9d1d9;
            word-break: break-word;
            flex: 1;
        }
        .telemetry-empty {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.74rem;
            padding: 20px 10px;
        }
        .popover-footer {
            padding: 8px 12px;
            background: #161b22;
            border-top: 1px solid #30363d;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 6px;
        }
        .btn-popover {
            padding: 4px 10px;
            font-size: 0.72rem;
            font-weight: 600;
            border-radius: 4px;
            cursor: pointer;
            border: 1px solid var(--border);
            background: #21262d;
            color: var(--text);
            transition: all 0.15s;
        }
        .btn-popover-review {
            background: rgba(255, 234, 0, 0.2);
            color: var(--accent-yellow);
            border-color: var(--accent-yellow);
            font-weight: 700;
        }
        .btn-popover-report {
            background: rgba(0, 229, 255, 0.2);
            color: var(--accent-cyan);
            border-color: var(--accent-cyan);
            font-weight: 700;
        }
        .btn-popover-abort {
            background: rgba(255, 23, 68, 0.15);
            color: #ff5252;
            border-color: rgba(255, 23, 68, 0.4);
        }
        .btn-popover-dismiss {
            background: rgba(0, 230, 118, 0.15);
            color: #00e676;
            border-color: rgba(0, 230, 118, 0.4);
        }
        .btn-popover-link {
            background: none;
            border: none;
            color: var(--text-muted);
            font-size: 0.68rem;
            cursor: pointer;
            text-decoration: underline;
            padding: 2px 4px;
        }
        .btn-popover-link:hover {
            color: var(--accent-cyan);
        }
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
            <button id="headerBtnUpsample" class="btn-bookmark" style="display: none; background: rgba(0, 230, 118, 0.15); border-color: rgba(0, 230, 118, 0.45); color: #00e676; font-weight: 600;" onclick="openUpsampleModalForCurrentFolder()" title="Launch AcoustiSinc Upsampling Studio for this folder">⚡ Upsample Album</button>

            <!-- Active Header Status Widget with UI-Native Telemetry Dropdown -->
            <div id="headerUpsampleContainer" class="header-upsample-container" style="display: none;">
                <div id="headerUpsampleWidget" class="header-upsample-widget" onclick="toggleUpsampleDropdown(event)" title="Click or hover to view real-time DSP telemetry stream">
                    <span id="headerUpsampleDot" class="upsample-status-dot"></span>
                    <div style="display: flex; flex-direction: column; min-width: 0; max-width: 220px;">
                        <span id="headerUpsampleTitle" class="upsample-status-title">⚡ Upsampling...</span>
                        <span id="headerUpsampleSub" style="font-size: 0.65rem; color: var(--text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Initializing</span>
                    </div>
                    <span id="headerUpsampleBadge" class="brand-badge" style="font-size: 0.68rem; padding: 1px 5px;">0%</span>
                    <div class="upsample-mini-bar">
                        <div id="headerUpsampleFill" class="upsample-mini-fill"></div>
                    </div>
                    <span id="headerUpsampleArrow" style="font-size: 0.65rem; color: var(--text-muted); margin-left: 2px;">▼</span>
                </div>

                <!-- Dropdown Popover Card (Revealed on Hover / Click) -->
                <div id="headerUpsampleDropdown" class="upsample-popover-card" onclick="event.stopPropagation()">
                    <div class="popover-header">
                        <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
                            <div style="display: flex; align-items: center; gap: 6px; min-width: 0;">
                                <span style="font-size: 0.85rem;">⚡</span>
                                <span id="popoverHeaderTitle" style="font-weight: 700; font-size: 0.82rem; color: var(--text-heading); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">AcoustiSinc Studio</span>
                            </div>
                            <span id="popoverTimeElapsed" style="font-size: 0.70rem; color: var(--text-muted); font-family: monospace; flex-shrink: 0;">0:00 elapsed</span>
                        </div>
                        <div id="popoverCurrentTrack" style="font-size: 0.76rem; color: var(--accent-cyan); font-weight: 600; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                            Preparing album scan...
                        </div>
                        <div class="popover-progress-row">
                            <div class="popover-progress-bar">
                                <div id="popoverProgressFill" class="popover-progress-fill"></div>
                            </div>
                            <span id="popoverProgressPct" style="font-size: 0.72rem; font-weight: 700; color: var(--text-heading); min-width: 32px; text-align: right;">0%</span>
                        </div>
                    </div>

                    <!-- UI Native Telemetry Stream -->
                    <div class="popover-body">
                        <div class="popover-stream-header">
                            <span>📡 Real-Time DSP Telemetry</span>
                            <span id="popoverEventCount" style="font-size: 0.68rem; color: var(--text-muted);">0 events</span>
                        </div>
                        <div id="popoverEventStream" class="popover-events-list">
                            <div class="telemetry-empty">Waiting for telemetry stream...</div>
                        </div>
                    </div>

                    <!-- Popover Actions Footer -->
                    <div class="popover-footer">
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <button class="btn-popover btn-popover-review" id="popoverReviewBtn" onclick="reopenActivePromptModal()" style="display: none;">
                                ⚠️ Review
                            </button>
                            <button class="btn-popover btn-popover-report" id="popoverReportBtn" onclick="viewGeneratedReport()" style="display: none;">
                                📊 Report
                            </button>
                            <button class="btn-popover btn-popover-abort" id="popoverAbortBtn" onclick="cancelUpsampleJob()">
                                ⛔ Abort
                            </button>
                            <button class="btn-popover btn-popover-dismiss" id="popoverDismissBtn" onclick="dismissUpsampleHeaderWidget()" style="display: none;">
                                ✓ Dismiss
                            </button>
                        </div>
                        <button class="btn-popover-link" onclick="toggleUpsampleLogModal()" title="View Raw Terminal Logs">
                            📜 Raw Logs
                        </button>
                    </div>
                </div>
            </div>

            <button id="headerBtnSummary" class="btn-bookmark" style="display: none; background: rgba(0, 229, 255, 0.15); border-color: rgba(0, 229, 255, 0.45); color: var(--accent-cyan); font-weight: 600;" onclick="openAlbumSummary()" title="View Album Analysis Summary">📋 Album Summary</button>
            <button class="btn-bookmark" onclick="loadDirectory('/mnt/PrimaryFS/FLAC_music/music', true)">📁 Music Library</button>
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
                <div style="display: flex; gap: 6px; align-items: center; margin-bottom: 8px;">
                    <input type="text" class="search-box" id="pathBar" placeholder="Path..." value="" style="margin-bottom: 0; flex: 1;" onkeydown="if(event.key==='Enter') navigateToPathBar()" onchange="navigateToPathBar()" />
                    <button class="btn-bookmark" onclick="openFolderPickerForMainPath()" title="Browse Directory" style="padding: 5px 9px; font-size: 0.78rem; white-space: nowrap; height: 32px; display: flex; align-items: center; gap: 4px;">📁 Browse</button>
                </div>
                <input type="text" class="search-box" id="searchBox" placeholder="Filter albums & tracks..." />
                <div id="albumSummaryBanner" style="display: none; padding: 6px 10px; background: rgba(0, 229, 255, 0.08); border: 1px solid rgba(0, 229, 255, 0.3); border-radius: 5px; justify-content: space-between; align-items: center; gap: 8px;">
                    <div style="display: flex; align-items: center; gap: 6px; overflow: hidden; min-width: 0;">
                        <span style="font-size: 0.85rem; flex-shrink: 0;">📋</span>
                        <span id="albumSummaryBannerLabel" style="font-size: 0.75rem; font-weight: 600; color: var(--accent-cyan); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Album Analysis Summary</span>
                    </div>
                    <button class="btn-bookmark" style="padding: 2px 8px; font-size: 0.72rem; flex-shrink: 0; background: var(--surface); color: var(--accent-cyan); border-color: var(--accent-cyan);" onclick="openAlbumSummary()">View</button>
                </div>
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
                                <div id="actionRecButtons" style="display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap;">
                                    <button class="btn-bookmark" style="background: rgba(0, 230, 118, 0.15); color: #00e676; border-color: rgba(0, 230, 118, 0.4); font-weight: 600; font-size: 0.78rem; padding: 4px 10px;" onclick="quickUpsampleCurrentTrack()">⚡ Upsample Track (Recommended)</button>
                                    <button class="btn-bookmark" style="background: rgba(0, 229, 255, 0.12); color: var(--accent-cyan); border-color: rgba(0, 229, 255, 0.35); font-weight: 600; font-size: 0.78rem; padding: 4px 10px;" onclick="openUpsampleModalForTrack()">⚙️ Custom Track Recipe...</button>
                                    <button class="btn-bookmark" style="background: rgba(255, 234, 0, 0.12); color: var(--accent-yellow); border-color: rgba(255, 234, 0, 0.35); font-weight: 600; font-size: 0.78rem; padding: 4px 10px;" onclick="openUpsampleModalForCurrentFolder()">⚡ Upsample Entire Album...</button>
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
                        <div style="display: flex; gap: 6px; align-items: center; flex-wrap: wrap;">
                            <div class="zoom-btn-group" title="Time (X-Axis) Zoom: Shift + Wheel or Click">
                                <span class="zoom-axis-label">X: Time</span>
                                <button class="btn-zoom" onclick="zoomSpecX(1.0/1.3)" title="Zoom Out Time (X)">−</button>
                                <button class="btn-zoom" onclick="zoomSpecX(1.3)" title="Zoom In Time (X)">+</button>
                            </div>
                            <div class="zoom-btn-group" title="Frequency (Y-Axis) Zoom: Ctrl + Wheel or Click">
                                <span class="zoom-axis-label">Y: Freq</span>
                                <button class="btn-zoom" onclick="zoomSpecY(1.0/1.3)" title="Zoom Out Frequency (Y)">−</button>
                                <button class="btn-zoom" onclick="zoomSpecY(1.3)" title="Zoom In Frequency (Y)">+</button>
                            </div>
                            <button class="btn-bookmark" onclick="setSpecPreset('native')" title="Fit to Native Container Bandwidth">Native Nyquist</button>
                            <button class="btn-bookmark" onclick="resetSpecZoom()" title="Reset to Full Normalised Bandwidth">Full Band</button>
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
                        <div style="display: flex; gap: 6px; align-items: center; flex-wrap: wrap;">
                            <div class="zoom-btn-group" title="Frequency (X-Axis) Zoom: Shift + Wheel or Click">
                                <span class="zoom-axis-label">X: Freq</span>
                                <button class="btn-zoom" onclick="zoomCurveX(1.0/1.3)" title="Zoom Out Frequency (X)">−</button>
                                <button class="btn-zoom" onclick="zoomCurveX(1.3)" title="Zoom In Frequency (X)">+</button>
                            </div>
                            <div class="zoom-btn-group" title="Amplitude (Y-Axis) Zoom: Ctrl + Wheel or Click">
                                <span class="zoom-axis-label">Y: dBFS</span>
                                <button class="btn-zoom" onclick="zoomCurveY(1.0/1.3)" title="Zoom Out dB Level (Y)">−</button>
                                <button class="btn-zoom" onclick="zoomCurveY(1.3)" title="Zoom In dB Level (Y)">+</button>
                            </div>
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


    <!-- Folder Browser / Directory Picker Modal -->
    <div id="folderPickerModal" class="modal-backdrop" style="display: none; z-index: 10050;" onclick="if(event.target===this) closeFolderPicker()">
        <div class="modal-content" style="max-width: 720px; height: 78vh; display: flex; flex-direction: column; border-radius: 12px; border: 1px solid #30363d; box-shadow: 0 20px 50px rgba(0,0,0,0.85); background: #0d1117;">
            <div class="modal-header" style="padding: 14px 20px; border-bottom: 1px solid #30363d;">
                <div style="display: flex; align-items: center; gap: 10px; min-width: 0;">
                    <span style="font-size: 1.1rem; font-weight: 700; color: var(--text-heading);">📁 Select Directory</span>
                </div>
                <button class="btn-bookmark" onclick="closeFolderPicker()" style="font-size: 1.2rem; padding: 2px 10px; line-height: 1;">&times;</button>
            </div>
            
            <!-- Breadcrumbs Navigation Bar -->
            <div style="padding: 8px 16px; background: #161b22; border-bottom: 1px solid #30363d; font-size: 0.80rem; overflow-x: auto; white-space: nowrap;" id="folderPickerBreadcrumbs">
                <span class="crumb" onclick="loadFolderPickerDirectory('/')">🏠 /</span>
            </div>

            <!-- Path Bar & Action Buttons -->
            <div style="padding: 10px 16px; background: #11141a; border-bottom: 1px solid #30363d; display: flex; gap: 8px; align-items: center;">
                <input type="text" id="folderPickerPathBar" class="form-input" style="font-family: monospace; font-size: 0.82rem; height: 36px;" onkeydown="if(event.key==='Enter') loadFolderPickerDirectory(this.value.trim())" />
                <button class="btn-bookmark" onclick="loadFolderPickerDirectory(document.getElementById('folderPickerPathBar').value.trim())" style="padding: 0 14px; height: 36px;">Go</button>
                <button class="btn-bookmark" onclick="createNewFolderInPicker()" style="padding: 0 14px; height: 36px; color: var(--accent-cyan); border-color: rgba(0, 229, 255, 0.4); white-space: nowrap;">+ New Folder</button>
            </div>

            <!-- Folder Items List -->
            <div class="modal-body" style="flex: 1; overflow-y: auto; padding: 8px 12px;" id="folderPickerList">
                <div style="padding: 28px; text-align: center; color: var(--text-muted);">
                    <div class="spinner" style="margin: 0 auto 10px;"></div>
                    Loading directory contents...
                </div>
            </div>

            <!-- Bottom Action Footer -->
            <div style="padding: 12px 18px; background: #161b22; border-top: 1px solid #30363d; display: flex; justify-content: space-between; align-items: center; gap: 10px;">
                <button class="btn-bookmark" onclick="closeFolderPicker()" style="padding: 7px 16px;">Cancel</button>
                <button class="btn-bookmark" onclick="selectFolderPickerChoice()" style="background: rgba(0, 230, 118, 0.2); border-color: rgba(0, 230, 118, 0.55); color: #00e676; font-weight: 700; padding: 7px 20px; font-size: 0.88rem;">✓ Select Current Folder</button>
            </div>
        </div>
    </div>
    <div id="albumSummaryModal" class="modal-backdrop" style="display: none;" onclick="if(event.target===this) closeAlbumSummaryModal()">
        <div class="modal-content">
            <div class="modal-header">
                <div style="display: flex; align-items: center; gap: 10px; min-width: 0;">
                    <span style="font-size: 1.05rem; font-weight: 700; color: var(--text-heading); white-space: nowrap;">📋 Album Analysis Summary</span>
                    <span id="modalSummaryBadge" class="brand-badge" style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">ALBUM_REPORT</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px; flex-shrink: 0;">
                    <button class="btn-bookmark" id="btnCopySummary" onclick="copyAlbumSummaryText()">📋 Copy</button>
                    <button class="btn-bookmark" id="modalOpenExternalBtn" onclick="openAlbumSummaryExternal()" style="display: none;">↗️ Open Tab</button>
                    <button class="btn-bookmark" onclick="closeAlbumSummaryModal()" style="font-size: 1.2rem; padding: 2px 10px; line-height: 1;">&times;</button>
                </div>
            </div>
            <div class="modal-body" id="modalSummaryBody">
                <div style="text-align: center; padding: 40px; color: var(--text-muted);">
                    <div class="spinner" style="margin: 0 auto 10px;"></div>
                    Loading album analysis summary...
                </div>
            </div>
        </div>
    </div>

    <!-- Upsample Configuration & Interactive Decision Modal -->
    <div id="upsampleModal" class="modal-backdrop" style="display: none;" onclick="if(event.target===this) closeUpsampleModal()">
        <div class="modal-content modal-upsample-dialog">
            <div class="modal-header" style="padding: 14px 22px; border-bottom: 1px solid #30363d;">
                <div style="display: flex; align-items: center; gap: 10px; min-width: 0;">
                    <span style="font-size: 1.15rem; font-weight: 700; color: var(--text-heading); white-space: nowrap;">⚡ AcoustiSinc Interactive Upsampling Studio</span>
                    <span id="upsampleModalScopeBadge" class="brand-badge" style="background: rgba(0, 230, 118, 0.15); border-color: rgba(0, 230, 118, 0.4); color: #00e676;">ALBUM BATCH</span>
                </div>
                <button class="btn-bookmark" onclick="closeUpsampleModal()" style="font-size: 1.2rem; padding: 2px 10px; line-height: 1;">&times;</button>
            </div>
            <div class="modal-body" style="padding: 20px 24px; overflow-y: auto;">
                
                <!-- Preparing & Headroom Scan View (Shown while Track 1 is being audited) -->
                <div id="upsampleModalPreparingView" style="display: none; padding: 40px 20px; text-align: center;">
                    <div class="spinner" style="width: 52px; height: 52px; border-width: 4px; margin: 0 auto 20px; border-top-color: var(--accent-cyan);"></div>
                    <h3 id="preparingModalTitle" style="color: var(--text-heading); font-size: 1.2rem; margin-bottom: 10px; font-weight: 700;">🔬 Auditing Audio Master & Scanning Headroom...</h3>
                    <p id="preparingModalDesc" style="color: var(--text-muted); font-size: 0.88rem; max-width: 520px; margin: 0 auto 24px; line-height: 1.55;">
                        Performing 64-bit double-precision FFT frequency analysis, detecting brickwall reconstruction knee frequencies, and formulating optimal DSP upsampling recipe...
                    </p>
                    <div style="display: inline-flex; align-items: center; gap: 10px; background: #161b22; padding: 8px 18px; border-radius: 20px; border: 1px solid #30363d; font-size: 0.82rem; color: var(--accent-yellow);">
                        <span style="display: inline-block; width: 9px; height: 9px; border-radius: 50%; background: var(--accent-yellow); box-shadow: 0 0 8px var(--accent-yellow); animation: pulse-glow 1s infinite;"></span>
                        <span id="preparingModalStatusText">Auditing Track 1 Spectrum & Inter-Sample Headroom</span>
                    </div>
                    <div style="margin-top: 32px;">
                        <button class="btn-decision btn-decision-abort" onclick="cancelUpsampleJob()" style="max-width: 220px; margin: 0 auto; padding: 10px 18px; display: inline-flex;">
                            <span style="font-size: 0.90rem;">⛔ [Q] Abort Upsampling</span>
                        </button>
                    </div>
                </div>

                <!-- Interactive Recipe Recommendation & Decision Controls View -->
                <div id="upsampleModalInteractiveView">
                    <!-- 1. Forensic Audit & Recommendation Details Card -->
                    <div id="promptAuditCard" class="prompt-audit-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap;">
                            <span style="font-size: 0.78rem; font-weight: 700; color: var(--accent-cyan); letter-spacing: 0.5px;">🔬 FORENSIC AUDIT & RECIPE RECOMMENDATION</span>
                            <span id="promptProvLabel" class="provenance-tag badge-provenance-native" style="font-size: 0.72rem; padding: 2px 10px;">RECOMMENDED RECIPE</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: baseline; gap: 8px; flex-wrap: wrap; margin-top: 2px;">
                            <div id="promptTrackTitle" style="font-weight: 700; font-size: 1.05rem; color: var(--text-heading);">Track Name</div>
                            <div id="promptFormatMeta" style="font-size: 0.78rem; color: var(--text-muted); font-family: monospace;">44.1 kHz • 2 Channels • 24-bit PCM</div>
                        </div>
                        <div id="promptMetricsRow" style="font-size: 0.78rem; color: #8b949e;">Cutoff Knee: -- | Stopband: --</div>
                        
                        <div style="background: rgba(0, 229, 255, 0.06); border: 1px solid rgba(0, 229, 255, 0.35); border-radius: 6px; padding: 10px 14px; margin: 4px 0; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                            <div style="flex: 1; min-width: 240px;">
                                <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 3px;">
                                    <span style="font-size: 0.76rem; font-weight: 700; color: var(--accent-cyan);">Audited Recipe:</span>
                                    <span id="promptActionDesc" style="font-size: 0.84rem; font-weight: 600; color: var(--text-heading);">Direct Sinc Resampling</span>
                                </div>
                                <div style="display: flex; align-items: center; gap: 6px; margin-top: 4px;">
                                    <span style="font-size: 0.74rem; color: var(--text-muted);">DSP CLI Flag:</span>
                                    <code class="action-code" id="promptDspFlag">--phase min --dither shibata</code>
                                </div>
                            </div>
                            <button type="button" class="btn-bookmark" id="btnDirectApplyRec" onclick="applyForensicRecommendationPreset()" style="background: rgba(0, 229, 255, 0.22); border: 1px solid var(--accent-cyan); color: var(--accent-cyan); font-weight: 700; font-size: 0.82rem; padding: 7px 16px; border-radius: 6px; cursor: pointer; display: flex; align-items: center; gap: 6px; box-shadow: 0 0 12px rgba(0, 229, 255, 0.25);" title="Apply audited recipe to all tiles [R]">
                                ✨ [R] Apply Recommended Recipe
                            </button>
                        </div>

                        <div style="font-size: 0.80rem; color: var(--text); line-height: 1.45; background: rgba(0, 0, 0, 0.25); border-radius: 6px; padding: 8px 12px;">
                            <span style="font-weight: 600; color: var(--accent-yellow);">💡 Technical Rationale: </span>
                            <span id="promptTechnicalRationale">Applies optimal 64-bit double precision sinc filter.</span>
                        </div>
                    </div>

                    <!-- Presets Pill Bar (for quick configurations) -->
                    <div class="preset-pill-group" id="presetPillGroup" style="margin-bottom: 12px;">
                        <button class="preset-pill active" id="presetBtnRecommended" onclick="applyForensicRecommendationPreset()">✨ [R] Forensic Recommendation</button>
                        <button class="preset-pill" id="presetBtnAudiophile4x" onclick="applyDefaultPreset('4x')">🎧 Standard 4x Audiophile</button>
                        <button class="preset-pill" id="presetBtnApodizing" onclick="applyDefaultPreset('apod')">🛡️ Apodizing Ringing-Filter (22.05k)</button>
                    </div>

                    <!-- DSP PIPELINE TILES GRID -->
                    
                    <!-- Row 1: Target Sample Rate (Tile 1) & Impulse Phase Topology (Tile 2) -->
                    <div class="form-grid-2" style="margin-bottom: 12px;">
                        <!-- Tile 1: Target Rate (2x2 Segmented Tile Grid) -->
                        <div class="dsp-tile-card">
                            <div class="dsp-tile-header">
                                <span>⚡ Target Sample Rate</span>
                                <span class="form-label-sub">Integral Sinc Multiplier</span>
                            </div>
                            <input type="hidden" id="upsampleRate" value="4x" />
                            <div class="tile-segmented-grid cols-2" id="tileRateGrid">
                                <div class="tile-option-btn active" data-val="4x" onclick="selectRateTile('4x')">
                                    <div class="tile-option-title">⚡ Auto 4x Native</div>
                                    <div class="tile-option-sub">176.4k / 192k (Auto Family)</div>
                                </div>
                                <div class="tile-option-btn" data-val="192000" onclick="selectRateTile('192000')">
                                    <div class="tile-option-title">192.0 kHz</div>
                                    <div class="tile-option-sub">4x from 48k Master Family</div>
                                </div>
                                <div class="tile-option-btn" data-val="176400" onclick="selectRateTile('176400')">
                                    <div class="tile-option-title">176.4 kHz</div>
                                    <div class="tile-option-sub">4x from 44.1k CD Family</div>
                                </div>
                                <div class="tile-option-btn" data-val="384000" onclick="selectRateTile('384000')">
                                    <div class="tile-option-title">🔥 8x Ultra Sinc</div>
                                    <div class="tile-option-sub">352.8k / 384k Bandwidth</div>
                                </div>
                            </div>
                        </div>

                        <!-- Tile 2: Phase Response Mode (A/B Visual Cards) -->
                        <div class="dsp-tile-card">
                            <div class="dsp-tile-header">
                                <span>Impulse Phase Topology</span>
                                <span class="form-label-sub">Time Symmetry</span>
                            </div>
                            <input type="hidden" id="upsamplePhase" value="min" />
                            <div class="phase-ab-grid">
                                <div class="phase-card-option active" id="phaseCardMin" onclick="selectPhaseMode('min')">
                                    <div class="phase-card-title">
                                        <span>Minimum Phase</span>
                                        <span style="font-size: 0.70rem; color: var(--accent-cyan); font-weight: 700;">Analog</span>
                                    </div>
                                    <div class="phase-card-sub">
                                        ⚡ <b>0.00% Pre-Ringing</b>. Natural causal decay. Eliminates studio ADC ringing.
                                    </div>
                                </div>
                                <div class="phase-card-option" id="phaseCardLinear" onclick="selectPhaseMode('linear')">
                                    <div class="phase-card-title">
                                        <span>Linear Phase</span>
                                        <span style="font-size: 0.70rem; color: var(--accent-pink); font-weight: 700;">0° Shift</span>
                                    </div>
                                    <div class="phase-card-sub">
                                        ⚖️ <b>Strict 0° Linear Phase</b>. Symmetric time impulse. Pristine acoustic coherence.
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Row 2: Reconstruction Cutoff & Apodizing Filter (Tile 3 - Full Width) -->
                    <div class="dsp-tile-card" style="margin-bottom: 12px;">
                        <div class="dsp-tile-header">
                            <span>🎯 Reconstruction Cutoff & Apodizing Filter</span>
                            <span id="cutoffHint" class="brand-badge" style="font-size: 0.70rem; padding: 2px 8px; text-transform: none;">Disabled (Full Nyquist)</span>
                        </div>
                        
                        <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
                            <div class="tile-option-btn" id="btnTileApodToggle" onclick="toggleApodizingTile()" style="min-width: 170px; padding: 7px 12px; cursor: pointer;">
                                <div class="tile-option-title">
                                    <span id="apodToggleIcon">🛡️</span>
                                    <span id="apodToggleText">Apodizing Filter</span>
                                </div>
                                <div class="tile-option-sub" id="apodToggleSub">Click to enable/disable</div>
                            </div>
                            <input type="checkbox" id="upsampleApodizingEnable" style="display: none;" onchange="syncApodizingTileUI()" />
                            
                            <div style="display: flex; align-items: center; gap: 6px; width: 145px; background: #090d13; border: 1px solid #30363d; border-radius: 6px; padding: 2px 8px;">
                                <input type="number" class="form-input" id="upsampleCutoffHz" placeholder="e.g. 20700" min="5000" max="384000" step="1" oninput="onCutoffInputChange()" onchange="onCutoffInputChange()" style="border: none; background: transparent; padding: 4px 0; height: 32px; font-weight: 700; font-size: 0.95rem; color: var(--accent-cyan); width: 100px;" />
                                <span style="font-size: 0.78rem; color: var(--text-muted); font-weight: 700;">Hz</span>
                            </div>

                            <!-- Interactive Frequency Range Slider -->
                            <div class="cutoff-slider-row" style="flex: 1; min-width: 200px;">
                                <input type="range" class="cutoff-slider" id="upsampleCutoffSlider" min="15000" max="48000" step="50" value="22050" oninput="onCutoffSliderChange(this.value)" />
                            </div>
                        </div>

                        <!-- Quick Frequency Snap Chips -->
                        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 6px; margin-top: 2px;">
                            <div class="cutoff-chip-group" id="cutoffChipGroup" style="margin: 0;">
                                <button type="button" class="cutoff-chip active" data-hz="" onclick="setCutoffPreset('')" title="Full Nyquist Bandwidth (No Lowpass Cutoff)">Nyquist (Off)</button>
                                <button type="button" class="cutoff-chip" data-hz="20700" onclick="setCutoffPreset(20700)" title="20.7 kHz Cutoff: Clean up legacy studio ADC ringing">20.7k (ADC)</button>
                                <button type="button" class="cutoff-chip" data-hz="21500" onclick="setCutoffPreset(21500)" title="21.5 kHz Cutoff: Purge ultrasonic alias mirrors">21.5k (Alias)</button>
                                <button type="button" class="cutoff-chip" data-hz="22050" onclick="setCutoffPreset(22050)" title="22.05 kHz Cutoff: Standard CD / 44.1k Master Apodizing">22.05k (CD Std)</button>
                                <button type="button" class="cutoff-chip" data-hz="24000" onclick="setCutoffPreset(24000)" title="24.0 kHz Cutoff: Standard 48k Master Apodizing">24.0k (48k Std)</button>
                                <button type="button" class="cutoff-chip" data-hz="44100" onclick="setCutoffPreset(44100)" title="44.1 kHz Cutoff: 88.2k/96k/176.4k Master Cutoff">44.1k (88/96k)</button>
                            </div>
                            <div id="cutoffRolloffDesc" style="font-size: 0.72rem; color: #8b949e; font-family: monospace;">
                                Passband: Flat to Nyquist • 64-Bit Sinc
                            </div>
                        </div>
                    </div>

                    <!-- Row 3: Transition Steepness, Dither, MQA Engine (3 Columns) -->
                    <div class="form-grid-3" style="margin-bottom: 12px;">
                        <!-- Tile 4: Transition Steepness -->
                        <div class="dsp-tile-card">
                            <div class="dsp-tile-header">
                                <span>Filter Steepness</span>
                                <span class="form-label-sub">Transition Band</span>
                            </div>
                            <input type="checkbox" id="upsampleSteep" style="display: none;" onchange="syncSteepTileUI()" />
                            <div class="tile-segmented-grid cols-2" id="tileSteepGrid">
                                <div class="tile-option-btn active" data-val="standard" onclick="selectSteepTile(false)">
                                    <div class="tile-option-title">Standard</div>
                                    <div class="tile-option-sub">2.0 kHz taper</div>
                                </div>
                                <div class="tile-option-btn" data-val="sharp" onclick="selectSteepTile(true)">
                                    <div class="tile-option-title">Brickwall</div>
                                    <div class="tile-option-sub">500 Hz knee</div>
                                </div>
                            </div>
                        </div>

                        <!-- Tile 5: Dither & Noise Shaping -->
                        <div class="dsp-tile-card">
                            <div class="dsp-tile-header">
                                <span>Noise Shaping & Dither</span>
                                <span class="form-label-sub">24-Bit Depth</span>
                            </div>
                            <input type="hidden" id="upsampleDither" value="shibata" />
                            <div class="tile-segmented-grid cols-3" id="tileDitherGrid">
                                <div class="tile-option-btn active" data-val="shibata" onclick="selectDitherTile('shibata')">
                                    <div class="tile-option-title" style="font-size: 0.76rem;">Shibata</div>
                                    <div class="tile-option-sub">>30k shift</div>
                                </div>
                                <div class="tile-option-btn" data-val="high_rate" onclick="selectDitherTile('high_rate')">
                                    <div class="tile-option-title" style="font-size: 0.76rem;">TPDF Flat</div>
                                    <div class="tile-option-sub">High rate</div>
                                </div>
                                <div class="tile-option-btn" data-val="none" onclick="selectDitherTile('none')">
                                    <div class="tile-option-title" style="font-size: 0.76rem;">None</div>
                                    <div class="tile-option-sub">Float64</div>
                                </div>
                            </div>
                        </div>

                        <!-- Tile 6: MQA Payload Engine -->
                        <div class="dsp-tile-card">
                            <div class="dsp-tile-header">
                                <span>MQA Payload Engine</span>
                                <span class="form-label-sub">Origami Policy</span>
                            </div>
                            <input type="hidden" id="upsampleMqa" value="adaptive" />
                            <div class="tile-segmented-grid cols-3" id="tileMqaGrid">
                                <div class="tile-option-btn active" data-val="adaptive" onclick="selectMqaTile('adaptive')">
                                    <div class="tile-option-title" style="font-size: 0.76rem;">Adaptive</div>
                                    <div class="tile-option-sub">Subband</div>
                                </div>
                                <div class="tile-option-btn" data-val="strip" onclick="selectMqaTile('strip')">
                                    <div class="tile-option-title" style="font-size: 0.76rem;">Strip</div>
                                    <div class="tile-option-sub">Clean LSB</div>
                                </div>
                                <div class="tile-option-btn" data-val="ignore" onclick="selectMqaTile('ignore')">
                                    <div class="tile-option-title" style="font-size: 0.76rem;">Ignore</div>
                                    <div class="tile-option-sub">Raw PCM</div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Row 4: Master Storage & Target Locations (Tile 7) -->
                    <div class="dsp-tile-card" style="margin-bottom: 8px;">
                        <div class="dsp-tile-header">
                            <span>📁 Audio Master & Target Directory</span>
                            <span class="form-label-sub">Lossless FLAC Library Storage</span>
                        </div>
                        
                        <div class="form-grid-2" style="gap: 10px;">
                            <div class="form-group">
                                <label class="form-label">
                                    <span>Source Master Location</span>
                                </label>
                                <div class="form-path-row">
                                    <input type="text" class="form-input" id="upsampleSrcPath" readonly style="height: 36px;" />
                                    <button class="btn-browse" onclick="openFolderPicker('upsampleSrcPath')" title="Browse Source Master Folder" style="height: 36px;">📁 Browse...</button>
                                </div>
                            </div>

                            <div class="form-group">
                                <label class="form-label">
                                    <span>Destination Output Directory</span>
                                </label>
                                <div class="form-path-row">
                                    <input type="text" class="form-input" id="upsampleDstDir" placeholder="/mnt/PrimaryFS/1xxK_min/music/..." style="height: 36px;" />
                                    <button class="btn-browse" onclick="openFolderPicker('upsampleDstDir')" title="Browse Target Destination Directory" style="height: 36px;">📁 Browse...</button>
                                </div>
                            </div>
                        </div>

                        <div style="display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-top: 4px; flex-wrap: wrap;">
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <span style="font-size: 0.76rem; color: var(--text-muted); font-weight: 600;">File Overwrite:</span>
                                <input type="hidden" id="upsampleOverwrite" value="on" />
                                <div class="tile-segmented-grid cols-2" id="tileOverwriteGrid" style="width: 220px;">
                                    <div class="tile-option-btn active" data-val="on" onclick="selectOverwriteTile('on')" style="padding: 4px 8px;">
                                        <div class="tile-option-title" style="font-size: 0.74rem;">Overwrite</div>
                                    </div>
                                    <div class="tile-option-btn" data-val="off" onclick="selectOverwriteTile('off')" style="padding: 4px 8px;">
                                        <div class="tile-option-title" style="font-size: 0.74rem;">Skip Existing</div>
                                    </div>
                                </div>
                            </div>
                            <div class="tile-option-btn active" id="btnTileReportToggle" onclick="toggleReportTile()" style="padding: 5px 12px; cursor: pointer;">
                                <div class="tile-option-title" style="font-size: 0.76rem;">
                                    <span>📊 Comparative HTML & Markdown Reports</span>
                                </div>
                                <input type="checkbox" id="upsampleReport" checked style="display: none;" />
                            </div>
                        </div>
                    </div>

                    <!-- Initial Launch Actions (Batch / Album Launching) -->
                    <div id="initialLaunchControls" class="decision-btn-grid" style="border-top: 1px solid #30363d; padding-top: 14px; margin-top: 8px;">
                        <button class="btn-decision btn-decision-primary" onclick="startUpsampleJob(true)" title="Launch interactive session, auditing and prompting per-track decisions">
                            <span style="font-size: 0.92rem;">▶️ [Y] Start Interactive Mode</span>
                            <span class="btn-decision-sub">Prompts per-track recipe & review in modal</span>
                        </button>
                        <button class="btn-decision btn-decision-auto" onclick="startUpsampleJob(false)" title="Automatically apply each track's forensic recommendation unattended">
                            <span style="font-size: 0.92rem;">⚡ [A] Auto-Upsample Album</span>
                            <span class="btn-decision-sub">Auto-applies recommendations unattended</span>
                        </button>
                        <button class="btn-decision btn-decision-lock" onclick="startUpsampleJob(false)" title="Upsample album using the custom parameters configured above">
                            <span style="font-size: 0.92rem;">❄️ [C] Upsample with Configured Settings</span>
                            <span class="btn-decision-sub">Applies settings above to all tracks</span>
                        </button>
                        <button class="btn-decision btn-decision-abort" onclick="closeUpsampleModal()" title="Cancel and close dialog">
                            <span style="font-size: 0.92rem;">⏹️ [K] Cancel</span>
                            <span class="btn-decision-sub">Dismiss this dialog</span>
                        </button>
                    </div>

                    <!-- Interactive Track Decision Actions Grid (Live Subprocess Prompts) -->
                    <div id="interactiveDecisionControls" class="decision-btn-grid" style="display: none; border-top: 1px solid #30363d; padding-top: 14px; margin-top: 8px;">
                        <button class="btn-decision btn-decision-primary" onclick="sendPromptChoice('y')" title="Accept recipe and process this track">
                            <span style="font-size: 0.92rem;">▶️ [Y] Process This Track</span>
                            <span class="btn-decision-sub">Process file, background & prompt next</span>
                        </button>
                        <button class="btn-decision btn-decision-lock" onclick="sendPromptChoice('c')" title="Apply this recipe to REST of album directory">
                            <span style="font-size: 0.92rem;">❄️ [C] Apply to REST of Album</span>
                            <span class="btn-decision-sub">Freeze recipe for all remaining tracks</span>
                        </button>
                        <button class="btn-decision btn-decision-auto" onclick="sendPromptChoice('a')" title="Adopt recommended recipes automatically for ALL remaining tracks">
                            <span style="font-size: 0.92rem;">⚡ [A] Auto-Apply ALL Remaining</span>
                            <span class="btn-decision-sub">Adopt recommendations unattended</span>
                        </button>
                        <button class="btn-decision btn-decision-skip" onclick="sendPromptChoice('s')" title="Skip this track and proceed to next">
                            <span style="font-size: 0.92rem;">⏭️ [S] Skip This Track</span>
                            <span class="btn-decision-sub">Skip file & move to next track</span>
                        </button>
                        <button class="btn-decision btn-decision-skip" onclick="sendPromptChoice('k')" title="Skip this track and ALL remaining in this album directory">
                            <span style="font-size: 0.92rem;">⏹️ [K] Skip REST of Album</span>
                            <span class="btn-decision-sub">Bypass remainder of album directory</span>
                        </button>
                        <button class="btn-decision" id="btnPromptViewSummary" onclick="openAlbumSummary()" style="background: rgba(0, 229, 255, 0.12); border: 1px solid rgba(0, 229, 255, 0.35); color: var(--accent-cyan);" title="View existing Album Analysis Summary">
                            <span style="font-size: 0.92rem;">📋 [V] View Album Summary</span>
                            <span class="btn-decision-sub">Read forensic album report</span>
                        </button>
                        <button class="btn-decision btn-decision-abort" onclick="sendPromptChoice('q')" style="grid-column: 1 / -1;" title="Abort the entire upsampling session">
                            <span style="font-size: 0.92rem;">⛔ [Q] Abort Upsampling Session</span>
                            <span class="btn-decision-sub">Gracefully terminate background process</span>
                        </button>
                    </div>
                </div>

            </div>
        </div>
    </div>

    <!-- Live Console Log Viewer Modal -->
    <div id="upsampleLogModal" class="modal-backdrop" style="display: none;" onclick="if(event.target===this) closeUpsampleLogModal()">
        <div class="modal-content" style="max-width: 800px; height: 75vh;">
            <div class="modal-header">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1rem; font-weight: 700; color: var(--text-heading);">📜 AcoustiSinc DSP Terminal Output</span>
                </div>
                <button class="btn-bookmark" onclick="closeUpsampleLogModal()" style="font-size: 1.2rem; padding: 2px 10px; line-height: 1;">&times;</button>
            </div>
            <div class="modal-body" style="padding: 0; background: #0a0c10; display: flex; flex-direction: column;">
                <div id="upsampleTerminalBox" style="flex: 1; overflow-y: auto; padding: 14px 18px; font-family: ui-monospace, SFMono-Regular, 'SF Mono', monospace; font-size: 0.74rem; color: #8b949e;">
                    <div id="upsampleTerminalLogs" style="display: flex; flex-direction: column; gap: 3px;"></div>
                </div>
            </div>
        </div>
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
                updateAlbumSummaryUI(directoryData.album_summary);
                const hasAudioFiles = directoryData.files && directoryData.files.length > 0;
                const headerUpsampleBtn = document.getElementById('headerBtnUpsample');
                if (headerUpsampleBtn) {
                    headerUpsampleBtn.style.display = hasAudioFiles ? 'inline-block' : 'none';
                }

                // Connect SSE stream for progressive fresh folder badges
                startFolderStream(currentPath, fresh);
            } catch (err) {
                updateAlbumSummaryUI(null);
                const headerUpsampleBtn = document.getElementById('headerBtnUpsample');
                if (headerUpsampleBtn) headerUpsampleBtn.style.display = 'none';
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
                const res = await fetch(`/api/analyze?path=${encodeURIComponent(file.path)}&fresh=0`);
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
                const durMin = Math.floor(data.duration_s / 60);
                const durSec = Math.floor(data.duration_s % 60).toString().padStart(2, '0');
                document.getElementById('trackMeta').textContent = `${data.sr.toLocaleString()} Hz | ${data.nyquist_khz.toFixed(1)} kHz Nyquist | ${durMin}:${durSec} (${data.duration_s.toFixed(1)}s) | ${data.analysis_time}s (${data.gpu_enabled ? 'GPU' : 'CPU'})`;
                
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
        // SHARED CANVAS HATCH PATTERN (OUT-OF-CONTAINER REGIONS)
        // ==========================================
        function drawHatchPattern(ctx, x, y, width, height, labelText = null) {
            if (width <= 0 || height <= 0) return;
            ctx.save();
            ctx.beginPath();
            ctx.rect(x, y, width, height);
            ctx.clip();

            // 1. Dark tinted background
            ctx.fillStyle = "rgba(10, 14, 20, 0.88)";
            ctx.fillRect(x, y, width, height);

            // 2. Diagonal stripes
            ctx.strokeStyle = "rgba(255, 171, 0, 0.08)";
            ctx.lineWidth = 1.5;
            const step = 14;
            for (let pos = x - height; pos < x + width + height; pos += step) {
                ctx.beginPath();
                ctx.moveTo(pos, y + height);
                ctx.lineTo(pos + height, y);
                ctx.stroke();
            }

            // 3. Centered subtle label / watermark
            if (labelText) {
                ctx.fillStyle = "rgba(255, 171, 0, 0.55)";
                ctx.font = "bold 9px monospace";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(labelText, x + width / 2, y + height / 2);
            }
            ctx.restore();
        }

        function getNormalisedMaxF(data) {
            if (!data) return 88.2;
            if (data.normalised_max_khz) return data.normalised_max_khz;
            const nyq = data.nyquist_khz || 22.05;
            if (nyq <= 22.05) return 88.2;
            if (nyq <= 24.0) return 96.0;
            if (nyq <= 44.1) return 88.2;
            if (nyq <= 48.0) return 96.0;
            if (nyq <= 88.2) return 88.2;
            if (nyq <= 96.0) return 96.0;
            return nyq * 1.04;
        }

        // ==========================================
        // SPECTROGRAM CANVAS WITH DUAL-AXIS ZOOM & NORMALISED SCALE
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
            specTMin = 0.0;
            specTMax = data.duration_s || 30.0;
            const normMax = getNormalisedMaxF(data);
            specFMin = 0.0;
            specFMax = normMax;
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
            const normMax = getNormalisedMaxF(currentAnalysis);
            specTMin = 0.0; specTMax = currentAnalysis.duration_s;
            specFMin = 0.0; specFMax = normMax;
            drawSpectrogram(specCanvas.getBoundingClientRect().width, specCanvas.getBoundingClientRect().height);
        }

        function setSpecPreset(type) {
            if (!currentAnalysis) return;
            if (type === 'native') {
                specFMin = 0.0;
                specFMax = currentAnalysis.nyquist_khz;
            } else {
                const normMax = getNormalisedMaxF(currentAnalysis);
                specFMin = 0.0;
                specFMax = normMax;
            }
            drawSpectrogram(specCanvas.getBoundingClientRect().width, specCanvas.getBoundingClientRect().height);
        }

        function zoomSpecX(factor, centerTRatio = 0.5) {
            if (!currentAnalysis) return;
            const curTW = specTMax - specTMin;
            const newTW = Math.max(0.2, Math.min(currentAnalysis.duration_s, curTW / factor));
            const centerT = specTMin + curTW * centerTRatio;
            specTMin = Math.max(0, centerT - newTW * centerTRatio);
            specTMax = Math.min(currentAnalysis.duration_s, specTMin + newTW);
            if (specTMax - specTMin < newTW) specTMin = Math.max(0, specTMax - newTW);
            drawSpectrogram(specCanvas.getBoundingClientRect().width, specCanvas.getBoundingClientRect().height);
        }

        function zoomSpecY(factor, centerFRatio = 0.5) {
            if (!currentAnalysis) return;
            const normMax = getNormalisedMaxF(currentAnalysis);
            const curFW = specFMax - specFMin;
            const newFW = Math.max(1.0, Math.min(normMax, curFW / factor));
            const centerF = specFMin + curFW * centerFRatio;
            specFMin = Math.max(0, centerF - newFW * centerFRatio);
            specFMax = Math.min(normMax, specFMin + newFW);
            if (specFMax - specFMin < newFW) specFMin = Math.max(0, specFMax - newFW);
            drawSpectrogram(specCanvas.getBoundingClientRect().width, specCanvas.getBoundingClientRect().height);
        }

        function zoomSpec(factor, centerTRatio = 0.5, centerFRatio = 0.5) {
            zoomSpecX(factor, centerTRatio);
            zoomSpecY(factor, centerFRatio);
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

            const nyqF = currentAnalysis.nyquist_khz;
            const normMax = getNormalisedMaxF(currentAnalysis);

            sCtx.save();
            sCtx.beginPath();
            sCtx.rect(padL, padT, plotW, plotH);
            sCtx.clip();

            // 1. Audio data range [0, nyqF] within currently visible window [specFMin, specFMax]
            const visibleAudioFMin = Math.max(specFMin, 0.0);
            const visibleAudioFMax = Math.min(specFMax, nyqF);

            if (visibleAudioFMax > visibleAudioFMin) {
                const destYTop = padT + (1.0 - (visibleAudioFMax - specFMin) / (specFMax - specFMin)) * plotH;
                const destYBottom = padT + (1.0 - (visibleAudioFMin - specFMin) / (specFMax - specFMin)) * plotH;
                const destH = destYBottom - destYTop;

                const sx = (specTMin / currentAnalysis.duration_s) * specImg.naturalWidth;
                const sw = ((specTMax - specTMin) / currentAnalysis.duration_s) * specImg.naturalWidth;
                const sy = (1.0 - (visibleAudioFMax / nyqF)) * specImg.naturalHeight;
                const sh = ((visibleAudioFMax - visibleAudioFMin) / nyqF) * specImg.naturalHeight;

                sCtx.imageSmoothingEnabled = true;
                sCtx.drawImage(specImg, sx, sy, sw, sh, padL, destYTop, plotW, destH);
            }

            // 2. Out-of-container range [nyqF, specFMax] (Diagonal hatch pattern)
            if (specFMax > nyqF) {
                const outFMin = Math.max(specFMin, nyqF);
                const outYTop = padT + (1.0 - (specFMax - specFMin) / (specFMax - specFMin)) * plotH;
                const outYBottom = padT + (1.0 - (outFMin - specFMin) / (specFMax - specFMin)) * plotH;
                const outH = outYBottom - outYTop;

                if (outH > 0) {
                    drawHatchPattern(sCtx, padL, outYTop, plotW, outH, `🔒 Out of Container Bandwidth (>${nyqF.toFixed(1)} kHz)`);
                }

                // Draw Container Nyquist separator line
                const yNyq = padT + (1.0 - (nyqF - specFMin) / (specFMax - specFMin)) * plotH;
                if (yNyq >= padT && yNyq <= padT + plotH) {
                    sCtx.strokeStyle = "rgba(255, 171, 0, 0.95)";
                    sCtx.lineWidth = 1.5;
                    sCtx.setLineDash([5, 3]);
                    sCtx.beginPath();
                    sCtx.moveTo(padL, yNyq);
                    sCtx.lineTo(padL + plotW, yNyq);
                    sCtx.stroke();
                    sCtx.setLineDash([]);
                    sCtx.fillStyle = "#ffab00";
                    sCtx.font = "bold 9px monospace";
                    sCtx.textAlign = "right";
                    sCtx.fillText(`${nyqF.toFixed(1)}k Container Nyquist`, padL + plotW - 6, yNyq - 4);
                }
            }

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

                specHudTime.textContent = curT.toFixed(2) + " s";
                specHudFreq.textContent = curF.toFixed(2) + " kHz (" + (curF * 1000).toFixed(0) + " Hz)";
                if (curF > nyqF) {
                    specHudDb.textContent = "N/A (Out of Range)";
                    specHudDb.style.color = "var(--accent-yellow)";
                } else {
                    const curDb = getSpectrogramDb(curT, curF);
                    specHudDb.textContent = curDb.toFixed(1) + " dBFS";
                    specHudDb.style.color = "";
                }
            }

            sCtx.restore();

            // Axes & Labels
            sCtx.strokeStyle = '#30363d';
            sCtx.lineWidth = 1;
            sCtx.strokeRect(padL, padT, plotW, plotH);

            sCtx.font = '10px monospace';
            sCtx.textAlign = 'right';
            sCtx.textBaseline = 'middle';

            const fRange = specFMax - specFMin;
            let fStep = 20;
            if (fRange <= 10) fStep = 2;
            else if (fRange <= 25) fStep = 5;
            else if (fRange <= 50) fStep = 10;
            else if (fRange <= 100) fStep = 20;
            else fStep = 40;

            for (let f = Math.ceil(specFMin / fStep) * fStep; f <= specFMax; f += fStep) {
                const y = padT + (1.0 - (f - specFMin) / (specFMax - specFMin)) * plotH;
                if (y >= padT - 2 && y <= padT + plotH + 2) {
                    sCtx.fillStyle = (f === nyqF || f === 20 || f === 88.2 || f === 96) ? '#00e5ff' : '#8b949e';
                    sCtx.fillText(`${f}k`, padL - 6, y);
                    sCtx.strokeStyle = 'rgba(255,255,255,0.06)';
                    sCtx.beginPath(); sCtx.moveTo(padL, y); sCtx.lineTo(padL + plotW, y); sCtx.stroke();
                }
            }

            sCtx.textAlign = 'center';
            sCtx.textBaseline = 'top';
            const tRange = specTMax - specTMin;
            let tStep = 10;
            if (tRange > 300) tStep = 60;
            else if (tRange > 120) tStep = 30;
            else if (tRange > 60) tStep = 15;
            else if (tRange > 20) tStep = 5;
            else if (tRange > 8) tStep = 2;
            else if (tRange > 2) tStep = 0.5;
            else tStep = 0.2;

            for (let t = Math.ceil(specTMin / tStep) * tStep; t <= specTMax; t += tStep) {
                const x = padL + ((t - specTMin) / (specTMax - specTMin)) * plotW;
                let label = '';
                if (tStep >= 60 || (tRange > 60 && tStep >= 15)) {
                    const m = Math.floor(t / 60);
                    const s = Math.floor(t % 60).toString().padStart(2, '0');
                    label = `${m}:${s}`;
                } else if (tStep >= 1) {
                    label = `${Math.round(t)}s`;
                } else {
                    label = `${t.toFixed(1)}s`;
                }
                sCtx.fillStyle = '#8b949e';
                sCtx.fillText(label, x, padT + plotH + 6);
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
                const normMax = getNormalisedMaxF(currentAnalysis);

                const dt = -(dx / plotW) * curTW;
                const df = (dy / plotH) * curFW;

                specTMin = Math.max(0, Math.min(currentAnalysis.duration_s - curTW, specInitTMin + dt));
                specTMax = specTMin + curTW;

                specFMin = Math.max(0, Math.min(normMax - curFW, specInitFMin + df));
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

                if (e.shiftKey || mY > padT + plotH * 0.75) {
                    zoomSpecX(factor, tRatio);
                } else if (e.ctrlKey || e.altKey || mX < padL + plotW * 0.25) {
                    zoomSpecY(factor, fRatio);
                } else {
                    zoomSpec(factor, tRatio, fRatio);
                }
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
        // SPECTRUM CURVE CANVAS WITH DUAL-AXIS ZOOM & NORMALISED SCALE
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
            const normMax = getNormalisedMaxF(data);
            curveFMin = 0.0; curveFMax = normMax;
            curveDbMin = -175.0; curveDbMax = 0.0;
            resizeCurveCanvas();
        }

        function resetCurveZoom() {
            if (!currentAnalysis) return;
            const normMax = getNormalisedMaxF(currentAnalysis);
            curveFMin = 0.0; curveFMax = normMax;
            curveDbMin = -175.0; curveDbMax = 0.0;
            drawCurve(curveCanvas.getBoundingClientRect().width, curveCanvas.getBoundingClientRect().height);
        }

        function setCurvePreset(type) {
            if (!currentAnalysis) return;
            const normMax = getNormalisedMaxF(currentAnalysis);
            if (type === 'audible') {
                curveFMin = 0.0; curveFMax = 20.0;
                curveDbMin = -120.0; curveDbMax = 0.0;
            } else if (type === 'cutoff') {
                curveFMin = 15.0; curveFMax = Math.min(30.0, normMax);
                curveDbMin = -175.0; curveDbMax = -40.0;
            } else if (type === 'ultrasonic') {
                curveFMin = 20.0; curveFMax = normMax;
                curveDbMin = -175.0; curveDbMax = -80.0;
            }
            drawCurve(curveCanvas.getBoundingClientRect().width, curveCanvas.getBoundingClientRect().height);
        }

        function zoomCurveX(factor, centerFRatio = 0.5) {
            if (!currentAnalysis) return;
            const normMax = getNormalisedMaxF(currentAnalysis);
            const curFW = curveFMax - curveFMin;
            const newFW = Math.max(1.0, Math.min(normMax, curFW / factor));
            const centerF = curveFMin + curFW * centerFRatio;
            curveFMin = Math.max(0, centerF - newFW * centerFRatio);
            curveFMax = Math.min(normMax, curveFMin + newFW);
            if (curveFMax - curveFMin < newFW) curveFMin = Math.max(0, curveFMax - newFW);
            drawCurve(curveCanvas.getBoundingClientRect().width, curveCanvas.getBoundingClientRect().height);
        }

        function zoomCurveY(factor, centerDbRatio = 0.5) {
            if (!currentAnalysis) return;
            const curDbW = curveDbMax - curveDbMin;
            const newDbW = Math.max(10.0, Math.min(175.0, curDbW / factor));
            const centerDb = curveDbMin + curDbW * centerDbRatio;
            curveDbMin = Math.max(-175.0, centerDb - newDbW * centerDbRatio);
            curveDbMax = Math.min(0.0, curveDbMin + newDbW);
            if (curveDbMax - curveDbMin < newDbW) curveDbMin = Math.max(-175.0, curveDbMax - newDbW);
            drawCurve(curveCanvas.getBoundingClientRect().width, curveCanvas.getBoundingClientRect().height);
        }

        function zoomCurve(factor, centerFRatio = 0.5, centerDbRatio = 0.5) {
            zoomCurveX(factor, centerFRatio);
            zoomCurveY(factor, centerDbRatio);
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
            const nyqF = currentAnalysis.nyquist_khz;

            cCtx.save();
            cCtx.beginPath();
            cCtx.rect(padL, padT, plotW, plotH);
            cCtx.clip();

            // 1. Out-of-container Nyquist Hatch Pattern
            if (curveFMax > nyqF) {
                const xNyq = freqToX(Math.max(curveFMin, nyqF), w, padL, padR);
                const hatchW = (padL + plotW) - xNyq;
                if (hatchW > 0) {
                    drawHatchPattern(cCtx, xNyq, padT, hatchW, plotH, `🔒 Out of Container Bandwidth (>${nyqF.toFixed(1)} kHz)`);
                }
            }

            // 2. File Container Nyquist Limit Line
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

            // 3. 20 kHz Audible Hearing Limit (shown only when container Nyquist > 20 kHz)
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

            // 4. Suspected Native Lineage Nyquist
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

            // Peak curve (Cyan) - drawn strictly up to container Nyquist
            cCtx.strokeStyle = "rgba(0, 229, 255, 0.9)";
            cCtx.lineWidth = 1.5;
            cCtx.beginPath();
            let first = true;
            for (let i = 0; i < freqs.length; i++) {
                if (freqs[i] < curveFMin || freqs[i] > curveFMax || freqs[i] > nyqF) continue;
                const x = freqToX(freqs[i], w, padL, padR);
                const y = dbToY(peaks[i], h, padT, padB);
                if (first) { cCtx.moveTo(x, y); first = false; }
                else { cCtx.lineTo(x, y); }
            }
            cCtx.stroke();

            // RMS curve (Magenta) - drawn strictly up to container Nyquist
            cCtx.strokeStyle = "rgba(255, 0, 127, 0.9)";
            cCtx.lineWidth = 1.5;
            cCtx.beginPath();
            first = true;
            for (let i = 0; i < freqs.length; i++) {
                if (freqs[i] < curveFMin || freqs[i] > curveFMax || freqs[i] > nyqF) continue;
                const x = freqToX(freqs[i], w, padL, padR);
                const y = dbToY(rms[i], h, padT, padB);
                if (first) { cCtx.moveTo(x, y); first = false; }
                else { cCtx.lineTo(x, y); }
            }
            cCtx.stroke();

            // Projected Filter Cutoff Response
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

                if (targetFreq > nyqF) {
                    hudFreq.textContent = `${targetFreq.toFixed(2)} kHz (${(targetFreq * 1000).toFixed(0)} Hz)`;
                    hudPeak.textContent = "N/A (Out of Range)";
                    hudPeak.style.color = "var(--accent-yellow)";
                    hudRMS.textContent = "N/A (Out of Range)";
                    hudRMS.style.color = "var(--accent-yellow)";
                    const hudProjItem = document.getElementById('hudProjItem');
                    if (hudProjItem) hudProjItem.style.display = 'none';
                } else {
                    hudPeak.style.color = "";
                    hudRMS.style.color = "";
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
                }
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
            const fRange = curveFMax - curveFMin;
            let fStep = 20;
            if (fRange <= 10) fStep = 2;
            else if (fRange <= 25) fStep = 5;
            else if (fRange <= 50) fStep = 10;
            else if (fRange <= 100) fStep = 20;
            else fStep = 40;

            for (let f = Math.ceil(curveFMin / fStep) * fStep; f <= curveFMax; f += fStep) {
                const x = freqToX(f, w, padL, padR);
                if (x >= padL - 2 && x <= padL + plotW + 2) {
                    cCtx.fillStyle = (f === nyqF || f === 20 || f === 88.2 || f === 96) ? '#00e5ff' : '#8b949e';
                    cCtx.fillText(`${f}k`, x, padT + plotH + 6);
                    cCtx.strokeStyle = 'rgba(255,255,255,0.06)';
                    cCtx.beginPath(); cCtx.moveTo(x, padT); cCtx.lineTo(x, padT + plotH); cCtx.stroke();
                }
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
                const normMax = getNormalisedMaxF(currentAnalysis);

                const df = -(dx / plotW) * curFW;
                const dDb = (dy / plotH) * curDbW;

                curveFMin = Math.max(0, Math.min(normMax - curFW, curveInitFMin + df));
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

                if (e.shiftKey || mY > padT + plotH * 0.75) {
                    zoomCurveX(factor, fRatio);
                } else if (e.ctrlKey || e.altKey || mX < padL + plotW * 0.25) {
                    zoomCurveY(factor, dbRatio);
                } else {
                    zoomCurve(factor, fRatio, dbRatio);
                }
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

        // ==========================================
        // Album Analysis Summary Modal Logic
        // ==========================================
        let currentAlbumSummary = null;
        let currentSummaryRawUrl = '';
        let currentSummaryText = '';

        function updateAlbumSummaryUI(summaryInfo) {
            currentAlbumSummary = summaryInfo;
            const banner = document.getElementById('albumSummaryBanner');
            const headerBtn = document.getElementById('headerBtnSummary');
            const bannerLabel = document.getElementById('albumSummaryBannerLabel');

            if (summaryInfo && summaryInfo.filename) {
                const labelText = summaryInfo.filename + (summaryInfo.is_mirror ? ' (Output)' : '');
                if (banner) {
                    banner.style.display = 'flex';
                    if (bannerLabel) bannerLabel.textContent = labelText;
                }
                if (headerBtn) {
                    headerBtn.style.display = 'inline-block';
                    headerBtn.title = `View ${labelText}`;
                }
            } else {
                if (banner) banner.style.display = 'none';
                if (headerBtn) headerBtn.style.display = 'none';
            }
        }

        async function openAlbumSummary() {
            const modal = document.getElementById('albumSummaryModal');
            const body = document.getElementById('modalSummaryBody');
            const badge = document.getElementById('modalSummaryBadge');
            const extBtn = document.getElementById('modalOpenExternalBtn');
            const copyBtn = document.getElementById('btnCopySummary');

            if (!modal) return;
            modal.style.display = 'flex';
            body.innerHTML = '<div style="text-align: center; padding: 40px; color: var(--text-muted);"><div class="spinner" style="margin: 0 auto 10px;"></div>Loading album analysis summary...</div>';
            if (copyBtn) copyBtn.textContent = '📋 Copy';

            try {
                const res = await fetch(`/api/album_summary?path=${encodeURIComponent(currentPath)}`);
                const data = await res.json();
                if (!data || !data.found) {
                    badge.textContent = 'NOT FOUND';
                    body.innerHTML = '<div style="padding: 40px; text-align: center; color: var(--text-muted);"><svg width="48" height="48" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="margin-bottom: 12px; opacity: 0.5;"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg><h3>No Album Analysis Summary Found</h3><p style="margin-top: 8px; font-size: 0.85rem;">No ALBUM_REPORT.html, ALBUM_REPORT.md, or summary report was found in this folder or its mirror.</p></div>';
                    extBtn.style.display = 'none';
                    return;
                }

                badge.textContent = data.filename + (data.is_mirror ? ' (Mirror Folder)' : '');
                currentSummaryRawUrl = data.raw_url;
                currentSummaryText = data.content || '';

                if (data.type === 'html') {
                    extBtn.style.display = 'inline-block';
                    body.innerHTML = `<iframe class="modal-iframe" src="${data.raw_url}" style="width: 100%; height: 100%; min-height: 72vh; border: none; border-radius: 6px; background: #0d1117;"></iframe>`;
                } else {
                    extBtn.style.display = 'none';
                    const rendered = renderMarkdownToHtml(data.content);
                    body.innerHTML = `<div class="markdown-rendered">${rendered}</div>`;
                }
            } catch (err) {
                body.innerHTML = `<div style="padding: 24px; color: var(--accent-red); text-align: center;">Failed to load album summary:<br><small>${escapeHtml(err.message)}</small></div>`;
            }
        }

        function closeAlbumSummaryModal() {
            const modal = document.getElementById('albumSummaryModal');
            if (modal) modal.style.display = 'none';
            const body = document.getElementById('modalSummaryBody');
            if (body) body.innerHTML = '';
        }

        function openAlbumSummaryExternal() {
            if (currentSummaryRawUrl) {
                window.open(currentSummaryRawUrl, '_blank');
            }
        }

        async function copyAlbumSummaryText() {
            const copyBtn = document.getElementById('btnCopySummary');
            if (!currentSummaryText) return;
            try {
                await navigator.clipboard.writeText(currentSummaryText);
                if (copyBtn) copyBtn.textContent = '✅ Copied!';
                setTimeout(() => { if (copyBtn) copyBtn.textContent = '📋 Copy'; }, 2000);
            } catch (err) {
                if (copyBtn) copyBtn.textContent = '❌ Failed';
            }
        }

        function renderMarkdownToHtml(md) {
            if (!md) return '';
            let lines = md.split('\\n');
            let html = [];
            let inTable = false;
            let tableHeaderDone = false;

            for (let line of lines) {
                let trimmed = line.trim();

                // Table parsing
                if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
                    let cells = trimmed.split('|').slice(1, -1).map(c => c.trim());
                    if (cells.every(c => /^:?-+:?$/.test(c))) {
                        tableHeaderDone = true;
                        continue;
                    }
                    if (!inTable) {
                        html.push('<table>');
                        inTable = true;
                        tableHeaderDone = false;
                    }
                    let tag = tableHeaderDone ? 'td' : 'th';
                    html.push('<tr>' + cells.map(c => `<${tag}>${formatInlineMd(c)}</${tag}>`).join('') + '</tr>');
                    continue;
                } else if (inTable) {
                    html.push('</table>');
                    inTable = false;
                    tableHeaderDone = false;
                }

                if (!trimmed) {
                    html.push('<div style="height: 10px;"></div>');
                    continue;
                }

                if (trimmed.startsWith('### ')) {
                    html.push(`<h3>${formatInlineMd(trimmed.substring(4))}</h3>`);
                } else if (trimmed.startsWith('## ')) {
                    html.push(`<h2>${formatInlineMd(trimmed.substring(3))}</h2>`);
                } else if (trimmed.startsWith('# ')) {
                    html.push(`<h1>${formatInlineMd(trimmed.substring(2))}</h1>`);
                } else if (trimmed === '---' || trimmed === '***') {
                    html.push('<hr>');
                } else if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
                    html.push(`<li>${formatInlineMd(trimmed.substring(2))}</li>`);
                } else {
                    html.push(`<p>${formatInlineMd(trimmed)}</p>`);
                }
            }
            if (inTable) html.push('</table>');
            return html.join('\\n');
        }

        function formatInlineMd(str) {
            return escapeHtml(str)
                .replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/\\*(.*?)\\*/g, '<em>$1</em>')
                .replace(/`(.*?)`/g, '<code>$1</code>');
        }

        // ==========================================
        // AcoustiSinc Upsampling Studio & Job Telemetry
        // ==========================================
        let upsamplePollTimer = null;
        let lastLogIdx = 0;
        let activeUpsampleJob = null;
        let upsampleTargetMode = 'album';
        let currentPromptData = null;
        let activePromptTrack = null;
        let folderPickerTargetInputId = 'upsampleDstDir';
        let folderPickerCurrentPath = '/mnt/PrimaryFS/FLAC_music/music';

        async function openFolderPicker(targetInputId) {
            folderPickerTargetInputId = targetInputId;
            const currentVal = document.getElementById(targetInputId) ? document.getElementById(targetInputId).value.trim() : '';
            folderPickerCurrentPath = currentVal || currentPath || '/mnt/PrimaryFS';
            const modal = document.getElementById('folderPickerModal');
            if (modal) modal.style.display = 'flex';
            await loadFolderPickerDirectory(folderPickerCurrentPath);
        }

        function openFolderPickerForMainPath() {
            openFolderPicker('pathBar');
        }

        function closeFolderPicker() {
            const modal = document.getElementById('folderPickerModal');
            if (modal) modal.style.display = 'none';
        }

        async function loadFolderPickerDirectory(targetPath) {
            folderPickerCurrentPath = targetPath;
            const pathBar = document.getElementById('folderPickerPathBar');
            const breadcrumbsEl = document.getElementById('folderPickerBreadcrumbs');
            const listEl = document.getElementById('folderPickerList');
            if (pathBar) pathBar.value = targetPath;
            if (listEl) {
                listEl.innerHTML = '<div style="padding: 28px; text-align: center; color: #8b949e;"><div class="spinner" style="margin: 0 auto 10px;"></div>Loading directory contents...</div>';
            }

            try {
                const res = await fetch(`/api/browse?path=${encodeURIComponent(targetPath)}&fresh=0`);
                const data = await res.json();
                if (!data || !listEl) return;

                folderPickerCurrentPath = data.current_path || targetPath;
                if (pathBar) pathBar.value = folderPickerCurrentPath;

                // Render Breadcrumbs
                if (breadcrumbsEl) {
                    let bHtml = '<span class="crumb" onclick="loadFolderPickerDirectory(\'/\')">🏠 /</span>';
                    if (data.breadcrumbs && data.breadcrumbs.length > 0) {
                        data.breadcrumbs.forEach(b => {
                            bHtml += ` <span style="color: #484f58;">/</span> <span class="crumb" onclick="loadFolderPickerDirectory('${b.path.replace(/'/g, "\\'")}')">${b.name}</span>`;
                        });
                    }
                    breadcrumbsEl.innerHTML = bHtml;
                }

                let html = '';
                if (data.parent_path && data.parent_path !== folderPickerCurrentPath) {
                    html += `<div class="folder-picker-item" onclick="loadFolderPickerDirectory('${data.parent_path.replace(/'/g, "\\'")}')">
                        <span style="font-size: 1.15rem;">📁</span>
                        <span style="font-weight: 600; color: var(--accent-cyan); flex: 1;">.. [Up to Parent Directory]</span>
                        <span style="font-size: 0.72rem; color: var(--text-muted);">${data.parent_path}</span>
                    </div>`;
                }

                const subfolders = data.folders || data.dirs || [];
                if (subfolders.length > 0) {
                    subfolders.forEach(d => {
                        const trackBadge = d.audio_count > 0 ? `<span class="badge-provenance-native" style="font-size: 0.68rem; padding: 1px 6px; margin-right: 8px;">${d.audio_count} track${d.audio_count > 1 ? 's' : ''}</span>` : '';
                        html += `<div class="folder-picker-item" onclick="loadFolderPickerDirectory('${d.path.replace(/'/g, "\\'")}')">
                            <span style="font-size: 1.15rem;">📁</span>
                            <span style="color: var(--text-heading); font-weight: 500; flex: 1;">${d.name}</span>
                            ${trackBadge}
                            <button class="btn-bookmark" style="padding: 2px 10px; font-size: 0.72rem; color: #00e676; border-color: rgba(0, 230, 118, 0.45); background: rgba(0, 230, 118, 0.1);" onclick="event.stopPropagation(); selectFolderPickerChoice('${d.path.replace(/'/g, "\\'")}')">Select</button>
                        </div>`;
                    });
                } else {
                    html += `<div style="padding: 28px; text-align: center; color: var(--text-muted); font-size: 0.85rem;">
                        <div style="font-size: 1.5rem; margin-bottom: 6px;">📁</div>
                        No subdirectories in this folder.<br>
                        <small style="color: #8b949e;">Click "✓ Select Current Folder" below to choose this location.</small>
                    </div>`;
                }
                listEl.innerHTML = html;
            } catch (err) {
                if (listEl) listEl.innerHTML = `<div style="padding: 20px; color: #ff5252; text-align: center;">Error loading directory: ${err.message}</div>`;
            }
        }

        async function createNewFolderInPicker() {
            const folderName = prompt('Enter new subfolder name:');
            if (!folderName || !folderName.trim()) return;
            const newPath = folderPickerCurrentPath.replace(/[/]+$/, '') + '/' + folderName.trim();
            try {
                const res = await fetch('/api/mkdir', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path: newPath })
                });
                const data = await res.json();
                if (data.status === 'ok') {
                    await loadFolderPickerDirectory(data.created_path);
                } else {
                    alert('Failed to create directory: ' + (data.message || 'Unknown error'));
                }
            } catch (err) {
                alert('Error creating directory: ' + err.message);
            }
        }

        function selectFolderPickerChoice(customPath = null) {
            const pathBar = document.getElementById('folderPickerPathBar');
            const chosenPath = customPath || (pathBar ? pathBar.value.trim() : '') || folderPickerCurrentPath;
            const targetInput = document.getElementById(folderPickerTargetInputId);
            if (targetInput && chosenPath) {
                targetInput.value = chosenPath;
            }
            closeFolderPicker();

            if (folderPickerTargetInputId === 'pathBar' && chosenPath) {
                loadDirectory(chosenPath, true);
            }
        }

        function renderInteractivePromptUI(pData) {
            const modal = document.getElementById('upsampleModal');
            const prepView = document.getElementById('upsampleModalPreparingView');
            const interView = document.getElementById('upsampleModalInteractiveView');
            const scopeBadge = document.getElementById('upsampleModalScopeBadge');
            const srcInput = document.getElementById('upsampleSrcPath');
            const dstInput = document.getElementById('upsampleDstDir');
            const initialLaunchControls = document.getElementById('initialLaunchControls');
            const interactiveDecisionControls = document.getElementById('interactiveDecisionControls');

            if (prepView) prepView.style.display = 'none';
            if (interView) interView.style.display = 'block';
            if (initialLaunchControls) initialLaunchControls.style.display = 'none';
            if (interactiveDecisionControls) interactiveDecisionControls.style.display = 'grid';

            scopeBadge.textContent = `TRACK ${pData.track_idx || 1} OF ${pData.total_tracks || 1}`;
            scopeBadge.style.background = 'rgba(255, 234, 0, 0.15)';
            scopeBadge.style.borderColor = 'rgba(255, 234, 0, 0.4)';
            scopeBadge.style.color = 'var(--accent-yellow)';

            srcInput.value = pData.filepath || '';
            if (pData.album_dir && (!dstInput.value || dstInput.value.trim() === '')) {
                fetch(`/api/upsample/dest_preview?path=${encodeURIComponent(pData.album_dir)}`)
                    .then(r => r.json())
                    .then(d => { if (d && d.dest_dir) dstInput.value = d.dest_dir; })
                    .catch(() => {});
            }

            // Populate Forensic Audit Card
            const recInfo = pData.rec_info || {};
            const recParams = pData.rec_params || {};
            const provInfo = pData.prov_info || {};
            const primary = provInfo.primary || {};
            const vis = provInfo.visual_morphology || {};
            const pk = vis.primary_knee || {};
            const purity = vis.stopband_purity || {};
            const srKhz = pData.sr ? (pData.sr / 1000).toFixed(1) : '44.1';

            document.getElementById('promptTrackTitle').textContent = `[Track ${pData.track_idx}/${pData.total_tracks}] ${pData.track_file || 'Track'}`;
            
            const provLabel = primary.label ? `${primary.label.toUpperCase()} [${primary.confidence || 'HIGH'} CONFIDENCE]` : (recInfo.action ? recInfo.action.toUpperCase() : 'STANDARD MASTER');
            const provEl = document.getElementById('promptProvLabel');
            provEl.textContent = provLabel;
            provEl.className = 'provenance-tag ' + (primary.label === 'Native Master' ? 'badge-provenance-native' : (primary.label === 'Lowpass Filtered' ? 'badge-provenance-lowpass' : 'badge-provenance-upsampled'));

            document.getElementById('promptFormatMeta').textContent = `Source Format: ${srKhz} kHz • 2 Channels • Master FLAC`;
            
            let metricText = '';
            if (pk && pk.is_brickwall_knee) {
                metricText += `📐 Brickwall Knee: ${pk.freq_khz ? pk.freq_khz.toFixed(1) : ''} kHz (Slope: ${pk.steepest_slope_db_per_khz ? pk.steepest_slope_db_per_khz.toFixed(1) : ''} dB/kHz, Drop: ${pk.drop_db ? pk.drop_db.toFixed(1) : ''} dB) • `;
            }
            if (purity && purity.has_stopband) {
                metricText += `📻 Stopband: ${purity.purity_label || 'Clean'} • `;
            }
            metricText += `📊 64-Bit Sinc Polyphase Reconstruction`;
            document.getElementById('promptMetricsRow').textContent = metricText;

            document.getElementById('promptActionDesc').textContent = recInfo.action ? `${recInfo.action}` : 'Direct 64-Bit Sinc Upsampling';
            document.getElementById('promptDspFlag').textContent = recInfo.dsp_params || '--phase min --dither shibata';
            
            // Detailed Technical Rationale
            if (recInfo.details) {
                document.getElementById('promptTechnicalRationale').textContent = recInfo.details;
            } else if (primary.label === 'Lowpass Filtered') {
                document.getElementById('promptTechnicalRationale').textContent = 'A steep lowpass reconstruction knee was detected. A minimum-phase apodizing sinc filter is recommended to eliminate pre-ringing impulse artefacts and suppress ultrasonic alias imaging without touching audible frequencies.';
            } else {
                document.getElementById('promptTechnicalRationale').textContent = 'Preserves pristine bit-perfect audio fidelity across audible octaves while providing optimal 64-bit double-precision sinc reconstruction with psychoacoustic 24-bit Shibata noise shaping.';
            }

            // Populate form overrides from recommended parameters
            selectRateTile('4x', true);
            selectPhaseMode(recParams.phase_mode || 'min', true);
            selectDitherTile(recParams.dither_mode || 'shibata', true);
            selectMqaTile(recParams.mqa_mode || 'adaptive', true);
            selectSteepTile(!!recParams.steep, true);

            if (recParams.cutoff_hz) {
                setCutoffPreset(Math.round(recParams.cutoff_hz));
            } else if (recParams.apodizing) {
                setCutoffPreset(22050);
            } else {
                setCutoffPreset('');
            }

            const btnSummary = document.getElementById('btnPromptViewSummary');
            if (btnSummary) {
                btnSummary.style.display = pData.has_summary ? 'flex' : 'none';
            }

            if (modal) modal.style.display = 'flex';
        }

        async function startUpsampleAlbumBatch(interactiveMode = true) {
            if (!currentPath) {
                alert('Please navigate to an album directory first.');
                return;
            }

            let dstDir = '';
            try {
                const res = await fetch(`/api/upsample/dest_preview?path=${encodeURIComponent(currentPath)}`);
                const data = await res.json();
                if (data && data.dest_dir) {
                    dstDir = data.dest_dir;
                }
            } catch (err) {}

            const payload = {
                source_path: currentPath,
                dest_dir: dstDir,
                rate: '4x',
                phase: 'min',
                apodizing: false,
                cutoff_hz: null,
                steep: false,
                dither: 'shibata',
                mqa: 'adaptive',
                overwrite: 'on',
                report: true,
                interactive: interactiveMode,
                use_recommended: true
            };

            try {
                updateHeaderUpsampleUI({ 
                    status: 'running', 
                    stage: 'Scanning Headroom & Auditing Track 1...', 
                    progress_percent: 2, 
                    current_track: 'Auditing Track 1...',
                    track_index: 1,
                    total_tracks: directoryData && directoryData.files ? directoryData.files.length : 1
                });

                const res = await fetch('/api/upsample/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.status !== 'ok') {
                    dismissUpsampleHeaderWidget();
                    closeUpsampleModal();
                    alert('Failed to start upsampling session: ' + (data.message || 'Unknown error'));
                    return;
                }

                activeUpsampleJob = data;
                lastLogIdx = 0;
                activePromptTrack = null;
                startPollingUpsampleStatus();
            } catch (err) {
                dismissUpsampleHeaderWidget();
                closeUpsampleModal();
                alert('Error starting upsampler: ' + err.message);
            }
        }

        function openUpsampleModalForTrack() {
            if (!currentAnalysis || !currentAnalysis.filepath) {
                alert('Please select and analyze a track first.');
                return;
            }
            openUpsampleModal('track', currentAnalysis.filepath);
        }

        function openUpsampleModalForCurrentFolder() {
            if (activeUpsampleJob && activeUpsampleJob.status === 'waiting_for_input' && currentPromptData) {
                renderInteractivePromptUI(currentPromptData);
                const modal = document.getElementById('upsampleModal');
                if (modal) modal.style.display = 'flex';
                return;
            }
            if (activeUpsampleJob && activeUpsampleJob.status === 'running') {
                toggleUpsampleLogModal();
                return;
            }
            if (!currentPath) {
                alert('Please navigate to an album directory first.');
                return;
            }
            openUpsampleModal('album', currentPath);
        }

        function closeUpsampleModal() {
            const modal = document.getElementById('upsampleModal');
            if (modal) modal.style.display = 'none';
        }

        function selectPhaseMode(mode, fromPreset = false) {
            const phaseInput = document.getElementById('upsamplePhase');
            if (phaseInput) phaseInput.value = mode;
            const cardMin = document.getElementById('phaseCardMin');
            const cardLinear = document.getElementById('phaseCardLinear');
            if (cardMin) cardMin.classList.toggle('active', mode === 'min');
            if (cardLinear) cardLinear.classList.toggle('active', mode === 'linear');
            if (!fromPreset) setActivePresetPill(null);
        }

        function selectRateTile(val, fromPreset = false) {
            const input = document.getElementById('upsampleRate');
            if (input) input.value = val;
            const btns = document.querySelectorAll('#tileRateGrid .tile-option-btn');
            btns.forEach(btn => {
                btn.classList.toggle('active', btn.getAttribute('data-val') === String(val));
            });
            if (!fromPreset) setActivePresetPill(null);
        }

        function selectSteepTile(isSteep, fromPreset = false) {
            const input = document.getElementById('upsampleSteep');
            if (input) input.checked = isSteep;
            const btns = document.querySelectorAll('#tileSteepGrid .tile-option-btn');
            btns.forEach(btn => {
                const bVal = btn.getAttribute('data-val') === 'sharp';
                btn.classList.toggle('active', bVal === isSteep);
            });
            if (!fromPreset) setActivePresetPill(null);
        }

        function syncSteepTileUI() {
            const input = document.getElementById('upsampleSteep');
            selectSteepTile(input ? input.checked : false);
        }

        function selectDitherTile(val, fromPreset = false) {
            const input = document.getElementById('upsampleDither');
            if (input) input.value = val;
            const btns = document.querySelectorAll('#tileDitherGrid .tile-option-btn');
            btns.forEach(btn => {
                btn.classList.toggle('active', btn.getAttribute('data-val') === val);
            });
            if (!fromPreset) setActivePresetPill(null);
        }

        function selectMqaTile(val, fromPreset = false) {
            const input = document.getElementById('upsampleMqa');
            if (input) input.value = val;
            const btns = document.querySelectorAll('#tileMqaGrid .tile-option-btn');
            btns.forEach(btn => {
                btn.classList.toggle('active', btn.getAttribute('data-val') === val);
            });
            if (!fromPreset) setActivePresetPill(null);
        }

        function selectOverwriteTile(val) {
            const input = document.getElementById('upsampleOverwrite');
            if (input) input.value = val;
            const btns = document.querySelectorAll('#tileOverwriteGrid .tile-option-btn');
            btns.forEach(btn => {
                btn.classList.toggle('active', btn.getAttribute('data-val') === val);
            });
        }

        function toggleReportTile() {
            const input = document.getElementById('upsampleReport');
            const btn = document.getElementById('btnTileReportToggle');
            if (input) {
                input.checked = !input.checked;
                if (btn) btn.classList.toggle('active', input.checked);
            }
        }

        function toggleApodizingTile() {
            const enableCheck = document.getElementById('upsampleApodizingEnable');
            const cutoffInput = document.getElementById('upsampleCutoffHz');
            const cutoffSlider = document.getElementById('upsampleCutoffSlider');
            if (enableCheck) {
                enableCheck.checked = !enableCheck.checked;
                if (enableCheck.checked) {
                    if (!cutoffInput.value || parseFloat(cutoffInput.value) <= 0) {
                        cutoffInput.value = '22050';
                        if (cutoffSlider) cutoffSlider.value = 22050;
                    }
                } else {
                    cutoffInput.value = '';
                }
                onCutoffInputChange(true);
            }
        }

        function syncApodizingTileUI() {
            const enableCheck = document.getElementById('upsampleApodizingEnable');
            const btn = document.getElementById('btnTileApodToggle');
            const sub = document.getElementById('apodToggleSub');
            const isChecked = enableCheck ? enableCheck.checked : false;
            if (btn) btn.classList.toggle('active', isChecked);
            if (sub) sub.textContent = isChecked ? 'Filter active' : 'Click to enable';
        }

        function toggleApodizingInput() {
            toggleApodizingTile();
        }

        function onCutoffSliderChange(val) {
            const cutoffInput = document.getElementById('upsampleCutoffHz');
            if (cutoffInput) {
                cutoffInput.value = val;
            }
            onCutoffInputChange(false);
        }

        function onCutoffInputChange(syncSlider = true) {
            const enableCheck = document.getElementById('upsampleApodizingEnable');
            const cutoffInput = document.getElementById('upsampleCutoffHz');
            const cutoffSlider = document.getElementById('upsampleCutoffSlider');
            const cutoffHint = document.getElementById('cutoffHint');
            const rolloffDesc = document.getElementById('cutoffRolloffDesc');
            const val = cutoffInput.value.trim();
            const num = parseFloat(val);
            if (val && !isNaN(num) && num > 0) {
                enableCheck.checked = true;
                if (cutoffHint) {
                    cutoffHint.textContent = `Active (${num.toLocaleString()} Hz Knee)`;
                    cutoffHint.style.background = 'rgba(0, 229, 255, 0.15)';
                    cutoffHint.style.color = 'var(--accent-cyan)';
                    cutoffHint.style.borderColor = 'rgba(0, 229, 255, 0.4)';
                }
                if (syncSlider && cutoffSlider && num >= 15000 && num <= 48000) {
                    cutoffSlider.value = num;
                }
                if (rolloffDesc) {
                    const passbandTop = Math.max(10000, num - 2000);
                    rolloffDesc.textContent = `Passband flat to ${(passbandTop/1000).toFixed(1)}k • Taper to ${(num/1000).toFixed(2)}k • >140dB rejection`;
                }
            } else {
                enableCheck.checked = false;
                if (cutoffHint) {
                    cutoffHint.textContent = 'Disabled (Full Nyquist)';
                    cutoffHint.style.background = 'rgba(139, 148, 158, 0.1)';
                    cutoffHint.style.color = 'var(--text-muted)';
                    cutoffHint.style.borderColor = '#30363d';
                }
                if (rolloffDesc) {
                    rolloffDesc.textContent = 'Passband: Flat to Nyquist • 64-Bit Sinc';
                }
            }
            syncApodizingTileUI();
            updateCutoffQuickButtons();
        }

        function setCutoffPreset(hz) {
            const cutoffInput = document.getElementById('upsampleCutoffHz');
            const cutoffSlider = document.getElementById('upsampleCutoffSlider');
            if (!hz || hz <= 0) {
                cutoffInput.value = '';
            } else {
                cutoffInput.value = hz;
                if (cutoffSlider && hz >= 15000 && hz <= 48000) {
                    cutoffSlider.value = hz;
                }
            }
            onCutoffInputChange(false);
        }

        function updateCutoffQuickButtons() {
            const cutoffInput = document.getElementById('upsampleCutoffHz');
            const curVal = cutoffInput ? cutoffInput.value.trim() : '';
            const quickBtns = document.querySelectorAll('.cutoff-chip');
            quickBtns.forEach(btn => {
                const bHz = btn.getAttribute('data-hz') || '';
                btn.classList.toggle('active', bHz === curVal || (!bHz && !curVal));
            });
        }

        function applyForensicRecommendationPreset() {
            setActivePresetPill('presetBtnRecommended');
            selectRateTile('4x', true);
            selectDitherTile('shibata', true);
            selectMqaTile('adaptive', true);
            selectSteepTile(false, true);
            selectPhaseMode('min', true);
            setCutoffPreset('');

            if (currentAnalysis && currentAnalysis.provenance) {
                const p = currentAnalysis.provenance;
                const rec = p.recommendation || {};
                const paramsStr = rec.dsp_params || '';

                if (paramsStr.includes('--phase linear')) {
                    selectPhaseMode('linear', true);
                } else if (paramsStr.includes('--phase min') || paramsStr.includes('--filter min')) {
                    selectPhaseMode('min', true);
                }

                if (rec.filter_cutoff_khz) {
                    const hz = Math.round(rec.filter_cutoff_khz * 1000);
                    setCutoffPreset(hz);
                } else if (paramsStr.includes('--cutoff') || paramsStr.includes('--apod')) {
                    const m = paramsStr.match(new RegExp('--cutoff\\\\s+(\\\\d+)'));
                    setCutoffPreset(m ? parseInt(m[1]) : 22050);
                }

                if (paramsStr.includes('--steep')) {
                    selectSteepTile(true, true);
                }

                if (paramsStr.includes('--mqa strip')) {
                    selectMqaTile('strip', true);
                } else if (paramsStr.includes('--mqa ignore')) {
                    selectMqaTile('ignore', true);
                }
            }
        }

        function applyDefaultPreset(type) {
            selectRateTile('4x', true);
            selectDitherTile('shibata', true);
            selectMqaTile('adaptive', true);
            selectSteepTile(false, true);

            if (type === '4x') {
                setActivePresetPill('presetBtnAudiophile4x');
                selectPhaseMode('min', true);
                setCutoffPreset('');
            } else if (type === 'apod') {
                setActivePresetPill('presetBtnApodizing');
                selectPhaseMode('min', true);
                setCutoffPreset(22050);
            }
        }

        function setActivePresetPill(btnId) {
            ['presetBtnRecommended', 'presetBtnAudiophile4x', 'presetBtnApodizing'].forEach(id => {
                const el = document.getElementById(id);
                if (el) el.classList.toggle('active', id === btnId);
            });
        }

        async function quickUpsampleCurrentTrack() {
            if (!currentAnalysis || !currentAnalysis.filepath) {
                alert('Please select a track first.');
                return;
            }
            await openUpsampleModal('track', currentAnalysis.filepath);
            applyForensicRecommendationPreset();
            startUpsampleJob(true);
        }

        async function openUpsampleModal(mode = 'album', customSrcPath = null) {
            upsampleTargetMode = mode;
            const modal = document.getElementById('upsampleModal');
            const prepView = document.getElementById('upsampleModalPreparingView');
            const interView = document.getElementById('upsampleModalInteractiveView');
            const scopeBadge = document.getElementById('upsampleModalScopeBadge');
            const srcInput = document.getElementById('upsampleSrcPath');
            const dstInput = document.getElementById('upsampleDstDir');
            const initialLaunchControls = document.getElementById('initialLaunchControls');
            const interactiveDecisionControls = document.getElementById('interactiveDecisionControls');

            if (prepView) prepView.style.display = 'none';
            if (interView) interView.style.display = 'block';

            if (activeUpsampleJob && activeUpsampleJob.status === 'waiting_for_input' && currentPromptData) {
                renderInteractivePromptUI(currentPromptData);
                modal.style.display = 'flex';
                return;
            }

            let srcPath = customSrcPath;
            if (!srcPath) {
                if (mode === 'track') {
                    srcPath = currentAnalysis ? currentAnalysis.filepath : (directoryData && directoryData.files && directoryData.files[0] ? directoryData.files[0].path : '');
                } else {
                    srcPath = currentPath;
                }
            }

            if (!srcPath) {
                alert('Please select a track or navigate to an album folder first.');
                return;
            }

            srcInput.value = srcPath;
            scopeBadge.textContent = mode === 'track' ? 'SINGLE TRACK' : 'ALBUM BATCH';
            scopeBadge.style.background = mode === 'track' ? 'rgba(0, 229, 255, 0.15)' : 'rgba(0, 230, 118, 0.15)';
            scopeBadge.style.borderColor = mode === 'track' ? 'rgba(0, 229, 255, 0.4)' : 'rgba(0, 230, 118, 0.4)';
            scopeBadge.style.color = mode === 'track' ? 'var(--accent-cyan)' : '#00e676';

            if (initialLaunchControls) initialLaunchControls.style.display = 'grid';
            if (interactiveDecisionControls) interactiveDecisionControls.style.display = 'none';

            populateRecommendationCard(mode, srcPath);

            try {
                const res = await fetch(`/api/upsample/dest_preview?path=${encodeURIComponent(srcPath)}`);
                const data = await res.json();
                if (data && data.dest_dir) {
                    dstInput.value = data.dest_dir;
                }
            } catch (err) {
                dstInput.value = '';
            }

            applyForensicRecommendationPreset();
            modal.style.display = 'flex';
        }

        function populateRecommendationCard(mode, srcPath) {
            const trackTitleEl = document.getElementById('promptTrackTitle');
            const provLabelEl = document.getElementById('promptProvLabel');
            const formatMetaEl = document.getElementById('promptFormatMeta');
            const metricsRowEl = document.getElementById('promptMetricsRow');
            const actionDescEl = document.getElementById('promptActionDesc');
            const dspFlagEl = document.getElementById('promptDspFlag');
            const rationaleEl = document.getElementById('promptTechnicalRationale');

            const filename = srcPath.split('/').filter(Boolean).pop() || 'Item';
            trackTitleEl.textContent = mode === 'track' ? filename : `Album: ${filename}`;

            if (currentAnalysis && currentAnalysis.provenance) {
                const p = currentAnalysis.provenance;
                const rec = p.recommendation || {};
                const prim = p.primary || {};
                const vis = p.visual_morphology || {};
                const pk = vis.primary_knee || {};
                const purity = vis.stopband_purity || {};

                provLabelEl.textContent = prim.label ? `${prim.label.toUpperCase()} [${prim.confidence || 'HIGH'} CONFIDENCE]` : 'STANDARD MASTER';
                provLabelEl.className = 'provenance-tag ' + (prim.label === 'Native Master' ? 'badge-provenance-native' : prim.label === 'Lowpass Filtered' ? 'badge-provenance-lowpass' : 'badge-provenance-upsampled');
                
                const srKhz = currentAnalysis.sr ? (currentAnalysis.sr / 1000).toFixed(1) : '44.1';
                const ch = currentAnalysis.channels || 2;
                const bd = currentAnalysis.bit_depth || 24;
                formatMetaEl.textContent = `Source Format: ${srKhz} kHz • ${ch} Channels • ${bd}-bit PCM`;

                let metricText = '';
                if (pk && pk.is_brickwall_knee) {
                    metricText += `📐 Brickwall Knee: ${pk.freq_khz.toFixed(1)} kHz (Slope: ${pk.steepest_slope_db_per_khz.toFixed(1)} dB/kHz, Drop: ${pk.drop_db.toFixed(1)} dB) • `;
                }
                if (purity && purity.has_stopband) {
                    metricText += `📻 Stopband: ${purity.purity_label || 'Clean'} • `;
                }
                metricText += `📊 True Peak: ${currentAnalysis.true_peak_db ? currentAnalysis.true_peak_db.toFixed(1) + ' dBTP' : 'Safe'}`;
                metricsRowEl.textContent = metricText;

                actionDescEl.textContent = rec.action ? `${rec.action}: ${rec.details || ''}` : 'Direct 64-bit polyphase sinc upsampling with Shibata noise shaping.';
                dspFlagEl.textContent = rec.dsp_params || '--phase min --dither shibata';

                if (rec.details) {
                    rationaleEl.textContent = rec.details;
                } else if (prim.label === 'Lowpass Filtered') {
                    rationaleEl.textContent = 'A steep lowpass reconstruction cutoff was detected. A minimum-phase apodizing sinc filter is recommended to eliminate pre-ringing impulse artefacts and suppress ultrasonic alias imaging without touching audible frequencies.';
                } else {
                    rationaleEl.textContent = 'Preserves pristine bit-perfect audio fidelity across audible octaves while providing optimal 64-bit double-precision sinc reconstruction with psychoacoustic 24-bit Shibata noise shaping.';
                }
            } else {
                provLabelEl.textContent = 'ALBUM BATCH AUDIT';
                provLabelEl.className = 'provenance-tag badge-provenance-native';
                formatMetaEl.textContent = 'Multi-Track Forensic Upsampling Pipeline';
                metricsRowEl.textContent = 'Per-track dynamic spectral analysis and intersample headroom protection';
                actionDescEl.textContent = 'Interactive AcoustiSinc Upsampling';
                dspFlagEl.textContent = '--phase min --dither shibata';
                rationaleEl.textContent = 'Audits every track individually during batch execution to recommend optimal minimum-phase or apodizing recipes tailored to each master file.';
            }
        }

        function reopenActivePromptModal() {
            if (currentPromptData) {
                renderInteractivePromptUI(currentPromptData);
            }
            const modal = document.getElementById('upsampleModal');
            if (modal) modal.style.display = 'flex';
        }

        async function startUpsampleJob(interactiveMode = true) {
            const srcPath = document.getElementById('upsampleSrcPath').value.trim();
            const dstDir = document.getElementById('upsampleDstDir').value.trim();
            const rate = document.getElementById('upsampleRate').value;
            const phase = document.getElementById('upsamplePhase').value;
            const apodizingCheck = document.getElementById('upsampleApodizingEnable').checked;
            const cutoffVal = document.getElementById('upsampleCutoffHz').value.trim();
            const cutoffHz = cutoffVal && parseFloat(cutoffVal) > 0 ? parseFloat(cutoffVal) : null;
            const apodizingEnable = apodizingCheck || !!cutoffHz;
            const steep = document.getElementById('upsampleSteep').checked;
            const dither = document.getElementById('upsampleDither').value;
            const mqa = document.getElementById('upsampleMqa').value;
            const overwrite = document.getElementById('upsampleOverwrite').value;
            const report = document.getElementById('upsampleReport').checked;

            if (!srcPath) {
                alert('Source path is required.');
                return;
            }

            const payload = {
                source_path: srcPath,
                dest_dir: dstDir,
                rate: rate,
                phase: phase,
                apodizing: apodizingEnable,
                cutoff_hz: cutoffHz,
                steep: steep,
                dither: dither,
                mqa: mqa,
                overwrite: overwrite,
                report: report,
                interactive: interactiveMode,
                use_recommended: !interactiveMode
            };

            try {
                closeUpsampleModal();
                updateHeaderUpsampleUI({ 
                    status: 'running', 
                    stage: 'Scanning Headroom & Auditing...', 
                    progress_percent: 2, 
                    current_track: 'Initializing...' 
                });

                const res = await fetch('/api/upsample/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (data.status !== 'ok') {
                    dismissUpsampleHeaderWidget();
                    alert('Failed to start upsampling job: ' + (data.message || 'Unknown error'));
                    return;
                }

                activeUpsampleJob = data;
                lastLogIdx = 0;
                activePromptTrack = null;
                startPollingUpsampleStatus();
            } catch (err) {
                dismissUpsampleHeaderWidget();
                alert('Error submitting upsampling job: ' + err.message);
            }
        }

        async function sendPromptChoice(choice) {
            if (choice === 'q') {
                cancelUpsampleJob();
                return;
            }

            const phase = document.getElementById('upsamplePhase').value;
            const apodizingCheck = document.getElementById('upsampleApodizingEnable').checked;
            const cutoffVal = document.getElementById('upsampleCutoffHz').value.trim();
            const cutoffHz = cutoffVal && parseFloat(cutoffVal) > 0 ? parseFloat(cutoffVal) : null;
            const apodizingEnable = apodizingCheck || !!cutoffHz;
            const steep = document.getElementById('upsampleSteep').checked;
            const dither = document.getElementById('upsampleDither').value;
            const mqa = document.getElementById('upsampleMqa').value;

            const customParams = {
                phase_mode: phase,
                apodizing: apodizingEnable,
                dither_mode: dither,
                cutoff_hz: cutoffHz,
                mqa_mode: mqa,
                steep: steep
            };

            // Immediately background modal until next prompt is needed
            closeUpsampleModal();

            try {
                const res = await fetch('/api/upsample/respond', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        choice: choice,
                        custom_params: customParams
                    })
                });
                const data = await res.json();
                if (data.status !== 'ok') {
                    alert('Error submitting choice: ' + (data.message || 'Unknown error'));
                }
            } catch (err) {
                alert('Failed to send decision: ' + err.message);
            }
        }

        let activeTelemetryEvents = [];

        function parseTelemetryEvent(line) {
            const text = line.text || '';
            const t = line.time || '';
            
            // Filter out raw CLI ASCII dividers and repetitive terminal noise
            if (!text || text.match(/^=+$/) || text.match(/^-+$/) || text.match(/^>+$/) || text.includes('ACOUSTISINC: 64-BIT GPU')) {
                return null;
            }
            if (text.includes('Source Path') || text.includes('Destination Path') || text.includes('Rate Multiple') || text.includes('Quality:') || text.includes('Command line')) {
                return null;
            }

            let type = 'info';
            let tag = 'INFO';
            let cleanText = text;

            if (text.includes('GPU Sinc complete') || text.includes('Batched GPU Sinc') || text.includes('VkFFT')) {
                type = 'gpu';
                tag = '⚡ GPU Sinc';
                const m = text.match(/complete in ([0-9.]+s)/);
                cleanText = `64-bit GPU Sinc transform completed ${m ? '(' + m[1] + ')' : ''}`;
            } else if (text.includes('Noise Shaping completed') || text.includes('Shibata')) {
                type = 'dither';
                tag = '🎛️ Shibata';
                const m = text.match(/completed in ([0-9.]+s)/);
                cleanText = `Multi-core 24-bit psychoacoustic noise shaping ${m ? '(' + m[1] + ')' : ''}`;
            } else if (text.includes('Handed off to NVMe Writer') || text.includes('Async Writer') || text.includes('FLAC')) {
                type = 'io';
                tag = '💾 NVMe FLAC';
                cleanText = 'Lossless FLAC compression (Level 5) queued for background NVMe disk write';
            } else if (text.includes('Scanning') && text.includes('headroom')) {
                type = 'headroom';
                tag = '🔍 Headroom';
                cleanText = 'Scanning inter-sample headroom peaks across album tracks';
            } else if (text.includes('Gain Factor:') || text.includes('Dynamic Backoff') || text.includes('ABORTING CURRENT PASS')) {
                type = 'headroom';
                tag = '⚠️ Headroom';
                cleanText = text.replace(/>>>/g, '').trim();
            } else if (text.includes('Processing track:') || text.includes('--- Track [')) {
                type = 'info';
                tag = '🎵 Track';
                cleanText = text.replace(/---/g, '').replace(/Processing track:/g, '').trim();
            } else if (text.includes('Comparative Upsampling Report Generated') || text.includes('ALBUM_REPORT.html') || text.includes('Interactive HTML Report:')) {
                type = 'report';
                tag = '📊 Report';
                cleanText = text.includes('HTML:') ? text.split('HTML:')[1].trim() : 'Interactive forensic HTML & Markdown reports generated';
            } else if (text.includes('Tagging') || text.includes('ReplayGain')) {
                type = 'io';
                tag = '🏷️ Metadata';
                cleanText = 'Preserving metadata tags and computing album ReplayGain';
            } else if (text.includes('Error') || text.includes('Fatal') || text.includes('Exception')) {
                type = 'error';
                tag = '❌ Error';
                cleanText = text.trim();
            } else if (text.includes('Directory completed cleanly') || text.includes('Finished Successfully')) {
                type = 'report';
                tag = '✅ Complete';
                cleanText = 'Album upsampling & forensic analysis finished cleanly';
            } else {
                cleanText = text.replace(new RegExp('^[>\\\\-\\\\s]+'), '').trim();
                if (!cleanText || cleanText.length < 3) return null;
            }

            return {
                time: t,
                type: type,
                tag: tag,
                text: cleanText
            };
        }

        function toggleUpsampleDropdown(event) {
            if (event) event.stopPropagation();
            const container = document.getElementById('headerUpsampleContainer');
            if (container) {
                container.classList.toggle('is-open');
            }
        }

        document.addEventListener('click', (e) => {
            const container = document.getElementById('headerUpsampleContainer');
            if (container && !container.contains(e.target)) {
                container.classList.remove('is-open');
            }
        });

        function startPollingUpsampleStatus() {
            if (upsamplePollTimer) clearInterval(upsamplePollTimer);
            upsamplePollTimer = setInterval(pollUpsampleStatus, 700);
            pollUpsampleStatus();
        }

        async function pollUpsampleStatus() {
            try {
                const res = await fetch(`/api/upsample/status?since_log=${lastLogIdx}`);
                const data = await res.json();
                if (!data) return;
                activeUpsampleJob = data;

                // Append logs to terminal & UI telemetry stream
                if (data.logs && data.logs.length > 0) {
                    const logsContainer = document.getElementById('upsampleTerminalLogs');
                    const telemetryStream = document.getElementById('popoverEventStream');

                    data.logs.forEach(l => {
                        lastLogIdx = Math.max(lastLogIdx, l.idx);

                        // 1. Raw terminal log line
                        if (logsContainer) {
                            const lineEl = document.createElement('div');
                            lineEl.className = 'drawer-terminal-line' + (l.text.includes('Error') ? ' error' : l.text.includes('Complete') || l.text.includes('==') ? ' highlight' : '');
                            lineEl.textContent = `[${l.time}] ${l.text}`;
                            logsContainer.appendChild(lineEl);
                        }

                        // 2. UI-Native Telemetry Event
                        const ev = parseTelemetryEvent(l);
                        if (ev && telemetryStream) {
                            activeTelemetryEvents.push(ev);
                            if (activeTelemetryEvents.length > 50) activeTelemetryEvents.shift();

                            // Remove empty placeholder if present
                            const emptyPlaceholder = telemetryStream.querySelector('.telemetry-empty');
                            if (emptyPlaceholder) emptyPlaceholder.remove();

                            const evEl = document.createElement('div');
                            evEl.className = `telemetry-event event-${ev.type}`;
                            evEl.innerHTML = `
                                <span class="telemetry-time">${escapeHtml(ev.time)}</span>
                                <span class="telemetry-tag">${escapeHtml(ev.tag)}</span>
                                <span class="telemetry-text">${escapeHtml(ev.text)}</span>
                            `;
                            telemetryStream.appendChild(evEl);
                            telemetryStream.scrollTop = telemetryStream.scrollHeight;
                        }
                    });
                    
                    const terminal = document.getElementById('upsampleTerminalBox');
                    if (terminal) terminal.scrollTop = terminal.scrollHeight;
                }

                updateHeaderUpsampleUI(data);

                // Handle interactive decision prompt
                if (data.status === 'waiting_for_input' && data.prompt_data) {
                    currentPromptData = data.prompt_data;
                    const trackFile = data.prompt_data.track_file;
                    if (activePromptTrack !== trackFile) {
                        activePromptTrack = trackFile;
                        renderInteractivePromptUI(data.prompt_data);
                        const modal = document.getElementById('upsampleModal');
                        if (modal) modal.style.display = 'flex';
                    }
                } else if (data.status === 'running') {
                    currentPromptData = null;
                } else if (data.status === 'completed') {
                    currentPromptData = null;
                    closeUpsampleModal();
                } else if (data.status === 'failed' || data.status === 'cancelled') {
                    dismissUpsampleHeaderWidget();
                    closeUpsampleModal();
                }
            } catch (err) {}
        }

        function updateHeaderUpsampleUI(data) {
            const container = document.getElementById('headerUpsampleContainer');
            const btnUpsample = document.getElementById('headerBtnUpsample');
            const dot = document.getElementById('headerUpsampleDot');
            const title = document.getElementById('headerUpsampleTitle');
            const sub = document.getElementById('headerUpsampleSub');
            const badge = document.getElementById('headerUpsampleBadge');
            const fill = document.getElementById('headerUpsampleFill');
            
            // Popover elements
            const popTrack = document.getElementById('popoverCurrentTrack');
            const popTime = document.getElementById('popoverTimeElapsed');
            const popFill = document.getElementById('popoverProgressFill');
            const popPct = document.getElementById('popoverProgressPct');
            const popEventCount = document.getElementById('popoverEventCount');
            const popReviewBtn = document.getElementById('popoverReviewBtn');
            const popReportBtn = document.getElementById('popoverReportBtn');
            const popAbortBtn = document.getElementById('popoverAbortBtn');
            const popDismissBtn = document.getElementById('popoverDismissBtn');
            const recButtons = document.getElementById('actionRecButtons');

            if (!container) return;

            if (!data || data.status === 'idle' || data.status === 'cancelled') {
                dismissUpsampleHeaderWidget();
                return;
            }

            // Replace header button with active status widget to prevent overlap
            if (btnUpsample) btnUpsample.style.display = 'none';
            container.style.display = 'inline-flex';

            // Disable track recommendation buttons while an active job is processing
            if (recButtons && (data.status === 'running' || data.status === 'waiting_for_input')) {
                recButtons.querySelectorAll('button').forEach(b => { b.disabled = true; b.style.opacity = '0.4'; });
            } else if (recButtons) {
                recButtons.querySelectorAll('button').forEach(b => { b.disabled = false; b.style.opacity = '1'; });
            }

            const pct = Math.min(100, Math.max(0, data.progress_percent || 0));
            const pctStr = `${pct.toFixed(0)}%`;
            if (fill) fill.style.width = `${pct}%`;
            if (popFill) popFill.style.width = `${pct}%`;
            if (popPct) popPct.textContent = pctStr;

            const elapsedSec = data.elapsed_seconds || 0;
            const m = Math.floor(elapsedSec / 60);
            const s = Math.floor(elapsedSec % 60);
            const timeStr = `${m}:${('0' + s).slice(-2)} elapsed`;
            if (popTime) popTime.textContent = timeStr;
            if (popEventCount) popEventCount.textContent = `${activeTelemetryEvents.length} events`;

            if (data.status === 'waiting_for_input') {
                dot.style.background = 'var(--accent-yellow)';
                dot.style.boxShadow = '0 0 8px var(--accent-yellow)';
                dot.style.animation = 'pulse-glow 1s infinite ease-in-out';
                title.textContent = `⚠️ Input Required`;
                if (sub) sub.textContent = `[${data.track_index || 1}/${data.total_tracks || 1}] ${data.current_track || 'Track'}`;
                badge.textContent = `[${data.track_index || 1}/${data.total_tracks || 1}]`;
                badge.style.color = 'var(--accent-yellow)';
                if (popTrack) popTrack.textContent = `[Track ${data.track_index || 1}/${data.total_tracks || 1}] ${data.current_track || 'Track'}`;
                if (popReviewBtn) popReviewBtn.style.display = 'inline-block';
                if (popReportBtn) popReportBtn.style.display = 'none';
                if (popAbortBtn) popAbortBtn.style.display = 'inline-block';
                if (popDismissBtn) popDismissBtn.style.display = 'none';
            } else if (data.status === 'running') {
                dot.style.background = 'var(--accent-cyan)';
                dot.style.boxShadow = '0 0 8px var(--accent-cyan)';
                dot.style.animation = 'pulse-glow 1.2s infinite ease-in-out';
                const trkName = data.current_track ? (data.current_track.length > 22 ? data.current_track.substring(0, 20) + '...' : data.current_track) : 'Processing...';
                title.textContent = `⚡ [${data.track_index || 1}/${data.total_tracks || 1}] ${trkName}`;
                if (sub) sub.textContent = data.stage || '64-bit Sinc Resampling';
                badge.textContent = pctStr;
                badge.style.color = 'var(--accent-cyan)';
                if (popTrack) popTrack.textContent = `[Track ${data.track_index || 1}/${data.total_tracks || 1}] ${data.current_track || 'Processing...'}`;
                if (popReviewBtn) popReviewBtn.style.display = 'none';
                if (popReportBtn) popReportBtn.style.display = 'none';
                if (popAbortBtn) popAbortBtn.style.display = 'inline-block';
                if (popDismissBtn) popDismissBtn.style.display = 'none';
            } else if (data.status === 'completed') {
                dot.style.background = '#00e676';
                dot.style.boxShadow = '0 0 8px #00e676';
                dot.style.animation = 'none';
                title.textContent = '✅ Upsampling Complete';
                if (sub) sub.textContent = '100% Finished Successfully';
                badge.textContent = '100%';
                badge.style.color = '#00e676';
                if (fill) fill.style.width = '100%';
                if (popFill) popFill.style.width = '100%';
                if (popTrack) popTrack.textContent = 'All tracks processed and tagged with ReplayGain';
                if (popReviewBtn) popReviewBtn.style.display = 'none';
                if (popAbortBtn) popAbortBtn.style.display = 'none';
                if (popReportBtn) {
                    popReportBtn.style.display = 'inline-block';
                    popReportBtn.setAttribute('data-report-url', data.report_url || '');
                }
                if (popDismissBtn) popDismissBtn.style.display = 'inline-block';
            } else if (data.status === 'failed') {
                dot.style.background = '#ff5252';
                dot.style.boxShadow = '0 0 8px #ff5252';
                dot.style.animation = 'none';
                title.textContent = '❌ Upsampling Failed';
                if (sub) sub.textContent = data.error_message || 'Processing Error';
                badge.textContent = 'Error';
                badge.style.color = '#ff5252';
                if (popTrack) popTrack.textContent = data.error_message || 'Processing Error';
                if (popReviewBtn) popReviewBtn.style.display = 'none';
                if (popAbortBtn) popAbortBtn.style.display = 'none';
                if (popReportBtn) popReportBtn.style.display = 'none';
                if (popDismissBtn) popDismissBtn.style.display = 'inline-block';
            }
        }

        async function cancelUpsampleJob() {
            if (!confirm('Are you sure you want to abort the current upsampling job?')) return;
            try {
                const res = await fetch('/api/upsample/cancel', { method: 'POST' });
                const data = await res.json();
                if (data.status !== 'ok' && data.message !== 'No job is currently running.') {
                    alert('Failed to cancel job: ' + (data.message || 'Unknown error'));
                }
            } catch (err) {
                console.error('Error cancelling job:', err);
            } finally {
                closeUpsampleModal();
                dismissUpsampleHeaderWidget();
            }
        }

        function toggleUpsampleLogModal() {
            const modal = document.getElementById('upsampleLogModal');
            if (!modal) return;
            const isShown = modal.style.display !== 'none';
            modal.style.display = isShown ? 'none' : 'flex';
            if (!isShown) {
                const terminal = document.getElementById('upsampleTerminalBox');
                if (terminal) terminal.scrollTop = terminal.scrollHeight;
            }
        }

        function closeUpsampleLogModal() {
            const modal = document.getElementById('upsampleLogModal');
            if (modal) modal.style.display = 'none';
        }

        function dismissUpsampleHeaderWidget() {
            const container = document.getElementById('headerUpsampleContainer');
            if (container) {
                container.style.display = 'none';
                container.classList.remove('is-open');
            }
            const btnUpsample = document.getElementById('headerBtnUpsample');
            if (btnUpsample) {
                const hasFiles = !directoryData || !directoryData.files || directoryData.files.length > 0;
                btnUpsample.style.display = hasFiles ? 'inline-block' : 'none';
            }
            activeUpsampleJob = null;
            currentPromptData = null;
            activePromptTrack = null;
            activeTelemetryEvents = [];
            const telemetryStream = document.getElementById('popoverEventStream');
            if (telemetryStream) {
                telemetryStream.innerHTML = '<div class="telemetry-empty">Waiting for telemetry stream...</div>';
            }
            const recButtons = document.getElementById('actionRecButtons');
            if (recButtons) {
                recButtons.querySelectorAll('button').forEach(b => { b.disabled = false; b.style.opacity = '1'; });
            }
            if (upsamplePollTimer) {
                clearInterval(upsamplePollTimer);
                upsamplePollTimer = null;
            }
        }

        function viewGeneratedReport() {
            let reportUrl = null;
            const popReportBtn = document.getElementById('popoverReportBtn');
            if (popReportBtn && popReportBtn.getAttribute('data-report-url')) {
                reportUrl = popReportBtn.getAttribute('data-report-url');
            } else if (activeUpsampleJob && activeUpsampleJob.report_url) {
                reportUrl = activeUpsampleJob.report_url;
            }
            if (reportUrl) {
                const modal = document.getElementById('albumSummaryModal');
                const body = document.getElementById('modalSummaryBody');
                const badge = document.getElementById('modalSummaryBadge');
                const extBtn = document.getElementById('modalOpenExternalBtn');
                if (modal && body) {
                    modal.style.display = 'flex';
                    badge.textContent = 'ALBUM_REPORT.html (Freshly Mastered)';
                    extBtn.style.display = 'inline-block';
                    currentSummaryRawUrl = reportUrl;
                    body.innerHTML = `<iframe class="modal-iframe" src="${reportUrl}" style="width: 100%; height: 100%; min-height: 72vh; border: none; border-radius: 6px; background: #0d1117;"></iframe>`;
                }
            }
        }

        // Modal keyboard shortcuts
        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                closeAlbumSummaryModal();
                closeUpsampleModal();
                closeUpsampleLogModal();
                closeFolderPicker();
                return;
            }

            const upsampleModal = document.getElementById('upsampleModal');
            const isModalOpen = upsampleModal && upsampleModal.style.display !== 'none';
            const isTyping = ['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName);

            if (isModalOpen && !isTyping) {
                const k = e.key.toLowerCase();
                if (k === 'r') {
                    e.preventDefault();
                    applyForensicRecommendationPreset();
                } else if (k === 'y') {
                    e.preventDefault();
                    const interView = document.getElementById('interactiveDecisionControls');
                    if (interView && interView.style.display !== 'none') {
                        sendPromptChoice('y');
                    } else {
                        startUpsampleJob(true);
                    }
                } else if (k === 'a') {
                    e.preventDefault();
                    const interView = document.getElementById('interactiveDecisionControls');
                    if (interView && interView.style.display !== 'none') {
                        sendPromptChoice('a');
                    } else {
                        startUpsampleJob(false);
                    }
                } else if (k === 'c') {
                    e.preventDefault();
                    const interView = document.getElementById('interactiveDecisionControls');
                    if (interView && interView.style.display !== 'none') {
                        sendPromptChoice('c');
                    } else {
                        startUpsampleJob(false);
                    }
                } else if (k === 's') {
                    e.preventDefault();
                    const interView = document.getElementById('interactiveDecisionControls');
                    if (interView && interView.style.display !== 'none') {
                        sendPromptChoice('s');
                    }
                } else if (k === 'k') {
                    e.preventDefault();
                    const interView = document.getElementById('interactiveDecisionControls');
                    if (interView && interView.style.display !== 'none') {
                        sendPromptChoice('k');
                    } else {
                        closeUpsampleModal();
                    }
                } else if (k === 'q') {
                    e.preventDefault();
                    const interView = document.getElementById('interactiveDecisionControls');
                    if (interView && interView.style.display !== 'none') {
                        sendPromptChoice('q');
                    } else {
                        cancelUpsampleJob();
                    }
                }
            }
        });

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

        elif path == "/api/album_summary":
            target = params.get("path", [""])[0]
            summary_info = find_album_summary_in_dir(target)
            if not summary_info:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "found": False, "message": "No album analysis summary found"}).encode("utf-8"))
            else:
                try:
                    with open(summary_info["path"], "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    res = {
                        "status": "ok",
                        "found": True,
                        "filename": summary_info["filename"],
                        "path": summary_info["path"],
                        "type": summary_info["type"],
                        "is_mirror": summary_info.get("is_mirror", False),
                        "raw_url": f"/api/raw_summary?path={urllib.parse.quote(summary_info['path'])}",
                        "content": content
                    }
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(res).encode("utf-8"))
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))

        elif path == "/api/raw_summary":
            target = params.get("path", [""])[0]
            if not os.path.exists(target) or not os.path.isfile(target):
                self.send_response(404)
                self.end_headers()
                return

            mime = "text/html; charset=utf-8" if target.lower().endswith(".html") else "text/plain; charset=utf-8"
            try:
                with open(target, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception:
                self.send_response(500)
                self.end_headers()
        elif path == "/api/upsample/status":
            since_log = 0
            try:
                since_log = int(params.get("since_log", ["0"])[0])
            except Exception:
                pass
            status_data = upsample_job_mgr.get_status(since_idx=since_log)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status_data).encode("utf-8"))

        elif path == "/api/upsample/dest_preview":
            src = params.get("path", [""])[0]
            dest = derive_default_destination_dir(src)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "source_path": src, "dest_dir": dest}).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/upsample/start":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                config = json.loads(body)
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": f"Invalid JSON body: {e}"}).encode("utf-8"))
                return

            res = upsample_job_mgr.start_job(config)
            self.send_response(200 if res.get("status") == "ok" else 400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))

        elif path == "/api/upsample/respond":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                resp = json.loads(body)
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": f"Invalid JSON body: {e}"}).encode("utf-8"))
                return

            res = upsample_job_mgr.send_response(resp)
            self.send_response(200 if res.get("status") == "ok" else 400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))

        elif path == "/api/upsample/cancel":
            res = upsample_job_mgr.cancel_job()
            self.send_response(200 if res.get("status") == "ok" else 400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))

        elif path == "/api/mkdir":
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                req_data = json.loads(body)
                dir_path = req_data.get("path", "").strip()
                if not dir_path:
                    raise ValueError("Path cannot be empty")
                dir_path = os.path.abspath(os.path.expanduser(dir_path))
                os.makedirs(dir_path, exist_ok=True)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "created_path": dir_path}).encode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))

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
