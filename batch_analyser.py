#!/usr/bin/env python3
"""
================================================================================
BATCH SPECTRUM & FORENSIC ANALYZER (HTML5 BATCH RUNNER)
================================================================================
Recursively scans a target music directory, and for each subfolder containing
FLAC audio files:
- Creates a dedicated 'spectrum_analysis' subfolder in that album directory.
- Generates an interactive HTML5 forensic report for each track.
- Idempotent: Automatically skips tracks that already have an analysis report.
- Multiprocessed: Utilizes all CPU cores for high-speed batch processing.
- Live progress reporting with ETA, throughput, and forensic summary statistics.
================================================================================
"""

import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import soundfile as sf
import numpy as np

# Import core DSP and HTML5 generation engine from analyser
from analyser import analyze_audio_forensics, generate_html5_report


def process_single_track(flac_path, force=False):
    """
    Analyzes a single track and generates its HTML5 report in the spectrum_analysis subfolder.
    """
    try:
        dir_name = os.path.dirname(flac_path)
        base_name = os.path.splitext(os.path.basename(flac_path))[0]
        out_dir = os.path.join(dir_name, "spectrum_analysis")
        out_html = os.path.join(out_dir, f"{base_name}_spectrum.html")

        # Skip if already analyzed (and not forced)
        if not force and os.path.exists(out_html) and os.path.getsize(out_html) > 1000:
            return {
                'status': 'skipped',
                'path': flac_path,
                'out': out_html
            }

        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()

        # Strict 64-bit double precision read (up to 60 seconds)
        data, sr = sf.read(flac_path, dtype='float64', start=0, stop=60*192000)
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Apply smooth 50ms boundary taper to eliminate edge truncation step discontinuities
        taper_len = min(int(sr * 0.05), len(data) // 10)
        if taper_len > 0:
            taper = np.sin(np.linspace(0, np.pi/2, taper_len))**2
            data[:taper_len] *= taper
            data[-taper_len:] *= taper[::-1]

        spec_db, freqs, peak_dbfs, rms_dbfs, assessment_text = analyze_audio_forensics(data, sr)
        generate_html5_report(data, sr, flac_path, out_html, spec_db, freqs, peak_dbfs, rms_dbfs, assessment_text)

        # Parse main assessment badge
        verdict = "ANALYZED"
        for line in assessment_text.splitlines():
            if line.startswith("ASSESSMENT:"):
                verdict = line.replace("ASSESSMENT:", "").strip()
                break

        elapsed = time.time() - t0
        return {
            'status': 'success',
            'path': flac_path,
            'out': out_html,
            'time': elapsed,
            'verdict': verdict,
            'sr': sr
        }
    except Exception as e:
        return {
            'status': 'error',
            'path': flac_path,
            'error': str(e)
        }


def collect_flac_files(root_dir):
    """
    Recursively discovers all valid .flac audio files in the target tree.
    """
    flac_files = []
    print(f"Scanning '{root_dir}' for FLAC files...")
    for root, dirs, files in os.walk(root_dir):
        # Ignore already existing spectrum_analysis subfolders
        if os.path.basename(root) == "spectrum_analysis":
            continue
        for f in files:
            if f.lower().endswith(".flac") and not f.endswith(".flac.WIP") and not f.startswith("._"):
                flac_files.append(os.path.join(root, f))
    return sorted(flac_files)


def format_time(seconds):
    """Formats seconds into hh:mm:ss or mm:ss."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def main():
    parser = argparse.ArgumentParser(description="Batch Spectrum & Forensic Analyzer for Music Libraries")
    parser.add_argument("--root", default="/mnt/PrimaryFS/1xxK_min/music", help="Root directory containing FLAC tracks")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 4), help="Number of parallel worker processes")
    parser.add_argument("--force", action="store_true", help="Force re-generation even if report already exists")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of tracks to process")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root)
    if not os.path.exists(root_dir):
        print(f"[Error] Target directory does not exist: {root_dir}")
        sys.exit(1)

    start_wall_time = time.time()
    tracks = collect_flac_files(root_dir)
    total_tracks = len(tracks)

    if args.limit and args.limit < total_tracks:
        tracks = tracks[:args.limit]
        total_tracks = len(tracks)

    print(f"Found {total_tracks:,} FLAC files across library subfolders.")
    print(f"Parallel Workers: {args.workers} (CPU Processes)")
    print(f"Overwrite Mode  : {'FORCED' if args.force else 'SKIP EXISTING'}")
    print("=" * 80)

    processed_count = 0
    skipped_count = 0
    success_count = 0
    error_count = 0

    verdict_counts = {}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit tasks
        future_to_track = {executor.submit(process_single_track, track, args.force): track for track in tracks}

        for future in as_completed(future_to_track):
            processed_count += 1
            res = future.result()
            status = res['status']
            track_path = res['path']
            rel_path = os.path.relpath(track_path, root_dir)

            elapsed_total = time.time() - start_wall_time
            rate = (processed_count / elapsed_total) if elapsed_total > 0 else 0
            remaining_tracks = total_tracks - processed_count
            eta_s = (remaining_tracks / rate) if rate > 0 else 0

            progress_pct = (processed_count / total_tracks) * 100.0

            if status == 'skipped':
                skipped_count += 1
                # Output a compact skip progress every 20 skips or on first/last
                if skipped_count == 1 or skipped_count % 25 == 0 or processed_count == total_tracks:
                    print(f"[{processed_count:5d}/{total_tracks:5d}] ({progress_pct:5.1f}%) | [SKIP] Already analyzed | {skipped_count} skipped | ETA: {format_time(eta_s)}")
            elif status == 'success':
                success_count += 1
                verdict = res.get('verdict', 'ANALYZED')
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
                t_dur = res.get('time', 0.0)
                print(f"[{processed_count:5d}/{total_tracks:5d}] ({progress_pct:5.1f}%) | [DONE] {verdict} ({t_dur:.1f}s) | {rel_path} | ETA: {format_time(eta_s)}")
            elif status == 'error':
                error_count += 1
                err_msg = res.get('error', 'Unknown error')
                print(f"[{processed_count:5d}/{total_tracks:5d}] ({progress_pct:5.1f}%) | [ERROR] {rel_path} -> {err_msg}")

    total_time = time.time() - start_wall_time
    print("=" * 80)
    print("BATCH FORENSIC ANALYSIS SUMMARY")
    print(f"Total Tracks Evaluated: {total_tracks:,}")
    print(f"Newly Generated       : {success_count:,}")
    print(f"Skipped (Pre-existing): {skipped_count:,}")
    print(f"Errors / Failures     : {error_count:,}")
    print(f"Total Elapsed Time    : {format_time(total_time)} ({total_tracks/total_time:.2f} tracks/sec)")

    if verdict_counts:
        print("\nForensic Classification Breakdown:")
        for v, cnt in sorted(verdict_counts.items(), key=lambda x: -x[1]):
            print(f"  - {v:50s} : {cnt:,} tracks")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
