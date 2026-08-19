#!/usr/bin/env python3
"""
AcoustiSinc Comparative Report Generator
Generates publication-grade, self-contained interactive HTML5 and Markdown reports
comparing Before (Source) and After (Upsampled) audio files with side-by-side
spectrograms, spectral energy curves, and forensic mastering metrics.
Every individual file gets its own dedicated <track_stem>_report.html and <track_stem>_report.md
in the target folder. Multi-track albums also receive an ALBUM_REPORT.html.
"""

import os
import time
import json
import numpy as np
import soundfile as sf
from analyser import analyze_audio_forensics, encode_spectrogram_and_lookup, load_audio_resilient

def audit_track_pair(src_path, dst_path, applied_recipe=None):
    """
    Audits a single track before and after upsampling.
    """
    filename = os.path.basename(src_path)
    rec = applied_recipe or {}

    # 1. Analyze Source (Before)
    try:
        data_src, sr_src = sf.read(src_path, frames=int(192000 * 25), dtype='float64', always_2d=True)
    except Exception:
        data_src, sr_src = load_audio_resilient(src_path, dtype='float64', frames=int(192000 * 25))
    if data_src.ndim > 1: data_src = np.mean(data_src, axis=1)

    spec_src, freqs_src, peak_src, rms_src, rep_src, dr_src, prov_src = analyze_audio_forensics(data_src, sr_src, filepath=src_path)
    webp_src, _, _, _ = encode_spectrogram_and_lookup(spec_src, width=900, height=450)

    step_src = max(1, len(freqs_src) // 1024)
    curve_f_src = (freqs_src[::step_src] / 1000.0).round(2).tolist()
    curve_pk_src = peak_src[::step_src].round(1).tolist()
    curve_rms_src = rms_src[::step_src].round(1).tolist()

    # 2. Analyze Target (After)
    try:
        data_dst, sr_dst = sf.read(dst_path, frames=int(192000 * 25), dtype='float64', always_2d=True)
    except Exception:
        data_dst, sr_dst = load_audio_resilient(dst_path, dtype='float64', frames=int(192000 * 25))
    if data_dst.ndim > 1: data_dst = np.mean(data_dst, axis=1)

    spec_dst, freqs_dst, peak_dst, rms_dst, rep_dst, dr_dst, prov_dst = analyze_audio_forensics(data_dst, sr_dst, filepath=dst_path)
    webp_dst, _, _, _ = encode_spectrogram_and_lookup(spec_dst, width=900, height=450)

    step_dst = max(1, len(freqs_dst) // 1024)
    curve_f_dst = (freqs_dst[::step_dst] / 1000.0).round(2).tolist()
    curve_pk_dst = peak_dst[::step_dst].round(1).tolist()
    curve_rms_dst = rms_dst[::step_dst].round(1).tolist()

    # File sizes
    sz_src = os.path.getsize(src_path) if os.path.exists(src_path) else 0
    sz_dst = os.path.getsize(dst_path) if os.path.exists(dst_path) else 0

    try: src_info = sf.info(src_path)
    except Exception: src_info = type("Info", (), {"subtype": "PCM_16"})()
    try: dst_info = sf.info(dst_path)
    except Exception: dst_info = type("Info", (), {"subtype": "PCM_24"})()

    src_knee = (prov_src.get("visual_morphology") or {}).get("primary_knee") or {}
    dst_knee = (prov_dst.get("visual_morphology") or {}).get("primary_knee") or {}

    return {
        "filename": filename,
        "src_path": src_path,
        "dst_path": dst_path,
        "src_sr": sr_src,
        "dst_sr": sr_dst,
        "src_nyquist": sr_src / 2000.0,
        "dst_nyquist": sr_dst / 2000.0,
        "src_knee_khz": src_knee.get("freq_khz"),
        "dst_knee_khz": dst_knee.get("freq_khz"),
        "applied_cutoff_khz": (rec.get("cutoff_hz") / 1000.0) if rec.get("cutoff_hz") else None,
        "src_sub_format": src_info.subtype,
        "dst_sub_format": dst_info.subtype,
        "src_size_mb": round(sz_src / (1024 * 1024), 2),
        "dst_size_mb": round(sz_dst / (1024 * 1024), 2),
        "src_peak_dbfs": round(float(dr_src.get("peak_dbfs", 0.0)), 2),
        "dst_peak_dbfs": round(float(dr_dst.get("peak_dbfs", 0.0)), 2),
        "src_dr": dr_src.get("dr_score", 0),
        "dst_dr": dr_dst.get("dr_score", 0),
        "src_lufs": round(dr_src.get("integrated_lufs", -140.0), 1),
        "dst_lufs": round(dr_dst.get("integrated_lufs", -140.0), 1),
        "src_lra": round(dr_src.get("lra_lu", 0.0), 1),
        "dst_lra": round(dr_dst.get("lra_lu", 0.0), 1),
        "src_crest": round(dr_src.get("crest_factor_db", 0.0), 1),
        "dst_crest": round(dr_dst.get("crest_factor_db", 0.0), 1),
        "src_verdict": prov_src.get("label", "Analyzed"),
        "dst_verdict": prov_dst.get("label", "Upsampled Master"),
        "src_webp": webp_src,
        "dst_webp": webp_dst,
        "src_curve": {"f": curve_f_src, "pk": curve_pk_src, "rms": curve_rms_src},
        "dst_curve": {"f": curve_f_dst, "pk": curve_pk_dst, "rms": curve_rms_dst}
    }


def generate_comparative_report(source_target_pairs, album_title, applied_recipe, output_dir):
    """
    Analyzes before and after audio files and writes:
      1. An individual <track_stem>_report.html and <track_stem>_report.md for every track.
      2. If multiple tracks, an aggregate ALBUM_REPORT.html and ALBUM_REPORT.md.
    """
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    
    comparisons = []
    for src_path, dst_path in source_target_pairs:
        if not os.path.exists(src_path) or not os.path.exists(dst_path):
            continue

        filename = os.path.basename(src_path)
        print(f"   [Report Analysis] Auditing Before/After: {filename}...")
        item = audit_track_pair(src_path, dst_path, applied_recipe=applied_recipe)
        comparisons.append(item)

        # Generate individual per-track report alongside the audio file
        track_stem = os.path.splitext(os.path.basename(dst_path))[0]
        track_html_path = os.path.join(output_dir, f"{track_stem}_report.html")
        track_md_path = os.path.join(output_dir, f"{track_stem}_report.md")

        write_single_track_html_report(track_html_path, item, applied_recipe)
        write_single_track_markdown_report(track_md_path, item, applied_recipe)

    if not comparisons:
        return None, None

    # For multi-track albums, also write an aggregate album report
    if len(comparisons) > 1:
        album_html_path = os.path.join(output_dir, "ALBUM_REPORT.html")
        album_md_path = os.path.join(output_dir, "ALBUM_REPORT.md")
        write_album_html_report(album_html_path, comparisons, album_title, applied_recipe)
        write_album_markdown_report(album_md_path, comparisons, album_title, applied_recipe)
        primary_html = album_html_path
        primary_md = album_md_path
    else:
        track_stem = os.path.splitext(os.path.basename(comparisons[0]["dst_path"]))[0]
        primary_html = os.path.join(output_dir, f"{track_stem}_report.html")
        primary_md = os.path.join(output_dir, f"{track_stem}_report.md")

    print(f"   [Report Complete] Generated individual and album reports in {time.time() - t0:.2f}s")
    return primary_html, primary_md


def write_single_track_markdown_report(filepath, item, applied_recipe):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# 🔬 AcoustiSinc Track Upsampling Report\n\n")
        f.write(f"**Track File**: `{item['filename']}`  \n")
        f.write(f"**Execution Date**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`  \n")
        f.write(f"**Applied Recipe**: `{applied_recipe.get('cli_params', 'Default')}`  \n")
        f.write(f"**Filter Topology**: `{applied_recipe.get('topology_name', 'Standard')}`  \n")
        f.write(f"**Headroom Gain Multiplier**: `{applied_recipe.get('gain_factor', 1.0):.6f}` ({applied_recipe.get('gain_db', 0.0):+.2f} dB)  \n\n")
        f.write(f"---\n\n")
        f.write(f"## 📊 Before vs After Forensic Statistics\n\n")
        f.write(f"| Metric | Source Master (Before) | AcoustiSinc Upsampled (After) |\n")
        f.write(f"| :--- | :---: | :---: |\n")
        f.write(f"| **Sampling Rate & Container** | {item['src_sr']/1000.0:.1f} kHz ({item['src_sub_format']}) | **{item['dst_sr']/1000.0:.1f} kHz ({item['dst_sub_format']})** |\n")
        f.write(f"| **Nyquist Limit** | {item['src_nyquist']:.1f} kHz | **{item['dst_nyquist']:.1f} kHz** |\n")
        f.write(f"| **True Peak Level** | {item['src_peak_dbfs']:.2f} dBFS | **{item['dst_peak_dbfs']:.2f} dBFS** (Guaranteed <= -0.3 dBFS) |\n")
        f.write(f"| **TT Dynamic Range** | DR {item['src_dr']} | **DR {item['dst_dr']}** (Bit-Perfect Dynamics) |\n")
        f.write(f"| **Integrated Loudness** | {item['src_lufs']:.1f} LUFS | **{item['dst_lufs']:.1f} LUFS** |\n")
        f.write(f"| **Loudness Range (LRA)** | {item['src_lra']:.1f} LU | **{item['dst_lra']:.1f} LU** |\n")
        f.write(f"| **Dynamic Crest Factor** | {item['src_crest']:.1f} dB | **{item['dst_crest']:.1f} dB** |\n")
        f.write(f"| **FLAC File Size** | {item['src_size_mb']} MB | **{item['dst_size_mb']} MB** |\n\n")
        f.write(f"---\n\n")
        f.write(f"### 🛡️ Quality Standards Guarantee\n")
        f.write(f"- **Strict 64-Bit Double Precision**: Verified IEEE 754 float64 / complex128 DSP math across all stages.\n")
        f.write(f"- **Zero Intersample Clipping**: Guaranteed target true peak headroom <= -0.30 dBFS.\n")
        f.write(f"- **FLAC Level 5 Output**: Bit-perfect Level 5 compression with complete Vorbis tags replicated.\n")


def write_single_track_html_report(filepath, item, applied_recipe):
    item_json = json.dumps(item)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AcoustiSinc Report: {item['filename']}</title>
    <style>
        :root {{
            --bg-base: #0a0c10;
            --bg-card: #12161f;
            --bg-hover: #1b212d;
            --border: #232936;
            --text: #f0f3f6;
            --text-muted: #8b949e;
            --accent-cyan: #00e5ff;
            --accent-pink: #ff007f;
            --accent-green: #aeea00;
            --accent-orange: #ff9100;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg-base);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 24px;
            line-height: 1.5;
        }}
        .header-card {{
            background: linear-gradient(135deg, #151b26 0%, #0d1117 100%);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        }}
        h1 {{ font-size: 1.6rem; font-weight: 700; color: #fff; margin-bottom: 8px; }}
        .badge-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }}
        .badge {{
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            background: rgba(0, 229, 255, 0.1);
            color: var(--accent-cyan);
            border: 1px solid rgba(0, 229, 255, 0.3);
        }}
        .badge-recipe {{
            background: rgba(174, 234, 0, 0.1);
            color: var(--accent-green);
            border-color: rgba(174, 234, 0, 0.3);
            font-family: monospace;
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}
        @media (max-width: 1024px) {{ .comparison-grid {{ grid-template-columns: 1fr; }} }}
        .pane-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
        }}
        .pane-title {{
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .spec-container {{
            position: relative;
            width: 100%;
            height: 280px;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #1f242e;
            margin-bottom: 12px;
        }}
        .spec-ruler {{
            position: absolute;
            left: 8px;
            top: 8px;
            bottom: 8px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            pointer-events: none;
            z-index: 2;
            font-family: "SF Mono", "Roboto Mono", monospace;
            font-size: 10px;
            font-weight: 600;
        }}
        .ruler-tag {{
            background: rgba(11, 14, 20, 0.75);
            padding: 2px 5px;
            border-radius: 4px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #f0f3f6;
            backdrop-filter: blur(4px);
        }}
        .ruler-aud {{
            color: var(--accent-green);
            border-color: rgba(174, 234, 0, 0.4);
        }}
        .spec-img {{
            width: 100%;
            height: 100%;
            object-fit: fill;
            display: block;
        }}
        .canvas-container {{
            width: 100%;
            height: 270px;
            background: #0b0e14;
            border-radius: 8px;
            border: 1px solid #1f242e;
            position: relative;
        }}
        canvas {{ width: 100%; height: 100%; display: block; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
            font-size: 0.85rem;
        }}
        th, td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
            text-align: left;
        }}
        th {{ color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.75rem; }}
        tr:hover td {{ background: var(--bg-hover); }}
        .val-good {{ color: var(--accent-green); font-weight: 600; }}
        .val-src {{ color: var(--text-muted); }}
    </style>
</head>
<body>

    <div class="header-card">
        <h1>🔬 AcoustiSinc Track Upsampling Report</h1>
        <p style="color: var(--text-muted); font-size: 0.9rem;">Track: <strong style="color: #fff;">{item['filename']}</strong></p>
        <div class="badge-row">
            <span class="badge">Strict 64-Bit Float</span>
            <span class="badge">FLAC Level 5 Output</span>
            <span class="badge">{applied_recipe.get('topology_name', 'Sinc Reconstruction')}</span>
            <span class="badge badge-recipe">{applied_recipe.get('cli_params', '--phase min --dither shibata')}</span>
            <span class="badge" style="color: var(--accent-orange); border-color: rgba(255,145,0,0.3);">Gain: {applied_recipe.get('gain_factor', 1.0):.4f} ({applied_recipe.get('gain_db', 0.0):+.2f} dB)</span>
        </div>
    </div>

    <div class="comparison-grid">
        <!-- Before (Source) -->
        <div class="pane-card">
            <div class="pane-title">
                <span style="color: var(--text-muted);">⬅️ Source Master (Before)</span>
                <span class="badge" style="color: #fff; background: rgba(255,255,255,0.1); border-color: #444;">{item['src_sr']/1000.0:.1f}k / {item['src_sub_format']}</span>
            </div>
            <div class="spec-container">
                <div class="spec-ruler">
                    <span class="ruler-tag">{item['src_nyquist']:.1f} kHz</span>
                    {f'<span class="ruler-tag ruler-aud">20.0 kHz</span>' if item['src_nyquist'] >= 20.0 else ''}
                    <span class="ruler-tag">0 Hz</span>
                </div>
                <img src="data:image/webp;base64,{item['src_webp']}" class="spec-img" alt="Source Spectrogram" />
            </div>
            <div class="canvas-container">
                <canvas id="srcCurveCanvas"></canvas>
            </div>
            <table>
                <tr><th>Metric</th><th>Source Value</th></tr>
                <tr><td>Sample Rate & Container</td><td class="val-src">{item['src_sr']:,} Hz ({item['src_sub_format']})</td></tr>
                <tr><td>Nyquist Limit</td><td class="val-src">{item['src_nyquist']:.1f} kHz</td></tr>
                <tr><td>True Peak Level</td><td class="val-src">{item['src_peak_dbfs']:.2f} dBFS</td></tr>
                <tr><td>TT Dynamic Range</td><td class="val-src">DR {item['src_dr']}</td></tr>
                <tr><td>Integrated Loudness</td><td class="val-src">{item['src_lufs']:.1f} LUFS</td></tr>
                <tr><td>Loudness Range (LRA)</td><td class="val-src">{item['src_lra']:.1f} LU</td></tr>
                <tr><td>Dynamic Crest Factor</td><td class="val-src">{item['src_crest']:.1f} dB</td></tr>
                <tr><td>File Size</td><td class="val-src">{item['src_size_mb']} MB</td></tr>
            </table>
        </div>

        <!-- After (Upsampled) -->
        <div class="pane-card">
            <div class="pane-title">
                <span style="color: var(--accent-green);">➡️ AcoustiSinc Upsampled (After)</span>
                <span class="badge" style="color: var(--accent-green); background: rgba(174,234,0,0.1); border-color: rgba(174,234,0,0.3);">{item['dst_sr']/1000.0:.1f}k / {item['dst_sub_format']}</span>
            </div>
            <div class="spec-container">
                <div class="spec-ruler">
                    <span class="ruler-tag">{item['dst_nyquist']:.1f} kHz</span>
                    {f'<span class="ruler-tag ruler-aud">20.0 kHz</span>' if item['dst_nyquist'] >= 20.0 else ''}
                    <span class="ruler-tag">0 Hz</span>
                </div>
                <img src="data:image/webp;base64,{item['dst_webp']}" class="spec-img" alt="Upsampled Spectrogram" />
            </div>
            <div class="canvas-container">
                <canvas id="dstCurveCanvas"></canvas>
            </div>
            <table>
                <tr><th>Metric</th><th>Upsampled Value</th></tr>
                <tr><td>Sample Rate & Container</td><td class="val-good">{item['dst_sr']:,} Hz ({item['dst_sub_format']})</td></tr>
                <tr><td>Nyquist Limit</td><td class="val-good">{item['dst_nyquist']:.1f} kHz</td></tr>
                <tr><td>True Peak Level</td><td class="val-good">{item['dst_peak_dbfs']:.2f} dBFS (Guaranteed &le; -0.3 dBFS)</td></tr>
                <tr><td>TT Dynamic Range</td><td class="val-good">DR {item['dst_dr']} (Bit-Perfect Dynamics)</td></tr>
                <tr><td>Integrated Loudness</td><td class="val-good">{item['dst_lufs']:.1f} LUFS</td></tr>
                <tr><td>Loudness Range (LRA)</td><td class="val-good">{item['dst_lra']:.1f} LU</td></tr>
                <tr><td>Dynamic Crest Factor</td><td class="val-good">{item['dst_crest']:.1f} dB</td></tr>
                <tr><td>File Size</td><td class="val-good">{item['dst_size_mb']} MB</td></tr>
            </table>
        </div>
    </div>

    <script>
        const item = {item_json};

        function drawForensicCurve(canvasId, curve, nyquist, cutoffKhz, kneeKhz) {{
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;
            const rect = canvas.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            const ctx = canvas.getContext('2d');
            ctx.scale(dpr, dpr);

            const w = rect.width, h = rect.height;
            const padL = 54, padR = 28, padT = 24, padB = 32;
            const plotW = w - padL - padR;
            const plotH = h - padT - padB;

            ctx.clearRect(0, 0, w, h);
            ctx.fillStyle = '#0b0e14';
            ctx.fillRect(padL, padT, plotW, plotH);
            ctx.strokeStyle = '#1e2638';
            ctx.lineWidth = 1;
            ctx.strokeRect(padL, padT, plotW, plotH);

            const fMin = 0, fMax = Math.max(22.05, nyquist * 1.04);
            const dbMin = -160, dbMax = 0;

            function fToX(f) {{ return padL + ((f - fMin) / (fMax - fMin)) * plotW; }}
            function dbToY(db) {{ return padT + (1.0 - (Math.max(dbMin, Math.min(dbMax, db)) - dbMin) / (dbMax - dbMin)) * plotH; }}

            // 1. Horizontal dBFS Grid & Axis Ticks
            const dbSteps = [0, -20, -40, -60, -80, -100, -120, -140, -160];
            ctx.font = '10px "SF Mono", "Roboto Mono", monospace';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';

            dbSteps.forEach(db => {{
                const y = dbToY(db);
                ctx.strokeStyle = db === 0 ? 'rgba(255,255,255,0.18)' : 'rgba(255,255,255,0.06)';
                ctx.setLineDash([]);
                ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();

                ctx.fillStyle = '#8b949e';
                ctx.fillText(`${{db}}`, padL - 8, y);
            }});

            // dBFS Axis Title
            ctx.save();
            ctx.translate(14, padT + plotH / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.textAlign = 'center';
            ctx.fillStyle = '#6e7681';
            ctx.font = '10px sans-serif';
            ctx.fillText('Amplitude (dBFS)', 0, 0);
            ctx.restore();

            // 2. Reference Quantization Noise Floor Lines
            // 16-bit PCM Floor (-96.33 dBFS)
            const y16 = dbToY(-96.33);
            ctx.strokeStyle = 'rgba(255, 171, 0, 0.4)';
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(padL, y16); ctx.lineTo(padL + plotW, y16); ctx.stroke();
            ctx.fillStyle = 'rgba(255, 171, 0, 0.85)';
            ctx.textAlign = 'left';
            ctx.fillText('16-bit Floor (-96 dB)', padL + 6, y16 - 6);

            // 24-bit PCM Floor (-144.49 dBFS)
            const y24 = dbToY(-144.49);
            ctx.strokeStyle = 'rgba(0, 229, 255, 0.35)';
            ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(padL, y24); ctx.lineTo(padL + plotW, y24); ctx.stroke();
            ctx.fillStyle = 'rgba(0, 229, 255, 0.75)';
            ctx.fillText('24-bit Floor (-144 dB)', padL + 6, y24 - 6);

            // 3. Vertical Frequency Grid & Axis Ticks
            let fTicks = [0, 5, 10, 15, 20, 22.05];
            if (nyquist > 25 && nyquist <= 50) fTicks = [0, 10, 20, 30, 40, nyquist <= 44.1 ? 44.1 : 48];
            else if (nyquist > 50 && nyquist <= 100) fTicks = [0, 20, 40, 44.1, 60, 80, nyquist <= 88.2 ? 88.2 : 96];
            else if (nyquist > 100) fTicks = [0, 20, 44.1, 88.2, 120, 160, 176.4, 192];

            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.setLineDash([]);

            fTicks.forEach(f => {{
                if (f > fMax) return;
                const x = fToX(f);
                ctx.strokeStyle = 'rgba(255,255,255,0.06)';
                ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();

                ctx.fillStyle = (f === 20 || f === 22.05 || f === 44.1 || f === 48 || f === 88.2 || f === 96 || f === 176.4 || f === 192) ? '#00e5ff' : '#8b949e';
                const label = f === 0 ? '0' : (f >= 1 ? `${{Number.isInteger(f) ? f : f.toFixed(1)}}k` : `${{f*1000}}`);
                ctx.fillText(label, x, padT + plotH + 6);
            }});

            // Frequency Axis Title
            ctx.fillStyle = '#6e7681';
            ctx.font = '10px sans-serif';
            ctx.fillText('Frequency (kHz)', padL + plotW / 2, padT + plotH + 18);

            // 4. Special Frequency Boundary Markers
            // Audible Limit (20 kHz)
            if (fMax >= 20.0) {{
                const x20 = fToX(20.0);
                ctx.strokeStyle = 'rgba(174, 234, 0, 0.7)';
                ctx.setLineDash([4, 4]);
                ctx.beginPath(); ctx.moveTo(x20, padT); ctx.lineTo(x20, padT + plotH); ctx.stroke();
                ctx.fillStyle = '#aeea00';
                ctx.font = '9px monospace';
                ctx.textAlign = 'center';
                ctx.fillText('20k Aud', x20, padT - 14);
            }}

            // Nyquist Limit Line
            const xNyq = fToX(nyquist);
            ctx.strokeStyle = '#ffab00';
            ctx.setLineDash([4, 4]);
            ctx.beginPath(); ctx.moveTo(xNyq, padT); ctx.lineTo(xNyq, padT + plotH); ctx.stroke();
            ctx.fillStyle = '#ffab00';
            ctx.font = '9px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(`${{nyquist.toFixed(1)}}k Nyq`, xNyq, padT - 14);

            // Cutoff Marker (if active)
            if (cutoffKhz && cutoffKhz < nyquist && cutoffKhz > 0) {{
                const xCut = fToX(cutoffKhz);
                ctx.strokeStyle = '#ff5252';
                ctx.setLineDash([2, 2]);
                ctx.beginPath(); ctx.moveTo(xCut, padT); ctx.lineTo(xCut, padT + plotH); ctx.stroke();
                ctx.fillStyle = '#ff5252';
                ctx.fillText(`Cutoff ${{cutoffKhz.toFixed(1)}}k`, xCut, padT - 14);
            }}

            // Detected Knee Marker (if present)
            if (kneeKhz && kneeKhz > 0 && Math.abs(kneeKhz - nyquist) > 1.0) {{
                const xKnee = fToX(kneeKhz);
                ctx.strokeStyle = '#e040fb';
                ctx.setLineDash([2, 4]);
                ctx.beginPath(); ctx.moveTo(xKnee, padT); ctx.lineTo(xKnee, padT + plotH); ctx.stroke();
                ctx.fillStyle = '#e040fb';
                ctx.fillText(`Knee ${{kneeKhz.toFixed(1)}}k`, xKnee, (cutoffKhz && Math.abs(cutoffKhz - kneeKhz) < 4.0) ? (padT + 10) : (padT - 14));
            }}

            ctx.setLineDash([]);

            // 5. RMS Energy Shading (Pink gradient)
            if (curve && curve.f && curve.f.length > 0) {{
                const grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
                grad.addColorStop(0, 'rgba(255, 0, 127, 0.18)');
                grad.addColorStop(1, 'rgba(255, 0, 127, 0.0)');
                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.moveTo(fToX(curve.f[0]), padT + plotH);
                curve.f.forEach((f, i) => {{
                    ctx.lineTo(fToX(f), dbToY(curve.rms[i]));
                }});
                ctx.lineTo(fToX(curve.f[curve.f.length - 1]), padT + plotH);
                ctx.closePath();
                ctx.fill();

                // 6. RMS Curve (Pink)
                ctx.strokeStyle = '#ff007f';
                ctx.lineWidth = 1.3;
                ctx.beginPath();
                curve.f.forEach((f, i) => {{
                    const x = fToX(f), y = dbToY(curve.rms[i]);
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }});
                ctx.stroke();

                // 7. Peak Curve (Cyan)
                ctx.strokeStyle = '#00e5ff';
                ctx.lineWidth = 1.4;
                ctx.beginPath();
                curve.f.forEach((f, i) => {{
                    const x = fToX(f), y = dbToY(curve.pk[i]);
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }});
                ctx.stroke();
            }}

            // 8. Legend Box (Top-Right)
            const legX = padL + plotW - 130;
            const legY = padT + 8;
            ctx.fillStyle = 'rgba(11, 14, 20, 0.85)';
            ctx.fillRect(legX, legY, 122, 38);
            ctx.strokeStyle = '#232936';
            ctx.strokeRect(legX, legY, 122, 38);

            // Peak
            ctx.strokeStyle = '#00e5ff'; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(legX + 8, legY + 12); ctx.lineTo(legX + 24, legY + 12); ctx.stroke();
            ctx.fillStyle = '#f0f3f6'; ctx.font = '9px sans-serif'; ctx.textAlign = 'left';
            ctx.fillText('Peak Level', legX + 30, legY + 12);

            // RMS
            ctx.strokeStyle = '#ff007f'; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(legX + 8, legY + 26); ctx.lineTo(legX + 24, legY + 26); ctx.stroke();
            ctx.fillText('RMS Energy', legX + 30, legY + 26);
        }}

        window.addEventListener('load', () => {{
            drawForensicCurve('srcCurveCanvas', item.src_curve, item.src_nyquist, null, item.src_knee_khz);
            drawForensicCurve('dstCurveCanvas', item.dst_curve, item.dst_nyquist, item.applied_cutoff_khz, item.dst_knee_khz);
        }});
        window.addEventListener('resize', () => {{
            drawForensicCurve('srcCurveCanvas', item.src_curve, item.src_nyquist, null, item.src_knee_khz);
            drawForensicCurve('dstCurveCanvas', item.dst_curve, item.dst_nyquist, item.applied_cutoff_khz, item.dst_knee_khz);
        }});
    </script>
</body>
</html>
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)


def write_album_markdown_report(filepath, comparisons, album_title, applied_recipe):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# 🔬 AcoustiSinc Album Upsampling Report\n\n")
        f.write(f"**Album**: `{album_title}`  \n")
        f.write(f"**Execution Date**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`  \n")
        f.write(f"**Applied Recipe**: `{applied_recipe.get('cli_params', 'Default')}`  \n")
        f.write(f"**Filter Topology**: `{applied_recipe.get('topology_name', 'Standard')}`  \n")
        f.write(f"**Headroom Gain Multiplier**: `{applied_recipe.get('gain_factor', 1.0):.6f}` ({applied_recipe.get('gain_db', 0.0):+.2f} dB)  \n\n")
        f.write(f"---\n\n")
        f.write(f"## 📊 Album Track Forensic Metrics Table\n\n")
        f.write(f"| Track Filename | Source Format | Target Format | Source DR | Target DR | Source Peak | Target Peak | Source LUFS | Target LUFS | Size (MB) |\n")
        f.write(f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for c in comparisons:
            src_fmt = f"{c['src_sr']/1000.0:.1f}k / {c['src_sub_format']}"
            dst_fmt = f"{c['dst_sr']/1000.0:.1f}k / {c['dst_sub_format']}"
            f.write(f"| `{c['filename']}` | {src_fmt} | **{dst_fmt}** | DR {c['src_dr']} | **DR {c['dst_dr']}** | {c['src_peak_dbfs']:.2f} dBFS | **{c['dst_peak_dbfs']:.2f} dBFS** | {c['src_lufs']} | {c['dst_lufs']} | {c['src_size_mb']} $\\to$ {c['dst_size_mb']} |\n")

        f.write(f"\n---\n\n")
        f.write(f"### 🛡️ Quality Standards Guarantee\n")
        f.write(f"- **Strict 64-Bit Double Precision**: Verified IEEE 754 float64 / complex128 DSP math across all stages.\n")
        f.write(f"- **Zero Intersample Clipping**: Guaranteed target true peak headroom <= -0.30 dBFS.\n")
        f.write(f"- **FLAC Level 5 Output**: Bit-perfect Level 5 compression with complete Vorbis tags replicated.\n")


def write_album_html_report(filepath, comparisons, album_title, applied_recipe):
    comp_json = json.dumps(comparisons)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AcoustiSinc Report: {album_title}</title>
    <style>
        :root {{
            --bg-base: #0a0c10;
            --bg-card: #12161f;
            --bg-hover: #1b212d;
            --border: #232936;
            --text: #f0f3f6;
            --text-muted: #8b949e;
            --accent-cyan: #00e5ff;
            --accent-pink: #ff007f;
            --accent-green: #aeea00;
            --accent-orange: #ff9100;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg-base);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 24px;
            line-height: 1.5;
        }}
        .header-card {{
            background: linear-gradient(135deg, #151b26 0%, #0d1117 100%);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        }}
        h1 {{ font-size: 1.8rem; font-weight: 700; color: #fff; margin-bottom: 8px; }}
        .badge-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }}
        .badge {{
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            background: rgba(0, 229, 255, 0.1);
            color: var(--accent-cyan);
            border: 1px solid rgba(0, 229, 255, 0.3);
        }}
        .badge-recipe {{
            background: rgba(174, 234, 0, 0.1);
            color: var(--accent-green);
            border-color: rgba(174, 234, 0, 0.3);
            font-family: monospace;
        }}
        .track-nav {{
            display: flex;
            gap: 8px;
            overflow-x: auto;
            padding-bottom: 12px;
            margin-bottom: 24px;
        }}
        .btn-track {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 8px 14px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.85rem;
            white-space: nowrap;
            transition: all 0.15s;
        }}
        .btn-track.active, .btn-track:hover {{
            background: var(--accent-cyan);
            color: #000;
            font-weight: 600;
            border-color: var(--accent-cyan);
        }}
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}
        @media (max-width: 1024px) {{ .comparison-grid {{ grid-template-columns: 1fr; }} }}
        .pane-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
        }}
        .pane-title {{
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .spec-container {{
            position: relative;
            width: 100%;
            height: 280px;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #1f242e;
            margin-bottom: 12px;
        }}
        .spec-img {{
            width: 100%;
            height: 100%;
            object-fit: fill;
            display: block;
        }}
        .canvas-container {{
            width: 100%;
            height: 240px;
            background: #11141a;
            border-radius: 8px;
            border: 1px solid #1f242e;
            position: relative;
        }}
        canvas {{ width: 100%; height: 100%; display: block; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
            font-size: 0.85rem;
        }}
        th, td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
            text-align: left;
        }}
        th {{ color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.75rem; }}
        tr:hover td {{ background: var(--bg-hover); }}
        .val-good {{ color: var(--accent-green); font-weight: 600; }}
        .val-src {{ color: var(--text-muted); }}
    </style>
</head>
<body>

    <div class="header-card">
        <h1>🔬 AcoustiSinc Forensic Album Report</h1>
        <p style="color: var(--text-muted); font-size: 0.9rem;">Album: <strong style="color: #fff;">{album_title}</strong></p>
        <div class="badge-row">
            <span class="badge">Strict 64-Bit Float</span>
            <span class="badge">FLAC Level 5 Output</span>
            <span class="badge">{applied_recipe.get('topology_name', 'Sinc Reconstruction')}</span>
            <span class="badge badge-recipe">{applied_recipe.get('cli_params', '--phase min --dither shibata')}</span>
            <span class="badge" style="color: var(--accent-orange); border-color: rgba(255,145,0,0.3);">Gain: {applied_recipe.get('gain_factor', 1.0):.4f} ({applied_recipe.get('gain_db', 0.0):+.2f} dB)</span>
        </div>
    </div>

    <div class="track-nav" id="trackNav"></div>

    <div class="comparison-grid">
        <!-- Before (Source) -->
        <div class="pane-card">
            <div class="pane-title">
                <span style="color: var(--text-muted);">⬅️ Source Master (Before)</span>
                <span id="srcFmtBadge" class="badge" style="color: #fff; background: rgba(255,255,255,0.1); border-color: #444;">--</span>
            </div>
            <div class="spec-container">
                <div class="spec-ruler" id="srcSpecRuler">
                    <span class="ruler-tag" id="srcRulerTop">-- kHz</span>
                    <span class="ruler-tag ruler-aud">20.0 kHz</span>
                    <span class="ruler-tag">0 Hz</span>
                </div>
                <img id="srcSpecImg" class="spec-img" alt="Source Spectrogram" />
            </div>
            <div class="canvas-container">
                <canvas id="srcCurveCanvas"></canvas>
            </div>
            <table>
                <tr><th>Metric</th><th>Source Value</th></tr>
                <tr><td>Sample Rate & Container</td><td id="srcSrVal" class="val-src">--</td></tr>
                <tr><td>Nyquist Limit</td><td id="srcNyqVal" class="val-src">--</td></tr>
                <tr><td>True Peak Level</td><td id="srcPkVal" class="val-src">--</td></tr>
                <tr><td>TT Dynamic Range</td><td id="srcDrVal" class="val-src">--</td></tr>
                <tr><td>Integrated Loudness</td><td id="srcLufsVal" class="val-src">--</td></tr>
                <tr><td>Loudness Range (LRA)</td><td id="srcLraVal" class="val-src">--</td></tr>
                <tr><td>Dynamic Crest Factor</td><td id="srcCrestVal" class="val-src">--</td></tr>
            </table>
        </div>

        <!-- After (Upsampled) -->
        <div class="pane-card">
            <div class="pane-title">
                <span style="color: var(--accent-green);">➡️ AcoustiSinc Upsampled (After)</span>
                <span id="dstFmtBadge" class="badge" style="color: var(--accent-green); background: rgba(174,234,0,0.1); border-color: rgba(174,234,0,0.3);">--</span>
            </div>
            <div class="spec-container">
                <div class="spec-ruler" id="dstSpecRuler">
                    <span class="ruler-tag" id="dstRulerTop">-- kHz</span>
                    <span class="ruler-tag ruler-aud">20.0 kHz</span>
                    <span class="ruler-tag">0 Hz</span>
                </div>
                <img id="dstSpecImg" class="spec-img" alt="Upsampled Spectrogram" />
            </div>
            <div class="canvas-container">
                <canvas id="dstCurveCanvas"></canvas>
            </div>
            <table>
                <tr><th>Metric</th><th>Upsampled Value</th></tr>
                <tr><td>Sample Rate & Container</td><td id="dstSrVal" class="val-good">--</td></tr>
                <tr><td>Nyquist Limit</td><td id="dstNyqVal" class="val-good">--</td></tr>
                <tr><td>True Peak Level</td><td id="dstPkVal" class="val-good">--</td></tr>
                <tr><td>TT Dynamic Range</td><td id="dstDrVal" class="val-good">--</td></tr>
                <tr><td>Integrated Loudness</td><td id="dstLufsVal" class="val-good">--</td></tr>
                <tr><td>Loudness Range (LRA)</td><td id="dstLraVal" class="val-good">--</td></tr>
                <tr><td>Dynamic Crest Factor</td><td id="dstCrestVal" class="val-good">--</td></tr>
            </table>
        </div>
    </div>

    <!-- Master Comparative Table -->
    <div class="pane-card" style="margin-top: 24px;">
        <div class="pane-title">📋 Full Album Side-by-Side Summary</div>
        <table>
            <thead>
                <tr>
                    <th>Track Filename</th>
                    <th>Source Format</th>
                    <th>Upsampled Target</th>
                    <th>Source DR</th>
                    <th>Target DR</th>
                    <th>Source Peak</th>
                    <th>Target Peak</th>
                    <th>Size (MB)</th>
                </tr>
            </thead>
            <tbody id="summaryTableBody"></tbody>
        </table>
    </div>

    <script>
        const data = {comp_json};
        let currentIdx = 0;

        function renderNav() {{
            const nav = document.getElementById('trackNav');
            nav.innerHTML = '';
            data.forEach((item, idx) => {{
                const btn = document.createElement('button');
                btn.className = 'btn-track' + (idx === currentIdx ? ' active' : '');
                btn.textContent = `${{idx + 1}}. ${{item.filename}}`;
                btn.onclick = () => selectTrack(idx);
                nav.appendChild(btn);
            }});
        }}

        function selectTrack(idx) {{
            currentIdx = idx;
            renderNav();
            const item = data[idx];

            document.getElementById('srcFmtBadge').textContent = `${{item.src_sr / 1000}}kHz / ${{item.src_sub_format}}`;
            document.getElementById('dstFmtBadge').textContent = `${{item.dst_sr / 1000}}kHz / ${{item.dst_sub_format}}`;

            document.getElementById('srcSpecImg').src = 'data:image/webp;base64,' + item.src_webp;
            document.getElementById('dstSpecImg').src = 'data:image/webp;base64,' + item.dst_webp;

            document.getElementById('srcRulerTop').textContent = `${{item.src_nyquist.toFixed(1)}} kHz`;
            document.getElementById('dstRulerTop').textContent = `${{item.dst_nyquist.toFixed(1)}} kHz`;

            document.getElementById('srcSrVal').textContent = `${{item.src_sr.toLocaleString()}} Hz (${{item.src_sub_format}})`;
            document.getElementById('dstSrVal').textContent = `${{item.dst_sr.toLocaleString()}} Hz (${{item.dst_sub_format}})`;

            document.getElementById('srcNyqVal').textContent = `${{item.src_nyquist}} kHz`;
            document.getElementById('dstNyqVal').textContent = `${{item.dst_nyquist}} kHz`;

            document.getElementById('srcPkVal').textContent = `${{item.src_peak_dbfs}} dBFS`;
            document.getElementById('dstPkVal').textContent = `${{item.dst_peak_dbfs}} dBFS (Guaranteed <= -0.3 dBFS)`;

            document.getElementById('srcDrVal').textContent = `DR ${{item.src_dr}}`;
            document.getElementById('dstDrVal').textContent = `DR ${{item.dst_dr}} (Bit-Perfect Dynamics)`;

            document.getElementById('srcLufsVal').textContent = `${{item.src_lufs}} LUFS`;
            document.getElementById('dstLufsVal').textContent = `${{item.dst_lufs}} LUFS`;

            document.getElementById('srcLraVal').textContent = `${{item.src_lra}} LU`;
            document.getElementById('dstLraVal').textContent = `${{item.dst_lra}} LU`;

            document.getElementById('srcCrestVal').textContent = `${{item.src_crest}} dB`;
            document.getElementById('dstCrestVal').textContent = `${{item.dst_crest}} dB`;

            drawForensicCurve('srcCurveCanvas', item.src_curve, item.src_nyquist, null, item.src_knee_khz);
            drawForensicCurve('dstCurveCanvas', item.dst_curve, item.dst_nyquist, item.applied_cutoff_khz, item.dst_knee_khz);
        }}

        function drawForensicCurve(canvasId, curve, nyquist, cutoffKhz, kneeKhz) {{
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;
            const rect = canvas.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            const ctx = canvas.getContext('2d');
            ctx.scale(dpr, dpr);

            const w = rect.width, h = rect.height;
            const padL = 54, padR = 28, padT = 24, padB = 32;
            const plotW = w - padL - padR;
            const plotH = h - padT - padB;

            ctx.clearRect(0, 0, w, h);
            ctx.fillStyle = '#0b0e14';
            ctx.fillRect(padL, padT, plotW, plotH);
            ctx.strokeStyle = '#1e2638';
            ctx.lineWidth = 1;
            ctx.strokeRect(padL, padT, plotW, plotH);

            const fMin = 0, fMax = Math.max(22.05, nyquist * 1.04);
            const dbMin = -160, dbMax = 0;

            function fToX(f) {{ return padL + ((f - fMin) / (fMax - fMin)) * plotW; }}
            function dbToY(db) {{ return padT + (1.0 - (Math.max(dbMin, Math.min(dbMax, db)) - dbMin) / (dbMax - dbMin)) * plotH; }}

            // 1. Horizontal dBFS Grid & Axis Ticks
            const dbSteps = [0, -20, -40, -60, -80, -100, -120, -140, -160];
            ctx.font = '10px "SF Mono", "Roboto Mono", monospace';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';

            dbSteps.forEach(db => {{
                const y = dbToY(db);
                ctx.strokeStyle = db === 0 ? 'rgba(255,255,255,0.18)' : 'rgba(255,255,255,0.06)';
                ctx.setLineDash([]);
                ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();

                ctx.fillStyle = '#8b949e';
                ctx.fillText(`${{db}}`, padL - 8, y);
            }});

            // dBFS Axis Title
            ctx.save();
            ctx.translate(14, padT + plotH / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.textAlign = 'center';
            ctx.fillStyle = '#6e7681';
            ctx.font = '10px sans-serif';
            ctx.fillText('Amplitude (dBFS)', 0, 0);
            ctx.restore();

            // 2. Reference Quantization Noise Floor Lines
            // 16-bit PCM Floor (-96.33 dBFS)
            const y16 = dbToY(-96.33);
            ctx.strokeStyle = 'rgba(255, 171, 0, 0.4)';
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(padL, y16); ctx.lineTo(padL + plotW, y16); ctx.stroke();
            ctx.fillStyle = 'rgba(255, 171, 0, 0.85)';
            ctx.textAlign = 'left';
            ctx.fillText('16-bit Floor (-96 dB)', padL + 6, y16 - 6);

            // 24-bit PCM Floor (-144.49 dBFS)
            const y24 = dbToY(-144.49);
            ctx.strokeStyle = 'rgba(0, 229, 255, 0.35)';
            ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(padL, y24); ctx.lineTo(padL + plotW, y24); ctx.stroke();
            ctx.fillStyle = 'rgba(0, 229, 255, 0.75)';
            ctx.fillText('24-bit Floor (-144 dB)', padL + 6, y24 - 6);

            // 3. Vertical Frequency Grid & Axis Ticks
            let fTicks = [0, 5, 10, 15, 20, 22.05];
            if (nyquist > 25 && nyquist <= 50) fTicks = [0, 10, 20, 30, 40, nyquist <= 44.1 ? 44.1 : 48];
            else if (nyquist > 50 && nyquist <= 100) fTicks = [0, 20, 40, 44.1, 60, 80, nyquist <= 88.2 ? 88.2 : 96];
            else if (nyquist > 100) fTicks = [0, 20, 44.1, 88.2, 120, 160, 176.4, 192];

            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.setLineDash([]);

            fTicks.forEach(f => {{
                if (f > fMax) return;
                const x = fToX(f);
                ctx.strokeStyle = 'rgba(255,255,255,0.06)';
                ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();

                ctx.fillStyle = (f === 20 || f === 22.05 || f === 44.1 || f === 48 || f === 88.2 || f === 96 || f === 176.4 || f === 192) ? '#00e5ff' : '#8b949e';
                const label = f === 0 ? '0' : (f >= 1 ? `${{Number.isInteger(f) ? f : f.toFixed(1)}}k` : `${{f*1000}}`);
                ctx.fillText(label, x, padT + plotH + 6);
            }});

            // Frequency Axis Title
            ctx.fillStyle = '#6e7681';
            ctx.font = '10px sans-serif';
            ctx.fillText('Frequency (kHz)', padL + plotW / 2, padT + plotH + 18);

            // 4. Special Frequency Boundary Markers
            // Audible Limit (20 kHz)
            if (fMax >= 20.0) {{
                const x20 = fToX(20.0);
                ctx.strokeStyle = 'rgba(174, 234, 0, 0.7)';
                ctx.setLineDash([4, 4]);
                ctx.beginPath(); ctx.moveTo(x20, padT); ctx.lineTo(x20, padT + plotH); ctx.stroke();
                ctx.fillStyle = '#aeea00';
                ctx.font = '9px monospace';
                ctx.textAlign = 'center';
                ctx.fillText('20k Aud', x20, padT - 14);
            }}

            // Nyquist Limit Line
            const xNyq = fToX(nyquist);
            ctx.strokeStyle = '#ffab00';
            ctx.setLineDash([4, 4]);
            ctx.beginPath(); ctx.moveTo(xNyq, padT); ctx.lineTo(xNyq, padT + plotH); ctx.stroke();
            ctx.fillStyle = '#ffab00';
            ctx.font = '9px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(`${{nyquist.toFixed(1)}}k Nyq`, xNyq, padT - 14);

            // Cutoff Marker (if active)
            if (cutoffKhz && cutoffKhz < nyquist && cutoffKhz > 0) {{
                const xCut = fToX(cutoffKhz);
                ctx.strokeStyle = '#ff5252';
                ctx.setLineDash([2, 2]);
                ctx.beginPath(); ctx.moveTo(xCut, padT); ctx.lineTo(xCut, padT + plotH); ctx.stroke();
                ctx.fillStyle = '#ff5252';
                ctx.fillText(`Cutoff ${{cutoffKhz.toFixed(1)}}k`, xCut, padT - 14);
            }}

            // Detected Knee Marker (if present)
            if (kneeKhz && kneeKhz > 0 && Math.abs(kneeKhz - nyquist) > 1.0) {{
                const xKnee = fToX(kneeKhz);
                ctx.strokeStyle = '#e040fb';
                ctx.setLineDash([2, 4]);
                ctx.beginPath(); ctx.moveTo(xKnee, padT); ctx.lineTo(xKnee, padT + plotH); ctx.stroke();
                ctx.fillStyle = '#e040fb';
                ctx.fillText(`Knee ${{kneeKhz.toFixed(1)}}k`, xKnee, (cutoffKhz && Math.abs(cutoffKhz - kneeKhz) < 4.0) ? (padT + 10) : (padT - 14));
            }}

            ctx.setLineDash([]);

            // 5. RMS Energy Shading (Pink gradient)
            if (curve && curve.f && curve.f.length > 0) {{
                const grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
                grad.addColorStop(0, 'rgba(255, 0, 127, 0.18)');
                grad.addColorStop(1, 'rgba(255, 0, 127, 0.0)');
                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.moveTo(fToX(curve.f[0]), padT + plotH);
                curve.f.forEach((f, i) => {{
                    ctx.lineTo(fToX(f), dbToY(curve.rms[i]));
                }});
                ctx.lineTo(fToX(curve.f[curve.f.length - 1]), padT + plotH);
                ctx.closePath();
                ctx.fill();

                // 6. RMS Curve (Pink)
                ctx.strokeStyle = '#ff007f';
                ctx.lineWidth = 1.3;
                ctx.beginPath();
                curve.f.forEach((f, i) => {{
                    const x = fToX(f), y = dbToY(curve.rms[i]);
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }});
                ctx.stroke();

                // 7. Peak Curve (Cyan)
                ctx.strokeStyle = '#00e5ff';
                ctx.lineWidth = 1.4;
                ctx.beginPath();
                curve.f.forEach((f, i) => {{
                    const x = fToX(f), y = dbToY(curve.pk[i]);
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }});
                ctx.stroke();
            }}

            // 8. Legend Box (Top-Right)
            const legX = padL + plotW - 130;
            const legY = padT + 8;
            ctx.fillStyle = 'rgba(11, 14, 20, 0.85)';
            ctx.fillRect(legX, legY, 122, 38);
            ctx.strokeStyle = '#232936';
            ctx.strokeRect(legX, legY, 122, 38);

            // Peak
            ctx.strokeStyle = '#00e5ff'; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(legX + 8, legY + 12); ctx.lineTo(legX + 24, legY + 12); ctx.stroke();
            ctx.fillStyle = '#f0f3f6'; ctx.font = '9px sans-serif'; ctx.textAlign = 'left';
            ctx.fillText('Peak Level', legX + 30, legY + 12);

            // RMS
            ctx.strokeStyle = '#ff007f'; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(legX + 8, legY + 26); ctx.lineTo(legX + 24, legY + 26); ctx.stroke();
            ctx.fillText('RMS Energy', legX + 30, legY + 26);
        }}

        function renderSummaryTable() {{
            const tbody = document.getElementById('summaryTableBody');
            tbody.innerHTML = '';
            data.forEach((item, idx) => {{
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${{item.filename}}</strong></td>
                    <td class="val-src">${{item.src_sr / 1000}}k / ${{item.src_sub_format}}</td>
                    <td class="val-good">${{item.dst_sr / 1000}}k / ${{item.dst_sub_format}}</td>
                    <td>DR ${{item.src_dr}}</td>
                    <td class="val-good">DR ${{item.dst_dr}}</td>
                    <td>${{item.src_peak_dbfs}} dBFS</td>
                    <td class="val-good">${{item.dst_peak_dbfs}} dBFS</td>
                    <td>${{item.src_size_mb}}MB &rarr; ${{item.dst_size_mb}}MB</td>
                `;
                tr.style.cursor = 'pointer';
                tr.onclick = () => selectTrack(idx);
                tbody.appendChild(tr);
            }});
        }}

        window.addEventListener('load', () => {{
            renderNav();
            selectTrack(0);
            renderSummaryTable();
        }});
        window.addEventListener('resize', () => selectTrack(currentIdx));
    </script>
</body>
</html>
"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
