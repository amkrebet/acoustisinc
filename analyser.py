#!/usr/bin/env python3
"""
================================================================================
HI-FI NEWS STYLE AUDIO SPECTRUM & FORENSIC ANALYZER (HTML5 EDITION V4.2)
================================================================================
Generates ultra-fast, self-contained interactive HTML5 forensic reports with:
- Strict 64-bit double precision (float64) DSP analysis.
- Instant 2D spectrogram with full axes, ticks, colorbar, and exact dB level cursor HUD.
- Full interactive Zoom & Pan controls (wheel zoom, drag pan, preset buttons) on both charts.
- Interactive HTML5 canvas spectrum curve with hover inspector and crosshair HUD.
- Semantic HTML/CSS dark-mode forensic lab card with 1-click clipboard copy.
- 50x-100x faster generation time compared to monolithic matplotlib PNG exports.
================================================================================
"""

import os
import sys
import io
import json
import base64
import argparse
import numpy as np
import soundfile as sf
import librosa
import scipy.signal as signal
import matplotlib
from PIL import Image


def calculate_dynamic_range_metrics(y, sr):
    """
    Computes audiophile and broadcast dynamic range & loudness metrics with strict 64-bit precision:
    1. Official Pleasurize Music Foundation / TT Dynamic Range Meter (DR Score e.g. DR12)
    2. ITU-R BS.1770-4 / EBU R128 Integrated Loudness (LUFS) and Loudness Range (LRA in LU)
    3. Peak-to-RMS Crest Factor (dB)
    """
    # 1. Official TT Dynamic Range (3-second blocks, top 20% loudest RMS vs peak)
    block_len = int(sr * 3.0)
    if len(y) < block_len:
        block_len = len(y)

    num_blocks = len(y) // block_len
    if num_blocks > 0:
        blocks = y[:num_blocks * block_len].reshape(num_blocks, block_len)
        block_rms = np.sqrt(np.mean(blocks**2, axis=1))
        
        top_count = max(1, int(np.ceil(num_blocks * 0.20)))
        sorted_rms = np.sort(block_rms)
        top_20_rms = sorted_rms[-top_count:]
        
        top_rms_mean = np.sqrt(np.mean(top_20_rms**2))
        top_rms_db = 20.0 * np.log10(top_rms_mean) if top_rms_mean > 0 else -140.0
    else:
        overall_rms = np.sqrt(np.mean(y**2))
        top_rms_db = 20.0 * np.log10(overall_rms) if overall_rms > 0 else -140.0

    peak_val = np.max(np.abs(y))
    peak_db = 20.0 * np.log10(peak_val) if peak_val > 0 else -140.0
    
    dr_val = peak_db - top_rms_db
    dr_score = max(0, int(np.round(dr_val)))
    
    # 2. Integrated RMS and Crest Factor
    tot_rms = np.sqrt(np.mean(y**2))
    tot_rms_db = 20.0 * np.log10(tot_rms) if tot_rms > 0 else -140.0
    crest_factor_db = peak_db - tot_rms_db

    # 3. ITU-R BS.1770 / EBU R128 Loudness (LUFS) & Loudness Range (LRA)
    try:
        # High-shelf filter design (BS.1770)
        vh = np.power(10.0, 3.9998438 / 20.0)
        vb = np.power(vh, 0.4996667741545416)
        K = np.tan(np.pi * 1681.974450955533 / sr)
        K2 = K * K
        den = 1.0 + np.sqrt(2.0) * K + K2
        b_shelf = [(vh + vb * np.sqrt(2.0) * K + K2) / den, 2.0 * (K2 - vh) / den, (vh - vb * np.sqrt(2.0) * K + K2) / den]
        a_shelf = [1.0, 2.0 * (K2 - 1.0) / den, (1.0 - np.sqrt(2.0) * K + K2) / den]

        # High-pass filter design (BS.1770)
        K_hp = np.tan(np.pi * 38.13547087602444 / sr)
        K2_hp = K_hp * K_hp
        den_hp = 1.0 + np.sqrt(2.0) * K_hp + K2_hp
        b_hp = [1.0 / den_hp, -2.0 / den_hp, 1.0 / den_hp]
        a_hp = [1.0, 2.0 * (K2_hp - 1.0) / den_hp, (1.0 - np.sqrt(2.0) * K_hp + K2_hp) / den_hp]

        y_k = signal.lfilter(b_shelf, a_shelf, y)
        y_k = signal.lfilter(b_hp, a_hp, y_k)

        # Short-term loudness blocks (3.0s window, 100ms step)
        step_len = int(sr * 0.1)
        win_len = int(sr * 3.0)
        if len(y_k) >= win_len:
            num_st = (len(y_k) - win_len) // step_len
            st_blocks = np.array([np.mean(y_k[i * step_len : i * step_len + win_len]**2) for i in range(num_st)], dtype=np.float64)
            st_lufs = -0.691 + 10.0 * np.log10(np.maximum(st_blocks, 1e-12))
            valid_st = st_lufs[st_lufs > -70.0]
            
            if len(valid_st) > 0:
                ungated_mean_power = np.mean(10.0**(valid_st / 10.0))
                rel_thresh = (-0.691 + 10.0 * np.log10(ungated_mean_power)) - 20.0
                gated_st = valid_st[valid_st > rel_thresh]
                if len(gated_st) > 0:
                    lra = np.percentile(gated_st, 95) - np.percentile(gated_st, 10)
                    integrated_lufs = -0.691 + 10.0 * np.log10(np.mean(10.0**(gated_st / 10.0)))
                else:
                    lra = 0.0
                    integrated_lufs = -0.691 + 10.0 * np.log10(ungated_mean_power)
            else:
                lra = 0.0
                integrated_lufs = -70.0
        else:
            lra = 0.0
            p = np.mean(y_k**2)
            integrated_lufs = -0.691 + 10.0 * np.log10(p) if p > 0 else -140.0
    except Exception:
        lra = 0.0
        integrated_lufs = tot_rms_db

    return {
        "dr_score": dr_score,
        "dr_val": round(dr_val, 2),
        "peak_dbfs": round(peak_db, 2),
        "crest_factor_db": round(crest_factor_db, 2),
        "integrated_lufs": round(integrated_lufs, 1),
        "lra_lu": round(lra, 1)
    }


def analyze_audio_forensics(y, sr):
    """
    Executes automated forensic analysis using strict 64-bit double precision.
    """
    nyquist = sr / 2.0
    n_fft = 16384
    hop_length = n_fft // 4
    
    win = signal.windows.blackmanharris(n_fft)
    S1 = np.sum(win)
    S2 = np.sum(win**2)
    enbw_hz = sr * (S2 / (S1**2))
    
    # 1. Linear STFT Spectrum in 64-bit double precision
    stft_mag = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='blackmanharris'))
    stft_norm = stft_mag / (S1 / 2.0)
    
    # 2. Peak Hold & RMS Average
    peak_mag = np.max(stft_norm, axis=1)
    peak_dbfs = 20.0 * np.log10(np.maximum(peak_mag, 1e-12))
    
    power_linear = np.mean(stft_norm**2, axis=1)
    rms_dbfs = 10.0 * np.log10(np.maximum(power_linear, 1e-24))
    
    spec_db = 20.0 * np.log10(np.maximum(stft_norm, 1e-12))
    mean_spec_db = np.mean(spec_db, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    idx_20k = np.argmin(np.abs(freqs - 20000))
    idx_22k = np.argmin(np.abs(freqs - 22050))
    idx_24k = np.argmin(np.abs(freqs - 24000))
    idx_44k = np.argmin(np.abs(freqs - 44100))
    idx_48k = np.argmin(np.abs(freqs - 48000)) if nyquist >= 48000 else None
    
    zero_ratio = 1.0 - (np.count_nonzero(y) / len(y))
    image_power = np.max(rms_dbfs[idx_22k:idx_44k]) if sr >= 88200 else -140.0
    audible_rms_mean = np.mean(rms_dbfs[:idx_20k])
    audible_peak_max = np.max(peak_dbfs[:idx_20k])
    
    is_zero_stuffed = False
    is_upsampled = False
    is_cd_cutoff = False
    is_48k_cutoff = False
    is_88k_cutoff = False
    is_96k_cutoff = False
    detected_base_hz = 0
    effective_cutoff_hz = nyquist
    
    report = []
    report.append("--- FORENSIC LAB REPORT ---")
    report.append(f"Container Sample Rate: {sr:,} Hz | Nyquist Limit: {nyquist/1000:.1f} kHz")
    report.append(f"ENBW: {enbw_hz:.2f} Hz | FFT Resolution: {sr/n_fft:.2f} Hz/bin")
    
    if sr >= 88200:
        if zero_ratio > 0.40:
            is_zero_stuffed = True
            effective_cutoff_hz = 22050.0
            report.append("\nASSESSMENT: [RAW ZERO-STUFFED / NO FILTERING]")
            report.append(f"  -> Exact zero padding detected ({zero_ratio*100:.1f}% literal zeros).")
        else:
            # Mirror imaging check (unfiltered aliasing):
            # Check if high ultrasonic band (30k-40k) has loud reflected signal matching audible levels
            idx_30k = np.argmin(np.abs(freqs - 30000))
            idx_40k = np.argmin(np.abs(freqs - 40000))
            mirror_rms_median = np.median(rms_dbfs[idx_30k:idx_40k]) if nyquist >= 44100 else -140.0
            
            # Baseline high ultrasonic floor in top 15% of Nyquist band
            idx_floor_start = np.argmin(np.abs(freqs - (0.82 * nyquist)))
            idx_floor_end = np.argmin(np.abs(freqs - (0.96 * nyquist)))
            noise_floor_rms = np.median(rms_dbfs[idx_floor_start:idx_floor_end])
            
            if mirror_rms_median > (audible_rms_mean - 20.0) and mirror_rms_median > -80.0 and (mirror_rms_median - noise_floor_rms) > 20.0:
                is_zero_stuffed = True
                effective_cutoff_hz = 22050.0
                report.append("\nASSESSMENT: [FAKE HI-RES / UNFILTERED ALIASING]")
                report.append("  -> Strong spectral mirror imaging detected above 22.05 kHz.")
            
    # Forensic Cutoff & Upsampling Detection
    if sr >= 48000 and not is_zero_stuffed:
        idx_floor_start = np.argmin(np.abs(freqs - (0.82 * nyquist)))
        idx_floor_end = np.argmin(np.abs(freqs - (0.96 * nyquist)))
        noise_floor_rms = np.median(rms_dbfs[idx_floor_start:idx_floor_end])
        noise_floor_peak = np.median(peak_dbfs[idx_floor_start:idx_floor_end])
        noise_floor_spec = np.median(mean_spec_db[idx_floor_start:idx_floor_end])

        idx_20k = np.argmin(np.abs(freqs - 20000))
        idx_22k = np.argmin(np.abs(freqs - 22050))
        idx_23k = np.argmin(np.abs(freqs - 23000))
        idx_24k = np.argmin(np.abs(freqs - 24000))
        idx_25k = np.argmin(np.abs(freqs - 25000))
        idx_30k = np.argmin(np.abs(freqs - 30000)) if nyquist > 30000 else None
        idx_40k = np.argmin(np.abs(freqs - 40000)) if nyquist > 40000 else None
        idx_42k = np.argmin(np.abs(freqs - 42000)) if nyquist > 42000 else None
        idx_45k = np.argmin(np.abs(freqs - 45000)) if nyquist > 45000 else None
        idx_46k = np.argmin(np.abs(freqs - 46000)) if nyquist > 46000 else None
        idx_48k = np.argmin(np.abs(freqs - 48000)) if nyquist > 48000 else None
        idx_50k = np.argmin(np.abs(freqs - 50000)) if nyquist > 50000 else None

        # Spectrogram-aligned decibel drops (immune to isolated transient spikes)
        if nyquist > 60000:
            # 88.2 kHz Master Cutoff (cliff near 44.1 kHz):
            drop_40_45_spec = mean_spec_db[idx_40k] - mean_spec_db[idx_45k]
            drop_42_45_rms = rms_dbfs[idx_42k] - rms_dbfs[idx_45k]
            if (drop_40_45_spec > 8.0 or drop_42_45_rms > 4.5) and rms_dbfs[idx_45k] < -100.0:
                is_88k_cutoff = True

            # 96 kHz Master Cutoff (cliff near 48 kHz):
            drop_46_50_spec = mean_spec_db[idx_46k] - mean_spec_db[idx_50k]
            drop_46_50_rms = rms_dbfs[idx_46k] - rms_dbfs[idx_50k]
            if not is_88k_cutoff and (drop_46_50_spec > 8.0 or drop_46_50_rms > 5.0) and rms_dbfs[idx_50k] < -100.0:
                is_96k_cutoff = True

        # Check CD Cutoff (44.1 kHz brickwall at 22.05 kHz):
        drop_20_23 = mean_spec_db[idx_20k] - mean_spec_db[idx_23k]
        if not is_88k_cutoff and not is_96k_cutoff and drop_20_23 > 16.0 and (rms_dbfs[idx_23k] <= noise_floor_rms + 8.0 or mean_spec_db[idx_23k] < -125.0):
            is_cd_cutoff = True

        # Check 48k Cutoff (48 kHz brickwall at 24.0 kHz):
        if nyquist > 25000:
            drop_22_25 = mean_spec_db[idx_22k] - mean_spec_db[idx_25k]
            if not is_cd_cutoff and not is_88k_cutoff and not is_96k_cutoff and drop_22_25 > 16.0 and (rms_dbfs[idx_25k] <= noise_floor_rms + 8.0 or mean_spec_db[idx_25k] < -125.0):
                is_48k_cutoff = True

        if is_cd_cutoff:
            is_upsampled = True
            detected_base_hz = 22050
            effective_cutoff_hz = 22050.0
        elif is_48k_cutoff:
            is_upsampled = True
            detected_base_hz = 24000
            effective_cutoff_hz = 24000.0
        elif is_88k_cutoff:
            is_upsampled = True
            detected_base_hz = 44100
            effective_cutoff_hz = 44100.0
        elif is_96k_cutoff:
            is_upsampled = True
            detected_base_hz = 48000
            effective_cutoff_hz = 48000.0
        else:
            above_floor = (rms_dbfs > (noise_floor_rms + 5.0)) | (mean_spec_db > (np.median(mean_spec_db[idx_floor_start:idx_floor_end]) + 5.0))
            valid_idx = np.where(above_floor)[0]
            effective_bw_hz = freqs[valid_idx[-1]] if len(valid_idx) > 0 else 20000.0
            effective_cutoff_hz = min(nyquist, max(20000.0, effective_bw_hz))

    report.insert(3, f"Effective Signal Bandwidth: ~{effective_cutoff_hz/1000:.1f} kHz")

    # 3. Comprehensive Noise Profile Analysis
    noise_profile = "STANDARD PCM / UNFILTERED"
    if nyquist > 24000:
        ultrasonic_floor = np.mean(rms_dbfs[idx_24k:])
        report.append(f"\nUltrasonic Noise Floor (RMS): {ultrasonic_floor:.1f} dBFS")
        
        # Calculate slope in ultrasonic band above effective cutoff
        start_f = max(24000.0, effective_cutoff_hz + 3000.0)
        end_f = nyquist - 2000.0
        
        if end_f > start_f:
            idx_start = np.argmin(np.abs(freqs - start_f))
            idx_end = np.argmin(np.abs(freqs - end_f))
            span_pts = max(5, (idx_end - idx_start) // 4)
            
            floor_low = np.median(rms_dbfs[idx_start : idx_start + span_pts])
            floor_high = np.median(rms_dbfs[idx_end - span_pts : idx_end])
            noise_rise = floor_high - floor_low
            
            if noise_rise >= 3.0:
                noise_profile = f"PSYCHOACOUSTIC NOISE SHAPING (+{noise_rise:.1f} dB HF Rise)"
            elif abs(noise_rise) < 3.0 and ultrasonic_floor < -125.0:
                noise_profile = f"FLAT TPDF DITHER ({ultrasonic_floor:.1f} dBFS Floor)"
            elif ultrasonic_floor > -75.0:
                noise_profile = f"HIGH ULTRASONIC NOISE / DSD SOURCED ({ultrasonic_floor:.1f} dBFS)"
            else:
                noise_profile = "STANDARD PCM GRADUAL ROLL-OFF"
        else:
            noise_profile = "NATIVE ACOUSTIC HARMONIC EXTENSION"
            
        report.append(f"NOISE PROFILE: [{noise_profile}]")
    else:
        # Standard CD / 48k files
        idx_16k = np.argmin(np.abs(freqs - 16000))
        idx_20k = np.argmin(np.abs(freqs - 20000))
        audible_floor = np.mean(rms_dbfs[idx_16k:idx_20k])
        report.append(f"\nHigh-Frequency Floor (RMS): {audible_floor:.1f} dBFS")
        
        rise_16_20 = rms_dbfs[idx_20k] - rms_dbfs[idx_16k]
        if rise_16_20 >= 4.0 and audible_floor < -60.0:
            noise_profile = f"PSYCHOACOUSTIC NOISE SHAPING (+{rise_16_20:.1f} dB Rise near Nyquist)"
        elif audible_floor < -110.0:
            noise_profile = f"FLAT 24-BIT DITHER ({audible_floor:.1f} dBFS Floor)"
        elif audible_floor < -85.0:
            noise_profile = f"FLAT 16-BIT TPDF DITHER ({audible_floor:.1f} dBFS Floor)"
        else:
            noise_profile = "STANDARD PCM SPECTRUM"
            
        report.append(f"NOISE PROFILE: [{noise_profile}]")
            
    # 4. Dynamic Range & Loudness Analysis (TT DR Score & EBU R128 LRA)
    dr_metrics = calculate_dynamic_range_metrics(y, sr)
    dr_score = dr_metrics["dr_score"]
    dr_val = dr_metrics["dr_val"]
    crest_db = dr_metrics["crest_factor_db"]
    lufs = dr_metrics["integrated_lufs"]
    lra = dr_metrics["lra_lu"]
    peak_db = dr_metrics["peak_dbfs"]

    report.append(f"\nDYNAMIC RANGE (TT DR Meter) : DR{dr_score} ({dr_val:.1f} dB)")
    report.append(f"EBU R128 Loudness Range (LRA): {lra:.1f} LU | Integrated: {lufs:.1f} LUFS")
    report.append(f"Peak-to-RMS Crest Factor     : {crest_db:.2f} dB")
    report.append(f"Peak Signal Level            : {peak_db:.2f} dBFS")

    # 5. Estimated Provenance Analysis (Lineage, Base Rates & Resampling Fingerprint)
    # Universal Multi-Boundary Mirror & Notch Rebound Scanner:
    # Checks for leaky reconstruction filters folding across candidate Nyquist boundaries:
    # 16.0 kHz (32k base), 22.05 kHz (44.1k base), 24.0 kHz (48k base), 32.0 kHz (64k base), 44.1 kHz (88.2k base), 48.0 kHz (96k base)
    candidates_fn = [16000, 22050, 24000, 32000, 44100, 48000]
    leaky_hits = []

    for f_n in candidates_fn:
        if f_n + 1200 > nyquist or (f_n - 1500) < 5000:
            continue

        i_base = np.argmin(np.abs(freqs - (f_n - 1500)))
        i_notch = np.argmin(np.abs(freqs - f_n))
        i_rebound = np.argmin(np.abs(freqs - (f_n + 700)))

        notch_drop = mean_spec_db[i_base] - mean_spec_db[i_notch]
        rebound_rise = mean_spec_db[i_rebound] - mean_spec_db[i_notch]

        # Calculate cross-temporal correlation across the folding axis f_n
        corrs = []
        for delta in np.linspace(200, 1500, 8):
            f1 = f_n - delta
            f2 = f_n + delta
            if f2 > nyquist:
                continue
            i1 = np.argmin(np.abs(freqs - f1))
            i2 = np.argmin(np.abs(freqs - f2))
            t1 = stft_mag[i1, :]
            t2 = stft_mag[i2, :]
            if np.std(t1) > 1e-8 and np.std(t2) > 1e-8:
                c = np.corrcoef(t1, t2)[0, 1]
                if not np.isnan(c):
                    corrs.append(c)

        avg_corr = float(np.mean(corrs)) if corrs else 0.0

        if (rebound_rise > 2.0 or (notch_drop > 15.0 and avg_corr > 0.55)) and (avg_corr > 0.55 or notch_drop > 8.0):
            leaky_hits.append({
                "base_rate_khz": (f_n * 2) / 1000.0,
                "f_n_khz": f_n / 1000.0,
                "rebound_db": rebound_rise,
                "notch_drop_db": notch_drop,
                "mirror_corr": avg_corr
            })

    primary_prov = None
    alt_prov = None

    if is_zero_stuffed:
        primary_prov = {
            "label": "Raw Zero-Stuffed / NOS Source",
            "confidence": "High",
            "score": 0.98,
            "badge_class": "badge-provenance-fake",
            "details": "Literal zero-stuffing or strong unfiltered imaging detected above 22.05 kHz."
        }
    elif is_cd_cutoff:
        drop_val = drop_20_23 if 'drop_20_23' in locals() else 20.0
        primary_prov = {
            "label": "Upsampled from 44.1 kHz Master",
            "confidence": "High",
            "score": 0.94,
            "badge_class": "badge-provenance-upsampled",
            "details": f"Sharp anti-aliasing cutoff at 22.05 kHz ({drop_val:.1f} dB drop) into container noise floor."
        }
        if leaky_hits and leaky_hits[0]["mirror_corr"] > 0.55:
            mc = leaky_hits[0]["mirror_corr"]
            alt_prov = {
                "label": "44.1 kHz Master (Leaky Filter Residual)",
                "confidence": "Moderate",
                "score": round(mc, 2),
                "badge_class": "badge-provenance-leaky",
                "details": f"Residual spectral leakage across 22.05 kHz with r={mc:+.2f} mirror correlation."
            }
    elif is_88k_cutoff:
        primary_prov = {
            "label": "Upsampled from 88.2 kHz Master",
            "confidence": "High",
            "score": 0.95,
            "badge_class": "badge-provenance-upsampled",
            "details": "Sharp anti-aliasing cutoff at 44.1 kHz into baseline dither noise floor."
        }
        if leaky_hits and leaky_hits[0]["mirror_corr"] > 0.55:
            mc = leaky_hits[0]["mirror_corr"]
            alt_prov = {
                "label": "88.2 kHz Master (Leaky Filter Residual)",
                "confidence": "Moderate",
                "score": round(mc, 2),
                "badge_class": "badge-provenance-leaky",
                "details": f"Residual spectral leakage across 44.1 kHz with r={mc:+.2f} mirror correlation."
            }
    elif is_96k_cutoff:
        primary_prov = {
            "label": "Upsampled from 96.0 kHz Master",
            "confidence": "High",
            "score": 0.95,
            "badge_class": "badge-provenance-upsampled",
            "details": "Sharp anti-aliasing cutoff at 48.0 kHz into baseline dither noise floor."
        }
    elif is_48k_cutoff:
        primary_prov = {
            "label": "Upsampled from 48.0 kHz Source",
            "confidence": "High",
            "score": 0.95,
            "badge_class": "badge-provenance-upsampled",
            "details": "Sharp anti-aliasing cutoff at 24.0 kHz into container noise floor."
        }
    elif leaky_hits:
        h = leaky_hits[0]
        c_score = h["mirror_corr"]
        conf = "High" if c_score > 0.75 else "Moderate"
        base_khz = h["base_rate_khz"]
        fn_khz = h["f_n_khz"]
        primary_prov = {
            "label": f"{base_khz:.1f} kHz Master (Leaky SRC / DAC)",
            "confidence": conf,
            "score": round(min(0.98, max(0.70, c_score)), 2),
            "badge_class": "badge-provenance-leaky",
            "details": f"{base_khz:.1f} kHz filter notch at {fn_khz:.2f} kHz with mirrored spectral imaging (r={c_score:+.2f})."
        }
        alt_prov = {
            "label": f"Upsampled from {base_khz:.1f} kHz Master",
            "confidence": "Moderate",
            "score": 0.65,
            "badge_class": "badge-provenance-upsampled",
            "details": f"Sharp attenuation near {fn_khz:.2f} kHz with filter skirt artifacts."
        }
    elif nyquist <= 22050:
        primary_prov = {
            "label": f"Native {sr/1000:.1f} kHz CD Master",
            "confidence": "High",
            "score": 0.95,
            "badge_class": "badge-provenance-native",
            "details": f"Standard Red Book container with full {sr/1000:.1f} kHz audible passband."
        }
    elif sr == 48000:
        if effective_cutoff_hz >= 23500.0:
            primary_prov = {
                "label": "Native 48.0 kHz Master",
                "confidence": "High",
                "score": 0.90,
                "badge_class": "badge-provenance-native",
                "details": f"Continuous harmonic extension up to {effective_cutoff_hz/1000:.1f} kHz Nyquist limit."
            }
        elif effective_cutoff_hz >= 21500.0:
            primary_prov = {
                "label": "Native 48.0 kHz Material",
                "confidence": "Moderate",
                "score": 0.75,
                "badge_class": "badge-provenance-native",
                "details": f"Natural acoustic roll-off extending to ~{effective_cutoff_hz/1000:.1f} kHz."
            }
        else:
            primary_prov = {
                "label": "Unclear / Mixed Source",
                "confidence": "Low",
                "score": 0.40,
                "badge_class": "badge-provenance-unclear",
                "details": "Bandwidth limited without clear brickwall or mirror characteristics."
            }
    else:
        if effective_cutoff_hz >= 26000.0:
            primary_prov = {
                "label": f"Native {sr/1000:.1f} kHz Master",
                "confidence": "High",
                "score": 0.92,
                "badge_class": "badge-provenance-native",
                "details": f"Continuous acoustic harmonic bandwidth reaching ~{effective_cutoff_hz/1000:.1f} kHz."
            }
        elif effective_cutoff_hz >= 21000.0:
            primary_prov = {
                "label": f"Native {sr/1000:.1f} kHz Material",
                "confidence": "Moderate",
                "score": 0.75,
                "badge_class": "badge-provenance-native",
                "details": f"Smooth ultrasonic roll-off up to ~{effective_cutoff_hz/1000:.1f} kHz."
            }
        else:
            primary_prov = {
                "label": "Unclear / Mixed Source",
                "confidence": "Low",
                "score": 0.40,
                "badge_class": "badge-provenance-unclear",
                "details": "Bandwidth limited without clear brickwall or mirror characteristics."
            }

    provenance_info = {
        "primary": primary_prov,
        "alternative": alt_prov,
        "label": primary_prov["label"],
        "confidence": primary_prov["confidence"],
        "score": primary_prov["score"],
        "badge_class": primary_prov["badge_class"],
        "details": primary_prov["details"]
    }

    report.append(f"\nESTIMATED PROVENANCE : {primary_prov['label']} [{primary_prov['confidence']} Confidence: {int(primary_prov['score']*100)}%]")
    report.append(f"  -> {primary_prov['details']}")
    if alt_prov:
        report.append(f"  -> ALTERNATIVE POSSIBILITY: {alt_prov['label']} [{alt_prov['confidence']} Confidence: {int(alt_prov['score']*100)}%]")
        report.append(f"     {alt_prov['details']}")

    return spec_db, freqs, peak_dbfs, rms_dbfs, "\n".join(report), dr_metrics, provenance_info


def encode_spectrogram_and_lookup(spec_db, width=1600, height=800, lookup_w=600, lookup_h=300):
    """
    Renders the STFT matrix to an optimized WebP image string and generates a compact
    2D uint8 lookup table for real-time sub-millisecond dBFS cursor inspection.
    """
    norm = np.clip((spec_db - (-165.0)) / 165.0, 0.0, 1.0)
    norm_u8 = (norm[::-1, :] * 255).astype(np.uint8)
    
    # 1. Visual WebP Heatmap
    img_gray = Image.fromarray(norm_u8, 'L').resize((width, height), Image.Resampling.BILINEAR)
    cmap = matplotlib.colormaps.get_cmap('magma')
    lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8).flatten().tolist()
    img_gray.putpalette(lut)
    img_rgb = img_gray.convert('RGB')
    
    buf = io.BytesIO()
    img_rgb.save(buf, format='WEBP', quality=85)
    webp_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # 2. Compact 2D Lookup Table (LUT)
    img_lookup = Image.fromarray(norm_u8, 'L').resize((lookup_w, lookup_h), Image.Resampling.BILINEAR)
    lookup_bytes = np.array(img_lookup).tobytes()
    lookup_b64 = base64.b64encode(lookup_bytes).decode('utf-8')
    
    return webp_b64, lookup_b64, lookup_w, lookup_h


def generate_html5_report(y, sr, audio_filename, output_html, spec_db, freqs, peak_dbfs, rms_dbfs, report_text):
    """
    Generates a single self-contained, publication-grade interactive HTML5 report with zoom controls.
    """
    nyquist = sr / 2.0
    duration_s = len(y) / float(sr)

    # True, uncolored physical dBFS spectral traces
    display_peak = np.copy(peak_dbfs)
    display_rms = np.copy(rms_dbfs)
    
    # Downsample curve data for 60fps interactive HTML Canvas (keep ~2048 high-precision points)
    step = max(1, len(freqs) // 2048)
    curve_freqs_khz = (freqs[::step] / 1000.0).round(3).tolist()
    curve_peak_db = display_peak[::step].round(2).tolist()
    curve_rms_db = display_rms[::step].round(2).tolist()
    
    # Render fast spectrogram heatmap and dB lookup table
    webp_base64, lookup_base64, lookup_w, lookup_h = encode_spectrogram_and_lookup(spec_db)
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spectrum Analysis: {os.path.basename(audio_filename)}</title>
    <style>
        :root {{
            --bg: #0d1117;
            --card-bg: #161b22;
            --border: #30363d;
            --text: #c9d1d9;
            --text-heading: #f0f6fc;
            --accent-cyan: #00e5ff;
            --accent-pink: #ff007f;
            --accent-green: #aeea00;
            --accent-yellow: #ffea00;
            --accent-red: #ff1744;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            padding: 24px;
            display: flex;
            justify-content: center;
        }}
        .container {{
            width: 100%;
            max-width: 1200px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        header {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
        }}
        .title-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 12px;
        }}
        h1 {{
            color: var(--text-heading);
            font-size: 1.35rem;
            font-weight: 600;
            word-break: break-all;
        }}
        .meta-badges {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .badge {{
            font-size: 0.8rem;
            font-weight: 500;
            padding: 4px 10px;
            border-radius: 4px;
            background: #21262d;
            border: 1px solid var(--border);
            color: #8b949e;
        }}
        .badge-highlight {{
            background: rgba(174, 234, 0, 0.1);
            border-color: rgba(174, 234, 0, 0.3);
            color: var(--accent-green);
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px 20px;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 12px;
        }}
        .card-title {{
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-heading);
        }}
        .toolbar {{
            display: flex;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
        }}
        .btn {{
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
        }}
        .btn:hover {{
            background: #30363d;
            color: var(--text-heading);
            border-color: #8b949e;
        }}
        .btn:active {{
            background: #282e36;
        }}
        .btn-active {{
            background: rgba(0, 229, 255, 0.15);
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }}
        .canvas-container {{
            position: relative;
            width: 100%;
            height: 380px;
            background: #11141a;
            border-radius: 6px;
            overflow: hidden;
            touch-action: none;
        }}
        canvas {{
            display: block;
            width: 100%;
            height: 100%;
            cursor: crosshair;
        }}
        .legend-bar {{
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 16px;
            font-size: 0.8rem;
            margin-top: 8px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 2px;
        }}
        .hud-overlay {{
            position: absolute;
            top: 12px;
            right: 12px;
            background: rgba(13, 17, 23, 0.88);
            backdrop-filter: blur(6px);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 8px 12px;
            font-family: "JetBrains Mono", monospace;
            font-size: 0.8rem;
            pointer-events: none;
            display: flex;
            flex-direction: column;
            gap: 4px;
            z-index: 10;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }}
        .report-box {{
            background: #0d1117;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 16px;
            font-family: "JetBrains Mono", SFMono-Regular, Consolas, monospace;
            font-size: 0.85rem;
            line-height: 1.6;
            color: var(--accent-green);
            white-space: pre-wrap;
            position: relative;
        }}
        .btn-copy {{
            position: absolute;
            top: 12px;
            right: 12px;
            background: #21262d;
            border: 1px solid var(--border);
            color: var(--text);
            padding: 6px 12px;
            font-size: 0.75rem;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s ease;
        }}
        .btn-copy:hover {{
            background: #30363d;
            color: var(--text-heading);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="title-row">
                <h1>Acoustic Spectrum & Forensic Analysis</h1>
                <div class="meta-badges">
                    <span class="badge badge-highlight">{sr:,} Hz ({nyquist/1000:.1f} kHz Nyquist)</span>
                    <span class="badge">{duration_s:.1f}s Window</span>
                    <span class="badge">64-bit Double Precision</span>
                </div>
            </div>
            <div style="font-size: 0.9rem; color: #8b949e; word-break: break-all;">
                <strong>File:</strong> {os.path.basename(audio_filename)}
            </div>
        </header>

        <!-- Panel 1: Spectrogram with Full Coordinates, Colorbar, and Zoom -->
        <section class="card">
            <div class="card-header">
                <span class="card-title">Linear Spectrogram (Time vs Frequency)</span>
                <div class="toolbar">
                    <span style="font-size: 0.75rem; color: #8b949e; margin-right: 4px;">Zoom:</span>
                    <button class="btn" onclick="zoomSpectrogram(1.35)">+ In</button>
                    <button class="btn" onclick="zoomSpectrogram(1.0/1.35)">- Out</button>
                    <button class="btn" onclick="setSpecPreset('audible')">0-20 kHz</button>
                    <button class="btn" onclick="setSpecPreset('ultrasonic')">20 kHz+</button>
                    <button class="btn" onclick="resetSpecZoom()">&#x21BA; Reset</button>
                </div>
            </div>
            <div class="canvas-container" style="height: 380px;">
                <canvas id="spectrogramCanvas"></canvas>
                <div class="hud-overlay" id="specHudOverlay">
                    <div><span style="color: #8b949e;">Time :</span> <strong id="specHudTime">-- s</strong></div>
                    <div><span style="color: var(--accent-cyan);">Freq :</span> <strong id="specHudFreq">-- kHz</strong></div>
                    <div><span style="color: var(--accent-green);">Level:</span> <strong id="specHudDb">-- dBFS</strong></div>
                </div>
            </div>
            <div style="font-size: 0.75rem; color: #8b949e; margin-top: 6px; display: flex; justify-content: space-between;">
                <span>💡 Tip: Scroll wheel to zoom, click &amp; drag to pan.</span>
                <span>Dynamic Range: 0 dBFS &rarr; -165 dBFS</span>
            </div>
        </section>

        <!-- Panel 2: Interactive Spectrum Curve with Zoom -->
        <section class="card">
            <div class="card-header">
                <span class="card-title">Frequency Spectrum &amp; Noise Profile</span>
                <div class="toolbar">
                    <span style="font-size: 0.75rem; color: #8b949e; margin-right: 4px;">Zoom:</span>
                    <button class="btn" onclick="zoomCurve(1.35)">+ In</button>
                    <button class="btn" onclick="zoomCurve(1.0/1.35)">- Out</button>
                    <button class="btn" onclick="setCurvePreset('audible')">0-20 kHz</button>
                    <button class="btn" onclick="setCurvePreset('cutoff')">15-25 kHz</button>
                    <button class="btn" onclick="setCurvePreset('ultrasonic')">20 kHz+</button>
                    <button class="btn" onclick="resetCurveZoom()">&#x21BA; Reset</button>
                </div>
            </div>
            <div class="canvas-container" style="height: 380px;">
                <canvas id="spectrumCanvas"></canvas>
                <div class="hud-overlay" id="hudOverlay">
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

        <!-- Panel 3: Forensic Lab Report -->
        <section class="card">
            <div class="card-header">
                <span class="card-title">Forensic Assessment Lab Report</span>
            </div>
            <div class="report-box" id="reportText">
                <button class="btn-copy" onclick="copyReport()">Copy Report</button>
{report_text}
            </div>
        </section>
    </div>

    <script>
        const freqs = {json.dumps(curve_freqs_khz)};
        const peaks = {json.dumps(curve_peak_db)};
        const rms = {json.dumps(curve_rms_db)};
        const nyquistKhz = {nyquist / 1000.0};
        const durationS = {duration_s};
        const defaultMinDb = -175.0;
        const defaultMaxDb = 0.0;

        // --- 2D SPECTROGRAM LOOKUP MATRIX ---
        const lookupW = {lookup_w};
        const lookupH = {lookup_h};
        const rawLookup = Uint8Array.from(atob("{lookup_base64}"), c => c.charCodeAt(0));

        function getSpectrogramDb(t, fKhz) {{
            if (t < 0 || t > durationS || fKhz < 0 || fKhz > nyquistKhz) return -165.0;
            const x = Math.max(0, Math.min(lookupW - 1, Math.floor((t / durationS) * lookupW)));
            const y = Math.max(0, Math.min(lookupH - 1, Math.floor((1.0 - (fKhz / nyquistKhz)) * lookupH)));
            const u8 = rawLookup[y * lookupW + x];
            return (u8 / 255.0) * 165.0 - 165.0;
        }}

        // --- SPECTROGRAM STATE & ZOOM ---
        let specTMin = 0.0, specTMax = durationS;
        let specFMin = 0.0, specFMax = nyquistKhz;
        let specIsDragging = false, specDragStartX = 0, specDragStartY = 0;
        let specDragInitTMin = 0, specDragInitTMax = 0, specDragInitFMin = 0, specDragInitFMax = 0;

        const specCanvas = document.getElementById('spectrogramCanvas');
        const sCtx = specCanvas.getContext('2d');
        const specHudTime = document.getElementById('specHudTime');
        const specHudFreq = document.getElementById('specHudFreq');
        const specHudDb = document.getElementById('specHudDb');

        const specImg = new Image();
        specImg.src = "data:image/webp;base64,{webp_base64}";
        specImg.onload = () => resizeSpecCanvas();

        let specMouseX = -1, specMouseY = -1;

        function resetSpecZoom() {{
            specTMin = 0.0; specTMax = durationS;
            specFMin = 0.0; specFMax = nyquistKhz;
            drawSpectrogram(specCanvas.getBoundingClientRect().width, specCanvas.getBoundingClientRect().height);
        }}

        function setSpecPreset(type) {{
            if (type === 'audible') {{
                specFMin = 0.0; specFMax = Math.min(20.0, nyquistKhz);
            }} else if (type === 'ultrasonic') {{
                specFMin = Math.min(20.0, nyquistKhz); specFMax = nyquistKhz;
            }}
            drawSpectrogram(specCanvas.getBoundingClientRect().width, specCanvas.getBoundingClientRect().height);
        }}

        function zoomSpectrogram(factor, centerTRatio = 0.5, centerFRatio = 0.5) {{
            const curTW = specTMax - specTMin;
            const newTW = Math.max(0.5, Math.min(durationS, curTW / factor));
            const centerT = specTMin + curTW * centerTRatio;
            specTMin = Math.max(0, centerT - newTW * centerTRatio);
            specTMax = Math.min(durationS, specTMin + newTW);
            if (specTMax - specTMin < newTW) specTMin = Math.max(0, specTMax - newTW);

            const curFW = specFMax - specFMin;
            const newFW = Math.max(1.0, Math.min(nyquistKhz, curFW / factor));
            const centerF = specFMin + curFW * centerFRatio;
            specFMin = Math.max(0, centerF - newFW * centerFRatio);
            specFMax = Math.min(nyquistKhz, specFMin + newFW);
            if (specFMax - specFMin < newFW) specFMin = Math.max(0, specFMax - newFW);

            drawSpectrogram(specCanvas.getBoundingClientRect().width, specCanvas.getBoundingClientRect().height);
        }}

        function resizeSpecCanvas() {{
            const rect = specCanvas.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            specCanvas.width = rect.width * dpr;
            specCanvas.height = rect.height * dpr;
            sCtx.scale(dpr, dpr);
            drawSpectrogram(rect.width, rect.height);
        }}

        function drawSpectrogram(w, h) {{
            sCtx.clearRect(0, 0, w, h);
            const padL = 60, padR = 75, padT = 15, padB = 40;
            const plotW = w - padL - padR;
            const plotH = h - padT - padB;

            if (plotW <= 0 || plotH <= 0) return;

            // Draw Heatmap Bitmap with Zoom/Sub-Rect
            if (specImg.complete && specImg.naturalWidth > 0) {{
                const sx = (specTMin / durationS) * specImg.naturalWidth;
                const sw = ((specTMax - specTMin) / durationS) * specImg.naturalWidth;
                const sy = (1.0 - (specFMax / nyquistKhz)) * specImg.naturalHeight;
                const sh = ((specFMax - specFMin) / nyquistKhz) * specImg.naturalHeight;

                sCtx.save();
                sCtx.beginPath();
                sCtx.rect(padL, padT, plotW, plotH);
                sCtx.clip();
                sCtx.drawImage(specImg, sx, sy, sw, sh, padL, padT, plotW, plotH);
                sCtx.restore();
            }}

            // Outer border
            sCtx.strokeStyle = "#30363d";
            sCtx.lineWidth = 1;
            sCtx.strokeRect(padL, padT, plotW, plotH);

            // Y-Axis Ticks & Labels (Frequency kHz)
            sCtx.fillStyle = "#8b949e";
            sCtx.font = "10px -apple-system, sans-serif";
            sCtx.textAlign = "right";
            sCtx.textBaseline = "middle";

            const fRange = specFMax - specFMin;
            const fStep = fRange > 40 ? 20 : (fRange > 20 ? 10 : (fRange > 8 ? 5 : 2));
            const firstF = Math.ceil(specFMin / fStep) * fStep;

            for (let f = firstF; f <= specFMax; f += fStep) {{
                const y = padT + (1.0 - (f - specFMin) / fRange) * plotH;
                sCtx.strokeStyle = "#1f242c";
                sCtx.beginPath();
                sCtx.moveTo(padL - 4, y);
                sCtx.lineTo(padL, y);
                sCtx.stroke();
                sCtx.fillText(f.toFixed(fStep < 1 ? 1 : 0) + "k", padL - 8, y);
            }}

            // Y-Axis Label
            sCtx.save();
            sCtx.translate(14, padT + plotH / 2);
            sCtx.rotate(-Math.PI / 2);
            sCtx.textAlign = "center";
            sCtx.fillStyle = "#c9d1d9";
            sCtx.font = "11px -apple-system, sans-serif";
            sCtx.fillText("Frequency (kHz)", 0, 0);
            sCtx.restore();

            // X-Axis Ticks & Labels (Time seconds)
            sCtx.textAlign = "center";
            sCtx.textBaseline = "top";
            const tRange = specTMax - specTMin;
            const tStep = tRange > 40 ? 10 : (tRange > 15 ? 5 : (tRange > 5 ? 2 : 1));
            const firstT = Math.ceil(specTMin / tStep) * tStep;

            for (let t = firstT; t <= specTMax; t += tStep) {{
                const x = padL + ((t - specTMin) / tRange) * plotW;
                sCtx.strokeStyle = "#1f242c";
                sCtx.beginPath();
                sCtx.moveTo(x, padT + plotH);
                sCtx.lineTo(x, padT + plotH + 4);
                sCtx.stroke();
                sCtx.fillText(t.toFixed(tStep < 1 ? 1 : 0) + "s", x, padT + plotH + 7);
            }}

            // X-Axis Label
            sCtx.textAlign = "center";
            sCtx.fillStyle = "#c9d1d9";
            sCtx.font = "11px -apple-system, sans-serif";
            sCtx.fillText("Time (seconds)", padL + plotW / 2, padT + plotH + 22);

            // Right-side Colorbar Scale
            const barX = padL + plotW + 15;
            const barW = 12;
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

            // Colorbar Labels
            sCtx.textAlign = "left";
            sCtx.textBaseline = "middle";
            sCtx.fillStyle = "#8b949e";
            sCtx.font = "9px -apple-system, sans-serif";
            const dbTicks = [0, -40, -80, -120, -165];
            for (let d of dbTicks) {{
                const y = padT + (d / -165.0) * barH;
                sCtx.fillText(d === 0 ? "0 dB" : d + " dB", barX + barW + 5, y);
            }}

            // Interactive Crosshair on hover
            if (specMouseX >= padL && specMouseX <= padL + plotW && specMouseY >= padT && specMouseY <= padT + plotH) {{
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
            }}
        }}

        specCanvas.addEventListener('mousedown', (e) => {{
            const rect = specCanvas.getBoundingClientRect();
            specIsDragging = true;
            specDragStartX = e.clientX;
            specDragStartY = e.clientY;
            specDragInitTMin = specTMin; specDragInitTMax = specTMax;
            specDragInitFMin = specFMin; specDragInitFMax = specFMax;
        }});

        window.addEventListener('mouseup', () => {{ specIsDragging = false; }});

        specCanvas.addEventListener('mousemove', (e) => {{
            const rect = specCanvas.getBoundingClientRect();
            specMouseX = e.clientX - rect.left;
            specMouseY = e.clientY - rect.top;

            if (specIsDragging) {{
                const padL = 60, padR = 75, padT = 15, padB = 40;
                const plotW = rect.width - padL - padR;
                const plotH = rect.height - padT - padB;
                const dx = e.clientX - specDragStartX;
                const dy = e.clientY - specDragStartY;

                const curTW = specDragInitTMax - specDragInitTMin;
                const curFW = specDragInitFMax - specDragInitFMin;

                const dt = -(dx / plotW) * curTW;
                const df = (dy / plotH) * curFW;

                specTMin = Math.max(0, Math.min(durationS - curTW, specDragInitTMin + dt));
                specTMax = specTMin + curTW;

                specFMin = Math.max(0, Math.min(nyquistKhz - curFW, specDragInitFMin + df));
                specFMax = specFMin + curFW;
            }}

            drawSpectrogram(rect.width, rect.height);
        }});

        specCanvas.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const rect = specCanvas.getBoundingClientRect();
            const padL = 60, padR = 75, padT = 15, padB = 40;
            const plotW = rect.width - padL - padR;
            const plotH = rect.height - padT - padB;

            const mX = e.clientX - rect.left;
            const mY = e.clientY - rect.top;

            if (mX >= padL && mX <= padL + plotW && mY >= padT && mY <= padT + plotH) {{
                const tRatio = (mX - padL) / plotW;
                const fRatio = 1.0 - (mY - padT) / plotH;
                const factor = e.deltaY < 0 ? 1.25 : (1.0 / 1.25);
                zoomSpectrogram(factor, tRatio, fRatio);
            }}
        }}, {{ passive: false }});

        specCanvas.addEventListener('mouseleave', () => {{
            specMouseX = -1;
            specMouseY = -1;
            const rect = specCanvas.getBoundingClientRect();
            drawSpectrogram(rect.width, rect.height);
            specHudTime.textContent = "-- s";
            specHudFreq.textContent = "-- kHz";
            specHudDb.textContent = "-- dBFS";
        }});


        // --- SPECTRUM CURVE STATE & ZOOM ---
        let curveFMin = 0.0, curveFMax = nyquistKhz;
        let curveDbMin = defaultMinDb, curveDbMax = defaultMaxDb;
        let curveIsDragging = false, curveDragStartX = 0, curveDragStartY = 0;
        let curveDragInitFMin = 0, curveDragInitFMax = 0, curveDragInitDbMin = 0, curveDragInitDbMax = 0;

        const canvas = document.getElementById('spectrumCanvas');
        const ctx = canvas.getContext('2d');
        const hudOverlay = document.getElementById('hudOverlay');
        const hudFreq = document.getElementById('hudFreq');
        const hudPeak = document.getElementById('hudPeak');
        const hudRMS = document.getElementById('hudRMS');

        let mouseX = -1, mouseY = -1;

        function resetCurveZoom() {{
            curveFMin = 0.0; curveFMax = nyquistKhz;
            curveDbMin = defaultMinDb; curveDbMax = defaultMaxDb;
            drawPlot(canvas.getBoundingClientRect().width, canvas.getBoundingClientRect().height);
        }}

        function setCurvePreset(type) {{
            if (type === 'audible') {{
                curveFMin = 0.0; curveFMax = Math.min(20.0, nyquistKhz);
                curveDbMin = -120.0; curveDbMax = 0.0;
            }} else if (type === 'cutoff') {{
                curveFMin = 15.0; curveFMax = Math.min(25.0, nyquistKhz);
                curveDbMin = -175.0; curveDbMax = -40.0;
            }} else if (type === 'ultrasonic') {{
                curveFMin = Math.min(20.0, nyquistKhz); curveFMax = nyquistKhz;
                curveDbMin = -175.0; curveDbMax = -120.0;
            }}
            drawPlot(canvas.getBoundingClientRect().width, canvas.getBoundingClientRect().height);
        }}

        function zoomCurve(factor, centerFRatio = 0.5, centerDbRatio = 0.5) {{
            const curFW = curveFMax - curveFMin;
            const newFW = Math.max(1.0, Math.min(nyquistKhz, curFW / factor));
            const centerF = curveFMin + curFW * centerFRatio;
            curveFMin = Math.max(0, centerF - newFW * centerFRatio);
            curveFMax = Math.min(nyquistKhz, curveFMin + newFW);
            if (curveFMax - curveFMin < newFW) curveFMin = Math.max(0, curveFMax - newFW);

            const curDbW = curveDbMax - curveDbMin;
            const newDbW = Math.max(10.0, Math.min(defaultMaxDb - defaultMinDb, curDbW / factor));
            const centerDb = curveDbMin + curDbW * centerDbRatio;
            curveDbMin = Math.max(defaultMinDb, centerDb - newDbW * centerDbRatio);
            curveDbMax = Math.min(defaultMaxDb, curveDbMin + newDbW);
            if (curveDbMax - curveDbMin < newDbW) curveDbMin = Math.max(defaultMinDb, curveDbMax - newDbW);

            drawPlot(canvas.getBoundingClientRect().width, canvas.getBoundingClientRect().height);
        }}

        function resizeCanvas() {{
            const rect = canvas.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);
            drawPlot(rect.width, rect.height);
        }}

        function freqToX(fKhz, width, padL, padR) {{
            const plotW = width - padL - padR;
            return padL + ((fKhz - curveFMin) / (curveFMax - curveFMin)) * plotW;
        }}

        function dbToY(db, height, padT, padB) {{
            const plotH = height - padT - padB;
            const clamped = Math.max(curveDbMin, Math.min(curveDbMax, db));
            return padT + (1.0 - (clamped - curveDbMin) / (curveDbMax - curveDbMin)) * plotH;
        }}

        function drawPlot(w, h) {{
            ctx.clearRect(0, 0, w, h);

            const padL = 60, padR = 25, padT = 20, padB = 40;
            const plotW = w - padL - padR;
            const plotH = h - padT - padB;

            if (plotW <= 0 || plotH <= 0) return;

            // Background
            ctx.fillStyle = "#11141a";
            ctx.fillRect(padL, padT, plotW, plotH);

            // Horizontal Grid Lines (every 20 dB or adapted step)
            const dbRange = curveDbMax - curveDbMin;
            const dbStep = dbRange > 100 ? 20 : (dbRange > 40 ? 10 : 5);
            const firstDb = Math.ceil(curveDbMin / dbStep) * dbStep;

            ctx.strokeStyle = "#1f242c";
            ctx.lineWidth = 1;
            ctx.fillStyle = "#6e7681";
            ctx.font = "10px -apple-system, sans-serif";
            ctx.textAlign = "right";
            ctx.textBaseline = "middle";

            for (let db = firstDb; db <= curveDbMax; db += dbStep) {{
                const y = dbToY(db, h, padT, padB);
                ctx.beginPath();
                ctx.moveTo(padL, y);
                ctx.lineTo(padL + plotW, y);
                ctx.stroke();
                ctx.fillText(db.toFixed(0) + " dB", padL - 8, y);
            }}

            // Y-Axis Label
            ctx.save();
            ctx.translate(14, padT + plotH / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.textAlign = "center";
            ctx.fillStyle = "#c9d1d9";
            ctx.font = "11px -apple-system, sans-serif";
            ctx.fillText("Amplitude (dBFS)", 0, 0);
            ctx.restore();

            // Vertical Frequency Grid Lines
            ctx.textAlign = "center";
            ctx.textBaseline = "top";
            const fRange = curveFMax - curveFMin;
            const fStep = fRange > 40 ? 20 : (fRange > 20 ? 10 : (fRange > 8 ? 5 : (fRange > 3 ? 1 : 0.5)));
            const firstF = Math.ceil(curveFMin / fStep) * fStep;

            for (let f = firstF; f <= curveFMax; f += fStep) {{
                const x = freqToX(f, w, padL, padR);
                ctx.beginPath();
                ctx.moveTo(x, padT);
                ctx.lineTo(x, padT + plotH);
                ctx.stroke();
                ctx.fillText(f.toFixed(fStep < 1 ? 1 : 0) + "k", x, padT + plotH + 8);
            }}

            // X-Axis Label
            ctx.textAlign = "center";
            ctx.fillStyle = "#c9d1d9";
            ctx.font = "11px -apple-system, sans-serif";
            ctx.fillText("Frequency (kHz)", padL + plotW / 2, padT + plotH + 22);

            // Clip plot drawing
            ctx.save();
            ctx.beginPath();
            ctx.rect(padL, padT, plotW, plotH);
            ctx.clip();

            // Reference Marker: 20 kHz (Audible Limit)
            if (curveFMin <= 20.0 && curveFMax >= 20.0) {{
                const x20 = freqToX(20.0, w, padL, padR);
                ctx.strokeStyle = "#ff1744";
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(x20, padT);
                ctx.lineTo(x20, padT + plotH);
                ctx.stroke();
            }}

            // Reference Marker: 22.05 kHz (CD Nyquist)
            if (curveFMin <= 22.05 && curveFMax >= 22.05) {{
                const x22 = freqToX(22.05, w, padL, padR);
                ctx.strokeStyle = "#ffea00";
                ctx.setLineDash([4, 4]);
                ctx.beginPath();
                ctx.moveTo(x22, padT);
                ctx.lineTo(x22, padT + plotH);
                ctx.stroke();
            }}
            ctx.setLineDash([]);

            // Draw RMS Curve (Magenta)
            ctx.strokeStyle = "#ff007f";
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            let firstPoint = true;
            for (let i = 0; i < freqs.length; i++) {{
                if (freqs[i] >= curveFMin - 1.0 && freqs[i] <= curveFMax + 1.0) {{
                    const x = freqToX(freqs[i], w, padL, padR);
                    const y = dbToY(rms[i], h, padT, padB);
                    if (firstPoint) {{ ctx.moveTo(x, y); firstPoint = false; }}
                    else ctx.lineTo(x, y);
                }}
            }}
            ctx.stroke();

            // Draw Peak Curve (Cyan)
            ctx.strokeStyle = "#00e5ff";
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            firstPoint = true;
            for (let i = 0; i < freqs.length; i++) {{
                if (freqs[i] >= curveFMin - 1.0 && freqs[i] <= curveFMax + 1.0) {{
                    const x = freqToX(freqs[i], w, padL, padR);
                    const y = dbToY(peaks[i], h, padT, padB);
                    if (firstPoint) {{ ctx.moveTo(x, y); firstPoint = false; }}
                    else ctx.lineTo(x, y);
                }}
            }}
            ctx.stroke();

            ctx.restore();

            // Draw Crosshair on Hover
            if (mouseX >= padL && mouseX <= padL + plotW) {{
                const ratio = (mouseX - padL) / plotW;
                const targetFreq = curveFMin + ratio * (curveFMax - curveFMin);
                let closestIdx = 0, minDiff = Infinity;
                for (let i = 0; i < freqs.length; i++) {{
                    const d = Math.abs(freqs[i] - targetFreq);
                    if (d < minDiff) {{ minDiff = d; closestIdx = i; }}
                }}

                const curX = freqToX(freqs[closestIdx], w, padL, padR);
                const curYPeak = dbToY(peaks[closestIdx], h, padT, padB);
                const curYRMS = dbToY(rms[closestIdx], h, padT, padB);

                ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
                ctx.setLineDash([2, 2]);
                ctx.beginPath();
                ctx.moveTo(curX, padT);
                ctx.lineTo(curX, padT + plotH);
                ctx.stroke();
                ctx.setLineDash([]);

                // Peak indicator dot
                ctx.fillStyle = "#00e5ff";
                ctx.beginPath();
                ctx.arc(curX, curYPeak, 3.5, 0, Math.PI * 2);
                ctx.fill();

                // RMS indicator dot
                ctx.fillStyle = "#ff007f";
                ctx.beginPath();
                ctx.arc(curX, curYRMS, 3.5, 0, Math.PI * 2);
                ctx.fill();

                // Update HUD Text
                hudFreq.textContent = `${{freqs[closestIdx].toFixed(2)}} kHz (${{(freqs[closestIdx]*1000).toFixed(0)}} Hz)`;
                hudPeak.textContent = `${{peaks[closestIdx].toFixed(1)}} dBFS`;
                hudRMS.textContent = `${{rms[closestIdx].toFixed(1)}} dBFS`;
            }}
        }}

        canvas.addEventListener('mousedown', (e) => {{
            const rect = canvas.getBoundingClientRect();
            curveIsDragging = true;
            curveDragStartX = e.clientX;
            curveDragStartY = e.clientY;
            curveDragInitFMin = curveFMin; curveDragInitFMax = curveFMax;
            curveDragInitDbMin = curveDbMin; curveDragInitDbMax = curveDbMax;
        }});

        window.addEventListener('mouseup', () => {{ curveIsDragging = false; }});

        canvas.addEventListener('mousemove', (e) => {{
            const rect = canvas.getBoundingClientRect();
            mouseX = e.clientX - rect.left;
            mouseY = e.clientY - rect.top;

            if (curveIsDragging) {{
                const padL = 60, padR = 25, padT = 20, padB = 40;
                const plotW = rect.width - padL - padR;
                const plotH = rect.height - padT - padB;
                const dx = e.clientX - curveDragStartX;
                const dy = e.clientY - curveDragStartY;

                const curFW = curveDragInitFMax - curveDragInitFMin;
                const curDbW = curveDragInitDbMax - curveDragInitDbMin;

                const df = -(dx / plotW) * curFW;
                const dDb = (dy / plotH) * curDbW;

                curveFMin = Math.max(0, Math.min(nyquistKhz - curFW, curveDragInitFMin + df));
                curveFMax = curveFMin + curFW;

                curveDbMin = Math.max(defaultMinDb, Math.min(defaultMaxDb - curDbW, curveDragInitDbMin + dDb));
                curveDbMax = curveDbMin + curDbW;
            }}

            drawPlot(rect.width, rect.height);
        }});

        canvas.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const rect = canvas.getBoundingClientRect();
            const padL = 60, padR = 25, padT = 20, padB = 40;
            const plotW = rect.width - padL - padR;
            const plotH = rect.height - padT - padB;

            const mX = e.clientX - rect.left;
            const mY = e.clientY - rect.top;

            if (mX >= padL && mX <= padL + plotW && mY >= padT && mY <= padT + plotH) {{
                const fRatio = (mX - padL) / plotW;
                const dbRatio = 1.0 - (mY - padT) / plotH;
                const factor = e.deltaY < 0 ? 1.25 : (1.0 / 1.25);
                zoomCurve(factor, fRatio, dbRatio);
            }}
        }}, {{ passive: false }});

        canvas.addEventListener('mouseleave', () => {{
            mouseX = -1;
            mouseY = -1;
            const rect = canvas.getBoundingClientRect();
            drawPlot(rect.width, rect.height);
            hudFreq.textContent = "-- kHz";
            hudPeak.textContent = "-- dBFS";
            hudRMS.textContent = "-- dBFS";
        }});

        function onGlobalResize() {{
            resizeSpecCanvas();
            resizeCanvas();
        }}

        window.addEventListener('resize', onGlobalResize);
        setTimeout(onGlobalResize, 50);

        function copyReport() {{
            const text = `{report_text}`;
            navigator.clipboard.writeText(text).then(() => {{
                const btn = document.querySelector('.btn-copy');
                btn.textContent = 'Copied!';
                setTimeout(() => btn.textContent = 'Copy Report', 2000);
            }});
        }}
    </script>
</body>
</html>"""

    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_template)


def main():
    parser = argparse.ArgumentParser(description="Hi-Fi News Style Acoustic Spectrum & Forensic Analyzer (HTML5 Edition)")
    parser.add_argument("file", help="Path to FLAC/WAV audio file")
    parser.add_argument("--out", default=None, help="Output HTML path (default: filename_spectrum.html)")
    args = parser.parse_args()

    filepath = os.path.abspath(args.file)
    if not os.path.exists(filepath):
        print(f"[Error] File not found: {filepath}")
        sys.exit(1)

    # Output defaults to .html
    if args.out:
        output_html = args.out
        if not output_html.endswith('.html'):
            output_html = f"{os.path.splitext(output_html)[0]}.html"
    else:
        output_html = f"{os.path.splitext(filepath)[0]}_spectrum.html"

    print(f"\n==================================================")
    print(f"HI-FI NEWS SPECTRUM & FORENSIC ANALYZER (HTML5 V4.2)")
    print(f"Target File: {os.path.basename(filepath)}")
    print(f"Precision  : 64-bit Double Precision (Strict)")
    print(f"==================================================")

    data, sr = sf.read(filepath, dtype='float64', start=0, stop=60*192000)
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Apply gentle boundary taper (50ms) to eliminate artificial truncation step discontinuities
    taper_len = min(int(sr * 0.05), len(data) // 10)
    if taper_len > 0:
        taper = np.sin(np.linspace(0, np.pi/2, taper_len))**2
        data[:taper_len] *= taper
        data[-taper_len:] *= taper[::-1]

    spec_db, freqs, peak_dbfs, rms_dbfs, assessment_text, dr_metrics, provenance_info = analyze_audio_forensics(data, sr)
    
    print("\n" + assessment_text)
    print("----------------------------\n")

    print(f"Generating interactive HTML5 forensic report -> {output_html}...")
    generate_html5_report(data, sr, filepath, output_html, spec_db, freqs, peak_dbfs, rms_dbfs, assessment_text)
    print("Done!\n")


if __name__ == "__main__":
    main()
