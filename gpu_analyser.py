#!/usr/bin/env python3
"""
GPU-Accelerated 64-Bit Double-Precision Forensic Audio Analyser
Supports PyOpenCL + PyVkFFT with automated, seamless CPU fallback.
"""

import os
import sys
import time
import logging
import numpy as np
import scipy.signal as signal
import librosa

import threading

logger = logging.getLogger("gpu_analyser")

# Attempt PyOpenCL & PyVkFFT imports
HAS_OPENCL = False
try:
    import pyopencl as cl
    import pyopencl.array as cla
    from pyvkfft.opencl import VkFFTApp
    HAS_OPENCL = True
except Exception as e:
    logger.warning(f"OpenCL/VkFFT not available: {e}. Falling back to CPU.")

# OpenCL 64-bit reduction and spectral math kernel
OPENCL_KERNEL_SRC = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void compute_spectral_reductions(
    __global const double2* stft_in,
    __global double* spec_db_out,
    __global double* peak_dbfs_out,
    __global double* rms_dbfs_out,
    const int n_frames,
    const int n_bins,
    const double inv_norm_half_s1
) {
    int bin = get_global_id(0);
    if (bin >= n_bins) return;

    double max_mag = 0.0;
    double sum_pwr = 0.0;

    for (int f = 0; f < n_frames; f++) {
        int idx = f * n_bins + bin;
        double2 val = stft_in[idx];
        double mag = sqrt(val.x * val.x + val.y * val.y) * inv_norm_half_s1;
        
        // Output format: (bins, frames) for direct display indexing
        double s_db = 20.0 * log10(max(mag, 1e-12));
        spec_db_out[bin * n_frames + f] = s_db;

        if (mag > max_mag) {
            max_mag = mag;
        }
        sum_pwr += mag * mag;
    }

    double peak_db = 20.0 * log10(max(max_mag, 1e-12));
    double rms_db = 10.0 * log10(max(sum_pwr / (double)n_frames, 1e-24));

    peak_dbfs_out[bin] = peak_db;
    rms_dbfs_out[bin] = rms_db;
}
"""


class GPUForensicEngine:
    """
    Singleton GPU acceleration engine for forensic audio analysis.
    Strictly retains 64-bit double precision across all pipelines.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GPUForensicEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.enabled = False
        self.device_name = "CPU Reference (No GPU)"
        self.ctx = None
        self.queue = None
        self.prg = None
        self.kernel = None
        self._cached_apps = {}
        self._lock = threading.Lock()

        if not HAS_OPENCL:
            logger.info("OpenCL not present; initialized in CPU fallback mode.")
            return

        try:
            # Discover platforms and devices
            platforms = cl.get_platforms()
            gpu_device = None
            for p in platforms:
                gpus = p.get_devices(device_type=cl.device_type.GPU)
                if gpus:
                    gpu_device = gpus[0]
                    break
            
            if gpu_device is None:
                # Check for any device with FP64
                for p in platforms:
                    devs = p.get_devices()
                    for d in devs:
                        if "cl_khr_fp64" in d.extensions:
                            gpu_device = d
                            break
                    if gpu_device:
                        break

            if gpu_device is None:
                logger.info("No OpenCL device with cl_khr_fp64 found; using CPU fallback.")
                return

            self.ctx = cl.Context([gpu_device])
            self.queue = cl.CommandQueue(self.ctx)
            self.device_name = gpu_device.name.strip()
            self.prg = cl.Program(self.ctx, OPENCL_KERNEL_SRC).build()
            self.kernel = cl.Kernel(self.prg, "compute_spectral_reductions")
            self.enabled = True
            logger.info(f"GPU Forensic Engine initialized on: {self.device_name} (64-Bit Float)")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenCL GPU context: {e}. Using CPU fallback.")
            self.enabled = False

    def clear_cache(self):
        """Releases cached VkFFT applications and cleans memory."""
        with self._lock:
            self._cached_apps.clear()

    def compute_stft_and_reductions(self, y: np.ndarray, sr: int, n_fft: int = 16384, hop_length: int = None):
        """
        Executes batched 64-bit STFT and 2D reduction on GPU with CPU fallback.
        Thread-safe across multiple concurrent background workers.
        """
        if hop_length is None:
            hop_length = max(4096, len(y) // 2048)

        if not self.enabled:
            return self._compute_cpu(y, sr, n_fft, hop_length)

        # Non-blocking lock attempt: if another thread is using the GPU, smoothly compute on CPU
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            return self._compute_cpu(y, sr, n_fft, hop_length)

        try:
            t0 = time.time()
            win = signal.windows.blackmanharris(n_fft, sym=False)
            S1 = np.sum(win)
            S2 = np.sum(win**2)
            enbw_hz = sr * (S2 / (S1**2))
            inv_norm_half_s1 = np.float64(1.0 / (S1 / 2.0))

            # Centered padding
            pad_width = n_fft // 2
            y_padded = np.pad(y, pad_width, mode='reflect')
            n_frames = 1 + (len(y_padded) - n_fft) // hop_length
            shape = (n_frames, n_fft)
            strides = (y_padded.strides[0] * hop_length, y_padded.strides[0])
            frames_view = np.lib.stride_tricks.as_strided(y_padded, shape=shape, strides=strides)
            
            # Apply window on CPU (fast vectorized)
            windowed_frames = np.ascontiguousarray(frames_view * win, dtype=np.float64)

            # GPU Buffer allocation
            n_bins = n_fft // 2 + 1
            d_frames = cla.to_device(self.queue, windowed_frames)
            d_stft = cla.empty(self.queue, (n_frames, n_bins), dtype=np.complex128)
            d_spec_db = cla.empty(self.queue, (n_bins, n_frames), dtype=np.float64)
            d_peak_dbfs = cla.empty(self.queue, n_bins, dtype=np.float64)
            d_rms_dbfs = cla.empty(self.queue, n_bins, dtype=np.float64)

            # Retrieve or create cached VkFFT app
            app_key = (n_frames, n_fft)
            if app_key not in self._cached_apps:
                self._cached_apps[app_key] = VkFFTApp(
                    d_frames.shape, d_frames.dtype, self.queue, ndim=1, r2c=True, inplace=False
                )
            app = self._cached_apps[app_key]

            # 1. GPU Batched R2C FFT
            app.fft(d_frames, d_stft)

            # 2. GPU Fused Reductions Kernel
            kernel = cl.Kernel(self.prg, "compute_spectral_reductions")
            kernel.set_args(
                d_stft.data,
                d_spec_db.data,
                d_peak_dbfs.data,
                d_rms_dbfs.data,
                np.int32(n_frames),
                np.int32(n_bins),
                inv_norm_half_s1
            )
            cl.enqueue_nd_range_kernel(self.queue, kernel, (n_bins,), None)
            self.queue.finish()

            # Retrieve results from GPU
            spec_db = d_spec_db.get()
            peak_dbfs = d_peak_dbfs.get()
            rms_dbfs = d_rms_dbfs.get()
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

            return spec_db, freqs, peak_dbfs, rms_dbfs, enbw_hz, "GPU (VkFFT FP64)"
        except Exception as e:
            logger.warning(f"GPU STFT execution failed: {e}. Falling back to CPU.")
            return self._compute_cpu(y, sr, n_fft, hop_length)
        finally:
            self._lock.release()

    def _compute_cpu(self, y: np.ndarray, sr: int, n_fft: int = 16384, hop_length: int = None):
        import scipy.fft as sfft
        if hop_length is None:
            hop_length = max(4096, len(y) // 2048)

        win = signal.windows.blackmanharris(n_fft, sym=False)
        S1 = np.sum(win)
        S2 = np.sum(win**2)
        enbw_hz = sr * (S2 / (S1**2))

        pad_width = n_fft // 2
        y_padded = np.pad(y, pad_width, mode='reflect')
        n_frames = 1 + (len(y_padded) - n_fft) // hop_length
        shape = (n_frames, n_fft)
        strides = (y_padded.strides[0] * hop_length, y_padded.strides[0])
        frames_view = np.lib.stride_tricks.as_strided(y_padded, shape=shape, strides=strides)
        
        windowed = frames_view * win
        stft = sfft.rfft(windowed, axis=-1, workers=-1)
        stft_norm = (np.abs(stft) / (S1 / 2.0)).T  # (n_bins, n_frames)

        peak_mag = np.max(stft_norm, axis=1)
        peak_dbfs = 20.0 * np.log10(np.maximum(peak_mag, 1e-12))

        power_linear = np.mean(stft_norm**2, axis=1)
        rms_dbfs = 10.0 * np.log10(np.maximum(power_linear, 1e-24))

        spec_db = 20.0 * np.log10(np.maximum(stft_norm, 1e-12))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        return spec_db, freqs, peak_dbfs, rms_dbfs, enbw_hz, "CPU Multi-Thread (SciPy FP64)"


# Global singleton instance
gpu_engine = GPUForensicEngine()


def analyze_audio_forensics_accelerated(y, sr, rules_path=None, filepath=None):
    """
    High-level accelerated forensic analyser combining GPU DSP with full rule assessment.
    """
    from analyser import calculate_dynamic_range_metrics
    from provenance_engine import (
        get_provenance_engine,
        DEFAULT_RULES_PATH,
        detect_mqa_signature,
        analyze_effective_bit_depth
    )

    t0 = time.time()
    n_fft = 16384

    # 1. Compute STFT and 2D Reductions on GPU (or CPU Fallback)
    spec_db, freqs, peak_dbfs, rms_dbfs, enbw_hz, backend_used = gpu_engine.compute_stft_and_reductions(
        y, sr, n_fft=n_fft, hop_length=None
    )

    # 2. Dynamic Range & EBU R128
    dr_metrics = calculate_dynamic_range_metrics(y, sr)

    # 3. Provenance Engine Assessment
    mean_spec_db = np.mean(spec_db, axis=1)
    nyquist = sr / 2.0
    zero_ratio = 1.0 - (np.count_nonzero(y) / len(y))
    pcm_int32 = (np.clip(y, -1.0, 1.0) * 2147483647.0).astype(np.int32)
    mqa_info = detect_mqa_signature(filepath=filepath, sr=sr, pcm_int32=pcm_int32) if filepath else None
    bitdepth_info = analyze_effective_bit_depth(filepath=filepath, pcm_int32=pcm_int32) if filepath else None
    engine = get_provenance_engine(rules_path or DEFAULT_RULES_PATH)
    
    provenance_info = engine.evaluate(
        sr=sr,
        nyquist=nyquist,
        freqs=freqs,
        mean_spec_db=mean_spec_db,
        rms_dbfs=rms_dbfs,
        peak_dbfs=peak_dbfs,
        stft_mag=spec_db,
        zero_ratio=zero_ratio,
        mqa_info=mqa_info,
        bitdepth_info=bitdepth_info
    )

    # 4. Noise Profile Classification
    noise_profile = "STANDARD PCM / UNFILTERED"
    idx_24k = np.argmin(np.abs(freqs - 24000)) if nyquist >= 24000 else 0
    if nyquist > 24000:
        ultrasonic_floor = np.mean(rms_dbfs[idx_24k:])
        eff_cutoff = provenance_info.get("effective_bw_hz", nyquist)
        start_f = max(24000.0, eff_cutoff + 3000.0)
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
    else:
        idx_16k = np.argmin(np.abs(freqs - 16000))
        idx_20k = np.argmin(np.abs(freqs - 20000))
        audible_floor = np.mean(rms_dbfs[idx_16k:idx_20k])
        rise_16_20 = rms_dbfs[idx_20k] - rms_dbfs[idx_16k]
        if rise_16_20 >= 4.0 and audible_floor < -60.0:
            noise_profile = f"PSYCHOACOUSTIC NOISE SHAPING (+{rise_16_20:.1f} dB Rise near Nyquist)"
        elif audible_floor < -110.0:
            noise_profile = f"FLAT 24-BIT DITHER ({audible_floor:.1f} dBFS Floor)"
        elif audible_floor < -85.0:
            noise_profile = f"FLAT 16-BIT TPDF DITHER ({audible_floor:.1f} dBFS Floor)"
        else:
            noise_profile = "STANDARD PCM SPECTRUM"

    # 5. Build Lab Assessment Text
    report = []
    report.append("--- FORENSIC LAB REPORT ---")
    report.append(f"Engine Backend: {backend_used} | Device: {gpu_engine.device_name}")
    report.append(f"Container Sample Rate: {sr:,} Hz | Nyquist Limit: {nyquist/1000:.1f} kHz")
    report.append(f"ENBW: {enbw_hz:.2f} Hz | FFT Resolution: {sr/n_fft:.2f} Hz/bin")
    
    eff_bw = provenance_info.get("effective_bw_hz", nyquist)
    report.append(f"Effective Signal Bandwidth: ~{eff_bw/1000:.1f} kHz")
    
    nf_rms = provenance_info.get("noise_floor_rms", -120.0)
    report.append(f"\nUltrasonic Noise Floor (RMS): {nf_rms:.1f} dBFS")
    report.append(f"NOISE PROFILE: [{noise_profile}]")

    report.append(f"\nDYNAMIC RANGE (TT DR Meter) : DR{dr_metrics.get('dr_score', 0)} ({dr_metrics.get('dr_val', 0.0):.1f} dB)")
    report.append(f"EBU R128 Loudness Range (LRA): {dr_metrics.get('lra_lu', 0.0):.1f} LU | Integrated: {dr_metrics.get('integrated_lufs', -140.0):.1f} LUFS")
    report.append(f"Peak-to-RMS Crest Factor     : {dr_metrics.get('crest_factor_db', 0.0):.2f} dB")
    report.append(f"Peak Signal Level            : {dr_metrics.get('peak_dbfs', 0.0):.2f} dBFS")

    if bitdepth_info:
        cb = bitdepth_info.get("container_bits", 16)
        eb = bitdepth_info.get("effective_bits", 16)
        tz = bitdepth_info.get("trailing_zero_bits", 0)
        summ = bitdepth_info.get("summary", "Standard PCM")
        report.append(f"\nBIT DEPTH RESOLUTION        : {cb}-bit Container -> {eb}-bit Effective ({tz} LSBs Inactive)")
        report.append(f"Bit Activity & LSB Profile   : {summ}")

    primary = provenance_info.get("primary", {})
    v_label = primary.get("label", "Native Master")
    v_conf = primary.get("confidence", "Medium")
    v_score = int(primary.get("score", 0.8) * 100)
    v_det = primary.get("details", "")

    vis = provenance_info.get("visual_morphology", {})
    if vis:
        pk = vis.get("primary_knee")
        knee_str = f"{pk['detected_knee_khz']} kHz (Steepness: {pk['steepest_slope_db_per_khz']} dB/kHz, Curvature: {pk['max_curvature']})" if pk else "Natural Rolloff (No Brickwall Knee)"
        r_val = vis.get("rhythmic_coherence", 0.0)
        r_type = "Authentic Wideband Transients" if r_val >= 0.45 else "Stationary Noise / Dither" if vis.get("is_stationary_ultrasonic") else "Standard Signal"
        r_str = f"r = {r_val:+.3f} ({r_type})"
        report.append(f"\nVISUAL MORPHOLOGY DYNAMICS  : Curvature Knee: {knee_str}")
        report.append(f"Rhythmic Cross-Correlation   : {r_str}")
        report.append(f"Temporal Dynamics (Variance) : Audible: {vis.get('audible_temporal_variance', 0.0):.1f} dB² | Ultrasonic: {vis.get('ultrasonic_temporal_variance', 0.0):.1f} dB²")

    report.append(f"\nESTIMATED PROVENANCE : {v_label} [{v_conf} Confidence: {v_score}%]")
    if v_det:
        report.append(f"  -> {v_det}")

    assessment_text = "\n".join(report)

    return spec_db, freqs, peak_dbfs, rms_dbfs, assessment_text, dr_metrics, provenance_info
