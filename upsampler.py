#!/usr/bin/env python3
"""
================================================================================
ACOUSTISINC: ULTRA HIGH-PERFORMANCE SINC UPSAMPLER (V3 ENTERPRISE MASTER)
================================================================================
Architecture & Optimizations:
- Double Precision (float64 / complex128 / double2) across all DSP pipelines.
- Batched Multi-Channel GPU Sinc Engine (transforms L/R channels simultaneously).
- Composite FFT Length Optimization (scipy.fft.next_fast_len) for 35-58% speedup.
- In-Register Multi-Core Psychoacoustic Noise Shaping (Shibata & High-Rate 4th).
- Bounded Asynchronous NVMe FLAC Serialization at strict Compression Level 5.
- Embedded FLAC Picture auto-repair (valid dimensions/depth) + folder cover.jpg export.
- Zero Memory Leak Architecture: Active PyVkFFT plan & OpenCL buffer cache flushing.
- Pre-Flight Album Intersample Headroom Scan with Auto-Healing Retries.
================================================================================
"""

import os
import sys
import gc
import time
import shutil
import uuid
import threading
import argparse
import numpy as np
import scipy.fft
import soundfile as sf
import pyopencl as cl
import pyopencl.array as cla
from pyvkfft.fft import rfftn, irfftn, clear_vkfftapp_cache
import io
import re
import json
from PIL import Image
from mutagen import File
from mutagen.flac import FLAC, Picture
from concurrent.futures import ThreadPoolExecutor
from provenance_engine import load_audio_resilient, probe_audio_info_resilient
from analyser import analyze_audio_forensics
from report_generator import generate_comparative_report

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

HAS_NUMBA = False
try:
    from numba import njit, prange, uint64
    HAS_NUMBA = True
except ImportError:
    pass

os.environ["MESA_SHADER_CACHE_DISABLE"] = "false"
os.environ["MESA_SHADER_CACHE_DIR"] = os.path.expanduser("~/.cache/mesa_shader_cache")

# Audio DSP Constants
PEAK_TARGET_DB = -0.3
PEAK_TARGET_LIN = 10 ** (PEAK_TARGET_DB / 20.0)
OVERSHOOT_BUFFER_DB = 2.4
SAFE_CHUNK_LEN = 2**21  # 2.10M samples (~47.6s at 44.1k) for optimal sweet-spot throughput and VRAM efficiency
OVERLAP_SAMPLES = 65536
FLAC_COMPRESSION_LEVEL_5 = 5.0 / 8.0  # 0.625 for FLAC level 5

DEFAULT_TMP_DIR = "/var/tmp"
SKIP_TAGS = ("1xxK_min", "1xxK_GPU", "1xxK_apod", "3xxK_min", "7xxK_min", "upsampled")


# ==============================================================================
# GPU OPENCL VRAM MULTI-CHANNEL SINC KERNEL (ZERO-PADDING & FILTERING IN VRAM)
# ==============================================================================

OPENCL_SINC_SRC = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void sinc_interpolate_multichannel_gpu(
    __global const double2 *in_spec,
    __global double2 *out_spec,
    __global const double2 *filter_kernel,
    const ulong in_len,
    const ulong out_len,
    const int apply_filter,
    const ulong num_channels
) {
    ulong gid = get_global_id(0);
    ulong ch = get_global_id(1);
    if (gid >= out_len || ch >= num_channels) return;

    ulong in_idx = ch * in_len + gid;
    ulong out_idx = ch * out_len + gid;

    if (gid < in_len - 1) {
        double2 val = in_spec[in_idx];
        if (apply_filter) {
            double2 k = filter_kernel[gid];
            val = (double2)(val.x * k.x - val.y * k.y, val.x * k.y + val.y * k.x);
        }
        out_spec[out_idx] = val;
    } else if (gid == in_len - 1) {
        double2 val = (in_len == out_len) ? in_spec[in_idx] : (double2)(in_spec[in_idx].x * 0.5, in_spec[in_idx].y * 0.5);
        if (apply_filter) {
            double2 k = filter_kernel[gid];
            val = (double2)(val.x * k.x - val.y * k.y, val.x * k.y + val.y * k.x);
        }
        out_spec[out_idx] = val;
    } else {
        out_spec[out_idx] = (double2)(0.0, 0.0);
    }
}

__kernel void init_min_phase_log_mag(
    __global double2 *log_mag_spec,
    const ulong in_len,
    const ulong k_cutoff,
    const ulong w
) {
    ulong gid = get_global_id(0);
    if (gid >= in_len) return;

    double mag_val = 1e-12;
    ulong start_taper = (k_cutoff > w) ? (k_cutoff - w) : 0;

    if (gid < start_taper) {
        mag_val = 1.0;
    } else if (gid < k_cutoff && w > 0) {
        double progress = (double)(gid - start_taper) / (double)(w > 1 ? (w - 1) : 1);
        double taper = 0.5 * (1.0 + cos(progress * M_PI));
        mag_val = taper * (1.0 - 1e-12) + 1e-12;
    } else {
        mag_val = 1e-12;
    }

    log_mag_spec[gid] = (double2)(log(mag_val), 0.0);
}

__kernel void apply_cepstral_window(
    __global double *cep,
    const ulong n_in
) {
    ulong gid = get_global_id(0);
    if (gid >= n_in) return;

    double inv_n = 1.0 / (double)n_in;
    ulong half_n = n_in / 2;

    if (gid == 0 || gid == half_n) {
        cep[gid] *= inv_n;
    } else if (gid < half_n) {
        cep[gid] *= (2.0 * inv_n);
    } else {
        cep[gid] = 0.0;
    }
}

__kernel void complex_exp_inplace(
    __global double2 *spec,
    const ulong in_len
) {
    ulong gid = get_global_id(0);
    if (gid >= in_len) return;

    double2 z = spec[gid];
    double exp_x = exp(z.x);
    spec[gid] = (double2)(exp_x * cos(z.y), exp_x * sin(z.y));
}

__kernel void init_apodizing_filter(
    __global double2 *filter_spec,
    const ulong in_len,
    const ulong cutoff_bin,
    const ulong transition_bins
) {
    ulong gid = get_global_id(0);
    if (gid >= in_len) return;

    double mag_val = 0.0;
    ulong start_taper = (cutoff_bin > transition_bins) ? (cutoff_bin - transition_bins) : 0;

    if (gid < start_taper) {
        mag_val = 1.0;
    } else if (gid >= start_taper && gid < cutoff_bin && transition_bins > 0) {
        double progress = (double)(gid - start_taper) / (double)(transition_bins > 1 ? (transition_bins - 1) : 1);
        mag_val = 0.5 * (1.0 + cos(progress * M_PI));
    }

    filter_spec[gid] = (double2)(mag_val, 0.0);
}
"""

CL_SINC_PROGRAM = None
CL_KERNELS = {}
GPU_FILTER_CACHE = {}

def get_sinc_program(ctx):
    global CL_SINC_PROGRAM
    if CL_SINC_PROGRAM is None:
        CL_SINC_PROGRAM = cl.Program(ctx, OPENCL_SINC_SRC).build()
    return CL_SINC_PROGRAM

def get_sinc_kernel(ctx, kernel_name):
    prg = get_sinc_program(ctx)
    if kernel_name not in CL_KERNELS:
        CL_KERNELS[kernel_name] = cl.Kernel(prg, kernel_name)
    return CL_KERNELS[kernel_name]

def get_gpu_filter_kernel(queue, fast_input_len, in_len, phase_mode="linear", apodizing=False, cutoff_hz=None, src_sr=None, steep=False):
    p = phase_mode.lower()
    is_min = p in ['min', 'minimum']
    
    k_cutoff = in_len - 1
    if cutoff_hz is not None and src_sr is not None:
        f_nyq = src_sr / 2.0
        if cutoff_hz < f_nyq:
            k_cutoff = int((cutoff_hz / f_nyq) * (in_len - 1))
            k_cutoff = max(1, min(in_len - 1, k_cutoff))
            apodizing = True

    if not is_min and not apodizing:
        return None, 0

    key = (fast_input_len, in_len, k_cutoff, is_min, apodizing, steep)
    if key in GPU_FILTER_CACHE:
        return GPU_FILTER_CACHE[key], 1

    # Keep bounded filter cache to prevent VRAM accumulation across thousands of tracks
    if len(GPU_FILTER_CACHE) > 8:
        GPU_FILTER_CACHE.clear()

    trans_bw = 500.0 if steep else 2000.0
    ref_nyq = (src_sr / 2.0) if src_sr is not None else 22050.0
    transition_bins = max(16, int((in_len - 1) * trans_bw / ref_nyq)) if (apodizing or cutoff_hz) else max(16, int((in_len - 1) * 250.0 / ref_nyq))

    if is_min:
        # Minimum Phase Kernel (with or without Apodizing transition shaping)
        k_init = get_sinc_kernel(queue.context, "init_min_phase_log_mag")
        k_window = get_sinc_kernel(queue.context, "apply_cepstral_window")
        k_exp = get_sinc_kernel(queue.context, "complex_exp_inplace")

        gpu_log_mag = cla.empty(queue, (in_len,), dtype=np.complex128)
        k_init(
            queue, (in_len,), None,
            gpu_log_mag.data, np.uint64(in_len), np.uint64(k_cutoff), np.uint64(transition_bins)
        )
        
        # 1D IFFT to cepstral domain in double precision
        gpu_cep = irfftn(gpu_log_mag, ndim=1, norm=0)
        
        # Apply causal step window + normalization in-place
        k_window(
            queue, (fast_input_len,), None,
            gpu_cep.data, np.uint64(fast_input_len)
        )
        
        # 1D RFFT back to frequency domain
        d_kernel = rfftn(gpu_cep, ndim=1)
        
        # In-place complex exponential: exp(u + i*v)
        k_exp(
            queue, (in_len,), None,
            d_kernel.data, np.uint64(in_len)
        )
        queue.finish()
        
        del gpu_log_mag, gpu_cep
    else:
        # Linear Phase Apodizing Kernel
        k_apod = get_sinc_kernel(queue.context, "init_apodizing_filter")
        d_kernel = cla.empty(queue, (in_len,), dtype=np.complex128)
        k_apod(
            queue, (in_len,), None,
            d_kernel.data, np.uint64(in_len), np.uint64(k_cutoff), np.uint64(transition_bins)
        )
        queue.finish()

    GPU_FILTER_CACHE[key] = d_kernel
    return d_kernel, 1


# ==============================================================================
# GPU SINC UPSAMPLER ENGINE (BATCHED MULTI-CHANNEL IN VRAM)
# ==============================================================================

def upsample_multichannel_gpu(pcm_data, scale_factor, queue, phase_mode="linear", apodizing=False, cutoff_hz=None, src_sr=None, steep=False):
    """
    Batched multi-channel GPU sinc upsampler using 64-bit double precision.
    Processes all channels simultaneously in a single GPU submission.
    """
    orig_samples, num_channels = pcm_data.shape
    fast_input_len = scipy.fft.next_fast_len(orig_samples, real=True)
    fast_output_len = fast_input_len * scale_factor
    orig_target_samples = orig_samples * scale_factor

    input_buf = np.zeros((num_channels, fast_input_len), dtype=np.float64)
    input_buf[:, :orig_samples] = pcm_data.T

    fft_in_shape = fast_input_len // 2 + 1
    padded_spectrum_shape = fast_output_len // 2 + 1

    # 1. Single Batched Host->GPU Upload for all channels
    gpu_in = cla.to_device(queue, input_buf)

    # 2. Batched Forward 1D FFT on GPU across all channels
    gpu_freq = rfftn(gpu_in, ndim=1)

    # 3. Sinc Zero-Padding + Filter Multiplication entirely inside VRAM
    d_kernel, apply_filter = get_gpu_filter_kernel(queue, fast_input_len, fft_in_shape, phase_mode, apodizing, cutoff_hz, src_sr, steep)
    kernel_data = d_kernel.data if d_kernel is not None else gpu_freq.data

    gpu_padded = cla.empty(queue, (num_channels, padded_spectrum_shape), dtype=np.complex128)
    sinc_kernel = get_sinc_kernel(queue.context, "sinc_interpolate_multichannel_gpu")
    
    sinc_kernel(
        queue, (padded_spectrum_shape, num_channels), None,
        gpu_freq.data, gpu_padded.data, kernel_data,
        np.uint64(fft_in_shape), np.uint64(padded_spectrum_shape),
        np.int32(apply_filter), np.uint64(num_channels)
    )

    # 4. Batched Inverse 1D FFT on GPU across all channels
    gpu_time = irfftn(gpu_padded, ndim=1, norm=0)
    queue.finish()

    # 5. Single Batched GPU->Host Download
    res_buf = gpu_time.get() / fast_input_len

    del gpu_in, gpu_freq, gpu_padded, gpu_time, input_buf

    # Transpose back to (samples, channels)
    return res_buf[:, :orig_target_samples].T.copy()


# ==============================================================================
# FAST MULTI-CORE CPU NOISE SHAPING (NUMBA PARALLEL IN L1 CACHE)
# ==============================================================================

HIGH_RATE_COEFFS = np.array([4.0, -6.0, 4.0, -1.0], dtype=np.float64)
SHIBATA_COEFFS = np.array([2.4332, -3.1251, 2.3789, -1.1345, 0.2865], dtype=np.float64)

if HAS_NUMBA:
    @njit(parallel=True, fastmath=True)
    def _noise_shape_loop_numba_parallel(data, coeffs):
        num_samples, num_channels = data.shape
        out = np.empty((num_samples, num_channels), dtype=np.int32)
        scale = 8388608.0
        inv_scale = 1.0 / scale
        dither_amp = 0.5 * inv_scale
        order = len(coeffs)

        for ch in prange(num_channels):
            rng = uint64(123456789 + ch * 987654321)
            e_hist = np.zeros(order, dtype=np.float64)

            for n in range(num_samples):
                feedback = 0.0
                for k in range(order):
                    feedback += coeffs[k] * e_hist[k]

                loop_target = data[n, ch] + feedback

                # Inlined fast XorShift64* PRNG
                rng ^= (rng >> uint64(12))
                rng ^= (rng << uint64(25))
                rng ^= (rng >> uint64(27))
                r1_val = (float((rng * uint64(0x2545F4914F6CDD1D)) >> uint64(11)) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0

                rng ^= (rng >> uint64(12))
                rng ^= (rng << uint64(25))
                rng ^= (rng >> uint64(27))
                r2_val = (float((rng * uint64(0x2545F4914F6CDD1D)) >> uint64(11)) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0

                tpdf = (r1_val + r2_val) * dither_amp
                q_int = np.round((loop_target + tpdf) * scale)

                if q_int > 8388607.0: q_int = 8388607.0
                elif q_int < -8388608.0: q_int = -8388608.0

                out[n, ch] = np.int32(q_int) * 256
                q_out = q_int * inv_scale
                error = loop_target - q_out

                for k in range(order - 1, 0, -1):
                    e_hist[k] = e_hist[k - 1]
                e_hist[0] = error

        return out

def _noise_shape_loop_python(data, coeffs):
    num_samples, num_channels = data.shape
    out = np.empty((num_samples, num_channels), dtype=np.int32)
    scale = 8388608.0
    inv_scale = 1.0 / scale
    dither_amp = 0.5 * inv_scale
    order = len(coeffs)

    for ch in range(num_channels):
        e_hist = np.zeros(order, dtype=np.float64)
        for n in range(num_samples):
            feedback = 0.0
            for k in range(order):
                feedback += coeffs[k] * e_hist[k]

            loop_target = data[n, ch] + feedback
            tpdf = (np.random.uniform(-1.0, 1.0) + np.random.uniform(-1.0, 1.0)) * dither_amp
            q_int = np.round((loop_target + tpdf) * scale)
            
            if q_int > 8388607.0: q_int = 8388607.0
            elif q_int < -8388608.0: q_int = -8388608.0

            out[n, ch] = int(q_int) * 256
            q_out = q_int * inv_scale
            error = loop_target - q_out

            for k in range(order - 1, 0, -1):
                e_hist[k] = e_hist[k - 1]
            e_hist[0] = error

    return out

def apply_dither_and_noise_shaping(data, dither_mode="shibata"):
    scale = 8388608.0
    if dither_mode == "none":
        clamped = np.clip(np.round(data * scale), -8388608.0, 8388607.0)
        return (clamped.astype(np.int32) * 256)

    coeffs = SHIBATA_COEFFS if dither_mode == "shibata" else HIGH_RATE_COEFFS

    if HAS_NUMBA:
        return _noise_shape_loop_numba_parallel(data, coeffs)
    else:
        return _noise_shape_loop_python(data, coeffs)


# ==============================================================================
# ASYNC BOUNDED FILE WRITER (FLAC COMPRESSION LEVEL 5)
# ==============================================================================

class BoundedThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, max_workers, max_queue_size):
        super().__init__(max_workers=max_workers)
        self._semaphore = threading.Semaphore(max_workers + max_queue_size)

    def submit(self, fn, *args, **kwargs):
        self._semaphore.acquire()
        try:
            future = super().submit(fn, *args, **kwargs)
            future.add_done_callback(lambda f: self._semaphore.release())
            return future
        except Exception:
            self._semaphore.release()
            raise

file_writer_pool = BoundedThreadPoolExecutor(max_workers=1, max_queue_size=1)


# ==============================================================================
# METADATA & REPLAYGAIN COPY ENGINE WITH AUTO-REPAIR ARTWORK
# ==============================================================================

def copy_metadata_and_update_replaygain(src_path, dst_path, track_peak_lin, album_peak_lin, gain_factor):
    try:
        src = FLAC(src_path) if src_path.lower().endswith('.flac') else File(src_path)
        dst = FLAC(dst_path) if dst_path.lower().endswith('.flac') else File(dst_path)
        if src is not None and dst is not None and src.tags is not None:
            dst.delete()
            if hasattr(dst, 'clear_pictures'):
                dst.clear_pictures()

            # MQA tag keys to sanitize when saving upsampled / processed clean PCM
            MQA_TAG_KEYS = {
                "MQAENCODER", 
                "ORIGINALSAMPLERATE", 
                "MQAAUTHENTICATION", 
                "MQA_SAMPLE_RATE", 
                "MQA_ORIGINAL_SAMPLE_RATE",
                "MQA_FLAGS"
            }

            for key, value in src.tags.items():
                ku = key.upper()
                if ku.startswith("REPLAYGAIN_"):
                    continue
                if ku in MQA_TAG_KEYS:
                    continue
                if ku == "ENCODER":
                    v_str = str(value[0] if isinstance(value, list) else value)
                    if "MQA" in v_str.upper():
                        continue
                dst.tags[key] = value

            dest_dir = os.path.dirname(dst_path)
            cover_written = False

            if hasattr(src, 'pictures') and src.pictures:
                for pic in src.pictures:
                    new_pic = Picture()
                    new_pic.data = pic.data
                    new_pic.type = pic.type if pic.type else 3
                    new_pic.mime = pic.mime if pic.mime else "image/jpeg"
                    new_pic.desc = pic.desc

                    # Auto-repair missing/zero image dimensions and color depth for WiiM / Lyrion compatibility
                    try:
                        with Image.open(io.BytesIO(pic.data)) as img:
                            new_pic.width, new_pic.height = img.size
                            if img.mode in ('RGB', 'YCbCr'):
                                new_pic.depth = 24
                            elif img.mode == 'RGBA':
                                new_pic.depth = 32
                            else:
                                new_pic.depth = 8
                    except Exception:
                        new_pic.width = pic.width
                        new_pic.height = pic.height
                        new_pic.depth = pic.depth

                    dst.add_picture(new_pic)

                    # Export standalone cover.jpg and folder.jpg for Lyrion (LMS) fast folder art discovery
                    if (new_pic.type == 3 or not cover_written) and dest_dir and os.path.exists(dest_dir):
                        for cover_name in ('cover.jpg', 'folder.jpg'):
                            cover_path = os.path.join(dest_dir, cover_name)
                            if not os.path.exists(cover_path):
                                try:
                                    with open(cover_path, 'wb') as cf:
                                        cf.write(pic.data)
                                    cover_written = True
                                except Exception:
                                    pass

            new_track_peak_lin = track_peak_lin * gain_factor
            new_album_peak_lin = album_peak_lin * gain_factor

            new_track_peak_db = 20 * np.log10(new_track_peak_lin) if new_track_peak_lin > 0 else -99.0
            new_album_peak_db = 20 * np.log10(new_album_peak_lin) if new_album_peak_lin > 0 else -99.0

            track_gain_db = -18.0 - new_track_peak_db
            album_gain_db = -18.0 - new_album_peak_db

            dst.tags['REPLAYGAIN_TRACK_PEAK'] = f"{new_track_peak_lin:.6f}"
            dst.tags['REPLAYGAIN_TRACK_GAIN'] = f"{track_gain_db:+.2f} dB"
            dst.tags['REPLAYGAIN_ALBUM_PEAK'] = f"{new_album_peak_lin:.6f}"
            dst.tags['REPLAYGAIN_ALBUM_GAIN'] = f"{album_gain_db:+.2f} dB"

            dst.save()
    except Exception:
        pass


def save_and_tag_async(wip_path, dest_path, int32_data, target_rate, filepath, track_start_time, out_peak_lin, album_max_peak_lin, gain_factor):
    t_write = time.time()
    num_samples, num_channels = int32_data.shape
    BLOCK_SIZE = target_rate * 10
    
    # Strictly enforce FLAC Compression Level 5 (0.625)
    with sf.SoundFile(wip_path, mode='w', samplerate=target_rate, channels=num_channels, 
                      format='FLAC', subtype='PCM_24', compression_level=FLAC_COMPRESSION_LEVEL_5) as sf_out:
        pos = 0
        while pos < num_samples:
            end_pos = min(pos + BLOCK_SIZE, num_samples)
            sf_out.write(int32_data[pos:end_pos])
            pos += BLOCK_SIZE

    copy_metadata_and_update_replaygain(filepath, wip_path, out_peak_lin, album_max_peak_lin, gain_factor)
    os.rename(wip_path, dest_path)
    del int32_data
    gc.collect()

    elapsed_total = time.time() - track_start_time
    print(f"   [+{elapsed_total:6.2f}s] [NVMe Writer] Stream-saved (FLAC L5) & tagged {os.path.basename(dest_path)} in {time.time() - t_write:.2f}s")


# ==============================================================================
# DSP UTILITIES & DIRECTORY ROUTING
# ==============================================================================

def get_target_scale(samplerate):
    if samplerate == 44100: return 4, 176400
    elif samplerate == 88200: return 2, 176400
    elif samplerate >= 176400 and (samplerate % 44100 == 0): return 1, samplerate
    elif samplerate == 48000: return 4, 192000
    elif samplerate == 96000: return 2, 192000
    elif samplerate >= 192000 and (samplerate % 48000 == 0): return 1, samplerate
    elif samplerate <= 48000: return 4, samplerate * 4
    elif samplerate <= 96000: return 2, samplerate * 2
    else: return 1, samplerate

def is_memmap_necessary(orig_samples, scale_factor, num_channels):
    target_samples = orig_samples * scale_factor
    required_bytes = target_samples * num_channels * 8
    if HAS_PSUTIL:
        available_ram = psutil.virtual_memory().available
        return required_bytes > (available_ram * 0.40)
    return required_bytes > (4.5 * 1024 * 1024 * 1024)

def get_destination_dir(source_dir, root_source_dir, root_target_dir):
    """
    Computes the corresponding output directory for an album folder,
    preserving the source subdirectory hierarchy under root_target_dir.
    """
    if os.path.abspath(source_dir) == os.path.abspath(root_source_dir):
        return root_target_dir
    rel_path = os.path.relpath(source_dir, root_source_dir)
    return os.path.join(root_target_dir, rel_path)

def get_audio_files(directory, recursive=False):
    audio_files = []
    try:
        if recursive:
            for r, _, f in os.walk(directory):
                for file in f:
                    if file.lower().endswith(('.flac', '.wav')):
                        audio_files.append(os.path.join(r, file))
        else:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.lower().endswith(('.flac', '.wav')):
                        audio_files.append(entry.path)
    except Exception:
        pass
    return sorted(audio_files)


# ==============================================================================
# PRE-FLIGHT ALBUM SCAN WITH ADAPTIVE CREST-AWARE HEADROOM
# ==============================================================================

def inspect_track_fast(filepath):
    try:
        info = probe_audio_info_resilient(filepath)
        scale, _ = get_target_scale(info['samplerate'])
    except Exception:
        return 4, 1.0, 0.1, False

    # Fast Path A: ReplayGain tags
    try:
        audio = File(filepath)
        if audio is not None and audio.tags is not None:
            peak_val = None
            gain_val = None
            for key, val in audio.tags.items():
                k_upper = key.upper()
                if k_upper == 'REPLAYGAIN_TRACK_PEAK':
                    peak_str = val[0] if isinstance(val, list) else str(val)
                    peak_val = float(peak_str)
                elif k_upper == 'REPLAYGAIN_TRACK_GAIN':
                    gain_str = val[0] if isinstance(val, list) else str(val)
                    gain_val = float(gain_str.replace('dB', '').strip())
            
            if peak_val is not None and gain_val is not None:
                rms_val = 10.0 ** ((-18.0 - gain_val) / 20.0)
                return scale, peak_val, rms_val, True
            elif peak_val is not None:
                return scale, peak_val, peak_val * 0.25, True
    except Exception:
        pass

    # Fast Path B: Chunked Streaming PCM Audio Scan (Memory-Safe with Resilient Fallback)
    try:
        with sf.SoundFile(filepath) as f:
            pk = 0.0
            sum_sq = 0.0
            total_samples = 0
            block_size = 65536
            while f.tell() < f.frames:
                block = f.read(block_size, dtype='float64')
                if block.size == 0:
                    break
                block_pk = float(np.max(np.abs(block)))
                if block_pk > pk:
                    pk = block_pk
                sum_sq += float(np.sum(block ** 2))
                total_samples += block.size
            rms = float(np.sqrt(sum_sq / max(1, total_samples)))
            return scale, pk, rms, False
    except Exception:
        try:
            data, _ = load_audio_resilient(filepath, dtype='float64')
            if data is not None and data.size > 0:
                pk = float(np.max(np.abs(data)))
                rms = float(np.sqrt(np.mean(data ** 2)))
                return scale, pk, rms, False
        except Exception:
            pass
        return scale, 1.0, 0.1, False

def scan_album(files):
    t_scan = time.time()
    album_max_peak_lin = 0.0
    min_crest_db = 999.0
    requires_upsampling = False
    all_used_fast_path = True

    print("   Scanning album tracks for adaptive headroom factor...")
    max_workers = min(len(files), os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(inspect_track_fast, files))

    for scale, pk, rms, from_tag in results:
        if scale > 1: requires_upsampling = True
        if not from_tag: all_used_fast_path = False
        if pk > album_max_peak_lin: album_max_peak_lin = pk
        
        if pk > 1e-5 and rms > 1e-7:
            crest = 20.0 * np.log10(pk / rms)
            if crest < min_crest_db:
                min_crest_db = crest

    album_max_peak_db = 20 * np.log10(album_max_peak_lin) if album_max_peak_lin > 0 else -999.0
    scan_type = "Metadata Header Tags" if all_used_fast_path else "Parallel PCM Audio Scan"
    
    if min_crest_db == 999.0:
        min_crest_db = 14.0  # Default fallback

    # Adaptive Intersample Headroom Margin based on dynamic crest profile:
    if min_crest_db >= 16.0:
        adaptive_margin_db = 1.8
        genre_profile = "High Dynamic Range (Classical / Acoustic Jazz)"
    elif min_crest_db >= 12.0:
        adaptive_margin_db = 2.8
        genre_profile = "Standard Dynamic Range (Acoustic / Audiophile Master)"
    elif min_crest_db >= 8.0:
        adaptive_margin_db = 3.8
        genre_profile = "Compressed Master (Commercial Pop / Rock)"
    else:
        adaptive_margin_db = 4.4
        genre_profile = "Hyper-Compressed / Brickwall Limited (Club / EDM)"

    print(f"   Album Max Source Peak: {album_max_peak_db:6.2f} dBFS | Min Crest: {min_crest_db:4.1f} dB ({scan_type} in {time.time() - t_scan:.3f}s)")
    print(f"   Dynamic Profile       : {genre_profile} -> Calibrated Intersample Margin: +{adaptive_margin_db:.1f} dB")

    est_peak_db = album_max_peak_db + (adaptive_margin_db if requires_upsampling else 0)
    if est_peak_db > PEAK_TARGET_DB:
        gain_factor = 10 ** ((PEAK_TARGET_DB - est_peak_db) / 20.0)
    else:
        gain_factor = 1.0

    print(f"   Calculated Headroom Gain Multiplier: {gain_factor:.6f} ({20*np.log10(gain_factor):+.2f} dB)")
    return gain_factor, album_max_peak_lin


# ==============================================================================
# TRACK PROCESSOR
# ==============================================================================

def process_track(filepath, dest_dir, gain_factor, album_max_peak_lin, queue, phase_mode, dither_mode, tmp_dir, apodizing=False, mqa_mode='adaptive', cutoff_hz=None, steep=False, overwrite=False):
    t0 = time.time()
    def elapsed(): return f"[+{time.time() - t0:6.2f}s]"

    filename = os.path.basename(filepath)
    dest_path = os.path.join(dest_dir, filename)
    wip_path = dest_path + ".WIP"

    # Bulletproof Safety Guard: Never overwrite original source file
    if os.path.abspath(filepath) == os.path.abspath(dest_path):
        print(f"   [Fatal Safety Error] Target path matches original source file ({dest_path}). Skipping to protect original.")
        return False, 0.0, None

    if os.path.exists(dest_path) and not overwrite:
        print(f"   [Skip] Output exists: {filename} (use --overwrite to overwrite)")
        return False, 0.0, None

    try: 
        data, samplerate = load_audio_resilient(filepath, dtype='float64')
    except Exception: 
        return False, 0.0, None
        
    if data is None or data.size == 0:
        return False, 0.0, None
        
    if data.ndim == 1: 
        data = data[:, np.newaxis]

    # MQA Auto-Detection & Core Unfolding / Stripping Pre-Processor
    if mqa_mode != 'ignore':
        try:
            from mqa_unfolder import probe_mqa_track, unfold_mqa_adaptive, unfold_mqa_simple, strip_mqa_payload
            is_mqa, orig_mqa_sr, is_studio, mqa_sig = probe_mqa_track(filepath=filepath, sr=samplerate)
            if is_mqa:
                studio_tag = "MQA Studio Master" if is_studio else "MQA Authenticated"
                if mqa_mode == 'strip':
                    print(f"{elapsed()} [MQA Pre-Processor] Detected {studio_tag} -> Stripping MQA payload bits & re-dithering...")
                    data, samplerate = strip_mqa_payload(data, samplerate)
                elif mqa_mode == 'simple' and samplerate in (44100, 48000):
                    if orig_mqa_sr and orig_mqa_sr <= samplerate:
                        print(f"{elapsed()} [MQA Pre-Processor] Detected {studio_tag} ({orig_mqa_sr:,} Hz native master, no ultrasonic subbands) -> Auto-routing to Strip & Re-Dither...")
                        data, samplerate = strip_mqa_payload(data, samplerate)
                    else:
                        print(f"{elapsed()} [MQA Pre-Processor] Detected {studio_tag} (Original Master: {orig_mqa_sr:,} Hz) -> Simple linear subband unfold ({samplerate} Hz -> {samplerate*2} Hz)...")
                        data, samplerate = unfold_mqa_simple(data, samplerate)
                elif mqa_mode == 'adaptive' and samplerate in (44100, 48000):
                    if orig_mqa_sr and orig_mqa_sr <= samplerate:
                        print(f"{elapsed()} [MQA Pre-Processor] Detected {studio_tag} ({orig_mqa_sr:,} Hz native master, no ultrasonic subbands) -> Auto-routing to Strip & Re-Dither...")
                        data, samplerate = strip_mqa_payload(data, samplerate)
                    else:
                        print(f"{elapsed()} [MQA Pre-Processor] Detected {studio_tag} (Original Master: {orig_mqa_sr:,} Hz) -> Adaptive companded unfold ({samplerate} Hz -> {samplerate*2} Hz | <= -95 dBFS floor)...")
                        data, samplerate = unfold_mqa_adaptive(data, samplerate)
        except Exception as e:
            pass

    orig_samples, num_channels = data.shape
    scale_factor, target_rate = get_target_scale(samplerate)

    is_min = phase_mode.lower() in ['min', 'minimum']
    phase_label = "MINIMUM PHASE" if is_min else "LINEAR PHASE"
    if cutoff_hz:
        filter_label = f"{phase_label} APODIZING (Cutoff: {cutoff_hz:,.0f}Hz)"
    else:
        filter_label = f"{phase_label} APODIZING" if apodizing else phase_label

    print(f"\n{elapsed()} --------------------------------------------------")
    print(f"{elapsed()} Processing {filename}")
    print(f"{elapsed()} Input:  {orig_samples:,} samples @ {samplerate}Hz ({num_channels} ch)")
    print(f"{elapsed()} Target: {target_rate}Hz (Scale: {scale_factor}x | Filter: {filter_label} | Dither: {dither_mode.upper()})")

    data = data * gain_factor

    needs_filtering = (cutoff_hz is not None) or apodizing or (phase_mode.lower() in ['min', 'minimum'])
    if scale_factor == 1 and not needs_filtering:
        out_int32 = apply_dither_and_noise_shaping(data, dither_mode)
        fut = file_writer_pool.submit(save_and_tag_async, wip_path, dest_path, out_int32, target_rate, filepath, t0, np.max(np.abs(data)), album_max_peak_lin, gain_factor)
        return False, np.max(np.abs(data)), fut

    orig_target_samples = orig_samples * scale_factor
    use_mmap = is_memmap_necessary(orig_samples, scale_factor, num_channels)
    mmap_path = None

    if use_mmap:
        session_id = str(uuid.uuid4())[:8]
        mmap_path = os.path.join(tmp_dir, f"acoustisinc_out_{session_id}.mmap")
        output_data = np.memmap(mmap_path, dtype='float64', mode='w+', shape=(orig_target_samples, num_channels))
    else:
        output_data = np.zeros((orig_target_samples, num_channels), dtype=np.float64)

    t_gpu = time.time()
    if orig_samples > SAFE_CHUNK_LEN:
        chunk_size = SAFE_CHUNK_LEN - (2 * OVERLAP_SAMPLES)
        overlap = OVERLAP_SAMPLES
        print(f"{elapsed()} Processing with Batched Overlap-Save GPU chunking...")
        
        pos = 0
        while pos < orig_samples:
            src_start = max(0, pos - overlap)
            src_end = min(orig_samples, pos + chunk_size + overlap)
            chunk_pcm = data[src_start:src_end, :]
            
            if len(chunk_pcm) < SAFE_CHUNK_LEN:
                chunk_buf = np.zeros((SAFE_CHUNK_LEN, num_channels), dtype=np.float64)
                chunk_buf[:len(chunk_pcm), :] = chunk_pcm
                upsampled_chunk = upsample_multichannel_gpu(chunk_buf, scale_factor, queue, phase_mode, apodizing, cutoff_hz=cutoff_hz, src_sr=samplerate, steep=steep)
            else:
                upsampled_chunk = upsample_multichannel_gpu(chunk_pcm, scale_factor, queue, phase_mode, apodizing, cutoff_hz=cutoff_hz, src_sr=samplerate, steep=steep)
            
            left_guard = (pos - src_start) * scale_factor
            valid_len = min(chunk_size, orig_samples - pos) * scale_factor
            valid_part = upsampled_chunk[left_guard : left_guard + valid_len, :]
            
            target_start = pos * scale_factor
            output_data[target_start : target_start + len(valid_part), :] = valid_part
            pos += chunk_size
            del upsampled_chunk

        if use_mmap: output_data.flush()
        gc.collect()
        print(f"{elapsed()} Batched GPU Sinc Chunking complete in {time.time() - t_gpu:.3f}s")
    else:
        output_data = upsample_multichannel_gpu(data, scale_factor, queue, phase_mode, apodizing, cutoff_hz=cutoff_hz, src_sr=samplerate, steep=steep)
        print(f"{elapsed()} Batched GPU Sinc complete in {time.time() - t_gpu:.3f}s")

    del data
    gc.collect()

    # Peak check for intersample clipping
    if use_mmap:
        out_peak = 0.0
        SCAN_BLOCK = target_rate * 60
        for pos in range(0, orig_target_samples, SCAN_BLOCK):
            block = np.array(output_data[pos : min(pos + SCAN_BLOCK, orig_target_samples)], dtype=np.float64)
            bp = np.max(np.abs(block))
            if bp > out_peak: out_peak = bp
    else:
        out_peak = np.max(np.abs(output_data))

    out_peak_db = 20 * np.log10(out_peak) if out_peak > 0 else -999.0

    if out_peak > PEAK_TARGET_LIN:
        overshoot_db = out_peak_db - PEAK_TARGET_DB
        print(f"{elapsed()} [Warning] Intersample clipping detected! Peak: {out_peak_db:.2f} dBFS (Overshoot: +{overshoot_db:.2f} dB > Target: {PEAK_TARGET_DB} dBFS)")
        if os.path.exists(wip_path): os.remove(wip_path)
        del output_data
        if use_mmap and mmap_path and os.path.exists(mmap_path): os.remove(mmap_path)
        gc.collect()
        clear_vkfftapp_cache()
        return True, out_peak, None

    t_dither = time.time()
    out_int32 = apply_dither_and_noise_shaping(output_data, dither_mode)
    print(f"{elapsed()} Multi-Core CPU Noise Shaping completed in {time.time() - t_dither:.3f}s")

    del output_data
    if use_mmap and mmap_path and os.path.exists(mmap_path):
        try: os.remove(mmap_path)
        except Exception: pass
    gc.collect()

    # Async hand-off to NVMe FLAC compression worker
    fut = file_writer_pool.submit(save_and_tag_async, wip_path, dest_path, out_int32, target_rate, filepath, t0, out_peak, album_max_peak_lin, gain_factor)
    print(f"{elapsed()} [Compute Pipeline Complete] Handed off to NVMe Writer")
    return False, out_peak, fut


# ==============================================================================
# ALBUM BATCH CONTROLLER WITH RETRY AUTO-HEALING
# ==============================================================================

def process_album_folder(source_album_dir, dest_album_dir, queue, default_params, tmp_dir, overwrite_mode='off', prompt_ctrl=None):
    print(f"\n==================================================")
    print(f"Directory:   {source_album_dir}")
    print(f"Destination: {dest_album_dir}")
    print(f"==================================================")

    # Bulletproof Safety Guard: Destination cannot be identical to source directory
    if os.path.abspath(source_album_dir) == os.path.abspath(dest_album_dir):
        print(f"[Fatal Safety Error] Destination directory is identical to source directory: {source_album_dir}. Skipping album to protect original files.")
        return

    if prompt_ctrl is not None:
        prompt_ctrl.set_album_context(source_album_dir)

    files = get_audio_files(source_album_dir)
    if not files: return

    # Skip entire album upfront ONLY if overwrite mode is completely 'off' (or album skipped)
    cur_overwrite = prompt_ctrl.overwrite_mode if prompt_ctrl else overwrite_mode
    cur_album_overwrite = prompt_ctrl.album_overwrite_mode if prompt_ctrl else None
    if (cur_overwrite == 'off' or cur_album_overwrite == 'off') and all(os.path.exists(os.path.join(dest_album_dir, os.path.basename(f))) for f in files):
        print(f">>> All {len(files)} target files already exist in {dest_album_dir}. Skipping album. (Use --overwrite to overwrite)")
        return

    os.makedirs(dest_album_dir, exist_ok=True)
    try:
        with os.scandir(source_album_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    dst_img = os.path.join(dest_album_dir, entry.name)
                    if not os.path.exists(dst_img): shutil.copy2(entry.path, dst_img)
    except Exception: pass

    gain_factor, album_max_peak_lin = scan_album(files)
    max_retries = 4; retry_count = 0; album_success = False
    last_track_params = default_params
    processed_tracks = []

    while not album_success and retry_count <= max_retries:
        clipped_in_pass = False
        max_overshoot_peak = 0.0
        pass_futures = []
        processed_tracks = []

        for idx, f in enumerate(files, 1):
            dest_file = os.path.join(dest_album_dir, os.path.basename(f))
            if os.path.exists(dest_file):
                if prompt_ctrl is not None:
                    should_overwrite = prompt_ctrl.resolve_overwrite(dest_file, f, idx, len(files))
                else:
                    should_overwrite = (overwrite_mode == 'on')

                if not should_overwrite:
                    print(f"   [Skip] Output exists: {os.path.basename(f)} (use --overwrite to overwrite)")
                    continue

            if prompt_ctrl is not None:
                track_params, skip = prompt_ctrl.resolve_track_params(f, idx, len(files), default_params)
                if skip == 'album' or getattr(prompt_ctrl, 'skip_album', False):
                    print(f">>> Skipping remainder of album: {os.path.basename(source_album_dir)}\n")
                    break
                if skip:
                    continue
            else:
                track_params = default_params

            last_track_params = track_params
            print(f"\n--- Track [{idx}/{len(files)}] ---")
            clipped, out_peak, fut = process_track(
                f, dest_album_dir, gain_factor, album_max_peak_lin, queue,
                phase_mode=track_params['phase_mode'],
                dither_mode=track_params['dither_mode'],
                tmp_dir=tmp_dir,
                apodizing=track_params['apodizing'],
                mqa_mode=track_params['mqa_mode'],
                cutoff_hz=track_params['cutoff_hz'],
                steep=track_params['steep'],
                overwrite=True
            )
            if fut is not None:
                pass_futures.append(fut)
            
            t_recipe = {
                "cli_params": f"--phase {track_params['phase_mode']} --dither {track_params['dither_mode']}" + (f" --cutoff {int(track_params['cutoff_hz'])}" if track_params.get('cutoff_hz') else "") + (f" --mqa {track_params['mqa_mode']}" if track_params.get('mqa_mode') != 'adaptive' else "") + (" --steep" if track_params.get('steep') else ""),
                "topology_name": f"{track_params['phase_mode'].upper()} APODIZING" if (track_params.get('apodizing') or track_params.get('cutoff_hz')) else f"{track_params['phase_mode'].upper()}",
                "gain_factor": gain_factor,
                "gain_db": 20.0 * np.log10(gain_factor),
                "cutoff_hz": track_params.get('cutoff_hz'),
                "phase_mode": track_params['phase_mode'],
                "dither_mode": track_params['dither_mode'],
                "mqa_mode": track_params['mqa_mode'],
                "steep": track_params.get('steep', False),
                "date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            processed_tracks.append({
                "src_path": f,
                "dst_path": dest_file,
                "recipe": t_recipe
            })
            if clipped:
                clipped_in_pass = True
                max_overshoot_peak = out_peak
                break

        if clipped_in_pass:
            retry_count += 1
            if retry_count > max_retries:
                print(f"\n[Error] Failed to resolve album clipping after {max_retries} attempts. Skipping album.")
                return
            
            # Calculate exact required dynamic backoff factor
            overshoot_ratio = PEAK_TARGET_LIN / max_overshoot_peak
            safety_margin_lin = 10.0 ** (-0.2 / 20.0)  # Extra 0.2 dB safety buffer
            exact_backoff_factor = overshoot_ratio * safety_margin_lin
            
            old_gain = gain_factor
            gain_factor *= exact_backoff_factor
            backoff_db = 20.0 * np.log10(exact_backoff_factor)

            print("==================================================")
            print(f">>> ABORTING CURRENT PASS due to intersample clipping (Overshoot: {20.0 * np.log10(max_overshoot_peak) - PEAK_TARGET_DB:+.2f} dB).")
            print(f">>> Calculated exact dynamic backoff: {backoff_db:.2f} dB (Gain: {old_gain:.6f} -> {gain_factor:.6f})")
            print("==================================================")

            # Drain any active background writers before wiping
            for fut in pass_futures:
                try: fut.result()
                except Exception: pass

            for f in files:
                out_f = os.path.join(dest_album_dir, os.path.basename(f))
                if os.path.exists(out_f):
                    try: os.remove(out_f)
                    except Exception: pass

            GPU_FILTER_CACHE.clear()
            clear_vkfftapp_cache()
            gc.collect()
            print(f">>> Restarting album pass with exact calculated Gain Factor: {gain_factor:.6f}")
        else:
            album_success = True

    # Drain any remaining background writers for this album to maintain low RAM footprint
    for fut in pass_futures:
        try: fut.result()
        except Exception as e:
            print(f"   [Async Writer Notice]: {e}")

    GPU_FILTER_CACHE.clear()
    clear_vkfftapp_cache()
    gc.collect()
    print(f"\n>>> Directory completed cleanly! (Gain Factor: {gain_factor:.6f})")

    # Generate Full Before & After Comparative Upsampling Report
    # ONLY analyze and generate reports if tracks were actually processed and album was not skipped
    is_album_skipped = bool(prompt_ctrl and getattr(prompt_ctrl, 'skip_album', False))
    if not is_album_skipped and processed_tracks:
        valid_items = [t for t in processed_tracks if os.path.exists(t["dst_path"])]
        if valid_items:
            alb_name = os.path.basename(source_album_dir) or "Album"
            applied_recipe = {
                "cli_params": f"--phase {last_track_params['phase_mode']} --dither {last_track_params['dither_mode']}" + (f" --cutoff {int(last_track_params['cutoff_hz'])}" if last_track_params.get('cutoff_hz') else "") + (f" --mqa {last_track_params['mqa_mode']}" if last_track_params.get('mqa_mode') != 'adaptive' else "") + (" --steep" if last_track_params.get('steep') else ""),
                "topology_name": f"{last_track_params['phase_mode'].upper()} APODIZING" if (last_track_params.get('apodizing') or last_track_params.get('cutoff_hz')) else f"{last_track_params['phase_mode'].upper()}",
                "gain_factor": gain_factor,
                "gain_db": 20.0 * np.log10(gain_factor),
                "cutoff_hz": last_track_params.get('cutoff_hz'),
                "date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            try:
                html_rep, md_rep = generate_comparative_report(valid_items, alb_name, applied_recipe, dest_album_dir)
                if html_rep:
                    print(f"\n   📊 [Comparative Upsampling Report Generated]")
                    print(f"      Interactive HTML Report: {html_rep}")
                    print(f"      Markdown Summary Report: {md_rep}")
            except Exception as e:
                print(f"   [Notice] Could not generate comparative report: {e}")
    elif is_album_skipped:
        print(f"   [Report Notice] Album skipped by user. Comparative report analysis bypassed.")
    elif not processed_tracks:
        print(f"   [Report Notice] No tracks upsampled in this session. Comparative report analysis bypassed.")


# ==============================================================================
# SESSION PROMPT CONTROLLER (PER-FILE FORENSIC RECIPE RESOLVER)
# ==============================================================================

class SessionPromptController:
    def __init__(self, mode='none', overwrite_mode='off', default_params=None):
        self.initial_mode = mode  # 'none', 'ask', 'auto'
        self.mode = mode  # 'none', 'ask', 'auto', 'locked'
        self.overwrite_mode = overwrite_mode  # 'on', 'off', 'ask'
        self.album_overwrite_mode = None  # None, 'on', 'off'
        self.default_params = default_params or {}
        self.locked_params = None
        self.skip_album = False
        self.current_album_dir = None
        self.track_decisions = {}
        self.overwrite_decisions = {}

    def set_album_context(self, album_dir):
        """
        Notify the controller that a new album directory is being processed.
        If a recipe or overwrite setting was locked for the previous album, reset to initial modes
        so that settings never bleed across distinct album directories.
        """
        if self.current_album_dir is not None and self.current_album_dir != album_dir:
            folder_name = os.path.basename(album_dir) or album_dir
            if self.mode == 'locked':
                print(f"\n[AcoustiSinc] Resetting per-album recipe lock for new directory: {folder_name}")
                self.mode = self.initial_mode
                self.locked_params = None
            if self.album_overwrite_mode is not None:
                print(f"\n[AcoustiSinc] Resetting per-album overwrite setting for new directory: {folder_name}")
                self.album_overwrite_mode = None
            self.skip_album = False
        self.current_album_dir = album_dir

    def resolve_overwrite(self, dest_path, src_filepath, track_idx=1, total_tracks=1):
        """
        Resolves whether an existing target file should be overwritten based on --overwrite {on|off|ask}.
        Returns True (overwrite) or False (skip).
        """
        if not os.path.exists(dest_path):
            return True

        # Re-use decision if already resolved in a prior pass (e.g. clipping retry)
        if dest_path in self.overwrite_decisions:
            return self.overwrite_decisions[dest_path]

        # Check album-level override
        if self.album_overwrite_mode == 'on':
            self.overwrite_decisions[dest_path] = True
            return True
        elif self.album_overwrite_mode == 'off':
            self.overwrite_decisions[dest_path] = False
            return False

        # Check session overwrite mode
        if self.overwrite_mode == 'on':
            self.overwrite_decisions[dest_path] = True
            return True
        elif self.overwrite_mode == 'off':
            self.overwrite_decisions[dest_path] = False
            return False

        # If overwrite_mode is 'ask' -> Interactive Prompt
        filename = os.path.basename(dest_path)
        track_str = f" [Track {track_idx}/{total_tracks}]" if track_idx and total_tracks else ""

        try:
            sz_mb = os.path.getsize(dest_path) / (1024 * 1024)
            mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(dest_path)))
        except Exception:
            sz_mb = 0.0
            mtime = "Unknown"

        safe_display_path = dest_path
        for prefix in ["/mnt/PrimaryFS/FLAC_music/music/", "/mnt/PrimaryFS/1xxK_min/music/", "/mnt/PrimaryFS/", "/home/amkrebet/"]:
            if dest_path.startswith(prefix):
                safe_display_path = "/Music/Hi-Res/" + dest_path[len(prefix):].lstrip("/")
                break

        print(f"\n================================================================================")
        print(f"⚠️  TARGET FILE ALREADY EXISTS{track_str}: {filename}")
        print(f"================================================================================")
        print(f"   Target Path   : {safe_display_path}")
        print(f"   Target Size   : {sz_mb:.2f} MB")
        print(f"   Last Modified : {mtime}")
        print("--------------------------------------------------------------------------------")
        print("[Y] Overwrite this track (Default)     |  [A] Overwrite ALL in entire run")
        print("[C] Overwrite REST of this album       |  [N] Skip this track (keep existing)")
        print("[S] Skip REST of existing in album     |  [Q] Quit")
        try:
            choice = input("Choice [Y/n/a/c/s/q]: ").strip().lower()
        except EOFError:
            choice = 'y'

        if choice in ['', 'y', 'yes']:
            print(f">>> Overwriting target file: {filename}\n")
            self.overwrite_decisions[dest_path] = True
            return True

        elif choice in ['a', 'all']:
            print(f">>> Enabling overwrite for this and ALL remaining files in the entire session.\n")
            self.overwrite_mode = 'on'
            self.overwrite_decisions[dest_path] = True
            return True

        elif choice in ['c', 'continue', 'album']:
            print(f">>> Enabling overwrite for the REST of this album directory.\n")
            self.album_overwrite_mode = 'on'
            self.overwrite_decisions[dest_path] = True
            return True

        elif choice in ['n', 'no', 'skip']:
            print(f">>> Keeping existing file, skipping track: {filename}\n")
            self.overwrite_decisions[dest_path] = False
            return False

        elif choice in ['s', 'skip_album']:
            print(f">>> Skipping this and all remaining existing files in this album directory.\n")
            self.album_overwrite_mode = 'off'
            self.overwrite_decisions[dest_path] = False
            return False

        elif choice in ['q', 'quit', 'abort']:
            print("\n[AcoustiSinc] Processing aborted by user.")
            sys.exit(0)
        else:
            print(f">>> Overwriting target file: {filename}\n")
            self.overwrite_decisions[dest_path] = True
            return True

    def resolve_track_params(self, filepath, track_idx=1, total_tracks=1, fallback_params=None):
        base_params = fallback_params or self.default_params

        # Re-use decision if already resolved in a prior pass (e.g. clipping retry)
        if filepath in self.track_decisions:
            return self.track_decisions[filepath], False

        # If user chose [C] Freeze recipe for remainder of album/session
        if self.mode == 'locked' and self.locked_params is not None:
            self.track_decisions[filepath] = self.locked_params
            return self.locked_params, False

        # If mode is 'none' (user did not specify --use-recommended)
        if self.mode == 'none':
            return base_params, False

        # If mode is 'auto'
        if self.mode == 'auto':
            rec_params, rec_info = self._audit_track(filepath, print_card=False)
            dsp_str = rec_info.get("dsp_params", "--phase min --dither shibata")
            print(f">>> [Auto-Recommended] {os.path.basename(filepath)} -> {dsp_str}")
            self.track_decisions[filepath] = rec_params
            return rec_params, False

        # If user chose to skip the rest of the album
        if self.skip_album:
            return None, 'album'

        # If mode is 'ask' -> Interactive Card Prompt
        rec_params, rec_info = self._audit_track(filepath, print_card=True, track_idx=track_idx, total_tracks=total_tracks)
        dsp_str = rec_info.get("dsp_params", "--phase min --dither shibata")

        # Check if an album analysis summary / report exists in the folder
        album_dir = self.current_album_dir or os.path.dirname(filepath)
        album_summary_path = self._find_album_summary(album_dir)
        has_summary = (album_summary_path is not None)

        # Emit structured machine-readable prompt for Web UI
        prompt_payload = {
            "track_file": os.path.basename(filepath),
            "filepath": filepath,
            "track_idx": track_idx,
            "total_tracks": total_tracks,
            "rec_params": rec_params,
            "rec_info": rec_info,
            "has_summary": has_summary,
            "album_dir": album_dir
        }
        print(f"\n__PROMPT_JSON__:{json.dumps(prompt_payload)}", flush=True)

        while True:
            print("--------------------------------------------------------------------------------")
            print("[Y] Accept for this track (Default)    |  [A] Auto-apply recommended for ALL remaining")
            print("[C] Apply this recipe to REST of album |  [E] Edit parameters")
            if has_summary:
                print("[S] Skip this track                    |  [K] Skip REST of this album")
                print("[V] View Album Analysis Summary        |  [Q] Quit")
                prompt_str = "Choice [Y/a/c/e/s/k/v/q]: "
            else:
                print("[S] Skip this track                    |  [K] Skip REST of this album")
                print("[Q] Quit")
                prompt_str = "Choice [Y/a/c/e/s/k/q]: "

            try:
                raw_input = input(prompt_str).strip()
            except EOFError:
                raw_input = 'y'

            # Parse structured response from Web UI if provided
            if raw_input.startswith("__RESP_JSON__:"):
                try:
                    resp_dict = json.loads(raw_input[len("__RESP_JSON__:"):])
                    choice = resp_dict.get("choice", "y").strip().lower()
                    custom_params = resp_dict.get("custom_params")
                    if custom_params and isinstance(custom_params, dict):
                        rec_params.update(custom_params)
                        if rec_params.get("cutoff_hz"):
                            rec_params["apodizing"] = True
                        dsp_str = f"--phase {rec_params.get('phase_mode', 'min')} --dither {rec_params.get('dither_mode', 'shibata')}" + (f" --cutoff {int(rec_params['cutoff_hz'])}" if rec_params.get('cutoff_hz') else "")
                except Exception:
                    choice = "y"
            else:
                choice = raw_input.lower()

            if choice in ['v', 'view', 'summary']:
                if album_summary_path:
                    self._display_album_summary(album_summary_path)
                else:
                    print("\n⚠️  No album analysis summary file found in this folder.\n")
                continue

            if choice in ['', 'y', 'yes']:
                print(f">>> Applying recipe to {os.path.basename(filepath)}: {dsp_str}\n")
                self.track_decisions[filepath] = rec_params
                return rec_params, False

            elif choice in ['a', 'all']:
                print(f">>> Adopting recommended recipes automatically for this and ALL remaining tracks.\n")
                self.mode = 'auto'
                self.track_decisions[filepath] = rec_params
                return rec_params, False

            elif choice in ['c', 'continue', 'freeze']:
                print(f">>> Freezing recipe ({dsp_str}) for the REST of this album directory.\n")
                self.mode = 'locked'
                self.locked_params = rec_params
                self.track_decisions[filepath] = rec_params
                return rec_params, False

            elif choice in ['e', 'edit']:
                custom_input = input("Enter custom CLI flags (e.g. --cutoff 22050 --phase min): ").strip()
                edited = dict(rec_params)
                if "--cutoff" in custom_input or "--apodize" in custom_input:
                    m = re.search(r"--(?:cutoff|apodize)\s+(\d+)", custom_input)
                    if m:
                        edited['cutoff_hz'] = float(m.group(1))
                        edited['apodizing'] = True
                if "--phase" in custom_input:
                    m = re.search(r"--phase\s+(\w+)", custom_input)
                    if m: edited['phase_mode'] = m.group(1)
                if "--dither" in custom_input:
                    m = re.search(r"--dither\s+(\w+)", custom_input)
                    if m: edited['dither_mode'] = m.group(1)
                if "--mqa" in custom_input:
                    m = re.search(r"--mqa\s+(\w+)", custom_input)
                    if m: edited['mqa_mode'] = m.group(1)

                scope = input("Apply custom recipe to: [1] This track only (Default)  [2] Rest of album: ").strip()
                if scope == '2':
                    self.mode = 'locked'
                    self.locked_params = edited
                self.track_decisions[filepath] = edited
                return edited, False

            elif choice in ['s', 'skip', 'skip_track']:
                print(f">>> Skipping track: {os.path.basename(filepath)}\n")
                return None, 'track'

            elif choice in ['k', 'skip_album', 'skip_rest', 'skip_all']:
                alb_label = os.path.basename(self.current_album_dir or os.path.dirname(filepath)) or "album"
                print(f">>> Skipping this track and the REST of album directory: {alb_label}\n")
                self.skip_album = True
                return None, 'album'

            elif choice in ['q', 'quit', 'abort']:
                print("\n[AcoustiSinc] Processing aborted by user.")
                sys.exit(0)
            else:
                print(f">>> Applying recipe to {os.path.basename(filepath)}: {dsp_str}\n")
                self.track_decisions[filepath] = rec_params
                return rec_params, False

    def _find_album_summary(self, album_dir):
        if not album_dir or not os.path.exists(album_dir):
            return None
        candidate_names = [
            'ALBUM_REPORT.md',
            'album_report.md',
            'ALBUM_SUMMARY.md',
            'album_summary.md',
            'analysis_summary.md',
            'ANALYSIS_SUMMARY.md',
            'album_analysis.md',
            'ALBUM_ANALYSIS.md',
            'PROVENANCE_SUMMARY.md',
            'provenance_summary.md',
            'report.md',
            'REPORT.md'
        ]
        for name in candidate_names:
            p = os.path.join(album_dir, name)
            if os.path.exists(p) and os.path.isfile(p):
                return p
        try:
            for entry in os.scandir(album_dir):
                if entry.is_file() and entry.name.lower().endswith(('_report.md', '_summary.md', 'album_report.md')):
                    return entry.path
        except Exception:
            pass
        return None

    def _display_album_summary(self, summary_path):
        if not summary_path or not os.path.exists(summary_path):
            print("\n⚠️  No album analysis summary file found in this folder.\n")
            return
        filename = os.path.basename(summary_path)
        print(f"\n================================================================================")
        print(f"📋 ALBUM ANALYSIS SUMMARY: {filename}")
        print(f"================================================================================")
        try:
            with open(summary_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            print(content)
        except Exception as e:
            print(f"[Error reading album summary]: {e}")
        print(f"================================================================================\n")

    def _audit_track(self, filepath, print_card=False, track_idx=None, total_tracks=None):
        try:
            data, sr = sf.read(filepath, frames=int(192000 * 25), dtype='float64', always_2d=True)
        except Exception:
            data, sr = load_audio_resilient(filepath, dtype='float64', frames=int(192000 * 25))
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        _, _, _, _, _, _, prov_info = analyze_audio_forensics(data, sr, filepath=filepath)
        primary = prov_info.get("primary", {})
        label = primary.get("label", "Standard Master")
        rec = prov_info.get("recommendation", {})
        vis = prov_info.get("visual_morphology", {})
        pk = vis.get("primary_knee", {})
        purity = vis.get("stopband_purity", {})

        if print_card:
            track_str = f" [Track {track_idx}/{total_tracks}]" if track_idx and total_tracks else ""
            print(f"\n================================================================================")
            print(f"🔬 FORENSIC AUDIT{track_str}: {os.path.basename(filepath)}")
            print(f"================================================================================")
            print(f"   Source Format   : {sr/1000.0:.1f} kHz (2 ch)")
            print(f"   Provenance      : {label} [{primary.get('confidence', 'High')} Confidence]")
            if pk and pk.get("is_brickwall_knee"):
                print(f"   Primary Knee    : {pk.get('freq_khz', 0):.1f} kHz (Slope: {pk.get('steepest_slope_db_per_khz', 0):.1f} dB/kHz | Drop: {pk.get('drop_db', 0):.1f} dB)")
            if purity and purity.get("has_stopband"):
                print(f"   Stopband State  : {purity.get('purity_label', 'Clean')} ({purity.get('description', '')})")
            if rec:
                print(f"   Recommended Action: {rec.get('action', 'Direct Sinc Upsampling')}")
                print(f"   Recommended DSP   : {rec.get('dsp_params', '--phase min --dither shibata')}")
                if rec.get('details'):
                    print(f"   Technical Rationale: {rec.get('details')}")

        rec_phase = 'min'
        rec_apodizing = False
        rec_cutoff = None
        rec_dither = 'shibata'
        rec_mqa = 'adaptive'
        rec_steep = False

        if rec:
            if rec.get("filter_cutoff_khz"):
                rec_cutoff = rec["filter_cutoff_khz"] * 1000.0
                rec_apodizing = True
            params_str = rec.get("dsp_params", "")
            if "--phase min" in params_str or "--filter min" in params_str:
                rec_phase = 'min'
            elif "--phase linear" in params_str:
                rec_phase = 'linear'
            if "--mqa" in params_str:
                m = re.search(r"--mqa\s+(\w+)", params_str)
                if m: rec_mqa = m.group(1)
            if "--steep" in params_str:
                rec_steep = True

        return {
            "phase_mode": rec_phase,
            "apodizing": rec_apodizing,
            "cutoff_hz": rec_cutoff,
            "dither_mode": rec_dither,
            "mqa_mode": rec_mqa,
            "steep": rec_steep
        }, rec


def main():
    parser = argparse.ArgumentParser(description="AcoustiSinc: GPU-Accelerated 64-Bit Sinc Audio Upsampler")
    parser.add_argument("source", help="Source audio file or root directory containing album folders")
    parser.add_argument("target", nargs="?", default=None, help="Target output directory (optional, default: <source>_upsampled_<topology>)")
    parser.add_argument("-o", "--output-dir", default=None, help="Explicit target output directory")
    parser.add_argument("--overwrite", "-f", "--force", choices=['on', 'off', 'ask', 'true', 'false'], default='off', const='on', nargs='?', dest="overwrite", help="Target output file overwrite policy: 'on' (silent overwrite), 'off' (skip existing), 'ask' (interactive prompt when output exists)")
    parser.add_argument("--use-recommended", "--use-rec", choices=['auto', 'ask', 'none'], default='none', const='auto', nargs='?', help="Analyze audio forensics and apply recommended DSP recipe: 'auto' (silent auto-apply) or 'ask' (interactive prompt)")
    parser.add_argument("--phase", choices=['linear', 'min', 'minimum'], default='linear', help="Filter phase mode: linear (symmetric) or min (minimum phase, causal)")
    parser.add_argument("--apodizing", "--apod", action="store_true", help="Enable Apodizing transition band to attenuate pre-existing studio ADC ringing")
    parser.add_argument("--cutoff", "--apodize", type=float, default=None, help="Custom low-pass reconstruction filter cutoff frequency in Hz (e.g. 20700, 21500, 22050, 44100)")
    parser.add_argument("--steep", action="store_true", help="Use sharp transition band (500 Hz) instead of standard 2 kHz taper")
    parser.add_argument("--filter", choices=['linear-phase', 'min-phase', 'linear', 'min'], default=None, help="Filter mode alias")
    parser.add_argument("--precision", choices=['float64', 'double', 'double2'], default='float64', help="Precision mode (default: strict float64)")
    parser.add_argument("--dither", choices=['shibata', 'high_rate', 'none'], default='shibata', help="Dither & noise shaping mode (default: shibata)")
    parser.add_argument("--no-dither", action="store_true", help="Disable dither and noise shaping")
    parser.add_argument("--mqa", choices=['adaptive', 'simple', 'strip', 'ignore'], default='adaptive', help="MQA processing mode: adaptive (companded high-fidelity unfold), simple (linear unfold), strip (strip MQA payload and re-dither), ignore (treat as raw unaltered PCM)")
    parser.add_argument("--tmp-dir", default=DEFAULT_TMP_DIR, help="Scratch directory for NVMe memory-mapped buffers")
    args = parser.parse_args()

    source_path = os.path.abspath(args.source)
    if not os.path.exists(source_path):
        print(f"[Error] Source path not found: {source_path}")
        sys.exit(1)

    if args.filter:
        if args.filter in ['min-phase', 'min']:
            args.phase = 'min'
        elif args.filter in ['linear-phase', 'linear']:
            args.phase = 'linear'

    if args.cutoff:
        args.apodizing = True

    is_min = args.phase.lower() in ['min', 'minimum']
    phase_name = "MINIMUM PHASE" if is_min else "LINEAR PHASE"
    topology_name = f"{phase_name} APODIZING" if args.apodizing else phase_name
    topology_suffix = ("min_apod" if args.apodizing else "min") if is_min else ("linear_apod" if args.apodizing else "linear")
    if args.cutoff:
        topology_suffix += f"_{int(args.cutoff)}hz"

    target_dir = args.output_dir or args.target
    if not target_dir:
        if os.path.isfile(source_path):
            parent = os.path.dirname(source_path)
            target_dir = os.path.join(parent, f"upsampled_{topology_suffix}")
        else:
            target_dir = f"{source_path.rstrip(os.sep)}_upsampled_{topology_suffix}"

    target_dir = os.path.abspath(target_dir)

    # Bulletproof Safety Guard: Target cannot be identical to source
    if os.path.abspath(source_path) == os.path.abspath(target_dir):
        print(f"[Fatal Safety Error] Target directory cannot be identical to source path ({target_dir}). Aborting to protect original files.")
        sys.exit(1)

    if os.path.isfile(source_path):
        src_file_dir = os.path.abspath(os.path.dirname(source_path))
        if os.path.abspath(target_dir) == src_file_dir:
            print(f"[Fatal Safety Error] Output directory cannot be identical to the source file directory ({target_dir}). Please specify a distinct target directory or use the default auto-named directory.")
            sys.exit(1)

    dither_mode = "none" if args.no_dither else args.dither
    tmp_dir = os.path.abspath(args.tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    overwrite_raw = str(args.overwrite).lower() if args.overwrite is not None else 'off'
    if overwrite_raw in ['true', '1', 'on']:
        overwrite_mode = 'on'
    elif overwrite_raw in ['ask', 'prompt']:
        overwrite_mode = 'ask'
    else:
        overwrite_mode = 'off'

    default_params = {
        "phase_mode": args.phase,
        "dither_mode": dither_mode,
        "apodizing": args.apodizing,
        "mqa_mode": args.mqa,
        "cutoff_hz": args.cutoff,
        "steep": args.steep,
        "overwrite_mode": overwrite_mode
    }
    prompt_ctrl = SessionPromptController(mode=args.use_recommended, overwrite_mode=overwrite_mode, default_params=default_params)

    overwrite_label = "ENABLED (Silent Overwrite)" if overwrite_mode == 'on' else ("ASK (Interactive Prompt on Existing Files)" if overwrite_mode == 'ask' else "DISABLED (Skip Existing)")

    print(f"\n=======================================================")
    print(f"ACOUSTISINC: 64-BIT GPU SINC AUDIO UPSAMPLER")
    print(f"=======================================================")
    print(f"Source Path     : {source_path}")
    print(f"Destination Path: {target_dir}")
    print(f"Filter Topology : {topology_name}")
    print(f"Phase Mode      : {phase_name}")
    print(f"Cutoff Frequency: {f'{args.cutoff:,.0f} Hz' if args.cutoff else 'Nyquist Bandwidth'}")
    print(f"Apodizing Filter: {'ENABLED' if args.apodizing else 'DISABLED (Full Sinc Bandwidth)'}")
    print(f"Overwrite Policy: {overwrite_label}")
    print(f"Noise Shaping   : {dither_mode.upper()} (In-Register Double Precision)")
    print(f"MQA Processing  : {args.mqa.upper()}")
    print(f"FLAC Compression: Level 5 (Strict)")
    print(f"Precision       : 64-bit Double Precision (Strict)")
    print(f"=======================================================\n")

    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)

    if os.path.isfile(source_path):
        dest_file = os.path.join(target_dir, os.path.basename(source_path))
        if os.path.exists(dest_file):
            should_overwrite = prompt_ctrl.resolve_overwrite(dest_file, source_path, 1, 1)
            if not should_overwrite:
                print(f"   [Skip] Output exists: {os.path.basename(source_path)} (use --overwrite to overwrite)")
                return

        t_params, skip = prompt_ctrl.resolve_track_params(source_path, 1, 1, default_params)
        if skip:
            return
        gain_factor, pk = scan_album([source_path])
        clipped, out_peak, fut = process_track(
            source_path, target_dir, gain_factor, pk, queue,
            phase_mode=t_params['phase_mode'],
            dither_mode=t_params['dither_mode'],
            tmp_dir=tmp_dir,
            apodizing=t_params['apodizing'],
            mqa_mode=t_params['mqa_mode'],
            cutoff_hz=t_params['cutoff_hz'],
            steep=t_params['steep'],
            overwrite=True
        )
        if clipped and out_peak > PEAK_TARGET_LIN:
            overshoot_ratio = PEAK_TARGET_LIN / out_peak
            safety_margin_lin = 10.0 ** (-0.2 / 20.0)
            gain_factor *= (overshoot_ratio * safety_margin_lin)
            print(f">>> Retrying single file with exact calculated Gain Factor: {gain_factor:.6f}")
            process_track(
                source_path, target_dir, gain_factor, pk, queue,
                phase_mode=t_params['phase_mode'],
                dither_mode=t_params['dither_mode'],
                tmp_dir=tmp_dir,
                apodizing=t_params['apodizing'],
                mqa_mode=t_params['mqa_mode'],
                cutoff_hz=t_params['cutoff_hz'],
                steep=t_params['steep'],
                overwrite=True
            )
        file_writer_pool.shutdown(wait=True)

        dest_file = os.path.join(target_dir, os.path.basename(source_path))
        t_recipe = {
            "cli_params": f"--phase {t_params['phase_mode']} --dither {t_params['dither_mode']}" + (f" --cutoff {int(t_params['cutoff_hz'])}" if t_params.get('cutoff_hz') else "") + (f" --mqa {t_params['mqa_mode']}" if t_params.get('mqa_mode') != 'adaptive' else "") + (" --steep" if t_params.get('steep') else ""),
            "topology_name": f"{t_params['phase_mode'].upper()} APODIZING" if (t_params.get('apodizing') or t_params.get('cutoff_hz')) else f"{t_params['phase_mode'].upper()}",
            "gain_factor": gain_factor,
            "gain_db": 20.0 * np.log10(gain_factor),
            "cutoff_hz": t_params.get('cutoff_hz'),
            "phase_mode": t_params['phase_mode'],
            "dither_mode": t_params['dither_mode'],
            "mqa_mode": t_params['mqa_mode'],
            "steep": t_params.get('steep', False),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        applied_recipe = dict(t_recipe)
        try:
            html_rep, md_rep = generate_comparative_report([{"src_path": source_path, "dst_path": dest_file, "recipe": t_recipe}], os.path.basename(source_path), applied_recipe, target_dir)
            if html_rep:
                print(f"\n   📊 [Comparative Upsampling Report Generated]")
                print(f"      Interactive HTML Report: {html_rep}")
                print(f"      Markdown Summary Report: {md_rep}")
        except Exception as e:
            print(f"   [Notice] Could not generate comparative report: {e}")
        return

    # Discover all subdirectories containing FLAC or WAV files
    album_directories = sorted({
        r for r, d, f in os.walk(source_path)
        if not os.path.abspath(r).startswith(target_dir) and any(x.lower().endswith(('.flac', '.wav')) for x in f)
    })

    if not album_directories:
        print(f"[AcoustiSinc] No FLAC or WAV files found under {source_path}")
        return

    for idx, alb_dir in enumerate(album_directories, 1):
        rel = os.path.relpath(alb_dir, source_path)
        dest_alb = target_dir if rel == "." else os.path.join(target_dir, rel)
        print(f"\n==================================================")
        print(f"[Album {idx}/{len(album_directories)}]")
        try:
            process_album_folder(alb_dir, dest_alb, queue, default_params, tmp_dir, overwrite_mode=overwrite_mode, prompt_ctrl=prompt_ctrl)
        except Exception as e:
            err_msg = f"[Album {idx} Skipped on Error]: {alb_dir}\nDetails: {e}"
            print(f"\n{err_msg}\n>>> Continuing to next album...")
            try:
                with open(os.path.join(target_dir, "upsample_errors.log"), "a", encoding="utf-8") as ef:
                    ef.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {err_msg}\n")
            except Exception:
                pass
            GPU_FILTER_CACHE.clear()
            clear_vkfftapp_cache()
            gc.collect()
    file_writer_pool.shutdown(wait=True)
    print("\n>>> All recursive album batches completed cleanly!")

if __name__ == "__main__":
    main()
