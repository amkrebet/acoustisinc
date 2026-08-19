"""
Provenance Rule Engine & Forensic Configurator
Strict 64-bit Double Precision (float64) DSP Pipeline Integration
"""

import os
import json
import subprocess
import soundfile as sf
import numpy as np


DEFAULT_RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "provenance_rules.json")


def probe_audio_info_resilient(filepath):
    """
    Probes audio file sample rate, channels, frames, and subtype.
    Uses soundfile as primary, falling back to ffprobe on header/stream corruption.
    Returns: dict with 'samplerate', 'channels', 'frames', 'subtype', 'duration'
    """
    try:
        info = sf.info(filepath)
        return {
            'samplerate': info.samplerate,
            'channels': info.channels,
            'frames': info.frames,
            'subtype': info.subtype,
            'duration': info.duration
        }
    except Exception:
        pass

    # Fallback to ffprobe
    try:
        cmd_probe = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels,duration,bits_per_raw_sample",
            "-of", "default=noprint_wrappers=1:nokey=1",
            filepath
        ]
        out = subprocess.check_output(cmd_probe, text=True, stderr=subprocess.DEVNULL, timeout=3.0).strip().split()
        if len(out) >= 2:
            sr = int(out[0])
            channels = int(out[1])
            duration = float(out[2]) if len(out) > 2 and out[2] != 'N/A' else 0.0
            frames = int(duration * sr)
            bits = int(out[3]) if len(out) > 3 and out[3] != 'N/A' else 16
            subtype = f"PCM_{bits}"
            return {
                'samplerate': sr,
                'channels': channels,
                'frames': frames,
                'subtype': subtype,
                'duration': duration
            }
    except Exception:
        pass

    return {
        'samplerate': 44100,
        'channels': 2,
        'frames': 0,
        'subtype': 'PCM_16',
        'duration': 0.0
    }


def load_audio_resilient(filepath, dtype='float64', start=0, stop=None, frames=None):
    """
    Resilient Multi-Tier Audio Loader:
    1. Primary: Fast native libsndfile (soundfile).
    2. Fallback: Subprocess stream pipe via FFmpeg with error concealment for damaged headers,
       improper ID3 tags prepended to FLAC, corrupted seektables, or truncated streams.
    Returns: (data: np.ndarray, samplerate: int)
    """
    # Tier 1: Try Fast Native libsndfile
    try:
        if stop is not None or frames is not None:
            data, sr = sf.read(filepath, dtype=dtype, start=start, stop=stop, frames=frames, always_2d=True)
            if data is not None and data.size > 0:
                return data, sr
        else:
            data, sr = sf.read(filepath, dtype=dtype, always_2d=True)
            if data is not None and data.size > 0:
                return data, sr
    except Exception:
        pass

    # Tier 2: FFmpeg Error-Concealed Pipe Fallback
    try:
        info = probe_audio_info_resilient(filepath)
        sr = info['samplerate']
        channels = info['channels']

        if dtype == 'int32':
            fmt = 's32le'
            codec = 'pcm_s32le'
            np_dt = np.int32
        else:
            fmt = 'f64le'
            codec = 'pcm_f64le'
            np_dt = np.float64

        cmd_decode = ["ffmpeg", "-v", "error", "-i", filepath]
        if start > 0:
            cmd_decode.extend(["-ss", str(start / sr)])
        if frames is not None:
            cmd_decode.extend(["-frames:a", str(frames)])
        elif stop is not None and stop > start:
            cmd_decode.extend(["-t", str((stop - start) / sr)])

        cmd_decode.extend(["-f", fmt, "-acodec", codec, "-"])
        raw_bytes = subprocess.check_output(cmd_decode, stderr=subprocess.DEVNULL, timeout=10.0)
        if len(raw_bytes) > 0:
            data = np.frombuffer(raw_bytes, dtype=np_dt).reshape(-1, channels)
            return data, sr
    except Exception:
        pass

    return np.zeros((0, 2), dtype=np.float64 if dtype == 'float64' else np.int32), 44100


def original_sample_rate_decoder(c: int) -> int:
    base = 48000 if (c & 1) == 1 else 44100
    multiplier = 1 << (((c >> 3) & 1) | (((c >> 2) & 1) << 1) | (((c >> 1) & 1) << 2))
    if multiplier > 16:
        multiplier *= 2
    return base * multiplier


def detect_mqa_signature(filepath=None, pcm_int32=None, sr=44100):
    """
    Forensically detects MQA encoding:
    1. Metadata Tag scan (MQAENCODER, ORIGINALSAMPLERATE, ENCODER)
    2. 36-bit magic word (0xbe0498c88) XOR bitstream detection
    """
    details = {
        "is_mqa": False,
        "is_studio": False,
        "original_sr": None,
        "bit_position": None,
        "encoder": None,
        "method": None
    }

    # 1. Metadata check
    if filepath and os.path.exists(filepath):
        try:
            from mutagen.flac import FLAC
            audio = FLAC(filepath)
            if audio and audio.tags:
                for k, v in audio.tags.items():
                    ku = k.upper()
                    v_str = str(v[0] if isinstance(v, list) else v)
                    if "MQAENCODER" in ku:
                        details["is_mqa"] = True
                        details["encoder"] = v_str
                        details["method"] = "METADATA_TAG"
                    elif "ORIGINALSAMPLERATE" in ku:
                        try:
                            details["original_sr"] = int(v_str)
                            details["is_mqa"] = True
                        except Exception:
                            pass
                if details["is_mqa"]:
                    if details["original_sr"] is None:
                        details["original_sr"] = sr * 2
                    return details
        except Exception:
            pass

    # 2. Audio Bitstream check (36-bit magic word 0xbe0498c88)
    if filepath and os.path.exists(filepath) and pcm_int32 is None:
        try:
            info = probe_audio_info_resilient(filepath)
            frames = min(int(info['samplerate'] * 6.0), info['frames'] if info['frames'] > 0 else int(info['samplerate'] * 6.0))
            pcm_int32, _ = load_audio_resilient(filepath, frames=frames, dtype="int32")
        except Exception:
            pass

    if pcm_int32 is not None and pcm_int32.ndim == 2 and pcm_int32.shape[1] >= 2:
        try:
            s0 = pcm_int32[:, 0].astype(np.uint32)
            s1 = pcm_int32[:, 1].astype(np.uint32)
            xor_sig = s0 ^ s1
            MASK_36 = 0xFFFFFFFFF
            MAGIC = 0xbe0498c88

            for bit in (8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 1, 2, 3, 4, 5, 6, 7):
                stream = (xor_sig >> bit) & 1
                buffer = 0
                for i, b in enumerate(stream):
                    buffer = ((buffer << 1) | int(b)) & MASK_36
                    if buffer == MAGIC:
                        orsf = 0
                        for m in range(3, 8):
                            if (i + m) < len(stream):
                                j = int(stream[i + m])
                                orsf |= (j << (7 - m))
                        orig_sr = original_sample_rate_decoder(orsf)
                        prov = 0
                        for m in range(29, 34):
                            if (i + m) < len(stream):
                                j = int(stream[i + m])
                                prov |= (j << (33 - m))
                        is_studio = (prov > 8)
                        return {
                            "is_mqa": True,
                            "is_studio": is_studio,
                            "original_sr": orig_sr,
                            "bit_position": bit,
                            "encoder": "MQA Bitstream Payload",
                            "method": "BITSTREAM_SYNC"
                        }
        except Exception:
            pass

def analyze_effective_bit_depth(filepath=None, pcm_int32=None):
    """
    Forensically measures actual effective bit depth and detects LSB zero-padding or bit truncation.
    """
    res = {
        "container_bits": 16,
        "container_subtype": "PCM_16",
        "effective_bits": 16,
        "trailing_zero_bits": 0,
        "is_zero_padded": False,
        "zero_ratio_lsb8": 0.0,
        "summary": "16-bit Standard Audio"
    }

    if filepath and os.path.exists(filepath):
        try:
            info = probe_audio_info_resilient(filepath)
            res["container_subtype"] = info['subtype']
            res["container_bits"] = 24 if "24" in info['subtype'] else (32 if "32" in info['subtype'] else 16)
            if pcm_int32 is None:
                frames = min(int(info['samplerate'] * 30.0), info['frames'] if info['frames'] > 0 else int(info['samplerate'] * 30.0))
                pcm_int32, _ = load_audio_resilient(filepath, frames=frames, dtype="int32")
        except Exception:
            pass

    if pcm_int32 is not None:
        try:
            s24 = pcm_int32 >> 8
            non_zero = s24[s24 != 0]
            if len(non_zero) > 0:
                trailing_zeros = 0
                for bit in range(24):
                    mask = 1 << bit
                    if (np.abs(non_zero) & mask).any():
                        break
                    trailing_zeros += 1
                
                effective_bits = 24 - trailing_zeros
                lsb8_zeros = float(np.mean((non_zero & 0xFF) == 0))
                
                res["effective_bits"] = effective_bits
                res["trailing_zero_bits"] = trailing_zeros
                res["zero_ratio_lsb8"] = lsb8_zeros
                
                if res["container_bits"] >= 24 and trailing_zeros >= 4:
                    res["is_zero_padded"] = True
                    if trailing_zeros >= 8:
                        res["summary"] = f"Fake {res['container_bits']}-bit (16-bit Zero-Padded / 8 LSBs Inactive)"
                    else:
                        res["summary"] = f"20-bit Master in {res['container_bits']}-bit Container ({trailing_zeros} LSBs Inactive)"
                elif res["container_bits"] >= 24:
                    res["summary"] = f"True {res['container_bits']}-bit (Full Dynamic Resolution)"
                else:
                    res["summary"] = "Native 16-bit Red Book Master"
        except Exception:
            pass

    return res


class ProvenanceRuleEngine:
    """
    Interprets declarative forensic provenance rules to evaluate audio lineage,
    base rates, filter leakage, and upsampling artifacts.
    """

    def __init__(self, rules_path=None):
        self.rules_path = rules_path or DEFAULT_RULES_PATH
        self._rules = None
        self._last_mtime = 0
        self.load_rules()

    def load_rules(self, path=None):
        if path:
            self.rules_path = path

        if os.path.exists(self.rules_path):
            try:
                mtime = os.path.getmtime(self.rules_path)
                with open(self.rules_path, "r", encoding="utf-8") as f:
                    self._rules = json.load(f)
                self._last_mtime = mtime
                return True
            except Exception as e:
                print(f"[ProvenanceRuleEngine] Error loading {self.rules_path}: {e}")
                return False
        return False

    def check_and_reload(self):
        """Auto-reloads rules if the configuration file has been edited on disk."""
        if os.path.exists(self.rules_path):
            try:
                mtime = os.path.getmtime(self.rules_path)
                if mtime > self._last_mtime:
                    self.load_rules()
            except Exception:
                pass

    @property
    def rules(self):
        self.check_and_reload()
        return self._rules or {}

    def evaluate(self, sr, nyquist, freqs, mean_spec_db, rms_dbfs, peak_dbfs, stft_mag, zero_ratio=0.0, mqa_info=None, bitdepth_info=None):
        """
        Executes rule interpreter against extracted spectral features.
        Returns a structured provenance result dictionary.
        """
        rules = self.rules
        
        # Determine baseline noise floor from top 15% of Nyquist band
        idx_floor_start = np.argmin(np.abs(freqs - (0.82 * nyquist)))
        idx_floor_end = np.argmin(np.abs(freqs - (0.96 * nyquist)))
        noise_floor_rms = float(np.median(rms_dbfs[idx_floor_start:idx_floor_end]))
        noise_floor_spec = float(np.median(mean_spec_db[idx_floor_start:idx_floor_end]))
        
        # 1. Evaluate Zero-Stuffing Rule
        z_rule = rules.get("zero_stuffing", {})
        z_thresh = z_rule.get("zero_ratio_threshold", 0.40)
        is_zero_stuffed = False
        if zero_ratio > z_thresh:
            is_zero_stuffed = True
        elif nyquist >= 44100:
            idx_30k = np.argmin(np.abs(freqs - 30000))
            idx_40k = np.argmin(np.abs(freqs - 40000))
            mirror_rms_median = float(np.median(rms_dbfs[idx_30k:idx_40k]))
            idx_20k = np.argmin(np.abs(freqs - 20000))
            audible_rms_mean = float(np.mean(rms_dbfs[:idx_20k]))
            margin = z_rule.get("ultrasonic_mirror_margin_db", 20.0)
            floor_req = z_rule.get("ultrasonic_rms_floor_dbfs", -80.0)
            if mirror_rms_median > (audible_rms_mean - margin) and mirror_rms_median > floor_req and (mirror_rms_median - noise_floor_rms) > margin:
                is_zero_stuffed = True

        # 2. Evaluate Multi-Boundary Folding & Mirror Leakage Rules
        folding_candidates = rules.get("folding_boundaries", [])
        leaky_hits = []
        
        for cand in folding_candidates:
            f_n = cand.get("nyquist_hz", 22050)
            min_sr = cand.get("min_container_sr", 48000)
            if sr < min_sr or (f_n + 1200) > nyquist or (f_n - 1500) < 5000:
                continue

            i_base = np.argmin(np.abs(freqs - (f_n - 1500)))
            i_notch = np.argmin(np.abs(freqs - f_n))
            i_rebound = np.argmin(np.abs(freqs - (f_n + 700)))

            notch_drop = float(mean_spec_db[i_base] - mean_spec_db[i_notch])
            rebound_rise = float(mean_spec_db[i_rebound] - mean_spec_db[i_notch])

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
                        corrs.append(float(c))

            avg_corr = float(np.mean(corrs)) if corrs else 0.0
            rebound_thresh = cand.get("rebound_threshold_db", 2.0)
            notch_thresh = cand.get("notch_depth_threshold_db", 8.0)
            min_corr = cand.get("min_mirror_correlation", 0.55)
            high_corr = cand.get("high_confidence_corr", 0.75)

            if (rebound_rise > rebound_thresh or (notch_drop > 15.0 and avg_corr > min_corr)) and (avg_corr > min_corr or notch_drop > notch_thresh):
                base_khz = cand.get("base_rate_hz", f_n * 2) / 1000.0
                conf = "High" if avg_corr >= high_corr else "Moderate"
                leaky_hits.append({
                    "name": cand.get("name", f"{base_khz:.1f} kHz Master"),
                    "base_rate_khz": base_khz,
                    "f_n_khz": f_n / 1000.0,
                    "rebound_db": rebound_rise,
                    "notch_drop_db": notch_drop,
                    "mirror_corr": avg_corr,
                    "confidence": conf,
                    "score": round(min(0.98, max(0.70, avg_corr)), 2),
                    "badge_class": cand.get("badge_class", "badge-provenance-leaky")
                })

        # 3. Extract Visual Morphology Metrics (Curvature Knees, Temporal Variance, Rhythmic Coherence, Stopband Purity)
        visual_metrics = extract_visual_morphology(freqs, mean_spec_db, rms_dbfs, peak_dbfs, stft_mag, sr, nyquist)
        rhythmic_corr = visual_metrics.get("rhythmic_coherence", 0.0)
        is_stat_ultra = visual_metrics.get("is_stationary_ultrasonic", False)
        prime_knee = visual_metrics.get("primary_knee")
        purity_info = visual_metrics.get("stopband_purity", {})
        is_messy_stopband = purity_info.get("is_messy", False)

        # 4. Evaluate Cutoff & Upsampling Rules
        cutoff_rules = rules.get("cutoff_rules", [])
        matched_cutoff = None
        
        for c_rule in cutoff_rules:
            min_c_sr = c_rule.get("min_container_sr", 0)
            min_c_nyquist = c_rule.get("min_container_nyquist_hz", 0)
            if sr < min_c_sr or nyquist < min_c_nyquist:
                continue

            f_low = c_rule.get("f_low_hz")
            f_high = c_rule.get("f_high_hz")
            if f_high > nyquist:
                continue

            i_low = np.argmin(np.abs(freqs - f_low))
            i_high = np.argmin(np.abs(freqs - f_high))

            spec_drop = float(mean_spec_db[i_low] - mean_spec_db[i_high])
            rms_drop = float(rms_dbfs[i_low] - rms_dbfs[i_high])

            min_spec_drop = c_rule.get("min_spec_drop_db", 10.0)
            min_rms_drop = c_rule.get("min_rms_drop_db", 0.0)
            max_stopband = c_rule.get("max_stopband_level_dbfs", None)
            floor_margin = c_rule.get("noise_floor_margin_db", None)
            abs_ceiling = c_rule.get("absolute_noise_ceiling_dbfs", None)

            # Local stopband check (5-8 kHz window above cutoff) to remain robust against distant ultrasonic dither humps
            stop_end_f = min(nyquist, f_high + 7500.0)
            i_stop_end = np.argmin(np.abs(freqs - stop_end_f))
            local_stop_rms = float(np.mean(rms_dbfs[i_high:i_stop_end])) if i_stop_end > i_high else float(rms_dbfs[i_high])

            max_stopband_ceiling = max_stopband if max_stopband is not None else -102.0

            is_match = False
            if (spec_drop >= min_spec_drop or (min_rms_drop > 0 and rms_drop >= min_rms_drop)) and local_stop_rms < max_stopband_ceiling:
                is_match = True
                if floor_margin is not None and abs_ceiling is not None:
                    if not (local_stop_rms <= noise_floor_rms + floor_margin or mean_spec_db[i_high] < abs_ceiling):
                        is_match = False

                # Prevent false positives on continuous organic/ambient rolloffs in 48kHz containers
                if c_rule.get("id") == "cutoff_44k" and sr == 48000:
                    if rhythmic_corr >= 0.65 and not is_stat_ultra and local_stop_rms > -150.0 and rms_drop < 28.0:
                        is_match = False

            if is_match:
                matched_cutoff = c_rule
                break

        # 5. Measure Effective Signal Bandwidth for Native Assessment
        above_floor = (rms_dbfs > (noise_floor_rms + 5.0)) | (mean_spec_db > (noise_floor_spec + 5.0))
        valid_idx = np.where(above_floor)[0]
        effective_bw_hz = float(freqs[valid_idx[-1]]) if len(valid_idx) > 0 else 20000.0

        # 6. Dual Verdict Assembler
        native_cfg = rules.get("native_rules", {})
        alt_cfg = rules.get("alternative_rules", {})
        enable_alts = alt_cfg.get("enable_alternative_hypotheses", True)

        primary_prov = None
        alt_prov = None

        if mqa_info and mqa_info.get("is_mqa"):
            mqa_cfg = rules.get("mqa_rules", {})
            orig_sr = mqa_info.get("original_sr") or (sr * 2)
            is_studio = mqa_info.get("is_studio", False)
            tag_name = "MQA Studio Master" if is_studio else "MQA Authenticated Master"
            meth = mqa_info.get("method", "BITSTREAM")
            primary_prov = {
                "label": f"{tag_name} (Orig: {orig_sr/1000:.1f} kHz)",
                "confidence": mqa_cfg.get("confidence", "High"),
                "score": mqa_cfg.get("confidence_score", 0.99),
                "badge_class": mqa_cfg.get("badge_class", "badge-provenance-mqa"),
                "details": f"Folded MQA ultrasonic subband detected via {meth}. Original Master: {orig_sr/1000:.1f} kHz."
            }
        elif is_zero_stuffed:
            z_spec = rules.get("zero_stuffing", {})
            primary_prov = {
                "label": "Raw Zero-Stuffed / NOS Source",
                "confidence": z_spec.get("confidence", "High"),
                "score": z_spec.get("confidence_score", 0.98),
                "badge_class": z_spec.get("badge_class", "badge-provenance-fake"),
                "details": z_spec.get("details", "Literal zero-stuffing or strong unfiltered imaging detected above 22.05 kHz.")
            }
        elif matched_cutoff is not None:
            c_name = matched_cutoff.get("name", "Upsampled Source")
            c_conf = matched_cutoff.get("confidence", "High")
            c_score = matched_cutoff.get("confidence_score", 0.95)
            c_badge = matched_cutoff.get("badge_class", "badge-provenance-upsampled")
            f_boundary_khz = (matched_cutoff.get("target_base_hz", 44100) / 2.0) / 1000.0
            
            primary_prov = {
                "label": c_name,
                "confidence": c_conf,
                "score": c_score,
                "badge_class": c_badge,
                "details": f"Sharp anti-aliasing cutoff near {f_boundary_khz:.2f} kHz into container noise floor."
            }

            if enable_alts:
                # Check for secondary partial cutoff at 22.05 kHz (e.g. 44.1k stems mixed into 88.2k master)
                i19 = np.argmin(np.abs(freqs - 19500))
                i23 = np.argmin(np.abs(freqs - 23000))
                drop_22k = float(rms_dbfs[i19] - rms_dbfs[i23])
                if matched_cutoff.get("id") in ["cutoff_88k", "cutoff_96k"] and drop_22k >= 7.0:
                    alt_prov = {
                        "label": "Upsampled from 44.1 kHz Master (or Mixed 44.1k Stems)",
                        "confidence": "Moderate",
                        "score": 0.65,
                        "badge_class": "badge-provenance-upsampled",
                        "details": f"Secondary spectral attenuation of {drop_22k:.1f} dB near 22.05 kHz suggests 44.1 kHz source elements mixed into the master."
                    }
                elif leaky_hits and leaky_hits[0]["mirror_corr"] >= alt_cfg.get("mirror_leakage_min_corr", 0.55):
                    top_leaky = leaky_hits[0]
                    alt_prov = {
                        "label": f"{top_leaky['base_rate_khz']:.1f} kHz Master (Leaky Filter Residual)",
                        "confidence": top_leaky["confidence"],
                        "score": top_leaky["score"],
                        "badge_class": top_leaky["badge_class"],
                        "details": f"Residual spectral leakage across {top_leaky['f_n_khz']:.2f} kHz with r={top_leaky['mirror_corr']:+.2f} mirror correlation."
                    }
        elif leaky_hits:
            h = leaky_hits[0]
            primary_prov = {
                "label": f"{h['base_rate_khz']:.1f} kHz Master (Leaky SRC / DAC)",
                "confidence": h["confidence"],
                "score": h["score"],
                "badge_class": h["badge_class"],
                "details": f"{h['base_rate_khz']:.1f} kHz filter notch at {h['f_n_khz']:.2f} kHz with mirrored spectral imaging (r={h['mirror_corr']:+.2f})."
            }
            if enable_alts:
                alt_prov = {
                    "label": f"Upsampled from {h['base_rate_khz']:.1f} kHz Master",
                    "confidence": "Moderate",
                    "score": 0.65,
                    "badge_class": "badge-provenance-upsampled",
                    "details": f"Sharp attenuation near {h['f_n_khz']:.2f} kHz with filter skirt artifacts."
                }
        elif nyquist <= native_cfg.get("redbook_max_nyquist_hz", 22050):
            primary_prov = {
                "label": f"Native {sr/1000:.1f} kHz CD Master",
                "confidence": "High",
                "score": 0.95,
                "badge_class": "badge-provenance-native",
                "details": f"Standard Red Book container with full {sr/1000:.1f} kHz audible passband."
            }
        elif sr == 48000:
            # Check for smooth continuous wideband extension past 22.05 kHz
            i19 = np.argmin(np.abs(freqs - 19500))
            i22 = np.argmin(np.abs(freqs - 22050))
            drop_to_22 = float(rms_dbfs[i19] - rms_dbfs[i22])
            
            if (drop_to_22 <= 9.0 and rhythmic_corr >= 0.60) or effective_bw_hz >= native_cfg.get("sr_48k_high_confidence_bw_hz", 23500):
                primary_prov = {
                    "label": "Native 48.0 kHz Master",
                    "confidence": "High",
                    "score": 0.92,
                    "badge_class": "badge-provenance-native",
                    "details": f"Continuous wideband harmonic extension (r = {rhythmic_corr:+.2f}) with natural acoustic rolloff past 22.05 kHz."
                }
            elif effective_bw_hz >= native_cfg.get("sr_48k_mod_confidence_bw_hz", 21500) or rhythmic_corr >= 0.50:
                primary_prov = {
                    "label": "Native 48.0 kHz Material",
                    "confidence": "Moderate",
                    "score": 0.78,
                    "badge_class": "badge-provenance-native",
                    "details": f"Natural acoustic roll-off extending to ~{effective_bw_hz/1000:.1f} kHz with dynamic correlation."
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
            if effective_bw_hz >= native_cfg.get("hires_high_confidence_bw_hz", 26000):
                primary_prov = {
                    "label": f"Native {sr/1000:.1f} kHz Master",
                    "confidence": "High",
                    "score": 0.92,
                    "badge_class": "badge-provenance-native",
                    "details": f"Continuous acoustic harmonic bandwidth reaching ~{effective_bw_hz/1000:.1f} kHz."
                }
            elif effective_bw_hz >= native_cfg.get("hires_mod_confidence_bw_hz", 21000):
                primary_prov = {
                    "label": f"Native {sr/1000:.1f} kHz Material",
                    "confidence": "Moderate",
                    "score": 0.75,
                    "badge_class": "badge-provenance-native",
                    "details": f"Smooth ultrasonic roll-off up to ~{effective_bw_hz/1000:.1f} kHz."
                }
        # Check for zero-padded bit depth
        if bitdepth_info and bitdepth_info.get("is_zero_padded"):
            tz = bitdepth_info.get("trailing_zero_bits", 8)
            eff = bitdepth_info.get("effective_bits", 16)
            cb = bitdepth_info.get("container_bits", 24)
            pad_summary = f"{eff}-bit Padded"
            
            if primary_prov is not None:
                primary_prov["details"] += f" Note: Container is zero-padded ({tz} inactive LSBs / true {eff}-bit)."
                if "Native" in primary_prov["label"]:
                    primary_prov["label"] += f" [{pad_summary}]"
            else:
                primary_prov = {
                    "label": f"Fake {cb}-bit Container ({eff}-bit Zero-Padded)",
                    "confidence": "High",
                    "score": 0.99,
                    "badge_class": "badge-provenance-fake",
                    "details": f"Container claims {cb}-bit resolution, but the lower {tz} bits are 100% inactive. Effective resolution is {eff}-bit."
                }

        suspected_base_sr_hz = None
        suspected_nyquist_hz = None
        if matched_cutoff is not None:
            suspected_base_sr_hz = matched_cutoff.get("target_base_hz")
            suspected_nyquist_hz = (suspected_base_sr_hz / 2.0) if suspected_base_sr_hz else None
        elif leaky_hits:
            suspected_base_sr_hz = int(leaky_hits[0]["base_rate_khz"] * 1000)
            suspected_nyquist_hz = leaky_hits[0]["f_n_khz"] * 1000.0

        # Check for elevated ultrasonic noise/rebound or messy stopband hash
        has_ultrasonic_noise = False
        if matched_cutoff is not None:
            f_boundary = matched_cutoff.get("f_high_hz", 24000)
            i_stop = np.argmin(np.abs(freqs - (f_boundary + 2000)))
            i_stop_end = np.argmin(np.abs(freqs - min(nyquist, f_boundary + 8000)))
            stop_level = float(np.mean(rms_dbfs[i_stop:i_stop_end])) if i_stop_end > i_stop else float(rms_dbfs[i_stop])
            
            if (i_stop_end + 5) < len(rms_dbfs):
                max_ultra_level = float(np.max(rms_dbfs[i_stop_end:]))
                if max_ultra_level > (stop_level + 4.5) and max_ultra_level > -128.0:
                    has_ultrasonic_noise = True
            
            if (is_stat_ultra or is_messy_stopband) and not has_ultrasonic_noise:
                has_ultrasonic_noise = True

        recommendation = generate_dsp_recommendation(
            primary_prov, bitdepth_info, mqa_info, sr, nyquist, effective_bw_hz, has_ultrasonic_noise=has_ultrasonic_noise, purity_info=purity_info, visual_morphology=visual_metrics
        )

        return {
            "primary": primary_prov,
            "alternative": alt_prov,
            "label": primary_prov["label"],
            "confidence": primary_prov["confidence"],
            "score": primary_prov["score"],
            "badge_class": primary_prov["badge_class"],
            "details": primary_prov["details"],
            "effective_bw_hz": effective_bw_hz,
            "suspected_base_sr_hz": suspected_base_sr_hz,
            "suspected_nyquist_hz": suspected_nyquist_hz,
            "recommendation": recommendation,
            "visual_morphology": visual_metrics,
            "noise_floor_rms": noise_floor_rms,
            "noise_floor_spec": noise_floor_spec,
            "bitdepth": bitdepth_info
        }


def extract_visual_morphology(freqs, mean_spec_db, rms_dbfs, peak_dbfs, stft_mag, sr, nyquist):
    """
    Emulates human visual perception of spectrograms and spectrum curves.
    Uses strict 64-bit double precision across all gradient, curvature, variance,
    and cross-band envelope correlation calculations.
    """
    from scipy.signal import find_peaks
    f_khz = freqs / 1000.0
    
    # 1. First & Second Derivative (Slope and Curvature "Knee" Detection)
    bin_hz = sr / (len(freqs) * 2 - 2) if len(freqs) > 1 else 10.0
    sigma_bins = max(3, int(450.0 / bin_hz))
    x = np.arange(-3*sigma_bins, 3*sigma_bins + 1)
    gauss_kernel = np.exp(-0.5 * (x / sigma_bins)**2)
    gauss_kernel /= np.sum(gauss_kernel)

    smooth_rms = np.convolve(rms_dbfs, gauss_kernel, mode="same")
    d1 = np.gradient(smooth_rms, f_khz) # slope in dB/kHz
    d2 = np.gradient(d1, f_khz)        # curvature in dB/kHz^2

    min_idx = np.argmin(np.abs(f_khz - 15.0))
    max_idx = np.argmin(np.abs(f_khz - (nyquist/1000.0 - 0.2)))

    # Global unconstrained peak detection across the full spectrum
    d2_sub = d2[min_idx:max_idx]
    peaks, _ = find_peaks(d2_sub, prominence=0.30, distance=int(1800.0/bin_hz))

    detected_knees = []
    for p in peaks:
        idx = min_idx + p
        f_k = float(f_khz[idx])
        
        i_pre = max(0, idx - int(2000.0/bin_hz))
        pre_slope = float(np.min(d1[i_pre:idx]))
        
        i_post = min(len(d1)-1, idx + int(2000.0/bin_hz))
        post_slope = float(np.mean(d1[idx:i_post]))
        
        drop = float(smooth_rms[i_pre] - smooth_rms[idx])
        level = float(smooth_rms[idx])
        
        if pre_slope <= -3.0 and drop >= 4.0:
            nyq_match = None
            for cand_nyq, cand_name in [(20.5, "Legacy ADC"), (22.05, "44.1k"), (24.0, "48k"), (44.1, "88.2k"), (48.0, "96k")]:
                if abs(f_k - cand_nyq) <= 2.5:
                    nyq_match = cand_name
                    break
            
            detected_knees.append({
                "freq_khz": round(f_k, 2),
                "detected_knee_khz": round(f_k, 2),
                "level_dbfs": round(level, 1),
                "pre_slope_db_per_khz": round(pre_slope, 1),
                "steepest_slope_db_per_khz": round(pre_slope, 1),
                "post_slope_db_per_khz": round(post_slope, 1),
                "drop_db": round(drop, 1),
                "max_curvature": round(float(d2[idx]), 2),
                "matched_nyquist": nyq_match,
                "is_brickwall_knee": bool(pre_slope <= -4.5 or drop >= 7.0)
            })

    # Sort knees in frequency order
    detected_knees.sort(key=lambda k: k["freq_khz"])
    
    # Primary knee is the most significant drop/curvature (prioritizing brickwall drops)
    significant_knees = sorted(detected_knees, key=lambda k: (1.5 if k["is_brickwall_knee"] else 1.0) * k["drop_db"] * abs(k["pre_slope_db_per_khz"]), reverse=True)
    primary_knee = significant_knees[0] if significant_knees else None

    # 2. Temporal Variance & Stationary Banding Analysis (Music Dynamics vs Constant Dither/Noise)
    stft_db_slices = 20.0 * np.log10(np.maximum(stft_mag / (np.max(stft_mag) + 1e-12), 1e-12))
    temp_variance = np.var(stft_db_slices, axis=1)
    
    idx_aud = np.where((f_khz >= 1.0) & (f_khz <= 15.0))[0]
    audible_variance = float(np.mean(temp_variance[idx_aud])) if len(idx_aud) > 0 else 25.0
    
    if nyquist >= 40000:
        idx_ult = np.where((f_khz >= 25.0) & (f_khz <= (nyquist/1000.0 - 2.0)))[0]
    elif nyquist >= 23000:
        idx_ult = np.where((f_khz >= 20.5) & (f_khz <= (nyquist/1000.0 - 0.2)))[0]
    else:
        idx_ult = np.where((f_khz >= 18.0) & (f_khz <= (nyquist/1000.0 - 0.2)))[0]
    ultrasonic_variance = float(np.mean(temp_variance[idx_ult])) if len(idx_ult) > 0 else 0.0
    
    # 3. Cross-Band Rhythmic Coherence / Correlation
    rhythmic_coherence = 0.0
    if len(idx_aud) > 0 and len(idx_ult) > 0:
        env_audible = np.mean(stft_db_slices[idx_aud, :], axis=0)
        env_ultra = np.mean(stft_db_slices[idx_ult, :], axis=0)
        
        std_a = np.std(env_audible)
        std_u = np.std(env_ultra)
        if std_a > 1e-6 and std_u > 1e-6:
            r = np.corrcoef(env_audible, env_ultra)[0, 1]
            if not np.isnan(r):
                rhythmic_coherence = float(r)

    # 4. Stationary Dither / Synthetic Noise Banding Check
    is_stationary_ultrasonic = bool(ultrasonic_variance < 5.0 and rhythmic_coherence < 0.25 and len(idx_ult) > 0)
    
    # 5. Stopband Noise Purity & Cleanliness Analysis
    stopband_purity = {
        "has_stopband": False,
        "is_messy": False,
        "purity_label": "AUTHENTIC WIDEBAND HARMONICS",
        "description": "Continuous musical harmonics extending across the full container bandwidth.",
        "crest_db": 0.0,
        "max_peak_dbfs": float(np.max(peak_dbfs[idx_ult])) if len(idx_ult) > 0 else -140.0,
        "median_rms_dbfs": float(np.median(rms_dbfs[idx_ult])) if len(idx_ult) > 0 else -140.0,
        "temporal_variance": float(ultrasonic_variance),
        "undulation_db": 0.0
    }
    
    # A true stopband ONLY exists if an anti-aliasing cutoff knee was detected below Nyquist
    if primary_knee and primary_knee.get("detected_knee_khz", 0) > 0:
        stop_start_khz = primary_knee["detected_knee_khz"] + 1.5
        i_stop = np.where((f_khz >= stop_start_khz) & (f_khz <= (nyquist/1000.0 - 1.0)))[0]
        
        if len(i_stop) > 8:
            rms_stop = rms_dbfs[i_stop]
            peak_stop = peak_dbfs[i_stop]
            
            median_rms = float(np.median(rms_stop))
            median_peak = float(np.median(peak_stop))
            max_peak = float(np.max(peak_stop))
            peak_to_floor_crest = float(max_peak - median_rms)
            
            stft_slices = 20.0 * np.log10(np.maximum(stft_mag[i_stop, :], 1e-12))
            temp_var = float(np.mean(np.var(stft_slices, axis=1)))
            
            win_size = min(30, max(5, len(rms_stop) // 4))
            smooth_stop = np.convolve(rms_stop, np.ones(win_size)/win_size, mode="valid")
            undulation_db = float(np.max(smooth_stop) - np.min(smooth_stop)) if len(smooth_stop) > 0 else 0.0
            
            is_messy = bool((peak_to_floor_crest >= 24.0 or temp_var >= 50.0 or undulation_db >= 4.0 or max_peak >= -106.0) and median_rms > -135.0 and max_peak > -120.0)
            
            if is_messy:
                p_label = "MESSY / SPURIOUS HASH"
                p_desc = f"Stopband above {stop_start_khz:.1f} kHz contains sporadic bursts (Variance {temp_var:.1f} dB²) and transient spurs up to {max_peak:.1f} dBFS (+{peak_to_floor_crest:.1f} dB above floor)."
            elif median_rms <= -135.0 or max_peak <= -120.0:
                p_label = "PRISTINE DIGITAL BLACK"
                p_desc = f"Cleaned stopband above {stop_start_khz:.1f} kHz in deep digital silence ({median_rms:.1f} dBFS) with inaudible noise floor."
            elif median_rms > -118.0:
                p_label = "ELEVATED DITHER HUMP"
                p_desc = f"Elevated ultrasonic stopband floor above {stop_start_khz:.1f} kHz ({median_rms:.1f} dBFS) with {undulation_db:.1f} dB undulation."
            else:
                p_label = "PRISTINE / UNIFORM DITHER"
                p_desc = f"Uniform, flat dither floor above {stop_start_khz:.1f} kHz ({median_rms:.1f} dBFS) with low crest ({peak_to_floor_crest:.1f} dB)."
                
            stopband_purity = {
                "has_stopband": True,
                "is_messy": is_messy,
                "purity_label": p_label,
                "description": p_desc,
                "crest_db": round(peak_to_floor_crest, 1),
                "max_peak_dbfs": round(max_peak, 1),
                "median_rms_dbfs": round(median_rms, 1),
                "temporal_variance": round(temp_var, 1),
                "undulation_db": round(undulation_db, 1)
            }

    return {
        "detected_knees": detected_knees,
        "primary_knee": primary_knee,
        "audible_temporal_variance": round(audible_variance, 2),
        "ultrasonic_temporal_variance": round(ultrasonic_variance, 2),
        "rhythmic_coherence": round(rhythmic_coherence, 3),
        "is_stationary_ultrasonic": is_stationary_ultrasonic,
        "stopband_purity": stopband_purity
    }


def generate_dsp_recommendation(primary_prov, bitdepth_info, mqa_info, sr, nyquist, effective_bw_hz, has_ultrasonic_noise=False, purity_info=None, visual_morphology=None):
    if not primary_prov:
        return None
        
    conf = (primary_prov.get("confidence") or "").strip().lower()
    score = primary_prov.get("score", 0.8)
    
    # If the highest confidence is low, do not return any DSP recommendation
    if conf == "low" or score < 0.50:
        return None

    is_potential = bool(conf in ["moderate", "medium"] or score < 0.85)
    action_prefix = "Potential" if is_potential else "Recommended"
    
    label = (primary_prov or {}).get("label", "")
    purity = purity_info or {}
    is_messy = purity.get("is_messy", False)
    max_p = purity.get("max_peak_dbfs", -100.0)
    crest_p = purity.get("crest_db", 0.0)
    
    if "Upsampled from 44.1 kHz" in label or ("Upsampled" in label and "44.1" in label):
        if has_ultrasonic_noise or is_messy:
            noise_reason = f"transient spurs up to {max_p:.1f} dBFS (+{crest_p:.1f} dB crest)" if is_messy else "elevated 16-bit noise and dither artifacts"
            return {
                "action_type": action_prefix,
                "is_potential": is_potential,
                "filter_cutoff_khz": 22.05,
                "action": "Apodizing Low-Pass @ 22.05 kHz (Replace Legacy Cutoff & Cleanse Noise)",
                "action_short": "Apodize @ 22.05k",
                "risk_level": "Minimal Risk — Recommended",
                "risk_class": "risk-minimal",
                "details": f"Source signal brickwalls at 22.05 kHz, but stopband contains {noise_reason}. When expanding bit-depth or re-upsampling, apply a minimum-phase apodizing filter at 22.05 kHz to replace the legacy transition band, eliminate filter ringing, and purge 16-bit noise before 24-bit Shibata noise shaping.",
                "dsp_params": "--cutoff 22050 --phase min --dither shibata"
            }
        else:
            return {
                "action_type": action_prefix,
                "is_potential": is_potential,
                "filter_cutoff_khz": 22.05,
                "action": "Direct Processing (or Apodize @ 22.05 kHz for Bit-Depth Expansion)",
                "action_short": "Direct / Apodize @ 22.05k",
                "risk_level": "Zero Risk",
                "risk_class": "risk-zero",
                "details": "Clean brickwall cutoff at 22.05 kHz. For direct decimation, process directly. When expanding bit-depth or re-upsampling, applying a minimum-phase apodizing filter at 22.05 kHz improves the transition band, removes legacy ADC ringing, and pre-cleans the noise floor before Shibata dither shaping.",
                "dsp_params": "--cutoff 22050 --phase min --dither shibata"
            }
    elif "Upsampled from 48.0 kHz" in label or ("Upsampled" in label and "48.0" in label):
        if has_ultrasonic_noise or is_messy:
            noise_reason = f"transient spurs up to {max_p:.1f} dBFS" if is_messy else "elevated synthetic noise/dither humps"
            return {
                "action_type": action_prefix,
                "is_potential": is_potential,
                "filter_cutoff_khz": 24.0,
                "action": "Apodizing Low-Pass @ 24.0 kHz (Replace Legacy Cutoff & Cleanse Noise)",
                "action_short": "Apodize @ 24.0k",
                "risk_level": "Minimal Risk — Recommended",
                "risk_class": "risk-minimal",
                "details": f"Musical harmonics terminate at 24.0 kHz, but stopband contains {noise_reason}. When expanding bit-depth or re-upsampling, apply a minimum-phase apodizing filter at 24.0 kHz to replace the legacy transition band and purge ultrasonic hash before 24-bit Shibata shaping.",
                "dsp_params": "--cutoff 24000 --phase min --dither shibata"
            }
        else:
            return {
                "action_type": action_prefix,
                "is_potential": is_potential,
                "filter_cutoff_khz": 24.0,
                "action": "Direct Processing (or Apodize @ 24.0 kHz for Bit-Depth Expansion)",
                "action_short": "Direct / Apodize @ 24k",
                "risk_level": "Zero Risk",
                "risk_class": "risk-zero",
                "details": "Clean brickwall cutoff at 24.0 kHz. For direct decimation, process directly. When expanding bit-depth, applying a minimum-phase apodizing filter at 24.0 kHz replaces the legacy transition band and eliminates ringing before 24-bit noise shaping.",
                "dsp_params": "--cutoff 24000 --phase min --dither shibata"
            }
    elif "Upsampled from 88.2 kHz" in label or "Upsampled from 96.0 kHz" in label:
        fn = 44.1 if "88.2" in label else 48.0
        if has_ultrasonic_noise or is_messy:
            noise_reason = f"messy noise hash and transient spurs up to {max_p:.1f} dBFS (+{crest_p:.1f} dB crest)" if is_messy else "elevated ultrasonic noise"
            return {
                "action_type": action_prefix,
                "is_potential": is_potential,
                "filter_cutoff_khz": fn,
                "action": f"Pre-Filter Low-Pass @ {fn} kHz (Strip Ultrasonic Noise & Spurs)",
                "action_short": f"Low-Pass @ {fn}k",
                "risk_level": "Minimal Risk",
                "risk_class": "risk-minimal",
                "details": f"Source signal brickwalls near {fn} kHz, but stopband contains {noise_reason}. Apply a low-pass filter at {fn} kHz before re-upsampling.",
                "dsp_params": f"--cutoff {int(fn*1000)} --phase min --dither shibata"
            }
        else:
            return {
                "action_type": action_prefix,
                "is_potential": is_potential,
                "filter_cutoff_khz": fn,
                "action": f"Apodizing Low-Pass @ {fn} kHz (Cleanse Ultrasonic Stopband & Re-Upsample)",
                "action_short": f"Apodize @ {fn}k",
                "risk_level": "Minimal Risk — Recommended",
                "risk_class": "risk-minimal",
                "details": f"Clean brickwall cutoff detected near {fn} kHz with no legitimate musical harmonics above {fn} kHz. When re-upsampling or expanding bit-depth, applying a minimum-phase apodizing filter at {fn} kHz cleanses unneeded ultrasonic imaging and pre-cleans the noise floor before Shibata dither shaping.",
                "dsp_params": f"--cutoff {int(fn*1000)} --phase min --dither shibata"
            }
    elif "Leaky" in label:
        fn = (primary_prov or {}).get("suspected_nyquist_hz", 22050) / 1000.0 if primary_prov else 22.05
        f_cut = max(20.0, fn - 0.6)
        return {
            "action_type": action_prefix,
            "is_potential": is_potential,
            "filter_cutoff_khz": round(f_cut, 1),
            "action": f"Apodizing Low-Pass Filter @ {f_cut:.1f} kHz",
            "action_short": f"Apodize @ {f_cut:.1f}k",
            "risk_level": "Low Risk",
            "risk_class": "risk-low",
            "details": f"Mirrored ultrasonic imaging aliases detected above {fn:.2f} kHz. Apply an apodizing minimum-phase filter with stopband notch at {fn:.2f} kHz to remove ultrasonic imaging.",
            "dsp_params": f"--apodize {int(f_cut*1000)}"
        }
    elif "Zero-Stuffed" in label or "NOS" in label:
        return {
            "action_type": action_prefix,
            "is_potential": is_potential,
            "filter_cutoff_khz": 21.5,
            "action": "Sharp Reconstruction Low-Pass @ 21.5 kHz",
            "action_short": "Brickwall LP @ 21.5k",
            "risk_level": "Low Risk",
            "risk_class": "risk-low",
            "details": "Unfiltered mirror images detected across ultrasonic spectrum. Apply steep low-pass brickwall filter at 21.5 kHz to eliminate imaging aliases.",
            "dsp_params": "--cutoff 21500 --steep"
        }
    elif "MQA" in label or (mqa_info and mqa_info.get("is_mqa")):
        return {
            "action_type": action_prefix,
            "is_potential": is_potential,
            "action": "Adaptive MQA Subband Unfold + Psychoacoustic Gating",
            "action_short": "Adaptive MQA Unfold",
            "risk_level": "Minimal Risk",
            "risk_class": "risk-minimal",
            "details": "MQA subband packaging detected. Unfold with psychoacoustic noise-gating to reconstruct transient details while suppressing high-frequency quantization noise.",
            "dsp_params": "--mqa adaptive --dither shibata"
        }
    elif bitdepth_info and bitdepth_info.get("is_zero_padded"):
        tz = bitdepth_info.get("trailing_zero_bits", 8)
        eff = bitdepth_info.get("effective_bits", 16)
        return {
            "action_type": action_prefix,
            "is_potential": is_potential,
            "action": f"Direct 64-Bit Processing (Lossless Truncation of {tz} Inactive LSBs)",
            "action_short": f"Lossless {eff}b Process",
            "risk_level": "Zero Risk",
            "risk_class": "risk-zero",
            "details": f"Container is zero-padded ({eff}-bit audio in 24-bit container). Processing directly in 64-bit float is lossless. Re-dither to true 24-bit with Shibata shaping on output.",
            "dsp_params": "--precision float64 --dither shibata"
        }
    elif "Native" in label:
        pk = (visual_morphology or {}).get("primary_knee")
        if pk and pk.get("is_brickwall_knee") and 19.5 <= pk.get("freq_khz", 0) <= 21.6:
            fk = pk["freq_khz"]
            return {
                "action_type": action_prefix,
                "is_potential": is_potential,
                "filter_cutoff_khz": round(fk, 1),
                "action": f"Apodizing Low-Pass @ {fk:.1f} kHz (Replace Legacy ADC Cutoff & Cleanse Noise)",
                "action_short": f"Apodize @ {fk:.1f}k",
                "risk_level": "Minimal Risk — Recommended",
                "risk_class": "risk-minimal",
                "details": f"Original studio ADC anti-aliasing filter knee detected at {fk:.1f} kHz (plunging {pk.get('drop_db', 0):.1f} dB into noise floor). Applying a minimum-phase apodizing filter at {fk:.1f} kHz replaces the steep legacy brickwall transition, eliminates ADC filter ringing, and purges empty noise above {fk:.1f} kHz before 24-bit Shibata upsampling.",
                "dsp_params": f"--cutoff {int(fk*1000)} --phase min --dither shibata"
            }
        else:
            return {
                "action_type": action_prefix,
                "is_potential": is_potential,
                "action": "Direct Polyphase Sinc Upsampling (No Pre-Filtering Required)",
                "action_short": "Direct Upsampling",
                "risk_level": "Zero Risk",
                "risk_class": "risk-zero",
                "details": "Authentic wideband master with clean acoustic harmonics extending across the container spectrum. Process directly using minimum-phase polyphase sinc interpolation and Shibata noise shaping.",
                "dsp_params": "--phase min --dither shibata"
            }
    else:
        return {
            "action_type": action_prefix,
            "is_potential": is_potential,
            "action": "Standard Minimum-Phase Sinc Reconstruction",
            "action_short": "Min-Phase Sinc",
            "risk_level": "Low Risk",
            "risk_class": "risk-low",
            "details": "Standard acoustic profile. Apply minimum-phase sinc reconstruction to preserve time-domain impulse response.",
            "dsp_params": "--phase min --dither shibata"
        }


# Singleton engine instance
ENGINE = ProvenanceRuleEngine()


def get_provenance_engine(rules_path=None):
    if rules_path:
        return ProvenanceRuleEngine(rules_path)
    return ENGINE


if __name__ == "__main__":
    import argparse
    import soundfile as sf
    import librosa
    from scipy import signal

    parser = argparse.ArgumentParser(description="Hi-Res Audio Forensic Provenance Engine & Rules Tester")
    parser.add_argument("file", nargs="?", help="FLAC or WAV file to evaluate against rules")
    parser.add_argument("--rules", default=DEFAULT_RULES_PATH, help="Path to provenance_rules.json")
    parser.add_argument("--show-rules", action="store_true", help="Print active rule set")
    args = parser.parse_args()

    engine = ProvenanceRuleEngine(args.rules)

    if args.show_rules:
        print(json.dumps(engine.rules, indent=2))
        exit(0)

    if args.file:
        print(f"\n=======================================================")
        print(f"🔬 FORENSIC PROVENANCE RULES EVALUATION")
        print(f"Target : {os.path.basename(args.file)}")
        print(f"Rules  : {args.rules}")
        print(f"=======================================================")

        data, sr = sf.read(args.file, dtype='float64', start=0, stop=60*192000)
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        nyquist = sr / 2.0
        n_fft = 8192
        hop_length = n_fft // 4
        win = signal.windows.blackmanharris(n_fft)
        S1 = np.sum(win)
        stft = librosa.stft(data, n_fft=n_fft, hop_length=hop_length, window="blackmanharris")
        stft_mag = np.abs(stft)
        stft_norm = stft_mag / (S1 / 2.0)
        spec_db = 20.0 * np.log10(np.maximum(stft_norm, 1e-12))
        mean_spec_db = np.mean(spec_db, axis=1)
        power_linear = np.mean(stft_norm**2, axis=1)
        rms_dbfs = 10.0 * np.log10(np.maximum(power_linear, 1e-24))
        peak_mag = np.max(stft_norm, axis=1)
        peak_dbfs = 20.0 * np.log10(np.maximum(peak_mag, 1e-12))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        zero_ratio = 1.0 - (np.count_nonzero(data) / len(data))

        res = engine.evaluate(sr, nyquist, freqs, mean_spec_db, rms_dbfs, peak_dbfs, stft_mag, zero_ratio)

        print(f"\nMOST LIKELY PROVENANCE : {res['primary']['label']}")
        print(f"Confidence             : {res['primary']['confidence']} ({int(res['primary']['score']*100)}%)")
        print(f"Details                : {res['primary']['details']}")
        
        if res['alternative']:
            print(f"\nALTERNATIVE POSSIBILITY: {res['alternative']['label']}")
            print(f"Confidence             : {res['alternative']['confidence']} ({int(res['alternative']['score']*100)}%)")
            print(f"Details                : {res['alternative']['details']}")
        else:
            print(f"\nALTERNATIVE POSSIBILITY: None (Decisive verdict)")
        print(f"=======================================================\n")
    else:
        parser.print_help()
