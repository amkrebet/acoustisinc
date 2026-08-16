"""
Provenance Rule Engine & Forensic Configurator
Strict 64-bit Double Precision (float64) DSP Pipeline Integration
"""

import os
import json
import numpy as np


DEFAULT_RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "provenance_rules.json")


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
            import soundfile as sf
            info = sf.info(filepath)
            frames = min(int(info.samplerate * 6.0), info.frames)
            pcm_int32, _ = sf.read(filepath, frames=frames, dtype="int32", always_2d=True)
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
            import soundfile as sf
            info = sf.info(filepath)
            res["container_subtype"] = info.subtype
            res["container_bits"] = 24 if "24" in info.subtype else (32 if "32" in info.subtype else 16)
            if pcm_int32 is None:
                frames = min(int(info.samplerate * 30.0), info.frames)
                pcm_int32, _ = sf.read(filepath, frames=frames, dtype="int32", always_2d=True)
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

        # 3. Evaluate Cutoff & Upsampling Rules
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

            is_match = False
            if spec_drop >= min_spec_drop or (min_rms_drop > 0 and rms_drop >= min_rms_drop):
                is_match = True
                if max_stopband is not None and rms_dbfs[i_high] > max_stopband:
                    is_match = False
                if floor_margin is not None and abs_ceiling is not None:
                    if not (rms_dbfs[i_high] <= noise_floor_rms + floor_margin or mean_spec_db[i_high] < abs_ceiling):
                        is_match = False

            if is_match:
                matched_cutoff = c_rule
                break

        # 4. Measure Effective Signal Bandwidth for Native Assessment
        above_floor = (rms_dbfs > (noise_floor_rms + 5.0)) | (mean_spec_db > (noise_floor_spec + 5.0))
        valid_idx = np.where(above_floor)[0]
        effective_bw_hz = float(freqs[valid_idx[-1]]) if len(valid_idx) > 0 else 20000.0

        # 5. Dual Verdict Assembler
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

            if enable_alts and leaky_hits:
                top_leaky = leaky_hits[0]
                if top_leaky["mirror_corr"] >= alt_cfg.get("mirror_leakage_min_corr", 0.55):
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
            if effective_bw_hz >= native_cfg.get("sr_48k_high_confidence_bw_hz", 23500):
                primary_prov = {
                    "label": "Native 48.0 kHz Master",
                    "confidence": "High",
                    "score": 0.90,
                    "badge_class": "badge-provenance-native",
                    "details": f"Continuous harmonic extension up to {effective_bw_hz/1000:.1f} kHz Nyquist limit."
                }
            elif effective_bw_hz >= native_cfg.get("sr_48k_mod_confidence_bw_hz", 21500):
                primary_prov = {
                    "label": "Native 48.0 kHz Material",
                    "confidence": "Moderate",
                    "score": 0.75,
                    "badge_class": "badge-provenance-native",
                    "details": f"Natural acoustic roll-off extending to ~{effective_bw_hz/1000:.1f} kHz."
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

        return {
            "primary": primary_prov,
            "alternative": alt_prov,
            "label": primary_prov["label"],
            "confidence": primary_prov["confidence"],
            "score": primary_prov["score"],
            "badge_class": primary_prov["badge_class"],
            "details": primary_prov["details"],
            "effective_bw_hz": effective_bw_hz,
            "noise_floor_rms": noise_floor_rms,
            "noise_floor_spec": noise_floor_spec,
            "bitdepth": bitdepth_info
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
