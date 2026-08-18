# 🎼 MQA Forensic Detection, Unfolding & Stripping

> **Comprehensive Technical Guide to AcoustiSinc's MQA Bit-Plane Forensics, Core Unfolding, Psychoacoustic Gating, and LSB De-Hashing.**

---

## 🌟 Overview

Master Quality Authenticated (MQA) is a hierarchical lossy-to-lossless packing technology that encapsulates high-frequency ultrasonic subbands into the least significant bits (LSBs) of standard $44.1\text{ kHz}$ or $48.0\text{ kHz}$ PCM containers. 

AcoustiSinc provides an advanced **64-bit double-precision MQA DSP pipeline** capable of deep bit-plane detection, adaptive psychoacoustic unfolding, and purist noise stripping:

```
                            MQA SIGNAL PROCESSING MODES
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
   --mqa adaptive (Default)        --mqa strip                  --mqa simple
   ─────────────────────────       ───────────                  ────────────
   • Core subband unfold           • Strips LSB hash stream     • Standard linear
   • Psychoacoustic noise gate     • Re-dithers to 24-bit TPDF    subband unfold
   • Reconstructs ultrasonic       • Restores pitch-black       • Retains elevated
     harmonics without hiss          noise floor (-144 dBFS)      noise floor
```

---

## 🔬 Dual-Layer Forensic Detection

AcoustiSinc scans files for MQA encoding through two independent layers:

1. **Vorbis Comment & Header Scan**:
   * Inspects FLAC tags for `MQAENCODER`, `ORIGINALSAMPLERATE`, and `ENCODER` strings.
2. **Deep Bit-Plane XOR Sync Search**:
   * Scans the $L \oplus R$ bitstream across the lower 8 bits for the official **36-bit MQA synchronization magic word** (`0xBE0498C88`).
   * Decodes the original master sample rate (e.g., $88.2\text{k}$, $96\text{k}$, $192\text{k}$, $352.8\text{k}$, or $705.6\text{k}$ DXD) and the provenance flag (**MQA Studio** vs. **MQA Authenticated**).
   * Generates real-time visual badges (`[MQA STUDIO]` or `[MQA]`) in the Interactive Web Explorer.

---

## 🎛️ Processing Modes & CLI Usage

AcoustiSinc supports four distinct processing strategies via the `--mqa` parameter:

```bash
# 1. Adaptive Unfold (Default - Best for Classical, Jazz & High-Resolution Acoustic Masters)
python upsampler.py "/Music/Hi-Res/Album" --mqa adaptive

# 2. MQA Noise Stripping (Best for Pop, Rock, Electronic & Purist Reference Playback)
python upsampler.py "/Music/Hi-Res/Album" --mqa strip

# 3. Simple Linear Unfold (Historical emulation)
python upsampler.py "/Music/Hi-Res/Album" --mqa simple

# 4. Raw Ignore (Unaltered legacy PCM)
python upsampler.py "/Music/Hi-Res/Album" --mqa ignore
```

---

## 📊 Trade-Off Comparison Matrix

| Mode | Soundstage & Air | Background Blackness | Distortion Level | Best Musical Fit |
| :--- | :--- | :--- | :--- | :--- |
| **`adaptive`** *(Default)* | ⭐⭐⭐⭐⭐ **Maximum** | ⭐⭐⭐⭐ **High** | ⭐⭐⭐⭐ **Low** | Acoustic recordings with genuine ultrasonic harmonics (2L, ECM, live acoustic). |
| **`strip`** | ⭐⭐⭐⭐ **Very Good** | ⭐⭐⭐⭐⭐ **Pitch Black** | ⭐⭐⭐⭐⭐ **Lowest** | Studio pop, rock, electronic, or questionable MQA provenance (eliminates LSB hash). |
| **`simple`** | ⭐⭐⭐⭐ **Good** | ⭐⭐ **Grainy / Hazy** | ⭐⭐ **Higher Noise** | Historical reference / raw MQA subband emulation (retains $-48\text{ dBFS}$ floor). |
| **`ignore`** | ⭐⭐⭐ **Standard** | ⭐⭐⭐ **Slight LSB Hash** | ⭐⭐⭐ **Moderate** | Exact legacy DAC playback without modification. |

---

## 🔍 Technical Details: Why Does MQA Add Noise?

* **16-Bit MQA-CDs**:
  * In 16-bit MQA-CD tracks, Bit 16 (the lowest bit) is continuously overwritten with pseudo-random MQA packet control streams. On a standard non-MQA DAC, this toggles like **correlated high-frequency hash noise**, degrading effective dynamic range from $96\text{ dB}$ down to $\approx 84\text{--}88\text{ dB}$.
* **24-Bit MQA Releases**:
  * The lower 8 bits (bits 0–7) hold the compressed lossy subband payload instead of physical analog dither. If unfolded without noise-gating, quantization noise from the 8-bit subband leaks into the audible spectrum.
* **The `--mqa strip` Solution**:
  * Zero-masks the pseudo-random packet payload and applies **64-bit TPDF dither**, restoring a pure $24\text{-bit}$ linear PCM container with a pitch-black $-115\text{ dBFS}$ to $-144\text{ dBFS}$ noise floor.
