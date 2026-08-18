# 📊 Library Forensic Provenance Audit & Pre-Processing Guide

> **Comprehensive Guide to Forensic Spectral Signatures, Fake Hi-Res Identification, Apodizing Pre-Filtering Recipes, and Integer Decimation Strategies.**

---

## 🌟 Overview

Spectral forensic auditing analyzes the physical high-frequency behavior of digital audio files to verify whether a high-resolution container ($48\text{ kHz}$, $88.2\text{ kHz}$, $96\text{ kHz}$, $176.4\text{ kHz}$, or $192.0\text{ kHz}$) contains authentic wideband acoustic harmonics, or represents an upscaled CD master, zero-stuffed container, or leaky DAC/SRC transfer.

This guide outlines the mathematical rationale and pre-processing recipes for handling common provenance anomalies before or during upsampling.

---

## 📊 Summary of Common Provenance Anomalies

```
                           FORENSIC SPECTRUM SIGNATURES
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
  FAKE HIGH-RATE CONTAINERS     LEAKY SRC / MIRROR IMAGING    NATIVE CONTINUOUS ROLLOFF
  ─────────────────────────     ──────────────────────────    ─────────────────────────
  • Sharp brickwall cutoff      • Mirrored alias reflection   • Gentle acoustic slope
    into digital black            past source Nyquist           to container Nyquist
  • Dead ultrasonic spectrum    • Causes intermodulation      • Active peaks & rhythm
  • Recipe: Apodize / Decimate  • Recipe: Apodize Notch LPF     coherence (r > +0.70)
```

---

## 🔬 In-Depth Pre-Processing Strategies

### Strategy 1: High-Rate Containers with Fake Ultrasonic Bandwidth ($176.4\text{k} / 192\text{k}$)
* **The Symptom**: A file packaged in a $176.4\text{ kHz}$ or $192.0\text{ kHz}$ container exhibits a sharp brickwall cutoff at $44.1\text{ kHz}$ or $48.0\text{ kHz}$, with digital silence or flat dither above it.
* **Why Non-Integer Decimation is Damaging ($176.4\text{ kHz} \to 96.0\text{ kHz}$)**:
  * Non-integer resampling ($\frac{96000}{176400} = \frac{80}{147}$) requires upsampling $80\times$ followed by decimation by $147$. This introduces two stages of polyphase sinc convolution and cascading passband ripple.
* **The Recommended Workflow**:
  * **Direct 64-Bit Apodizing Low-Pass Filtering at $176.4\text{ kHz}$**: Apply an apodizing sinc filter with raised-cosine transition centered at $44.1\text{ kHz}$ directly on the $176.4\text{ kHz}$ container (or integer decimate $2:1 \to 88.2\text{ kHz}$).
  * Completely eliminates empty ultrasonic band ringing and studio ADC artifacts without non-integer phase distortion.

---

### Strategy 2: 48.0 kHz Containers with 44.1 kHz Leaky Ultrasonic Mirrors
* **The Symptom**: Audio mastered from $44.1\text{ kHz}$ into a $48.0\text{ kHz}$ container with low-grade sample rate converters frequently exhibits **mirrored alias images** between $22.05\text{ kHz}$ and $24.0\text{ kHz}$.
* **The Risk**: Upsampling a leaky $48\text{ kHz}$ file to $192\text{ kHz}$ without pre-filtering amplifies the mirrored alias into the passband of downstream DACs, creating audible intermodulation distortion.
* **The Recommended Pre-Processing**:
  * Apply a **64-bit Apodizing Low-Pass Filter at $20.5\text{--}21.5\text{ kHz}$** ($-6\text{ dB}$ at $22.05\text{ kHz}$, $>140\text{ dB}$ attenuation at $24.0\text{ kHz}$) to eliminate the alias mirror before $4\times$ sinc interpolation to $192.0\text{ kHz}$.

```bash
python upsampler.py "/Music/Hi-Res/Album" --cutoff 21500 --phase min --dither shibata
```

---

### Strategy 3: Clean Integer Decimation ($2:1$ and $4:1$)
* For authentic integer-scaled files (e.g. clean zero-stuffed $44.1\text{k} \to 176.4\text{k}$ or $48\text{k} \to 192\text{k}$):
  1. Subsample every 4th sample with polyphase half-band rejection.
  2. Re-upsample with AcoustiSinc Minimum Phase Sinc and 5th-order Shibata noise shaping to eliminate the original legacy upsampler's ringing.

---

### Strategy 4: Bit-Depth Expansion with Apodization
* When upsampling 16-bit Red Book CD sources into 24-bit containers:
  * Applying a minimum-phase apodizing filter at $22.05\text{ kHz}$ cleans the 16-bit quantization floor and replaces harsh legacy ADC transition bands with a smooth monotonic step response before applying 24-bit Shibata noise shaping.
