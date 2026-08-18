# 🛡️ Intersample Headroom, Gain Architecture & Dynamic Range

> **Comprehensive Technical Guide to Intersample True-Peak Management, Headroom Auto-Healing, and 64-Bit TT Dynamic Range Scoring.**

---

## 🌟 The Physics of Intersample Overshoots

Digital audio samples represent discrete instantaneous voltage measurements. Band-limited sinc upsampling reconstructs the continuous analog curve between these discrete samples:

```
          Discrete Digital Samples                    Continuous Reconstructed Waveform
             [0 dBFS]       [0 dBFS]                       ▲  +2.4 dBFS True Peak Overshoot!
                ●              ●                          ╱ ╲
                │              │                         ╱   ╲
     ───────────┼──────────────┼─────────── 0 dBFS ─────●─────●───── 0 dBFS Ceiling
                │              │
```

When an audio recording has been mastered hot or heavily limited (compressed peaks near $0\text{ dBFS}$), reconstructing the continuous curve between adjacent full-scale samples produces **intersample true-peak overshoots** that frequently exceed $0\text{ dBFS}$ by $+1.5\text{ to }+4.0\text{ dB}$. If not managed with dynamic headroom, downstream DACs or fixed-point integer containers will severely clip and distort.

---

## 🛡️ AcoustiSinc 2-Stage Dynamic Gain Architecture

AcoustiSinc implements a 2-stage gain management system in strict **64-bit double precision (`float64`)** to guarantee zero intersample clipping while preserving maximum SNR:

```
                            2-STAGE GAIN ARCHITECTURE
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
  STAGE 1: ADAPTIVE PRE-SCAN                                STAGE 2: AUTO-HEALING
  ──────────────────────────                                ─────────────────────
  • Scans global album peak & crest factor                  • Verifies output true peak buffer
  • Dynamically calculates initial headroom                 • Captures exact overshoot dB
  • Preserves inter-track volume relationships              • Single-pass exact mathematical backoff
```

### Stage 1: Adaptive Crest-Aware Album Headroom Pre-Scan
Prior to upsampling, all audio tracks in an album folder are scanned in parallel to measure the global album peak (`album_max_peak_db`) and dynamic **Crest Factor ($\text{Peak} - \text{RMS}$)**:

* **Classical & Acoustic Jazz** ($\text{Crest} \ge 16\text{ dB}$): Margin = **$+1.8\text{ dB}$** (maximizes SNR and dynamic resolution).
* **Standard Audiophile Masters** ($\text{Crest } 12\text{--}16\text{ dB}$): Margin = **$+2.8\text{ dB}$**.
* **Commercial Pop & Rock** ($\text{Crest } 8\text{--}12\text{ dB}$): Margin = **$+3.8\text{ dB}$**.
* **Hyper-Compressed / Brickwall Masters** ($\text{Crest} < 8\text{ dB}$): Margin = **$+4.4\text{ dB}$** (prevents mid-album aborts).

$$\text{Initial Gain Factor} = \frac{10^{-0.3 / 20}}{10^{(\text{Album Peak}_{\text{dBFS}} + \text{Adaptive Margin}_{\text{dB}}) / 20}}$$

### Stage 2: Exact Overshoot Auto-Healing
Every upsampled output buffer is scanned in 64-bit float against the $-0.3\text{ dBFS}$ target ceiling (`PEAK_TARGET_DB`). If an extreme transient overshoots:

$$\text{Exact Backoff Factor} = \left(\frac{\text{Peak}_{\text{Target}}}{\text{Peak}_{\text{Overshoot}}}\right) \times 10^{-0.2 / 20}$$

The pass is cleanly restarted with this exact gain factor, guaranteeing compliance in a **single retry** without iterative volume loss.

---

## 🎚️ Audiophile Dynamic Range (DR) Scoring

AcoustiSinc computes official 64-bit TT Dynamic Range scores standardized by the [Pleasurize Music Foundation](https://dr.loudness-war.info) alongside broadcast mastering metrics:

$$\text{DR Score} = \text{Peak}_{\text{dBFS}} - \text{RMS}_{\text{Top 20\% (dBFS)}}$$

### Audiophile DR Rating Scale

| DR Score | Rating | Color Code | Musical Characteristics |
| :--- | :--- | :--- | :--- |
| **DR 14+** | **Pristine Dynamics** | 🟦 Cyan / Blue | Uncompressed audiophile masters (Classical, Acoustic Jazz, early vinyl/CD transfers). |
| **DR 10 – 13** | **Good Dynamics** | 🟩 Green | Open, dynamic recordings with natural transients and punch. |
| **DR 7 – 9** | **Moderate Compression** | 🟨 Yellow | Noticeable dynamic compression and limiting (typical commercial pop/rock). |
| **DR 1 – 6** | **Heavy Compression** | 🟥 Red | Severe brickwall limiting / Loudness War hyper-compression. |

### Broadcast Mastering Metrics
* **EBU R128 Loudness Range (LRA)**: Measures macro-dynamics (dynamic span in **LU / dB**) using dual-gated K-weighting filters (discarding silence below $-70\text{ LUFS}$ and relative gate at $-20\text{ LU}$).
* **Integrated Loudness ($\text{LUFS}$)**: ITU-R BS.1770-4 overall perceived loudness across the track.
* **Peak-to-RMS Crest Factor ($\text{dB}$)**: Headroom between overall RMS power and true peak level.
