# AcoustiSinc & Forensic Audio Lab

> **High-Performance GPU-Accelerated 64-Bit Sinc Audio Upsampler & Interactive Forensic Spectral Analyzer**

![AcoustiSinc Forensic Spectral Analysis](acoustisinc_forensic_analysis.png)

---

## 🌟 Overview

**AcoustiSinc** is an audiophile-grade digital signal processing (DSP) suite engineered for mastering-grade audio upsampling and forensic spectral authentication.

It delivers:
1. **Mathematical Purity**: Strict **64-bit double precision (`float64` / `complex128` / OpenCL `double2`)** maintained end-to-end across all FFTs, filtering kernels, and noise-shaping algorithms.
2. **GPU Sinc Upsampling**: High-throughput OpenCL-accelerated band-limited Whittaker–Shannon Sinc interpolation supporting **Linear Phase**, **Minimum Phase**, and **Apodizing** impulse responses.
3. **Advanced Psychoacoustic Noise Shaping**: 4th-order multi-band noise-shaping curves (Shibata & High-Rate) moving dither energy safely into the high ultrasonic spectrum ($>35\text{ kHz}$).
4. **Dynamic Gain Structure & Auto-Healing**: Pre-flight album headroom scanning with exact overshoot-calculated auto-healing guaranteeing zero intersample clipping.
5. **Forensic Cutoff & Fake Hi-Res Detection**: Automated acoustic analysis identifying upsampled CD masters, brick-wall filter cutoffs, zero-stuffing, and ultrasonic noise profiles.
6. **Audiophile Dynamic Range (DR) Scoring**: Official 64-bit Pleasurize Music Foundation (PMF) TT Dynamic Range Meter scoring alongside EBU R128 Loudness Range (LRA) and Integrated LUFS.
7. **Interactive Web Explorer**: A real-time, on-demand browser application providing dynamic library navigation, live DSP spectral analysis in under 1 second, interactive spectrograms with exact cursor dB level readouts, and in-browser lossless playback.

---

## 🚀 Key Features

### 1. Mastering-Grade Sinc Upsampler (`upsampler.py`)
- **Auto-Integer Ratio Scaling**: Upsamples $44.1\text{ kHz} \to 176.4\text{ kHz}$ ($4\times$) and $48\text{ kHz} \to 192\text{ kHz}$ ($4\times$) without fractional alias artifacts.
- **Filter Topologies**:
  - `linear`: True symmetric linear-phase band-limited Sinc reconstruction.
  - `min` / `minimum`: Causal minimum-phase reconstruction eliminating all pre-ringing for crisp transient attack.
  - `--apodizing`: Apodizing transition band to attenuate pre-existing ADC brick-wall ringing (can be combined with Linear or Minimum Phase).
- **Intersample Headroom Management**: Pre-flight album gain scans with exact overshoot auto-healing to guarantee **zero intersample clipping** ($0\text{ dBFS}$ true peak compliance).
- **Metadata & Album Art Preservation**: Lossless bit-perfect tag copying, ReplayGain retention, and embedded cover art extraction/sanitization.
- **Strict FLAC Compression Level 5**: Consistent balanced lossless compression.

### 2. Interactive Real-Time Forensic Explorer (`server.py`)
- **On-Demand Real-Time Analysis**: Dynamic library navigation with live 64-bit DSP forensic analysis in $\sim 0.8\text{s}$.
- **Hi-Fi News Style Spectral Inspection**: Full-track Peak Hold and RMS noise floor traces.
- **Automated Forensics & Cutoff Detection**:
  - `[FAKE HI-RES / UPSAMPLED CD SOURCE]`: Detects sharp $>45\text{ dB}$ brick-wall cutoffs near $22.05\text{ kHz}$.
  - `[UPSAMPLED 48 kHz / 96 kHz SOURCE]`: Identifies $24\text{ kHz}$ and $48\text{ kHz}$ legacy digital master cutoffs.
  - `[NATIVE HI-RES MATERIAL]`: Verifies continuous acoustic harmonic extension into ultrasonic frequencies.
  - `[NOISE PROFILE]`: Identifies Psychoacoustic Noise Shaping rise ($+6\text{ dB} \to +20\text{ dB}$ HF rise), Flat TPDF dither, or DSD ultrasonic humps.
  - `[BIT-DEPTH ZERO-PADDING]`: Measures Effective Bit Depth (ENOB) and detects fake 24-bit containers (16-bit zero-padded with 8 inactive LSBs or 20-bit HDCD transfers).
  - `[DYNAMIC RANGE]`: Official **TT Dynamic Range (DR Score e.g. DR12)**, **EBU R128 Loudness Range (LRA)**, and **Integrated LUFS**.
- **Interactive Heatmap Canvas**: Zoom/pan with mouse wheel and drag; cursor HUD reveals **exact Time (s), Frequency (kHz), and Level (dBFS)** anywhere on the canvas.
### 3. MQA Forensic Detection & Core Unfolding / Stripping Engine
- **Dual-Layer Forensic Detection**:
  - **Fast Metadata Scan**: Vorbis comments (`MQAENCODER`, `ORIGINALSAMPLERATE`, `ENCODER`).
  - **Deep Bit-Plane Sync**: Scans $L \oplus R$ XOR streams for the official 36-bit sync word (`0xBE0498C88`) and decodes original master rates and studio provenance flags.
  - **Visual Badging**: Violet `[MQA STUDIO]` or `[MQA]` badge with decoded original master sampling rate in the Web Explorer UI.
- **Configurable Processing Modes (`--mqa [mode]`)**:
  - **`adaptive`** *(Default)*: Companded high-fidelity unfold with psychoacoustic noise-gating to reconstruct ultrasonic harmonics while suppressing 8-bit quantization noise in quiet passages ($\le -95\text{ dBFS}$).
  - **`strip`**: Strips the noisy MQA pseudo-random bitstream hash from the LSBs, applies pristine 64-bit TPDF dither, and upsamples the pure baseband for maximum dynamic range and zero intermodulation distortion.
  - **`simple`**: Standard linear subband unfold ($2\times$ baseband).
  - **`ignore`**: Treats the audio as raw unaltered PCM.

---

## 🔬 Filter Topology, Roll-Off & Psychoacoustic Analysis

AcoustiSinc implements four distinct mathematical reconstruction filter topologies in strict 64-bit double precision (`float64` / `complex128` / OpenCL `double2`), catering to different recording provenances, musical genres, and acoustic preferences:

![AcoustiSinc Filter Topology Breakdown](acoustisinc_filter_topology_breakdown_v2.png)

### Topology Comparison Matrix

| Filter Topology | CLI Flags | Time-Domain Profile | Pre-Ringing | Post-Ringing | Phase Linearity | Roll-Off Geometry | Best Musical Fit |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **1. Linear Phase (Standard)** | `--phase linear` | Symmetric $\text{sinc}(x)$ | High (Symmetric) | Moderate | **Strict 0° Linear** | Sharp brickwall at $22.05\text{k}$ | Electronic, synthesizers, stems |
| **2. Linear Phase + Apodizing** | `--phase linear --apod` | Symmetric with cosine taper | Attenuated | Short | **Strict 0° Linear** | Smooth cosine taper from $20.0\text{k}$ | Hot/brickwalled pop & rock |
| **3. Minimum Phase (Standard)** | `--phase min` | Real-cepstrum causal sinc | **Zero (0.00%)** | Moderate | Minimum Phase (Analog) | Sharp brickwall at $22.05\text{k}$ | **Acoustic, jazz, classical, vocals** |
| **4. Minimum Phase + Apodizing** | `--phase min --apod` | Causal sinc with cosine taper | **Zero (0.00%)** | **Very Short** | Minimum Phase (Analog) | Smooth cosine taper from $20.0\text{k}$ | Early 1980s CDs / bright mixes |

---

### In-Depth Scenario Breakdown

#### 1. Linear Phase (Standard Sinc) — `--phase linear`
* **Filter & Time-Domain Response**: Perfectly symmetric around $t = 0$. Exactly 50% of the ringing energy occurs *before* the transient peak (pre-ringing) and 50% occurs *after* (post-ringing). Delivers absolute **zero phase distortion (0° phase shift)** and constant group delay across all frequencies ($20\text{ Hz} \to 20\text{ kHz}$).
* **Roll-Off & Ultrasonic Attenuation**: Razor-flat $\pm 0.00001\text{ dB}$ passband up to $20.0\text{ kHz}$ followed by a steep brickwall cutoff landing into a **$>140\text{ dB}$ stopband rejection floor**.
* **Noise Characteristics**: Coupled with 24-bit Shibata 5th-order psychoacoustic noise shaping, baseband quantization noise is pushed safely above $35\text{ kHz}$ with dynamic range $>120\text{ dB}$.
* **Psychoacoustic Impact**: Human hearing relies on *forward temporal masking* (masking sounds *after* a loud event by up to $100\text{--}200\text{ ms}$), but has near-zero *backward temporal masking* ($<5\text{ ms}$). Pre-ringing occurs before the note strikes and can be perceived as an artificial "digital smear" or lack of visceral transient bite. However, perfect phase linearity preserves precise spatial phase cues in multi-track synthesized mixes.

#### 2. Linear Phase + Apodizing — `--phase linear --apodizing`
* **Filter & Time-Domain Response**: Symmetric impulse with a raised-cosine apodizing window that substantially damps both pre-ringing and post-ringing amplitudes.
* **Roll-Off & Ultrasonic Attenuation**: Begins rolling off gently at $\approx 20.05\text{ kHz}$ ($-6\text{ dB}$ at $22.05\text{ kHz}$), eliminating Gibbs phenomenon overshoots and sharp corner reflections.
* **Noise Characteristics**: Produces lower intersample peak overshoot than standard brickwall sinc, preserving dynamic headroom.
* **Psychoacoustic Impact**: Many vintage 1980s/90s digital recordings were digitized through primitive brickwall studio ADCs that permanently captured filter ringing. The apodizing filter attenuates pre-existing studio ADC ripple, removing harsh treble glare while preserving 100% linear phase coherence.

#### 3. Minimum Phase (Causal Sinc) — `--phase min` *(Recommended Default)*
* **Filter & Time-Domain Response**: Generated via real-cepstral Hilbert transform factorization. Shifted entirely into the **causal domain** ($t \ge 0$) with **strictly 0.00% pre-ringing**. The signal is at absolute rest prior to the transient strike, followed by natural exponential post-ringing.
* **Roll-Off & Ultrasonic Attenuation**: Bit-perfect flat passband to $20.0\text{ kHz}$ with steep $>140\text{ dB}$ ultrasonic stopband rejection.
* **Noise Characteristics**: Optimal synergy with multi-threaded Shibata noise shaping; sample-by-sample feedback loops operate strictly forward in time.
* **Psychoacoustic Impact**: Because there is zero pre-echo, percussive transients (drums, acoustic guitar plucks, piano hammers) hit with maximum instantaneous speed and visceral physical punch. The post-ringing falls entirely within the ear's forward temporal masking window, rendering it completely inaudible.

#### 4. Minimum Phase + Apodizing — `--phase min --apodizing`
* **Filter & Time-Domain Response**: Combines **zero pre-ringing** with rapid cosine-tapered post-ringing damping (post-ringing decays in under half the time of standard minimum phase).
* **Roll-Off & Ultrasonic Attenuation**: Smooth cosine transition starting near $20.05\text{ kHz}$ with monotonic step response and minimal overshoot.
* **Noise Characteristics**: Lowest peak-to-average overshoot ratio of all four topologies.
* **Psychoacoustic Impact**: The most fatigue-free filter profile available. Produces a warm, silky top-end reminiscent of high-end analog tape playback. Transients remain fast without ever sounding brittle or analytical. Excellent for bright headphones, horn speakers, and long listening sessions.

---

## 🛡️ Intersample Headroom & Dynamic Gain Structure

Upsampling band-limited audio reconstructs the continuous analog waveform between discrete digital samples. On heavily mastered or brick-wall limited audio, this natural mathematical curve reconstruction creates **intersample true-peak overshoots** that easily exceed $0\text{ dBFS}$.

AcoustiSinc uses an intelligent **2-Stage Dynamic Gain Architecture** to guarantee absolute zero intersample clipping while preserving maximum dynamic range:

### Stage 1: Adaptive Crest-Aware Album Headroom Pre-Scan
- Prior to upsampling, all audio tracks in an album folder are scanned in parallel to measure the global album peak (`album_max_peak_db`) and dynamic **Crest Factor ($\text{Peak} - \text{RMS}$)**.
- The engine dynamically assigns a calibrated intersample headroom margin based on the recording's dynamic profile:
  - **Classical & Acoustic Jazz** ($\text{Crest} \ge 16\text{ dB}$): Calibrated to **$+1.8\text{ dB}$** to preserve maximum volume, resolution, and SNR.
  - **Standard Audiophile Masters** ($\text{Crest } 12 - 16\text{ dB}$): Calibrated to **$+2.8\text{ dB}$**.
  - **Commercial Pop & Rock** ($\text{Crest } 8 - 12\text{ dB}$): Calibrated to **$+3.8\text{ dB}$**.
  - **Hyper-Compressed / Brickwall Masters** ($\text{Crest} < 8\text{ dB}$): Calibrated to **$+4.4\text{ dB}$** to prevent mid-album aborts and retries.

$$\text{Initial Gain Factor} = \frac{10^{-0.3 / 20}}{10^{(\text{Album Peak}_{\text{dBFS}} + \text{Adaptive Margin}_{\text{dB}}) / 20}}$$

- Preserves 100% of relative inter-track volume relationships across the entire album.

### Stage 2: Exact Overshoot Auto-Healing
- Every upsampled track is scanned in 64-bit double precision across its full output buffer for true intersample peak compliance against the $-0.3\text{ dBFS}$ ceiling (`PEAK_TARGET_DB`).
- If an extreme track exceeds $-0.3\text{ dBFS}$ due to intense harmonic reconstruction, the engine captures the exact true peak overshoot and calculates the precise dynamic backoff:

$$\text{Exact Backoff Factor} = \left(\frac{\text{Peak}_{\text{Target}}}{\text{Peak}_{\text{Overshoot}}}\right) \times 10^{-0.2 / 20}$$

- The album pass is cleanly restarted with this exact gain factor, resolving the clipping in a **single retry** without unnecessary iterative volume loss.

---

## 🎚️ Audiophile Dynamic Range (DR) Scoring & Loudness Metrics

AcoustiSinc computes industry-standard audiophile dynamic range and broadcast loudness metrics in strict 64-bit double precision:

### TT Dynamic Range Meter (Official PMF Standard)
Standardized by the [Pleasurize Music Foundation](https://dr.loudness-war.info) and Foobar2000 (`foo_dynamic_range`), the TT DR meter evaluates crest dynamics across 3-second blocks:

$$\text{DR Score} = \text{Peak}_{\text{dBFS}} - \text{RMS}_{\text{Top 20\% (dBFS)}}$$

#### Audiophile DR Rating Scale:
| DR Score | Rating | Color Code | Characteristics |
| :--- | :--- | :--- | :--- |
| **DR 14+** | **Pristine Dynamics** | 🟦 Cyan / Blue | Uncompressed audiophile masters (Classical, Acoustic Jazz, early vinyl/CD transfers). |
| **DR 10 – 13** | **Good Dynamics** | 🟩 Green | Open, dynamic recordings with natural transients and high punch. |
| **DR 7 – 9** | **Moderate Compression** | 🟨 Yellow | Noticeable dynamic compression and limiting (typical commercial pop/rock masters). |
| **DR 1 – 6** | **Heavy Compression** | 🟥 Red | Severe brick-wall limiting / Loudness War hyper-compression. |

### Broadcast & Mastering Metrics
- **EBU R128 Loudness Range (LRA)**: Measures macro-dynamics (dynamic span in **LU / dB**) using dual-gated K-weighting filters (discarding silence below $-70\text{ LUFS}$ and relative gate at $-20\text{ LU}$).
- **Integrated Loudness ($\text{LUFS}$)**: ITU-R BS.1770-4 overall perceived loudness across the track.
- **Peak-to-RMS Crest Factor ($\text{dB}$)**: Total headroom between overall RMS power and true peak level.

---

## 📦 Installation & Requirements

### System Prerequisites
- **Python 3.10+**
- **OpenCL Runtime & GPU Driver** (e.g. AMD ROCm / Mesa OpenCL / NVIDIA CUDA OpenCL / Intel compute-runtime)
- Fast NVMe SSD storage for scratch memory mapping (recommended for multi-gigabyte album batches)

### Setup

```bash
# Clone repository
git clone https://github.com/amkrebet/acoustisinc.git
cd acoustisinc

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🛠️ Usage Guide

### 1. Running the Interactive Forensic Explorer (Web Application)

Launch the interactive web application server:

```bash
# Start browsing from a specific music directory
python server.py --root "/path/to/music" --port 8765

# Or start from the current working directory
python server.py
```

Open **`http://localhost:8765`** in your browser. Navigate directories using the path input bar, quick-jump buttons (`Start`, `Home`, `Root`), or the interactive sidebar tree. Click any track to run live DSP forensic analysis in under a second with interactive zooming, dB level inspection, color-coded DR badges, and lossless audio streaming.

#### Explorer Options
| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--root <path>` | *Optional* | `.` *(current directory)* | Initial directory path to display in the file browser. |
| `--port <number>` | *Optional* | `8765` | Local TCP port for the web interface. |
| `--host <ip>` | *Optional* | `0.0.0.0` | Network bind address (`0.0.0.0` allows LAN access, `127.0.0.1` for local-only). |

---

### 2. Running the GPU Sinc Upsampler (`upsampler.py`)

#### Basic Usage
Upsample a single track, an album folder, or an entire recursive music library with your preferred filter topology:

```bash
# 1. Minimum-Phase Apodizing (Recommended for Red Book CD 44.1k/48k - zero pre-ringing & removes ADC ringing)
python upsampler.py "/path/to/source_music" "/path/to/upsampled_music" --phase min --apodizing

# 2. Pure Minimum-Phase (Zero pre-ringing, full sinc bandwidth)
python upsampler.py "/path/to/source_music" "/path/to/upsampled_music" --phase min

# 3. Linear-Phase Apodizing (Symmetric linear phase, removes ADC ringing)
python upsampler.py "/path/to/source_music" "/path/to/upsampled_music" --phase linear --apodizing

# 4. Pure Linear-Phase Sinc (Default: symmetric phase, bit-perfect passband up to Nyquist)
python upsampler.py "/path/to/source_music"
```

#### Command-Line Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `source` | **Mandatory** | *(None)* | Path to a single audio track (`.flac`/`.wav`) or a root directory containing album folders. |
| `target` *(or `-o`, `--output-dir`)* | *Optional* | `<source>_upsampled_<topology>` | Destination root directory. Preserves the full source album subfolder hierarchy. |
| `--phase` | *Optional* | `linear` | Filter phase mode: `linear` (symmetric) or `min` (minimum phase / causal, zero pre-ringing). |
| `--apodizing`, `--apod` | *Optional* | `False` *(Off)* | Switch to enable Apodizing transition band to attenuate pre-existing studio ADC ringing. |
| `--dither` | *Optional* | `shibata` | 24-bit psychoacoustic noise shaping profile: `shibata`, `high_rate`, or `none`. |
| `--no-dither` | *Optional* | `False` | Flag to disable dither and noise shaping (raw truncation). |
| `--mqa` | *Optional* | `adaptive` | MQA processing mode: `adaptive` (companded high-fidelity unfold), `strip` (strip MQA payload and re-dither), `simple` (standard linear unfold), `ignore` (raw PCM). |
| `--tmp-dir` | *Optional* | `/tmp/upsample_scratch` | Custom fast NVMe scratch path for 64-bit memory-mapped buffers. |

---

## 🎼 MQA Processing Modes & Acoustic Trade-Offs

When an MQA-encoded track or MQA-CD rip is detected, AcoustiSinc offers four processing strategies tailored to different musical genres and audiophile preferences:

```bash
# 1. Adaptive Unfolding (Default - Best for Acoustic Jazz, Classical & Orchestral)
python upsampler.py "/path/to/mqa_music" --mqa adaptive

# 2. MQA Noise Stripping (Best for Pop, Rock, Electronic & Purist Playback)
python upsampler.py "/path/to/mqa_music" --mqa strip

# 3. Simple Linear Unfold (Raw historical MQA behavior)
python upsampler.py "/path/to/mqa_music" --mqa simple

# 4. Raw Ignore (Bit-for-bit uncleaned legacy PCM)
python upsampler.py "/path/to/mqa_music" --mqa ignore
```

### Trade-Off Comparison

| Mode | Soundstage & Air | Background Blackness | Distortion Level | Best Musical Fit |
| :--- | :--- | :--- | :--- | :--- |
| **`adaptive`** *(Default)* | ⭐⭐⭐⭐⭐ **Maximum** | ⭐⭐⭐⭐ **High** | ⭐⭐⭐⭐ **Low** | Acoustic recordings with genuine ultrasonic harmonics (2L, ECM, live acoustic). |
| **`strip`** | ⭐⭐⭐⭐ **Very Good** | ⭐⭐⭐⭐⭐ **Pitch Black** | ⭐⭐⭐⭐⭐ **Lowest** | Studio pop, rock, electronic, or questionable MQA provenance (eliminates LSB hash). |
| **`simple`** | ⭐⭐⭐⭐ **Good** | ⭐⭐ **Grainy / Hazy** | ⭐⭐ **Higher Noise** | Historical reference / raw MQA subband emulation (retains $-48\text{ dBFS}$ floor). |
| **`ignore`** | ⭐⭐⭐ **Standard** | ⭐⭐⭐ **Slight LSB Hash** | ⭐⭐⭐ **Moderate** | Exact legacy DAC playback without modification. |

#### Why Does MQA Add Noise Without Unfolding?
* **16-bit MQA-CDs**: Bit 16 is constantly overwritten with pseudo-random MQA packet data. On a standard non-MQA DAC, this toggles like **correlated high-frequency hash noise**, degrading effective SNR from $96\text{ dB}$ to $\approx 84\text{--}88\text{ dB}$.
* **24-bit MQA**: The lower 8 bits (bits 0–7) hold the compressed subband packet stream instead of physical audio dither.
* **The `--mqa strip` Solution**: Zero-masks the pseudo-random packet payload and applies **64-bit TPDF dither**, restoring a pure $24\text{bit}$ linear PCM container with a pitch-black $-115\text{ dBFS}$ to $-144\text{ dBFS}$ noise floor.

---

## 🔬 DSP Specifications & Quality Standards

| Parameter | Specification |
| :--- | :--- |
| **Computation Precision** | Strict IEEE 754 64-bit Double Precision (`float64` / `complex128`) |
| **Stopband Attenuation** | $>140\text{ dB}$ rejection |
| **Passband Ripple** | $< \pm 0.00001\text{ dB}$ ($0\text{ Hz} \to 20\text{ kHz}$) |
| **Intersample Headroom** | Guaranteed $\ge 0.3\text{ dBFS}$ margin with pre-scan gain normalization |
| **Headroom Healing** | Exact overshoot-calculated dynamic backoff on clipping retry |
| **Dither Resolution** | 24-bit TPDF with 4th-order Psychoacoustic Noise Shaping |
| **Dynamic Range Standards** | TT Dynamic Range Meter (PMF DR Score) & EBU R128 / ITU-R BS.1770-4 LRA |
| **FLAC Output** | Bit-perfect Level 5 ($0.625$) with complete Vorbis Comment & Picture replication |

---

## 📄 License

MIT License. See `LICENSE` for details.
