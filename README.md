# AcoustiSinc & Forensic Audio Lab

> **High-Performance GPU-Accelerated 64-Bit Sinc Audio Upsampler & Interactive Forensic Spectral Analyzer**

![AcoustiSinc Forensic Spectral Analysis](analyser_example.png)

---

## 🌟 Overview

**AcoustiSinc** is an audiophile-grade digital signal processing (DSP) suite engineered for mastering-grade audio upsampling and forensic spectral authentication.

It delivers:
1. **Mathematical Purity**: Strict **64-bit double precision (`float64` / `complex128` / OpenCL `double2`)** maintained end-to-end across all FFTs, filtering kernels, and noise-shaping algorithms.
2. **GPU Sinc Upsampling**: High-throughput OpenCL-accelerated band-limited Whittaker–Shannon Sinc interpolation supporting **Linear Phase**, **Minimum Phase**, and **Apodizing** impulse responses.
3. **Advanced Psychoacoustic Noise Shaping**: 4th-order multi-band noise-shaping curves (Shibata & High-Rate) moving dither energy safely into the high ultrasonic spectrum ($>35\text{ kHz}$).
4. **Forensic Cutoff & Fake Hi-Res Detection**: Automated acoustic analysis identifying upsampled CD masters, brick-wall filter cutoffs, zero-stuffing, filter phase signatures, and ultrasonic noise profiles.
5. **Audiophile Dynamic Range (DR) Scoring**: Official 64-bit Pleasurize Music Foundation (PMF) TT Dynamic Range Meter scoring alongside EBU R128 Loudness Range (LRA) and Integrated LUFS.
6. **Interactive Web Explorer**: A real-time, on-demand browser application providing dynamic library navigation, live DSP spectral analysis in under 1 second, interactive spectrograms with exact cursor dB level readouts, and in-browser lossless playback.

---

## 🚀 Key Features

### 1. Mastering-Grade Sinc Upsampler (`upsampler.py`)
- **Auto-Integer Ratio Scaling**: Upsamples $44.1\text{ kHz} \to 176.4\text{ kHz}$ ($4\times$) and $48\text{ kHz} \to 192\text{ kHz}$ ($4\times$) without fractional alias artifacts.
- **Phase Modes**:
  - `linear`: True symmetric linear-phase band-limited Sinc reconstruction.
  - `min` / `minimum`: Causal minimum-phase reconstruction eliminating all pre-ringing for crisp transient attack.
  - `apodizing`: Apodizing transition band to attenuate pre-existing ADC brick-wall ripple.
- **Intersample Headroom Management**: Pre-flight album gain scans with auto-healing to guarantee **zero intersample clipping** ($0\text{ dBFS}$ true peak compliance).
- **Metadata & Album Art Preservation**: Lossless bit-perfect tag copying, ReplayGain retention, and embedded cover art extraction/sanitization.
- **Strict FLAC Compression Level 5**: Consistent balanced lossless compression.

### 2. Interactive Real-Time Forensic Explorer (`server.py`)
- **On-Demand Real-Time Analysis**: Dynamic library navigation with live 64-bit DSP forensic analysis in $\sim 0.8\text{s}$.
- **Hi-Fi News Style Spectral Inspection**: Full-track Peak Hold and RMS noise floor traces.
- **Automated Forensics & Cutoff Detection**:
  - `[FAKE HI-RES / UPSAMPLED CD SOURCE]`: Detects sharp $>45\text{ dB}$ brick-wall cutoffs near $22.05\text{ kHz}$.
  - `[UPSAMPLED 48 kHz / 96 kHz SOURCE]`: Identifies $24\text{ kHz}$ and $48\text{ kHz}$ legacy digital master cutoffs.
  - `[NATIVE HI-RES MATERIAL]`: Verifies continuous acoustic harmonic extension into ultrasonic frequencies.
  - `[FILTER SIGNATURE]`: Measures transient asymmetry ratios to classify Linear Phase vs Minimum Phase filtering.
  - `[NOISE PROFILE]`: Identifies Psychoacoustic Noise Shaping rise ($+6\text{ dB} \to +20\text{ dB}$ HF rise), Flat TPDF dither, or DSD ultrasonic humps.
  - `[DYNAMIC RANGE]`: Official **TT Dynamic Range (DR Score e.g. DR12)**, **EBU R128 Loudness Range (LRA)**, and **Integrated LUFS**.
- **Interactive Heatmap Canvas**: Zoom/pan with mouse wheel and drag; cursor HUD reveals **exact Time (s), Frequency (kHz), and Level (dBFS)** anywhere on the canvas.
- **Built-in Lossless Audio Streaming**: Audition FLAC files directly in your browser while visually correlating audio transients with the spectrogram.

---

## 🎚️ Audiophile Dynamic Range (DR) Scoring & Loudness Metrics

AcoustiSinc computes industry-standard audiophile dynamic range and broadcast loudness metrics in strict 64-bit double precision:

### 1. TT Dynamic Range Meter (Official PMF Standard)
Standardized by the [Pleasurize Music Foundation](https://dr.loudness-war.info) and Foobar2000 (`foo_dynamic_range`), the TT DR meter evaluates crest dynamics across 3-second blocks:
$$\text{DR Score} = \text{Track Peak (dBFS)} - \text{Average RMS of Top 20\% Loudest Blocks (dBFS)}$$

#### Audiophile DR Rating Scale:
| DR Score | Rating | Color Code | Characteristics |
| :--- | :--- | :--- | :--- |
| **DR 14+** | **Pristine Dynamics** | 🟦 Cyan / Blue | Uncompressed audiophile masters (Classical, Acoustic Jazz, early vinyl/CD transfers). |
| **DR 10 – 13** | **Good Dynamics** | 🟩 Green | Open, dynamic recordings with natural transients and high punch. |
| **DR 7 – 9** | **Moderate Compression** | 🟨 Yellow | Noticeable dynamic compression and limiting (typical commercial pop/rock masters). |
| **DR 1 – 6** | **Heavy Compression** | 🟥 Red | Severe brick-wall limiting / Loudness War hyper-compression. |

### 2. Broadcast & Mastering Metrics
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
| `--tmp-dir` | *Optional* | `/tmp/upsample_scratch` | Custom fast NVMe scratch path for 64-bit memory-mapped buffers. |

> [!NOTE]
> **Batch Processing & Initial JIT Warm-Up**:
> When processing your first track in a batch, execution may take an extra 1–3 seconds while OpenCL JIT-compiles the double-precision GPU sinc kernels, PyVkFFT builds and caches GPU plans, and Numba JIT-compiles the multi-core noise-shaping loops.
> All subsequent tracks and recursive album folders run at full native GPU acceleration (~3–5× faster).

---

## 🔬 DSP Specifications & Quality Standards

| Parameter | Specification |
| :--- | :--- |
| **Computation Precision** | Strict IEEE 754 64-bit Double Precision (`float64` / `complex128`) |
| **Stopband Attenuation** | $>140\text{ dB}$ rejection |
| **Passband Ripple** | $< \pm 0.00001\text{ dB}$ ($0\text{ Hz} \to 20\text{ kHz}$) |
| **Intersample Headroom** | Guaranteed $\ge 0.3\text{ dBFS}$ margin with pre-scan gain normalization |
| **Dither Resolution** | 24-bit TPDF with 4th-order Psychoacoustic Noise Shaping |
| **Dynamic Range Standards** | TT Dynamic Range Meter (PMF DR Score) & EBU R128 / ITU-R BS.1770-4 LRA |
| **FLAC Output** | Bit-perfect Level 5 ($0.625$) with complete Vorbis Comment & Picture replication |

---

## 📄 License

MIT License. See `LICENSE` for details.
