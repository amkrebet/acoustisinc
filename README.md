# AcoustiSinc & Forensic Audio Lab

> **High-Performance GPU-Accelerated 64-Bit Sinc Audio Upsampler & Interactive Forensic Spectral Analyzer**

![AcoustiSinc Forensic Spectral Analysis](acoustisinc_forensic_analysis_v2.png)

---

## 🌟 Overview

**AcoustiSinc** is an audiophile-grade digital signal processing (DSP) suite engineered for mastering-grade audio upsampling and forensic spectral authentication.

* **Mathematical Purity**: Strict **64-bit double precision (`float64` / `complex128` / OpenCL `double2`)** across all FFTs, filtering kernels, and noise-shaping algorithms.
* **GPU Sinc Upsampling**: High-throughput OpenCL-accelerated band-limited Whittaker–Shannon Sinc interpolation supporting **Linear Phase**, **Minimum Phase**, and **Apodizing** impulse responses.
* **Psychoacoustic Noise Shaping**: Multi-band 5th-order Shibata and High-Rate noise shaping pushing dither energy safely into the high ultrasonic spectrum ($>35\text{ kHz}$).
* **Dynamic Headroom Auto-Healing**: Pre-flight album headroom scanning with exact overshoot-calculated auto-healing guaranteeing zero intersample clipping.
* **Forensic Authentication Engine**: Automated analysis identifying upsampled CD masters, brickwall filter cutoffs, zero-stuffing, and ultrasonic noise profiles.
* **Audiophile Dynamic Range (DR) Scoring**: Official 64-bit Pleasurize Music Foundation (PMF) TT Dynamic Range Meter scoring alongside EBU R128 Loudness Range (LRA) and Integrated LUFS.
* **Interactive Web Explorer & Tile Studio**: A real-time browser application providing dynamic library navigation, live sub-second forensic spectral analysis, interactive spectrogram HUD with exact cursor dB level readouts, in-browser lossless streaming, and an interactive **Tile-Oriented DSP Studio** for visual recipe design.
* **Decoupled Standalone Batch Engine**: Complete independence between the web UI and the high-throughput CLI batch upsampler (`upsampler.py`), allowing headless execution across thousands of albums via cron, systemd, or detached terminal sessions.

---

## 🚀 Quickstart Guide

### 1. Launch the Interactive Web Explorer & Tile Studio (`server.py`)

Explore your library, inspect spectral provenance, and interactively configure DSP upsampling recipes:

```bash
# Start browsing from your music collection
python server.py --root "/Music/Hi-Res" --port 8765
```

Open **`http://localhost:8765`** in your browser.

* **Tile-Oriented DSP Studio**: Modular tactile cards for Target Rate multiplier, Minimum Phase vs Linear Phase A/B comparison, logarithmic cutoff frequency slider ($15\text{ kHz} \leftrightarrow 48\text{ kHz}$), quick snap chips (`20.7k ADC`, `21.5k Alias`, `22.05k CD Std`, `24k`, `44.1k`), transition steepness, 24-bit Shibata noise shaping, and MQA payload policies.
* **Instant Recommendation Controls**: Dedicated `✨ [R] Apply Recommended Recipe` banner button and global `[R]` keyboard shortcut.
* **Live Forensic Badges**: Instantly tags tracks as `Native Hi-Res Master`, `Upsampled from 44.1k`, `Leaky SRC Mirror`, `MQA Studio`, or `Fake 24-bit (Zero-Padded)`.
* **Interactive Spectrogram HUD**: Sub-second Blackman-Harris STFT analysis with real-time cursor frequency and dBFS crosshair readouts.
* **Real-Time Telemetry Streaming**: Live progress percentage, per-track stage indicators, and scrolling console logs.
* **In-Browser Lossless Streaming**: Stream high-resolution FLAC files directly to your browser or external USB DAC.

---

### 2. Run the GPU Sinc Upsampler (`upsampler.py`)

Upsample individual tracks, album folders, or entire library collections as a **standalone background batch job** or interactive CLI session:

```bash
# 1. Automated Forensic Batch (Audit & auto-apply optimal recipes across all albums, skipping existing)
python upsampler.py "/Music/Hi-Res" "/Music/Hi-Res_Upsampled" --use-recommended=auto --overwrite=off

# 2. Interactive Forensic Batch (Audit Track 1 and prompt for per-track or album-wide recipes)
python upsampler.py "/Music/Hi-Res/Album" --use-recommended=ask

# 3. Minimum-Phase Apodizing with Custom Cutoff (e.g. 20.7 kHz legacy ADC ringing cleanup)
python upsampler.py "/Music/Hi-Res/Album" --cutoff 20700 --phase min --dither shibata

# 4. Pure Minimum-Phase (Zero pre-ringing, full sinc Nyquist bandwidth)
python upsampler.py "/Music/Hi-Res/Album" --phase min

# 5. Pure Linear-Phase Sinc (Default: strict 0° linear phase, bit-perfect passband)
python upsampler.py "/Music/Hi-Res/Album"
```

---

## 🛠️ CLI Options Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `source` | **Mandatory** | *(None)* | Path to a single audio track (`.flac`/`.wav`) or root directory containing album folders. |
| `target` *(or `-o`, `--output-dir`)* | *Optional* | `<source>_upsampled_<topology>` | Target destination directory. Multi-tier safety guards strictly prevent in-place overwrites. |
| `--use-recommended`, `--use-rec` | *Optional* | `none` | Audits audio provenance and resolves DSP recipe: `auto` (silent apply), `ask` (interactive prompt), or `none`. |
| `--overwrite`, `-f`, `--force` | *Optional* | `off` | Target output overwrite policy: `on` (silent overwrite), `off` (skip existing files), or `ask` (interactive prompt). |
| `--cutoff`, `--apodize` | *Optional* | `None` | Custom low-pass reconstruction filter cutoff frequency in Hz (e.g. `20700`, `21500`, `22050`, `44100`). |
| `--steep` | *Optional* | `False` | Uses a sharp transition band (500 Hz knee) instead of the standard 2 kHz cosine taper. |
| `--phase` | *Optional* | `linear` | Filter phase mode: `linear` (symmetric) or `min` (causal, 0.00% pre-ringing). |
| `--apodizing`, `--apod` | *Optional* | `False` | Enables raised-cosine apodizing transition band to attenuate pre-existing studio ADC ringing. |
| `--dither` | *Optional* | `shibata` | 24-bit psychoacoustic noise shaping profile: `shibata`, `high_rate`, or `none`. |
| `--no-dither` | *Optional* | `False` | Disables dither and noise shaping (raw 64-bit float truncation). |
| `--mqa` | *Optional* | `adaptive` | MQA processing: `adaptive` (companded high-fidelity unfold), `strip` (strip LSB hash and re-dither), `simple`, `ignore` (raw PCM). |
| `--tmp-dir` | *Optional* | `/tmp/upsample_scratch` | Fast NVMe scratch directory for 64-bit memory-mapped buffers. |

---

### 📊 Comparative Before & After Reports
Every upsampling run automatically generates self-contained comparative reports written directly to the target folder:
* **`ALBUM_REPORT.html`** / **`UPSAMPLING_REPORT.html`**: Interactive HTML5 report with side-by-side spectrograms, spectral energy distribution curves, and TT Dynamic Range meters.
* **`ALBUM_REPORT.md`** / **`UPSAMPLING_REPORT.md`**: Markdown summary table with track-by-track bit-depth, peak levels, and LUFS mastering receipts.

---

## 📚 Technical Deep-Dive Guides

For in-depth mathematical derivations, filter diagrams, and acoustic trade-off analyses, consult the dedicated guides:

| Document | Key Topics Covered |
| :--- | :--- |
| 🔬 **[Filter Topology & Psychoacoustics](docs/FILTER_TOPOLOGY_AND_PSYCHOACOUSTICS.md)** | Linear vs Minimum Phase, Standard vs Apodizing, time-domain impulse responses, pre-ringing elimination, and forward/backward auditory temporal masking. |
| 🛡️ **[Intersample Headroom & Dynamic Range](docs/HEADROOM_AND_DYNAMIC_RANGE.md)** | True-peak overshoots during sinc reconstruction, 2-stage adaptive crest pre-scanning, single-pass auto-healing math, and PMF TT Dynamic Range / EBU R128 scoring. |
| 🎼 **[MQA Forensics, Unfolding & Noise Stripping](docs/MQA_PROCESSING_AND_ACOUSTICS.md)** | 36-bit sync word bit-plane forensics, LSB hash noise characteristics, and comparative trade-offs between `adaptive`, `strip`, `simple`, and `ignore` modes. |
| 📊 **[Library Forensic Provenance Audit Guide](docs/FORENSIC_AUDIO_AUDIT_GUIDE.md)** | Real-world album forensics, cutoff detection, fake hi-res identification, apodization recipes for bit-depth expansion, and integer decimation strategies. |

---

## 🔬 DSP Specifications & Quality Standards

| Parameter | Specification |
| :--- | :--- |
| **Computation Precision** | Strict IEEE 754 64-bit Double Precision (`float64` / `complex128` / OpenCL `double2`) |
| **Stopband Attenuation** | $>140\text{ dB}$ rejection |
| **Passband Ripple** | $< \pm 0.00001\text{ dB}$ ($0\text{ Hz} \to 20\text{ kHz}$) |
| **Intersample Headroom** | Guaranteed $\ge 0.3\text{ dBFS}$ margin with pre-scan gain normalization |
| **Headroom Healing** | Exact overshoot-calculated dynamic backoff on clipping retry |
| **Dither Resolution** | 24-bit TPDF with 5th-order Shibata Psychoacoustic Noise Shaping |
| **Dynamic Range Standards** | TT Dynamic Range Meter (PMF DR Score) & EBU R128 / ITU-R BS.1770-4 LRA |
| **FLAC Output** | Bit-perfect Level 5 ($0.625$) with complete Vorbis Comment & Picture replication |

---

## 📦 Installation & Requirements

### System Prerequisites
- **Python 3.10+**
- **OpenCL Runtime & GPU Driver** (AMD ROCm / Mesa OpenCL / NVIDIA CUDA OpenCL / Intel compute-runtime)
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

## 📄 License

MIT License. See `LICENSE` for details.
