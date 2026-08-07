# Silentia

**End-to-end assistive communication pipeline for the deaf and hard-of-hearing community.**

Silentia fuses **computer vision, sign language recognition, large language models, and speech synthesis** into a single orchestrated pipeline that converts sign-language gestures captured from a video or webcam into natural, grammatically-correct, gender-matched spoken audio — in real time.

> Sign in. We speak for you.

---

## Overview

Silentia addresses a fundamental communication barrier: the gap between sign-language users and those who do not understand sign. The system watches a person signing, recognizes the gestures, repairs the inherently telegraphic structure of sign language with an LLM, and voices the result with a gender-matched synthesized voice.

The project is organized as **four independent components** coordinated by a lightweight **orchestrator**. Every component runs as an isolated subprocess with its own virtual environment, so components can be developed, versioned, and replaced independently.

```
                         ┌──────────────────────┐
                         │    Orchestrator      │
                         │  (config.yaml + CLI) │
                         └──────┬───────┬───────┘
                                │       │
                 ┌──────────────▼───┐ ┌─▼───────────────┐
                 │  Vision (GPU)    │ │  SLR (CPU)      │
                 │  gender label    │ │  sign → text    │
                 └────────┬─────────┘ └─┬───────────────┘
                          │             │
                          │    ┌────────▼────────┐
                          │    │  LLM            │
                          │    │  text repair    │
                          │    └────────┬────────┘
                          │             │
                 ┌────────▼─────────────▼──┐
                 │  TTS                    │
                 │  gender-matched speech  │
                 └─────────────────────────┘
```

## Pipeline architecture

The orchestrator executes the pipeline in three phases:

| Phase | Stage | Mode | Description |
|-------|-------|------|-------------|
| **1** | Vision | Parallel | Frame-gated face detection + gender classification (`male` / `female` / `no_face`) |
| **1** | SLR | Parallel | Sign-language recognition over the video → raw sign text sequence |
| **2** | LLM | Sequential | Grammar/vocabulary repair of the telegraphic sign text → natural spoken sentence |
| **3** | TTS | Sequential | Speech synthesis using a voice matched to the detected gender |

- **File mode** — process a pre-recorded video.
- **Webcam mode** — capture a fixed-duration clip live from a webcam, then run the pipeline.

---

## Components

### 🎥 Vision Component
Real-time **gender classification** from video using a ViT-B/16 classifier head.

- Face detection with configurable confidence thresholds and multi-face strategies
- Frame gating to skip non-informative frames
- Temporal smoothing for stable predictions
- GPU/CUDA and CPU device support

**Stack:** PyTorch · OpenCV · ViT-B/16

### 🤟 SLR Component — Sign Language Recognition
Classifies **100 common ASL signs** (`book`, `drink`, `go`, `mother`, `help`, …) into text sequences.

- Two model families: LSTM and BiLSTM sequence classifiers
- `single` (isolated sign) and `continuous` (sequence of signs) inference modes
- Checkpoint-based inference with label maps and top-k outputs
- CPU-first design

**Stack:** PyTorch · OpenCV · (Bi)LSTM sequence models

### 🧠 LLM Component — Text Repair
Repairs the missing articles, prepositions, and inflections that are inherent to sign language, producing natural, TTS-ready spoken English.

- **Multi-provider routing with failover**: Groq → OpenRouter → HuggingFace
- Deterministic prompt system with domain instructions (medical, legal, casual) and glossary substitution
- Input guards, response caching, retries, and markdown/quote stripping
- `standalone` and `system` modes; development / production environments

**Stack:** Python · Groq / OpenRouter / HuggingFace APIs · Pydantic

### 🔊 TTS Component — Text-to-Speech
Converts the repaired text into audio using a **gender-matched voice**.

- Provider priority: **Edge TTS → Coqui TTS**
- Speaker selection based on the gender predicted by the Vision component
- Adjustable speech rate and `file` / `play` output modes

**Stack:** Edge-TTS · Coqui TTS (VITS)

### 🎛️ Orchestrator
The control plane that wires the four components together.

- Subprocess isolation — each component uses its own venv; **zero code changes** to components
- Parallel Phase 1 (Vision + SLR) via thread pool, sequential Phases 2–3
- Per-stage timeouts, structured `PipelineResult` schemas, session-based temp artifacts and cleanup
- `config.yaml`-driven component paths, venvs, models, and timeouts

---

## Getting started

### Prerequisites

- Python 3.9+ (3.11 recommended)
- Each component installs into its **own** virtual environment (see each component's `requirements.txt`)
- API keys for LLM providers (Groq / OpenRouter / HuggingFace) — see `LLM component/.env.example`

### 1. Clone & install

```bash
git clone https://github.com/<your-org>/silentia.git
cd silentia

# Example — create and populate each component's venv
python -m venv "Vision component/venv" && "Vision component/venv/bin/pip" install -r "Vision component/requirements.txt"
python -m venv "SLR component/.venv"   && "SLR component/.venv/bin/pip"   install -r "SLR component/requirements.txt"
python -m venv "LLM component/.venv"   && "LLM component/.venv/bin/pip"   install -r "LLM component/requirements.txt"
python -m venv "TTS component/.venv"   && "TTS component/.venv/bin/pip"   install -r "TTS component/requirements.txt"

# Orchestrator deps (cv2, pyyaml)
python -m venv orchestrator/.venv && orchestrator/.venv/bin/pip install -r orchestrator/requirements.txt
```

### 2. Configure

Edit `orchestrator/config.yaml` so each component points at its local root, venv, checkpoint, and model paths, and drop model weights into the locations the components expect (see each component's `README`/docs).

### 3. Run the pipeline

```bash
# Process a pre-recorded video
python orchestrator/orchestrate.py --video path/to/sign.mp4

# Capture 10 seconds from webcam 0 and process live
python orchestrator/orchestrate.py --webcam --duration 10

# Continuous-sign mode, speech rate 1.2, play audio instead of saving
python orchestrator/orchestrate.py --video path/to/sign.mp4 \
  --slr-mode continuous --tts-mode play --tts-speed 1.2
```

**CLI reference**

| Flag | Default | Description |
|------|---------|-------------|
| `--video PATH` | — | Input video file (mutually exclusive with `--webcam`) |
| `--webcam [ID]` | — | Capture from webcam then process |
| `--slr-mode` | `single` | `single` or `continuous` SLR mode |
| `--duration` | `10.0` | Webcam capture duration (seconds) |
| `--tts-mode` | `file` | `file` (save WAV) or `play` (play audio) |
| `--tts-speed` | `1.0` | Speech rate, `0.5`–`2.0` |
| `--config` | `config.yaml` | Orchestrator config path |
| `--cleanup` | off | Delete session temp files after the run |

A structured JSON result is printed on completion:

```json
{
  "success": true,
  "vision": { "status": "completed", "result": { "label": "female", "confidence": 0.98 } },
  "slr":    { "status": "completed", "result": { "text": "mother like cook food", "top_k": [...] } },
  "llm":    { "status": "completed", "result": { "original_text": "mother like cook food", "corrected_text": "My mother likes to cook food." } },
  "tts":    { "status": "completed", "result": { "audio_path": ".../silentia_<session>.wav" } }
}
```

---

## Project structure

```
├── orchestrator/            # Pipeline control plane
│   ├── orchestrate.py       #  3-phase orchestrator (subprocess-based)
│   ├── schemas.py           #  PipelineInput / StageResult / PipelineResult
│   ├── config.yaml          #  component paths, venvs, models, timeouts
│   └── adapters/            #  per-component CLI adapters
├── "Vision component"/      # Gender detection (ViT-B/16 + OpenCV)
├── "SLR component"/         # Sign language recognition (BiLSTM, 100 signs)
├── "LLM component"/         # Sign-text grammar repair (multi-provider)
└── "TTS component"/         # Gender-matched speech synthesis (Edge/Coqui)
```

> **Note:** directories contain spaces (e.g. `Vision component/`). Quote paths in shell commands, as shown above.

Each component is self-contained: its own source, tests, demos, configs, block diagrams, and documentation PDFs.

---

## Project docs

| Component | Block diagram | Documentation |
|-----------|---------------|---------------|
| Vision | `Vision_Component_Documentation.pdf` | `Vision component/docs/` |
| SLR | `SLR_Block_Diagram.pdf` | `SLR_Component_Documentation.pdf` |
| LLM | `LLM_component_Block_diagram.pdf` | `LLM component/` |
| TTS | `TTS_component_Block_Diagram.pdf` | `TTS component/` |

---

## Testing

Each component ships its own `tests/` suite:

```bash
"Vision component/venv/bin/pytest"  "Vision component/tests"
"SLR component/.venv/bin/pytest"    "SLR component/tests"
"LLM component/.venv/bin/pytest"    "LLM component/tests"
"TTS component/.venv/bin/pytest"    "TTS component/tests"
```

---

## Roadmap

- [ ] Continuous-sign sentence-level decoding with beam search
- [ ] Expanded vocabulary beyond 100 signs
- [ ] Real-time (streaming) inference across the full pipeline
- [ ] Lip-sync / avatar output (visual feedback for sign-language users)
- [ ] Dockerized end-to-end deployment

---

## License

Distributed under the [MIT License](LICENSE).

## Acknowledgments

Built on the shoulders of PyTorch, OpenCV, HuggingFace, Groq, OpenRouter, Edge-TTS, and Coqui TTS.
