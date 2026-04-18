# IMPACT-CYCLE: Claim-Level Cross-Task Verification for Human-Light Video Scene Graph Refinement


IMPACT-CYCLE reframes long-video understanding as structured semantic memory construction plus claim-level verification. Instead of trusting one opaque video-to-LLM pass, the pipeline builds a frame-grounded scene graph, decomposes it into typed claims, verifies those claims through single-turn VQA, multi-turn VQA, and caption-based audit, then fuses the evidence into a refined graph that is easier to inspect, correct, and reuse for downstream reasoning.


<!-- Teaser figure suggestion:
Replace this comment with a pipeline figure showing:
video -> initial scene graph -> typed claims -> single-turn / multi-turn / caption verification -> role-aware fusion -> refined graph + human review queue
-->

## Table of Contents

- [Paper](#paper)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Pretrained Checkpoints and External Services](#pretrained-checkpoints-and-external-services)
- [Quickstart](#quickstart)
- [Interactive GUI](#interactive-gui)
- [Core Paper Scripts](#core-paper-scripts)
- [Documentation](#documentation)
- [Repository Structure](#repository-structure)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)
- [Citation](#citation)

## Paper

- arXiv: TODO

### Summary

IMPACT-CYCLE contributes:

1. A structured semantic-memory pipeline for turning raw video into scene-centric, temporally organized, editable metadata.
2. A cross-verification mechanism that compares metadata-grounded reasoning against direct video prompting.
3. A human-light refinement loop that escalates only uncertain claims instead of requiring full re-annotation.

## Installation

### Tested Smoke-Test Environment

The validated smoke-test path below was run on:

- Linux
- Python `3.9.7`
- CPU-only
- no local Qwen checkpoint
- no SAM3 checkpoint

### Create an Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pillow transformers matplotlib pytest
export PYTHONPATH=.
```

## Dataset Preparation

### What the Code Expects

The evaluation path expects:

- a PVSG-style annotation JSON at `data/pvsg.json`
- source videos under `data/vidor/videos`
- per-video mask PNGs under `data/vidor/masks/<video_id>/`

Recommended local layout:

```text
data/
├── pvsg.json
└── vidor/
    ├── videos/
    │   ├── 1203_8316378691.mp4
    │   └── ...
    └── masks/
        ├── 1203_8316378691/
        │   ├── 0000.png
        │   ├── 0001.png
        │   └── ...
        └── ...
```

### Download Sources

- VidOR official page: <https://xdshang.github.io/docs/vidor.html>
- PVSG official page: <https://jingkangyang.com/PVSG/>
- PVSG Hugging Face mirror: <https://huggingface.co/datasets/Jingkang/PVSG>
- OpenPVSG reference repo: <https://github.com/LilyDaytoy/OpenPVSG>

### Important Dataset Note

The paper text says "VidOR", but the checked-in evaluation code expects a PVSG-style annotation bundle and segmentation masks. In practice, treat the public reproduction path as:

- VidOR/PVSG videos
- PVSG-style `pvsg.json`
- PVSG-style mask directories

If you only have raw VidOR annotations, the current evaluation scripts are not sufficient.

### Optional: Precompute Stage-1 Detections

If you want to avoid rerunning the grounding backend during every ablation, precompute detections first:

```bash
python tools/precompute_sam3_detections.py \
  --videos_dir data/vidor/videos \
  --gt_json data/pvsg.json \
  --pipeline_config configs/impact_sg_pipeline.json \
  --ontology configs/impact_sg_ontology.json \
  --output_dir outputs/sam3_precompute \
  --max_videos 240 \
  --max_frames_per_video 5
```

This creates detection JSON files under:

```text
outputs/sam3_precompute/
├── _frame_cache/
└── frames/
    └── <video_id>/
        └── 000123.json
```

## Pretrained Checkpoints and External Services

### 1. Stage-1 Grounding Backend

The default scene-graph pipeline config uses an external command backend:

- `configs/impact_sg_pipeline.json`
- `configs/sam3_external_command.linux.json`
- `configs/sam3_runtime.linux.json`

The checked-in Linux runtime currently points to private local paths for:

- `repo_root`
- the SAM3 checkpoint

These must be updated before public reproduction.

**TODO**
- Add the official SAM3 checkpoint source.
- Add the exact expected checkpoint filename and hash.
- Add a one-command download/setup script.

### 2. Verification Model

#### Paper Setting

The PDF says the verifier is GPT-4V.

#### Current Code Default

The checked-in config defaults to a local `Qwen2.5-VL-3B-Instruct` path:

- `configs/impact_cycle.json -> local_verifier.model_id`

Official Qwen model page:

- <https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct>

If you want to use the current local-verifier path, download the model and update `configs/impact_cycle.json` accordingly.

#### API-Based Verifiers

The repo also supports API-backed verification through environment variables:

```bash
export IMPACT_OPENAI_API_KEY=YOUR_KEY
export IMPACT_GEMINI_API_KEY=YOUR_KEY
```

Important:

- the checked-in config has API mode disabled by default
- the paper/code match for GPT-4V is still a release note, not a turnkey checked-in config

## Quickstart

This is the fastest reproducible path in the current repo. It uses:

- the bundled sample image `tmp/sam3_probe.png`
- the mock backend
- no GPU
- no external checkpoints
- no API keys

```bash
mkdir -p /tmp/impact_readme_smoke

python tools/build_scene_graph.py \
  --image_id sam3_probe \
  --image_path tmp/sam3_probe.png \
  --backend_provider mock \
  --out /tmp/impact_readme_smoke/graph.json

python tools/generate_vqa.py \
  --scene_graph /tmp/impact_readme_smoke/graph.json \
  --out /tmp/impact_readme_smoke/vqa.json
```

Expected output on the current snapshot:

```text
[OK] scene graph saved: /tmp/impact_readme_smoke/graph.json
[INFO] nodes=20 edges=54
[OK] VQA saved: /tmp/impact_readme_smoke/vqa.json
[INFO] single=64 multi=80
```

Artifacts:

- `/tmp/impact_readme_smoke/graph.json`
- `/tmp/impact_readme_smoke/vqa.json`

Optional evaluation helpers for paired prediction / ground-truth files:

```bash
python tools/evaluate_scene_graph.py \
  --pred pred_graph.json \
  --gt gt_graph.json \
  --out graph_metrics.json

python tools/evaluate_vqa.py \
  --pred pred_vqa.json \
  --gt gt_vqa.json \
  --out vqa_metrics.json
```

## Interactive GUI

The repo still includes the PyQt desktop app used for inspection, debugging, and human-in-the-loop study flows:

```bash
python app.py
```

To also persist operation logs alongside saved outputs:

```bash
python app.py --oplog
```

The current GUI exposes four paper-relevant task panels behind one shared video player:

- `Video Scene Graph`
- `Single-turn VQA`
- `Multi-turn VQA`
- `Video Captioning`

This is useful for qualitative demos and bundle generation, but the paper-facing quantitative path is driven by the scripts below.

## Core Paper Scripts

### 1. Cycle Verification on an Existing Bundle

Use [`run_cycle_verify.py`](run_cycle_verify.py) when you already have a `scene_graph_bundle.json` and want to run claim-level verification plus graph refinement on selected frames:

```bash
python run_cycle_verify.py \
  --bundle log/<run_name>/scene_graph_bundle.json \
  --provider mock \
  --rounds 1 \
  --output log/<run_name>/cycle_results.json
```

Notes:

- `--frames 0 4 8` restricts verification to selected 0-based frame indices.
- `--low-quota` disables multi-turn and caption probes.
- Replace `mock` with `qwen25_vl`, `gemini_api`, or `chatgpt_api` after configuring the corresponding local model or API credentials.

### 2. Dataset-Level Evaluation / Ablations

Use [`run_vidor_gt_frame_eval.py`](run_vidor_gt_frame_eval.py) for the paper-style evaluation path over PVSG/VidOR data:

```bash
python run_vidor_gt_frame_eval.py \
  --videos_dir data/vidor/videos \
  --masks_dir data/vidor/masks \
  --gt_json data/pvsg.json \
  --provider mock \
  --pipeline_config configs/impact_sg_pipeline.json \
  --config configs/impact_cycle.json \
  --ontology configs/impact_sg_ontology.json \
  --output_dir outputs/pvsg_eval \
  --max_videos 5 \
  --max_frames_per_video 5 \
  --write_csv
```

Common variants:

- Add `--load_detections_from outputs/sam3_precompute` to reuse precomputed Stage-1 detections.
- Add `--skip_cycle` for a build-only ablation.
- Add `--low_quota` to disable expensive probes and reduce token usage.
- Add `--ablation_suite paper5` to run the bundled paper ablation sweep in one pass.

## Documentation

- [docs/gui_operator_guide.md](docs/gui_operator_guide.md): current GUI launch, shared controls, and four-task operator flow
- [docs/file_formats.md](docs/file_formats.md): retained file-format, validation-log, and oplog notes
- [docs/archive/legacy_annotation_manual.md](docs/archive/legacy_annotation_manual.md): archived Action Segmentation / HOI / PSR notes that are not part of the paper path
- [docs/interactive_action_segmentation_method_guide_20260320.md](docs/interactive_action_segmentation_method_guide_20260320.md): method-level explanation of the legacy EAST online-learning workflow

## Repository Structure

```text
.
├── app.py                        # PyQt desktop application
├── submission.pdf                # current paper submission
├── run_cycle_verify.py           # run cycle refinement on an existing scene-graph bundle
├── run_vidor_gt_frame_eval.py    # dataset-level evaluation and ablation sweep
├── parse_metrics.py              # ad hoc log parser; not release-ready
├── configs/                      # pipeline, ontology, cycle, and runtime configs
├── core/impact_sg/               # core IMPACT-CYCLE logic
│   ├── claim_graph.py            # claim decomposition and probe generation
│   ├── cycle_pipeline.py         # verification loop orchestration
│   ├── belief_update.py          # claim fusion and graph revision
│   ├── arbitration.py            # human-query generation
│   ├── captioning.py             # caption-audit prompts and claim votes
│   ├── eval_cycle.py             # verification metrics
│   ├── eval_scene_graph.py       # graph metrics
│   ├── eval_vqa.py               # VQA metrics
│   ├── pvsg_reference.py         # PVSG/VidOR-style reference loading
│   └── visual_verifier/policy.py # role-aware vote weighting
├── tools/
│   ├── build_scene_graph.py      # single-image scene-graph build
│   ├── generate_vqa.py           # graph-grounded VQA generation
│   ├── evaluate_scene_graph.py   # graph-vs-GT evaluation
│   ├── evaluate_vqa.py           # VQA evaluation
│   ├── precompute_sam3_detections.py
│   └── runners/run_in_env.py     # helper for alternate conda env profiles
├── ui/                           # GUI widgets and task panels
├── tests/                        # unit tests
└── tmp/sam3_probe.png            # bundled smoke-test image
```

## License

This repository currently includes an Apache-2.0 license:

- [LICENSE](LICENSE)

Please also check the licenses and terms of use for:

- VidOR
- PVSG
- Qwen checkpoints
- any external API-backed verifier you use

## Acknowledgments

This codebase builds on:

- VidOR
- PVSG
- PyQt5 and OpenCV for the GUI
- Qwen and API-backed MLLM verifiers for multimodal verification

## Contact

Corresponding author:

- Kunyu Peng — `kunyu.peng@kit.edu`

## Citation

```bibtex
@inproceedings{kong2026impact_cycle,
  title     = {IMPACT-Cycle: Claim-Level Cross-Task Verification for Human-Light Video Scene Graph Refinement},
  author    = {Kong, Weitong and Wen, Di and Peng, Kunyu and Schneider, David and Zhong, Zeyun and Jaus, Alexander and Marinov, Zdravko and Wei, Jiale and Liu, Ruiping and Zheng, Junwei and Chen, Yufan and Qi, Lei and Stiefelhagen, Rainer},
  booktitle = {IEEE International Conference on Systems, Man, and Cybernetics (SMC)},
  year      = {2026},
  note      = {Under review}
}
```
