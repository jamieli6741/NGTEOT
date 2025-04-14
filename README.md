# NGTEOT

**No Ground Truth Evaluation for Object Tracking (NGTEOT)** is a Python-based toolkit for evaluating object tracking performance in the absence of ground truth labels. This project supports comparing multiple trackers on real or synthetic videos and provides several no-GT evaluation metrics.

## Features

- Supports DeepSORT and filter-based trackers
- Multiple ground-truth-free evaluation metrics:
  - Appearance Similarity
  - Bounding Box Stability
  - Box Size Consistency
  - Drift Detection
- Video preprocessing and visualization tools
- Modular architecture for easy extension

## Project Structure

```
NGTEOT-main/
│
├── ECE613.py                    # Main script entry for evaluation
├── tracking/
│   └── deepsort_wrapper.py     # DeepSORT tracker wrapper
│
├── utils/
│   ├── evaluation.py           # Evaluation logic without GT
│   ├── video_utils.py          # Frame extraction and preprocessing
│   └── visualization.py        # Visualization of results
│
├── mars-small128.pb            # Pre-trained ReID model for DeepSORT
├── deepsort_req.txt            # Dependency requirements for DeepSORT
├── README.md                   # This file
└── CHANGELOG.md                # Project changelog
```

## Getting Started

### 1. Install Dependencies

It is recommended to use a dedicated conda environment:

```bash
conda create -n ngteot python=3.9
conda activate ngteot
pip install -r deepsort_req.txt
```

### 2. Run the Main Program

**Run with a single tracker**

```bash
python ECE613.py --video path_to_video.mp4 --tracker deepsort
```

Optional arguments:
- `--tracker`: Tracker type to use (options: `csrt`, `kcf`, `boosting`, `mil`, `tld`, `medianflow`, `mosse`, `deepsort`)
- `--compress`: Compression rate for frame sampling (default: 5)
- `--compute`: Platform to run on (`cpu` or `cuda`)
- `--deepsort_model`: Path to DeepSORT model (default: `mars-small128.pb`)

**Compare performance of multiple trackers**

```bash
python ECE613.py --video path_to_video.mp4 --metrics --trackers csrt mosse deepsort
```

This will evaluate all specified trackers on the same video and produce comparative performance metrics.

## Output

The program will generate:
- Result video with tracking and segmentation overlays
- Visual plots comparing different trackers' evaluation metrics

