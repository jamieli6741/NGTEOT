### 2025-03-31
#### Refactor
- Modularized tracking pipeline into multiple files (utils/, tracking/)

- Extracted:
  - video_utils.py: frame loading & saving 
  - visualization.py: metrics chart & CSV output 
  - evaluation.py: scoring logic & evaluator class

#### Improvements
- Added tqdm progress bars to frame loops
- Refactored composite score calculation with metric weighting

#### Todo
- Replace hardcoded tracker dict with dynamic factory from config
- Add config.yaml for all hyperparameters (video, tracker, metrics)
