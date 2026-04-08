
# ZuCo Eye-Tracking & Memory Analysis (Python)

This project analyzes eye-tracking data from the **ZuCo 2.0 – task1-NR (Normal Reading)** condition
and correlates gaze features with memory/comprehension performance.

## Folder Structure Expected
```
data/
  ├── task1-NR/        # ZuCo task1-NR .mat files (sub-01.mat, ...)
  ├── task_materials/  # sentence / word metadata
  ├── answers/         # comprehension answers
```

## How to Run
```bash
pip install numpy pandas scipy matplotlib
python main.py
```

Outputs include fixation statistics, heatmaps, and correlation results.
