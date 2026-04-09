import os
import pandas as pd
import matplotlib.pyplot as plt
import mat73

from utils import extract_sentence_metrics, load_answers

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEBUG = True          # 🔧 turn off when ready
DEBUG_N = 5           # 🔧 number of sentences to analyze in debug mode

PARTICIPANTS = ["YAC"]    # analyzing only YAC

DATA_DIR = "data/task1-NR/Matlab files"
ANSWERS_DIR = "data/answers"

# --------------------------------------------------
# Load sentence-level eye-tracking data
# --------------------------------------------------

all_sentence_metrics = []

for participant in PARTICIPANTS:
    mat_file = os.path.join(DATA_DIR, f"{participant}.mat")
    print(f"Loading eye-tracking data for {participant}...")

    mat = mat73.loadmat(mat_file)
    sent_df = extract_sentence_metrics(mat, participant)
    all_sentence_metrics.append(sent_df)

sentence_metrics = pd.concat(all_sentence_metrics, ignore_index=True)

# --------------------------------------------------
# Load memory / comprehension answers (NR only)
# --------------------------------------------------

answers = load_answers(ANSWERS_DIR, PARTICIPANTS)

# --------------------------------------------------
# Merge gaze and memory data
# --------------------------------------------------

merged = sentence_metrics.merge(
    answers,
    on=["subject", "sentence_id"],
    how="inner"
)

# --------------------------------------------------
# DEBUG: limit to first N rows
# --------------------------------------------------

if DEBUG:
    print(f"\nDEBUG MODE ON — limiting to first {DEBUG_N} sentences")
    merged = merged.head(DEBUG_N)

# --------------------------------------------------
# Force scalar conversion (MATLAB → Python safety)
# --------------------------------------------------

merged["mean_fixation_duration"] = merged["mean_fixation_duration"].apply(
    lambda x: float(x[0]) if hasattr(x, "__len__") else float(x)
)

merged["correct"] = merged["correct"].apply(
    lambda x: int(x[0]) if hasattr(x, "__len__") else int(x)
)

# --------------------------------------------------
# Sanity check (recommended during debugging)
# --------------------------------------------------

print("\nMerged dtypes:")
print(merged.dtypes)

print("\nMerged preview:")
print(merged)

# --------------------------------------------------
# Correlation analysis
# --------------------------------------------------

corr = merged["mean_fixation_duration"].corr(
    merged["correct"]
)

print("\nCorrelation between mean fixation duration and memory accuracy:")
print(corr)

# --------------------------------------------------
# Optional visualization (off by default in debug)
# --------------------------------------------------

if not DEBUG:
    plt.figure(figsize=(6, 5))
    plt.scatter(
        merged["mean_fixation_duration"],
        merged["correct"],
        alpha=0.5
    )
    plt.xlabel("Mean Fixation Duration (sentence-level)")
    plt.ylabel("Memory Accuracy")
    plt.title("Visual Attention vs Memory Performance (NR)")
    plt.tight_layout()
    plt.show()