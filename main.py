import os
import pandas as pd
import matplotlib.pyplot as plt
import mat73

from utils import extract_sentence_metrics, load_answers

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEBUG = True          # set to False for full analysis
DEBUG_N = 5           # only used when DEBUG=True

PARTICIPANTS = ["YAC"]

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
# Aggregate memory accuracy per sentence (Option A)
# --------------------------------------------------

memory_by_sentence = (
    answers
    .groupby(["subject", "sentence_id"])
    .agg(
        memory_accuracy=("correct", "mean")  # proportion correct
    )
    .reset_index()
)

# --------------------------------------------------
# Merge gaze and memory data (sentence-level)
# --------------------------------------------------

merged = sentence_metrics.merge(
    memory_by_sentence,
    on=["subject", "sentence_id"],
    how="inner"
)

# --------------------------------------------------
# DEBUG: limit rows for fast iteration
# --------------------------------------------------

if DEBUG:
    print(f"\nDEBUG MODE ON — limiting to first {DEBUG_N} sentences")
    merged = merged.head(DEBUG_N)

# --------------------------------------------------
# Force scalar conversion (MATLAB safety)
# --------------------------------------------------

merged["mean_fixation_duration"] = merged["mean_fixation_duration"].apply(
    lambda x: float(x[0]) if hasattr(x, "__len__") else float(x)
)

merged["memory_accuracy"] = merged["memory_accuracy"].astype(float)

# --------------------------------------------------
# Sanity checks
# --------------------------------------------------

print("\nMerged dtypes:")
print(merged.dtypes)

print("\nMerged preview:")
print(merged)

# --------------------------------------------------
# Correlation analysis (sentence-level)
# --------------------------------------------------

corr = merged["mean_fixation_duration"].corr(
    merged["memory_accuracy"]
)

print("\nCorrelation between mean fixation duration and memory accuracy:")
print(corr)

# --------------------------------------------------
# Optional visualization (off during DEBUG)
# --------------------------------------------------

if not DEBUG:
    plt.figure(figsize=(6, 5))
    plt.scatter(
        merged["mean_fixation_duration"],
        merged["memory_accuracy"],
        alpha=0.6
    )
    plt.xlabel("Mean Fixation Duration (sentence-level)")
    plt.ylabel("Memory Accuracy (proportion correct)")
    plt.title("Visual Attention vs Memory Performance (NR)")
    plt.tight_layout()
    plt.show()