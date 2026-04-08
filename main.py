import os
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from utils import extract_fixations, load_answers
import mat73

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PARTICIPANTS = ["YAC", "YAG"]

DATA_DIR = "data/task1-NR/Matlab files"
ANSWERS_DIR = "data/answers"

# --------------------------------------------------
# Load eye-tracking data (fixations)
# --------------------------------------------------

all_fixations = []

for participant in PARTICIPANTS:
    mat_file = os.path.join(DATA_DIR, f"{participant}.mat")
    print(f"Loading eye-tracking data for {participant}...")

    mat = mat73.loadmat(mat_file)
    fix_df = extract_fixations(mat, participant)
    all_fixations.append(fix_df)

fixations = pd.concat(all_fixations, ignore_index=True)

# --------------------------------------------------
# Load memory / comprehension answers (NR only)
# --------------------------------------------------

answers = load_answers(ANSWERS_DIR, PARTICIPANTS)

# --------------------------------------------------
# Merge eye-tracking and memory data
# --------------------------------------------------

merged = fixations.merge(
    answers,
    on=["subject", "sentence_id"],
    how="inner"
)

# --------------------------------------------------
# Compute fixation-based features
# --------------------------------------------------

features = (
    merged
    .groupby(["subject", "sentence_id"])
    .agg(
        mean_fixation_duration=("duration", "mean"),
        memory_accuracy=("correct", "mean")
    )
    .reset_index()
)

# --------------------------------------------------
# Correlation analysis
# --------------------------------------------------

corr = features["mean_fixation_duration"].corr(
    features["memory_accuracy"]
)

print("\nCorrelation between mean fixation duration and memory accuracy:")
print(corr)

# --------------------------------------------------
# Gaze heat map (visual attention dynamics)
# --------------------------------------------------

plt.figure(figsize=(6, 5))
plt.hexbin(
    fixations["x"],
    fixations["y"],
    gridsize=40,
    cmap="hot"
)
plt.colorbar(label="Fixation Density")
plt.title("Gaze Heat Map (YAC + YAG, Normal Reading)")
plt.xlabel("X Gaze Position")
plt.ylabel("Y Gaze Position")
plt.tight_layout()
plt.show()