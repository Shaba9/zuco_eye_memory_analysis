import os
import pandas as pd
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

OUTPUT_FILE = "analysis_output.txt"

# --------------------------------------------------
# Load sentence-level eye-tracking data
# --------------------------------------------------

all_sentence_metrics = []

with open(OUTPUT_FILE, "w") as f:

    for participant in PARTICIPANTS:
        mat_file = os.path.join(DATA_DIR, f"{participant}.mat")
        f.write(f"Loading eye-tracking data for {participant}...\n")

        mat = mat73.loadmat(mat_file)
        sent_df = extract_sentence_metrics(mat, participant)
        all_sentence_metrics.append(sent_df)

    sentence_metrics = pd.concat(all_sentence_metrics, ignore_index=True)

    # --------------------------------------------------
    # Load memory / comprehension answers
    # --------------------------------------------------

    answers = load_answers(ANSWERS_DIR, PARTICIPANTS)

    # --------------------------------------------------
    # Aggregate memory accuracy per sentence
    # --------------------------------------------------

    memory_by_sentence = (
        answers
        .groupby(["subject", "sentence_id"])
        .agg(memory_accuracy=("correct", "mean"))
        .reset_index()
    )

    # --------------------------------------------------
    # Merge gaze and memory data
    # --------------------------------------------------

    merged = sentence_metrics.merge(
        memory_by_sentence,
        on=["subject", "sentence_id"],
        how="inner"
    )

    if DEBUG:
        f.write(f"\nDEBUG MODE ON — limiting to first {DEBUG_N} sentences\n")
        merged = merged.head(DEBUG_N)

    # --------------------------------------------------
    # Force scalar conversion
    # --------------------------------------------------

    merged["mean_fixation_duration"] = merged["mean_fixation_duration"].apply(
        lambda x: float(x[0]) if hasattr(x, "__len__") else float(x)
    )

    merged["memory_accuracy"] = merged["memory_accuracy"].astype(float)

    # --------------------------------------------------
    # Write diagnostics
    # --------------------------------------------------

    f.write("\nMerged dtypes:\n")
    f.write(str(merged.dtypes) + "\n")

    f.write("\nMerged preview:\n")
    f.write(merged.to_string(index=False) + "\n")

    # --------------------------------------------------
    # Correlation analysis
    # --------------------------------------------------

    corr = merged["mean_fixation_duration"].corr(
        merged["memory_accuracy"]
    )

    f.write("\nCorrelation between mean fixation duration and memory accuracy:\n")
    f.write(str(corr) + "\n")

print(f"\nAnalysis complete. Results written to {OUTPUT_FILE}")