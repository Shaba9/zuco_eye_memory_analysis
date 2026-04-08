
import os
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from utils import extract_fixations, load_answers

DATA_DIR = "data/task1-NR"
ANSWERS_DIR = "data/answers"

all_fixations = []

for file in os.listdir(DATA_DIR):
    if file.endswith('.mat'):
        subject_id = file.replace('.mat', '')
        mat = loadmat(os.path.join(DATA_DIR, file))
        fix_df = extract_fixations(mat, subject_id)
        all_fixations.append(fix_df)

fixations = pd.concat(all_fixations, ignore_index=True)
answers = load_answers(ANSWERS_DIR)

merged = fixations.merge(answers, on=["subject", "sentence_id"], how="inner")

features = merged.groupby(["subject", "sentence_id"]).agg({
    "duration": "mean",
    "correct": "mean"
}).reset_index()

corr = features["duration"].corr(features["correct"])
print("Correlation between mean fixation duration and memory accuracy:", corr)

plt.hexbin(fixations["x"], fixations["y"], gridsize=40, cmap='hot')
plt.colorbar()
plt.title("Gaze Heatmap (All Subjects)")
plt.show()
