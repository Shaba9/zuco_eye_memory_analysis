import os
import pandas as pd
from scipy.io import loadmat


def extract_fixations(mat, subject_id):
    """
    Extract fixation-level eye-tracking data
    from a ZuCo task1-NR participant .mat file.
    """

    data = mat["data"]
    rows = []

    for sentence in data[0]:
        sentence_id = int(sentence["sentence_id"][0][0])
        fixations = sentence["fixations"][0]

        for f in fixations:
            rows.append({
                "subject": subject_id,
                "sentence_id": sentence_id,
                "x": float(f["x"][0][0]),
                "y": float(f["y"][0][0]),
                "duration": float(f["duration"][0][0])
            })

    return pd.DataFrame(rows)


def load_answers(path, participants):
    """
    Load Normal Reading (NR) comprehension answers
    for the selected participants.
    """

    all_answers = []

    for participant in participants:
        nr_file = os.path.join(path, participant, "NR.mat")
        print(f"Loading memory answers for {participant}...")

        mat = loadmat(nr_file)
        answers = mat["answers"]

        for row in answers:
            all_answers.append({
                "subject": participant,
                "sentence_id": int(row[1]),
                "correct": int(row[2] == 1)
            })

    return pd.DataFrame(all_answers)