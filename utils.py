import os
import pandas as pd
from scipy.io import loadmat


def extract_sentence_metrics(mat, subject_id):
    """
    Extract sentence-level eye-tracking metrics from ZuCo task1-NR data
    loaded via mat73.
    """

    subj = mat.get(subject_id, next(iter(mat.values())))

    mean_fix = subj["mean_t1"]

    df = pd.DataFrame({
        "subject": subject_id,
        "sentence_id": range(1, len(mean_fix) + 1),
        "mean_fixation_duration": mean_fix
    })

    return df


def load_answers(path, participants):
    """
    Load Normal Reading (NR) comprehension answers
    and force scalar values.
    """

    all_answers = []

    for participant in participants:
        nr_file = os.path.join(path, participant, "NR.mat")
        print(f"Loading memory answers for {participant}...")

        mat = loadmat(nr_file, squeeze_me=True)
        answers = mat["answers"]

        for row in answers:
            all_answers.append({
                "subject": participant,
                "sentence_id": int(row[1]),
                "correct": int(row[2].item() == 1)
            })

    return pd.DataFrame(all_answers)