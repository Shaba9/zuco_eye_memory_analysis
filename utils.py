
import os
import pandas as pd
from scipy.io import loadmat


def extract_fixations(mat, subject_id):
    data = mat['data']
    rows = []

    for sentence in data[0]:
        sentence_id = int(sentence['sentence_id'][0][0])
        fixations = sentence['fixations'][0]

        for f in fixations:
            rows.append({
                'subject': subject_id,
                'sentence_id': sentence_id,
                'x': float(f['x'][0][0]),
                'y': float(f['y'][0][0]),
                'duration': float(f['duration'][0][0])
            })

    return pd.DataFrame(rows)


def load_answers(path):
    all_ans = []

    for file in os.listdir(path):
        if file.endswith('.mat'):
            mat = loadmat(os.path.join(path, file))
            subject = file.split('_')[0]

            answers = mat['answers']
            for row in answers:
                all_ans.append({
                    'subject': subject,
                    'sentence_id': int(row[1]),
                    'correct': int(row[2] == 1)
                })

    return pd.DataFrame(all_ans)
