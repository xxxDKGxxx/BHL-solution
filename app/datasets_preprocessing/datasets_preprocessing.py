import os
import json
import csv
import glob
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

def load_json_data(dir_path: str):
    if not os.path.exists(dir_path):
        return []

    data = []
    for in_filename in glob.glob(os.path.join(dir_path, '*.jsonl')):
        with open(in_filename, 'r', encoding='utf-8') as infile:
            for line in infile:
                data.append(json.loads(line))
    return data

def make_pipeline(category: str) -> Pipeline:
    return Pipeline(steps=[
        ("json_modifying", JsonModifying(category)),
        ("csv_writing", CSVWriting(category))
    ])

class JsonModifying(BaseEstimator, TransformerMixin):
    def __init__(self, category: str):
        self.category = 'negative' if category not in ['bio', 'math', 'code'] else category

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        csv_rows = []

        for record in X:
            texts = record.get('texts', [])
            if not texts:
                continue

            question = texts[0]

            tags = record.get('tags', [])
            tags_str = ','.join(tags) if isinstance(tags, list) else str(tags)

            math = bio = code = 0
            if self.category == 'math':
                math = 1
            elif self.category == 'bio':
                bio = 1
            elif self.category == 'code':
                code = 1

            csv_rows.append([question, tags_str, math, bio, code])

        return csv_rows


class CSVWriting(BaseEstimator, TransformerMixin):
    def __init__(self, category: str):
        self.category = 'negative' if category not in ['bio', 'math', 'code'] else category

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if not (os.path.exists('csv_question_files')):
            os.mkdir('csv_question_files')

        output_filename = os.path.join('csv_question_files', self.category + '.csv')

        with open(output_filename, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['question', 'tags_str', 'math', 'bio', 'code'])
            for record in X:
                writer.writerow(record)
