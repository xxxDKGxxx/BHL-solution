import os, re
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

def make_pipeline(output_filename: str = "fact_generative_questions.csv") -> Pipeline:
    return Pipeline(steps=[
        ("json_modifying", JsonModifying()),
        ("csv_writing", CSVWriting(output_filename))
    ])

class JsonModifying(BaseEstimator, TransformerMixin):
    fact_categories = {
        "open_qa",
        "closed_qa",
        "classification",
        "information_extraction",
        "summarization",
    }

    not_fact_categories = {
        "creative_writing",
    }

    fact_words = [
        r"^what\b",
        r"^who\b",
        r"^when\b",
        r"^where\b",
        r"^which\b",
        r"^how many\b",
        r"^how much\b",
        r"^what causes\b",
        r"^what caused\b",
        r"^why\b",
        r"^how does\b",
        r"^how did\b",
        r"^did\b",
        r"^is it true\b",
        r"^are\b",
        r"^does\b",
        r"^name\b",
        r"^tell me whether\b",
        r"^classify\b",
        r"^categorize\b",
        r"^identify\b",
        r"^given\b",
        r"^according to\b",
        r"^based on\b",
    ]

    not_fact_words = [
        r"\bwrite\b",
        r"\bcompose\b",
        r"\bcreate\b",
        r"\bgenerate\b",
        r"\binvent\b",
        r"\bimagine\b",
        r"\bcome up with\b",
        r"\bthink up\b",
        r"\bmake up\b",
        r"\breview\b",
        r"\bfavorite\b",
        r"\bbest\b",
        r"\bgreatest\b",
        r"\bgood\b",
        r"\bbetter\b",
        r"\bshould\b",
        r"\bwould you\b",
        r"\bdo you think\b",
        r"\bopinion\b",
        r"\brecommend\b",
        r"\brecommendation\b",
        r"\bideas\b",
        r"\btips\b",
        r"\bways to\b",
        r"\bhow can i\b",
        r"\bhow should i\b",
        r"\bwhat should i\b",
        r"\bwhere should i\b",
        r"\bplan\b",
    ]

    def fit(self, X, y=None):
        return self

    def if_matches_any(self, text: str, patterns: list[str]) -> bool:
        return any(re.search(pattern, text) for pattern in patterns)

    def is_fact_not_fact(self, question: str, category: str):
        q = question.lower().strip()

        if self.if_matches_any(q, self.not_fact_words):
            return 0

        if self.if_matches_any(q, self.fact_words):
            return 1

        if category in self.not_fact_categories:
            return 0

        if category in self.fact_categories:
            return 1

        return None

    def transform(self, X, y=None):
        csv_rows = []

        for record in X:
            question = record.get("instruction", "")
            category = record.get("category", "")

            fact = self.is_fact_not_fact(
                question=question,
                category=category,
            )

            if fact is None:
                continue

            csv_rows.append([question, fact])

        return csv_rows


class CSVWriting(BaseEstimator, TransformerMixin):
    def __init__(self, output_filename: str):
        self.output_filename = output_filename

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        output_dir = 'csv_fact_generative'

        if not (os.path.exists(output_dir)):
            os.mkdir(output_dir)

        output_path = os.path.join(output_dir, self.output_filename)

        with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['question', 'fact'])
            for record in X:
                writer.writerow(record)

        return output_path
