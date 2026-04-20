import csv
import os
from collections import Counter

import nltk
from nltk import ngrams

VALID_LABELS = ["GEOGRAPHY", "MUSIC", "LITERATURE", "HISTORY", "SCIENCE"]

nltk.download("punkt_tab", quiet=True)
try:
    nltk.download("punkt", quiet=True)
except Exception:
    pass

os.makedirs("counts", exist_ok=True)
os.makedirs("counts2", exist_ok=True)


def preprocess_text(text):
    # Mantém texto original e pontuação; apenas tokeniza
    tokens = nltk.word_tokenize(text)
    return tokens


def main():
    categories = {label: [] for label in VALID_LABELS}

    with open("train.txt", "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')

        for row in reader:
            if not row or len(row) < 3:
                continue

            label = row[0].strip()
            if label not in VALID_LABELS:
                continue

            question = row[1].strip()
            answer = "\t".join(row[2:]).strip()

            text = f"{question} {answer}"
            tokens = preprocess_text(text)

            categories[label].extend(tokens)

    for label in VALID_LABELS:
        tokens = categories[label]

        # UNIGRAMAS -> counts
        unigram_counts = Counter(ngrams(tokens, 1))
        filename_uni = f"counts/unigrams_{label}.txt"

        with open(filename_uni, "w", encoding="utf-8") as f:
            f.write("Unigramas\n")
            for gram, freq in unigram_counts.most_common():
                f.write(f"{gram[0]} {freq}\n")

        # BIGRAMAS -> counts2
        bigram_counts = Counter(ngrams(tokens, 2))
        filename_bi = f"counts2/bigrams_{label}.txt"

        with open(filename_bi, "w", encoding="utf-8") as f:
            f.write("Bigramas\n")
            for gram, freq in bigram_counts.most_common():
                f.write(f"{gram[0]} {gram[1]} {freq}\n")

        print(f"Processado: {label} ({len(tokens)} tokens)")

if __name__ == "__main__":
    main()