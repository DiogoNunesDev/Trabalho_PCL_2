import csv
import math
import os
import re
from collections import Counter

import nltk
from nltk import ngrams

VALID_LABELS = ["GEOGRAPHY", "MUSIC", "LITERATURE", "HISTORY", "SCIENCE"]
YEAR_PLACEHOLDER = "_YEAR_"

TRAIN_FILE = "train.txt"
EVAL_FILE = "eval.txt"
TEST_FILE = "test-questions.txt"

RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "test-guess.txt")

def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\b\d{4}\b", YEAR_PLACEHOLDER, text)
    raw_tokens = nltk.word_tokenize(text)
    tokens = [t for t in raw_tokens if t.isalnum() or t == YEAR_PLACEHOLDER]

    return tokens


def read_labeled_qa_file(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')
        for row in reader:
            if not row or len(row) < 3:
                continue
            label = row[0].strip()
            if label not in VALID_LABELS:
                continue
            question = row[1].strip()
            answer = "\t".join(row[2:]).strip()
            rows.append({
                "label": label,
                "question": question,
                "answer": answer
            })

    return rows


def read_test_file(path):
    rows = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')

        for row in reader:
            if not row:
                continue

            if len(row) >= 2:
                question = row[0].strip()
                answer = "\t".join(row[1:]).strip()
            else:
                question = row[0].strip()
                answer = ""

            rows.append({
                "question": question,
                "answer": answer
            })

    return rows


def build_unigram_model(train_rows):
    tokens_per_label = {label: [] for label in VALID_LABELS}
    for row in train_rows:
        label = row["label"]
        question = row["question"]
        answer = row["answer"]
        q_tokens = preprocess_text(question)
        a_tokens = preprocess_text(answer)
        tokens_per_label[label].extend(q_tokens)
        tokens_per_label[label].extend(a_tokens)

    models = {}
    global_vocab = set()
    for label in VALID_LABELS:
        tokens = tokens_per_label[label]
        unigram_counts = Counter(ngrams(tokens, 1))
        counts = {}
        total_tokens = 0
        for gram, freq in unigram_counts.items():
            token = gram[0]
            counts[token] = freq
            global_vocab.add(token)
            total_tokens += freq
        models[label] = {
            "counts": counts,
            "total": total_tokens
        }
    return models, global_vocab

def classify_instance(question, answer, models, vocab_size):
    text = f"{question} {answer}".strip()
    tokens = preprocess_text(text)
    if not tokens:
        return "HISTORY"
    best_label = None
    max_log_prob = -float("inf")
    for label, data in models.items():
        counts = data["counts"]
        total = data["total"]
        log_prob = 0.0
        for token in tokens:
            count = counts.get(token, 0)
            log_prob += math.log((count + 1) / (total + vocab_size))
        if log_prob > max_log_prob:
            max_log_prob = log_prob
            best_label = label

    return best_label if best_label else "HISTORY"

def classify_test_set(test_rows, models, vocab_size):
    predictions = []
    for row in test_rows:
        pred = classify_instance(
            question=row["question"],
            answer=row["answer"],
            models=models,
            vocab_size=vocab_size
        )
        predictions.append(pred)

    return predictions

def save_predictions(path, predictions):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(predictions))
        if predictions:
            f.write("\n")

def main():
    ensure_nltk()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_rows = read_labeled_qa_file(TRAIN_FILE)
    eval_rows = read_labeled_qa_file(EVAL_FILE)
    final_train_rows = train_rows + eval_rows

    print(f"Instâncias de treino original: {len(train_rows)}")
    print(f"Instâncias de validação: {len(eval_rows)}")
    print(f"Instâncias de treino final (train + eval): {len(final_train_rows)}")

    models, vocab = build_unigram_model(final_train_rows)
    vocab_size = len(vocab)

    print(f"Tamanho do vocabulário final: {vocab_size}")
    print("Modelo escolhido: Unigramas com preprocessamento da Tarefa 3")

    test_rows = read_test_file(TEST_FILE)
    print(f"Instâncias de teste: {len(test_rows)}")

    predictions = classify_test_set(test_rows, models, vocab_size)
    save_predictions(OUTPUT_FILE, predictions)

    print(f"Previsões guardadas em: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()