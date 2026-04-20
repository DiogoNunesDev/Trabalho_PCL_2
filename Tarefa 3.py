import csv
import os
import re
import math
import sys
import subprocess
import tempfile
from collections import Counter
import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize

VALID_LABELS = ["GEOGRAPHY", "MUSIC", "LITERATURE", "HISTORY", "SCIENCE"]
YEAR_PLACEHOLDER = "_YEAR_"
TRAIN_INPUT = "train.txt"
EVAL_INPUT = "eval.txt"
DATA_DIR = "data-processed"
COUNTS_DIR = "counts2"
PREPROCESSED_TRAIN = os.path.join(DATA_DIR, "train_preprocessed.txt")
EVAL_QUESTIONS = "eval-questions.txt"
EVAL_LABELS = "eval-labels.txt"
INCLUDE_ANSWER_IN_INPUT = True
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(COUNTS_DIR, exist_ok=True)


def preprocess_text(text):

    text = text.lower()
    text = re.sub(r"\b\d{4}\b", YEAR_PLACEHOLDER, text)
    raw_tokens = word_tokenize(text)
    tokens = [t for t in raw_tokens if t.isalnum() or t == YEAR_PLACEHOLDER]
    return tokens

def build_train_models(
    train_input_path=TRAIN_INPUT,
    preprocessed_train_path=PREPROCESSED_TRAIN,
    counts_dir=COUNTS_DIR):

    rows = []
    tokens_por_label = {label: [] for label in VALID_LABELS}

    with open(train_input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')

        for row in reader:
            if not row or len(row) < 3:
                continue

            label = row[0].strip()
            if label not in VALID_LABELS:
                continue

            question = row[1].strip()
            answer = "\t".join(row[2:]).strip()

            q_tokens = preprocess_text(question)
            a_tokens = preprocess_text(answer)

            rows.append({
                "label": label,
                "question_pp": " ".join(q_tokens),
                "answer_pp": " ".join(a_tokens)
            })

            tokens_por_label[label].extend(q_tokens)
            tokens_por_label[label].extend(a_tokens)

    with open(preprocessed_train_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(
            fout,
            delimiter="\t",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL
        )
        for ex in rows:
            writer.writerow([
                ex["label"],
                ex["question_pp"],
                ex["answer_pp"]
            ])

    for label in VALID_LABELS:
        tokens = tokens_por_label[label]

        unigram_counts = Counter(ngrams(tokens, 1))
        filename_uni = os.path.join(counts_dir, f"unigrams_{label}.txt")
        with open(filename_uni, "w", encoding="utf-8") as f:
            f.write("Unigramas\n")
            for gram, freq in unigram_counts.most_common():
                f.write(f"{gram[0]} {freq}\n")

        bigram_counts = Counter(ngrams(tokens, 2))
        filename_bi = os.path.join(counts_dir, f"bigrams_{label}.txt")
        with open(filename_bi, "w", encoding="utf-8") as f:
            f.write("Bigramas\n")
            for gram, freq in bigram_counts.most_common():
                f.write(f"{gram[0]} {gram[1]} {freq}\n")

        print(f"Tarefa 3 - {label}: {len(tokens)} tokens")



def prepare_eval_data(
    eval_input_path=EVAL_INPUT,
    questions_output_path=EVAL_QUESTIONS,
    labels_output_path=EVAL_LABELS,
    include_answer=True):
    n_rows = 0

    with open(eval_input_path, "r", encoding="utf-8", newline="") as fin, \
         open(questions_output_path, "w", encoding="utf-8", newline="") as fq, \
         open(labels_output_path, "w", encoding="utf-8", newline="") as fl:

        reader = csv.reader(fin, delimiter="\t", quotechar='"')

        for row in reader:
            if not row or len(row) < 3:
                continue

            label = row[0].strip()
            if label not in VALID_LABELS:
                continue

            question = row[1].strip()
            answer = "\t".join(row[2:]).strip()

            if include_answer:
                fq.write(f"{question}\t{answer}\n")
            else:
                fq.write(f"{question}\n")

            fl.write(f"{label}\n")
            n_rows += 1

def load_models(directory, mode):
    models = {}
    global_vocab = set()

    if mode == "-unigrams":
        prefix = "unigrams_"
        is_bigram = False
    elif mode in ["-bigrams", "-smooth"]:
        prefix = "bigrams_"
        is_bigram = True
    else:
        raise ValueError(f"Modo inválido: {mode}")

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"A pasta '{directory}' não existe.")

    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue
        if not filename.startswith(prefix):
            continue

        filepath = os.path.join(directory, filename)
        label = os.path.splitext(filename)[0].split("_", 1)[1].upper()

        counts = {}
        total_tokens = 0

        with open(filepath, "r", encoding="utf-8") as f:
            next(f, None) 
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                freq = int(parts[-1])

                if is_bigram:
                    if len(parts) < 3:
                        continue
                    gram = (parts[0], parts[1])
                    global_vocab.add(parts[0])
                    global_vocab.add(parts[1])
                else:
                    gram = parts[0]
                    global_vocab.add(gram)

                counts[gram] = freq
                total_tokens += freq

        models[label] = {
            "counts": counts,
            "total": total_tokens
        }

    if not models:
        raise ValueError(
            f"Nenhum modelo foi carregado para o modo {mode} na pasta '{directory}'."
        )

    print(f"\nModelos carregados para {mode}: {list(models.keys())}")
    print(f"Tamanho do vocabulário: {len(global_vocab)}")

    return models, global_vocab


def run_classifier(mode, counts_dir, input_file):
    models, vocab = load_models(counts_dir, mode)
    v_size = len(vocab)
    results = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            tokens = preprocess_text(line)

            if not tokens:
                results.append("UNKNOWN")
                continue

            best_label = None
            max_log_prob = -float("inf")

            for label, data in models.items():
                log_prob = 0.0
                counts = data["counts"]
                total = data["total"]

                if mode == "-unigrams":
                    for t in tokens:
                        count = counts.get(t, 0)
                        log_prob += math.log((count + 1) / (total + v_size))

                elif mode == "-bigrams":
                    if len(tokens) < 2:
                        log_prob = -float("inf")
                    else:
                        for i in range(len(tokens) - 1):
                            bi = (tokens[i], tokens[i + 1])
                            count = counts.get(bi, 0)
                            if count > 0:
                                log_prob += math.log(count / total)
                            else:
                                log_prob += -100

                elif mode == "-smooth":
                    if len(tokens) < 2:
                        log_prob = -float("inf")
                    else:
                        for i in range(len(tokens) - 1):
                            bi = (tokens[i], tokens[i + 1])
                            count_bi = counts.get(bi, 0)
                            log_prob += math.log((count_bi + 1) / (total + v_size))

                if log_prob > max_log_prob:
                    max_log_prob = log_prob
                    best_label = label

            results.append(best_label if best_label else "UNKNOWN")

    return results


def save_predictions(path, predictions):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(predictions))


def evaluate_predictions(gold_path, pred_path):
    subprocess.run(
        [sys.executable, "evaluate.py", "-v", gold_path, pred_path],
        check=False
    )


def main():
    build_train_models()
    prepare_eval_data(
        eval_input_path=EVAL_INPUT,
        questions_output_path=EVAL_QUESTIONS,
        labels_output_path=EVAL_LABELS,
        include_answer=INCLUDE_ANSWER_IN_INPUT
    )
    modes = ["-unigrams", "-bigrams", "-smooth"]

    for mode in modes:
        print("\n" + "=" * 60)
        predictions = run_classifier(mode, COUNTS_DIR, EVAL_QUESTIONS)

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            delete=False,
        ) as temp_pred_file:
            temp_pred_path = temp_pred_file.name

        try:
            save_predictions(temp_pred_path, predictions)
            print(f"Resultados de avaliação para {mode}:")
            evaluate_predictions(EVAL_LABELS, temp_pred_path)
        finally:
            if os.path.exists(temp_pred_path):
                os.remove(temp_pred_path)


if __name__ == "__main__":
    main()