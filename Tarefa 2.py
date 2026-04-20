import csv
import os
import math
import sys
import subprocess
import tempfile
import nltk
from nltk.tokenize import word_tokenize

VALID_LABELS = ["GEOGRAPHY", "MUSIC", "LITERATURE", "HISTORY", "SCIENCE"]
EVAL_INPUT = "eval.txt"
EVAL_QUESTIONS = "eval-questions.txt"
EVAL_LABELS = "eval-labels.txt"
COUNTS_DIR = "counts"
INCLUDE_ANSWER_IN_INPUT = True


def prepare_eval_data(
    eval_input_path=EVAL_INPUT,
    questions_output_path=EVAL_QUESTIONS,
    labels_output_path=EVAL_LABELS,
    include_answer=True
):
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
            next(f, None)  # salta cabeçalho
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


def tokenize_input_line(line):
    return [t.lower() for t in word_tokenize(line) if t.isalnum()]


def run_classifier(mode, counts_dir, input_file):
    models, vocab = load_models(counts_dir, mode)
    v_size = len(vocab)
    results = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            tokens = tokenize_input_line(line)

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
    nltk.download("punkt_tab", quiet=True)

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