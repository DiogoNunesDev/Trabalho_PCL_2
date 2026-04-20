"""
Classificador n-grama: lê modelos (contagens por etiqueta) e etiqueta cada linha
de um ficheiro de questões+respostas (uma instância por linha).
"""
import argparse
import csv
import math
import os
import sys

import nltk
from nltk.tokenize import word_tokenize

VALID_LABELS = ["GEOGRAPHY", "MUSIC", "LITERATURE", "HISTORY", "SCIENCE"]


def tokenize_eval_line(line):
    return [t.lower() for t in word_tokenize(line) if t.isalnum()]


def vocabulary_from_train(train_path="train.txt"):
    """
    |V| para alisamento: tipos distintos em todo o texto de treino (pergunta+resposta),
    com a mesma tokenização que na classificação.
    """
    vocab = set()
    with open(train_path, "r", encoding="utf-8", newline="") as f:
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
            vocab.update(tokenize_eval_line(text))
    return vocab


def load_models(directory, is_bigram=False):
    models = {}
    global_vocab = set()
    prefix = "bigrams_" if is_bigram else "unigrams_"

    for filename in os.listdir(directory):
        if not filename.endswith(".txt") or not filename.startswith(prefix):
            continue
        # unigrams_GEOGRAPHY.txt -> GEOGRAPHY
        rest = filename[len(prefix) :]
        if not rest.endswith(".txt"):
            continue
        label = rest[: -len(".txt")]
        if label not in VALID_LABELS:
            continue

        counts = {}
        total_tokens = 0
        path = os.path.join(directory, filename)
        with open(path, "r", encoding="utf-8") as f:
            next(f)  # cabeçalho
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

        models[label] = {"counts": counts, "total": total_tokens}

    return models, global_vocab


def run_classifier(mode, counts_dir, input_file, train_path="train.txt"):
    is_bi = mode in ("-bigrams", "-smooth")
    models, vocab_from_counts = load_models(counts_dir, is_bigram=is_bi)

    if mode == "-smooth":
        train_vocab = vocabulary_from_train(train_path)
        v_size = len(train_vocab)
        if v_size == 0:
            raise ValueError("O vocabulário de treino está vazio")
    else:
        v_size = len(vocab_from_counts)

    results = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                results.append("UNKNOWN")
                continue

            tokens = tokenize_eval_line(line)
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
                        log_prob = 0.0
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
                        log_prob = 0.0
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


def main(argv=None):
    nltk.download("punkt_tab", quiet=True)

    parser = argparse.ArgumentParser(
        description="Classifica linhas (pergunta\\tresposta) com modelos n-grama por etiqueta."
    )
    mx = parser.add_mutually_exclusive_group(required=True)
    mx.add_argument(
        "-unigrams",
        action="store_const",
        const="-unigrams",
        dest="mode",
        help="Modelo de unigramas.",
    )
    mx.add_argument(
        "-bigrams",
        action="store_const",
        const="-bigrams",
        dest="mode",
        help="Modelo de bigramas sem alisamento.",
    )
    mx.add_argument(
        "-smooth",
        action="store_const",
        const="-smooth",
        dest="mode",
        help="Modelo de bigramas com alisamento (|V| a partir de todo o train.txt).",
    )
    parser.add_argument(
        "counts_dir",
        help="Diretoria com unigrams_<ETIQUETA>.txt e/ou bigrams_<ETIQUETA>.txt",
    )
    parser.add_argument(
        "questions_file",
        help="Ficheiro com uma questão (e resposta) por linha, ex.: eval-questions.txt",
    )
    parser.add_argument(
        "--train",
        default="train.txt",
        help="Treino para calcular |V| em -smooth (predefinição: train.txt).",
    )
    args = parser.parse_args(argv)

    if not os.path.isdir(args.counts_dir):
        print(f"Diretoria inexistente: {args.counts_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.questions_file):
        print(f"Ficheiro inexistente: {args.questions_file}", file=sys.stderr)
        sys.exit(1)

    preds = run_classifier(args.mode, args.counts_dir, args.questions_file, args.train)
    sys.stdout.write("\n".join(preds) + ("\n" if preds else ""))


if __name__ == "__main__":
    main()
