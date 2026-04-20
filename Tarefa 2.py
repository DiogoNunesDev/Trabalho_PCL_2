import csv
import subprocess
import sys

VALID_LABELS = ["GEOGRAPHY", "MUSIC", "LITERATURE", "HISTORY", "SCIENCE"]

EVAL_INPUT = "eval.txt"
EVAL_QUESTIONS = "eval-questions.txt"
EVAL_LABELS = "eval-labels.txt"

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

    print(f"Instâncias preparadas: {n_rows}")
    print(f"Ficheiro de questões: {questions_output_path}")
    print(f"Ficheiro de labels: {labels_output_path}")


def run_and_evaluate(mode, counts_dir, questions_file, labels_file, train_file="train.txt"):
    slug = mode.lstrip("-")
    result_file = f"result_{slug}.txt"

    with open(result_file, "w", encoding="utf-8") as fout:
        subprocess.run(
            [
                sys.executable,
                "lmclassifier.py",
                mode,
                counts_dir,
                questions_file,
                "--train",
                train_file
            ],
            stdout=fout,
            check=True
        )

    print("\n" + "=" * 60)
    print(f"Resultados de avaliação para {mode}:")

    subprocess.run(
        [sys.executable, "evaluate.py", "-v", labels_file, result_file],
        check=False
    )


def main():
    prepare_eval_data(
        eval_input_path=EVAL_INPUT,
        questions_output_path=EVAL_QUESTIONS,
        labels_output_path=EVAL_LABELS,
        include_answer=INCLUDE_ANSWER_IN_INPUT
    )

    mode_to_dir = {
        "-unigrams": "counts",
        "-bigrams": "counts2",
        "-smooth": "counts2"
    }

    for mode in ["-unigrams", "-bigrams", "-smooth"]:
        run_and_evaluate(
            mode=mode,
            counts_dir=mode_to_dir[mode],
            questions_file=EVAL_QUESTIONS,
            labels_file=EVAL_LABELS,
            train_file="train.txt"
        )


if __name__ == "__main__":
    main()