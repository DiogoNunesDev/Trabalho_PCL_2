import csv
import os
import random
from collections import Counter, defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from evaluate import classification_report


if torch.backends.mps.is_available():
    device = torch.device("mps")   # Apple Silicon
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
else:
    device = torch.device("cpu")   # CPU

print("Device used:", device)

MODEL_NAME = "google/flan-t5-base"
RESULTS_DIR = "Tarefa4-results"

VALID_LABELS = ["GEOGRAPHY", "MUSIC", "LITERATURE", "HISTORY", "SCIENCE"]

os.makedirs(RESULTS_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, token=False)
model.to(device)
model.eval()


def read_labeled_qa_file(path):
    rows = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')

        for row in reader:
            if not row:
                continue

            if len(row) >= 3 and row[0].strip() in VALID_LABELS:
                label = row[0].strip()
                question = row[1].strip()
                answer = "\t".join(row[2:]).strip()

                rows.append({
                    "label": label,
                    "question": question,
                    "answer": answer
                })

    return rows


def print_label_distribution(rows, title):
    counts = Counter(row["label"] for row in rows)
    print(f"\n{title}")
    print(f"Total: {len(rows)}")
    for label in VALID_LABELS:
        print(f"{label}: {counts.get(label, 0)}")


def select_balanced_subset(rows, n_per_label=4, seed=42):
    rng = random.Random(seed)
    buckets = defaultdict(list)

    for row in rows:
        buckets[row["label"]].append(row)

    subset = []
    for label in VALID_LABELS:
        if len(buckets[label]) < n_per_label:
            raise ValueError(f"Not enough examples for label {label}")
        subset.extend(rng.sample(buckets[label], n_per_label))

    rng.shuffle(subset)
    return subset


def select_few_shot_examples(train_rows, seed=123):
    rng = random.Random(seed)
    buckets = defaultdict(list)

    for row in train_rows:
        buckets[row["label"]].append(row)

    shots = []
    for label in VALID_LABELS:
        if not buckets[label]:
            raise ValueError(f"No training examples available for label {label}")
        shots.append(rng.choice(buckets[label]))

    return shots


def build_zero_shot_prompt(question, answer):
    return f"""
Classify the following Question+Answer pair into exactly one label.

Possible labels:
GEOGRAPHY, MUSIC, LITERATURE, HISTORY, SCIENCE

Rules:
- Return only one label.
- Do not explain.
- Do not output anything else.

Question: {question}
Answer: {answer}

Label:
""".strip()


def build_few_shot_prompt(question, answer, few_shot_examples):
    examples_text = "\n\n".join(
        [
            f"Question: {ex['question']}\nAnswer: {ex['answer']}\nLabel: {ex['label']}"
            for ex in few_shot_examples
        ]
    )

    return f"""
Classify the following Question+Answer pair into exactly one label.

Possible labels:
GEOGRAPHY, MUSIC, LITERATURE, HISTORY, SCIENCE

Rules:
- Return only one label.
- Do not explain.
- Do not output anything else.

Examples:

{examples_text}

Now classify this pair:

Question: {question}
Answer: {answer}

Label:
""".strip()


def normalize_prediction(text):
    if not text:
        return "UNKNOWN"

    text = text.strip().upper()

    for label in VALID_LABELS:
        if label in text:
            return label

    return "UNKNOWN"


def classify_with_local_llm(
    question,
    answer,
    prompt_type,
    few_shot_examples,
    stochastic=False,
    temperature=0.7,
    top_p=0.9
):
    if prompt_type == "few-shot":
        prompt = build_few_shot_prompt(question, answer, few_shot_examples)
    else:
        prompt = build_zero_shot_prompt(question, answer)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": 8,
        "do_sample": stochastic
    }

    if stochastic:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return normalize_prediction(text)


def accuracy_score_simple(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)


def write_labels(path, labels):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(labels))


def print_metrics_with_evaluate(title, y_true, y_pred):
    if title:
        print(f"\n{title}")

    try:
        print(
            classification_report(
                y_true,
                y_pred,
                labels=VALID_LABELS,
                target_names=VALID_LABELS,
                zero_division=0,
            )
        )
    except TypeError:
        result = classification_report(y_true, y_pred)
        if isinstance(result, dict) and "accuracy" in result:
            print(
                f"accuracy: {result['accuracy']:.4f} "
                f"({result['correct']}/{result['total']})"
            )
        else:
            print(result)


def evaluate_single_experiment(examples, prompt_type="zero-shot", few_shot_examples=None):
    truth = [ex["label"] for ex in examples]
    preds = []

    for i, ex in enumerate(examples, start=1):
        pred = classify_with_local_llm(
            question=ex["question"],
            answer=ex["answer"],
            prompt_type=prompt_type,
            few_shot_examples=few_shot_examples,
            stochastic=False
        )
        preds.append(pred)

        if i % 50 == 0 or i == len(examples):
            print(f"{prompt_type}: {i}/{len(examples)} instâncias processadas")

    return {
        "truth": truth,
        "predictions": preds,
    }


def run_experiment_consistency(examples, prompt_type="zero-shot", few_shot_examples=None, n_runs=20):
    all_runs = []

    for run_id in range(n_runs):
        preds = []

        for ex in examples:
            pred = classify_with_local_llm(
                question=ex["question"],
                answer=ex["answer"],
                prompt_type=prompt_type,
                few_shot_examples=few_shot_examples,
                stochastic=True,
                temperature=0.7,
                top_p=0.9
            )
            preds.append(pred)

        all_runs.append({
            "run_id": run_id + 1,
            "predictions": preds
        })

    return all_runs


def compute_consistency(run_results):
    prediction_runs = [r["predictions"] for r in run_results]
    n_runs = len(prediction_runs)
    n_examples = len(prediction_runs[0])

    exact_same_count = 0
    pairwise_matches = 0
    pairwise_total = 0
    majority_vote_predictions = []

    for i in range(n_examples):
        labels_i = [prediction_runs[r][i] for r in range(n_runs)]
        counts = Counter(labels_i)
        majority_vote_predictions.append(counts.most_common(1)[0][0])

        if len(set(labels_i)) == 1:
            exact_same_count += 1

        for a in range(n_runs):
            for b in range(a + 1, n_runs):
                pairwise_total += 1
                if labels_i[a] == labels_i[b]:
                    pairwise_matches += 1

    return {
        "exact_consistency": exact_same_count / n_examples if n_examples else 0.0,
        "pairwise_agreement": pairwise_matches / pairwise_total if pairwise_total else 1.0,
        "majority_vote_predictions": majority_vote_predictions
    }


def print_subset_predictions(title, examples, predictions):
    print(f"\n{title}")
    for ex, pred in zip(examples, predictions):
        print("-" * 80)
        print(f"Truth: {ex['label']}")
        print(f"Pred : {pred}")
        print(f"Q    : {ex['question']}")
        print(f"A    : {ex['answer']}")


def main():
    train_rows = read_labeled_qa_file("train.txt")
    eval_rows = read_labeled_qa_file("eval.txt")

    few_shot_examples = select_few_shot_examples(train_rows, seed=123)
    consistency_subset = select_balanced_subset(eval_rows, n_per_label=4, seed=42)

    print_label_distribution(eval_rows, "Eval completo")
    print_label_distribution(consistency_subset, "Subset para consistência")

    # Avaliação no eval completo
    print("\n" + "=" * 70)
    print("LLM no eval completo - zero-shot")
    zero_eval_full = evaluate_single_experiment(
        eval_rows,
        prompt_type="zero-shot",
        few_shot_examples=None
    )

    truth_full = zero_eval_full["truth"]
    write_labels(os.path.join(RESULTS_DIR, "result_zero_full.txt"), zero_eval_full["predictions"])

    print_metrics_with_evaluate(
        "LLM Zero-shot - Eval completo",
        truth_full,
        zero_eval_full["predictions"]
    )

    print("\n" + "=" * 70)
    print("LLM no eval completo - few-shot")
    few_eval_full = evaluate_single_experiment(
        eval_rows,
        prompt_type="few-shot",
        few_shot_examples=few_shot_examples
    )
    write_labels(os.path.join(RESULTS_DIR, "result_few_full.txt"), few_eval_full["predictions"])

    print_metrics_with_evaluate(
        "LLM Few-shot - Eval completo",
        truth_full,
        few_eval_full["predictions"]
    )

    # Consistência em subset de 20
    print("\n" + "=" * 70)
    print("Consistência do LLM - subset equilibrado de 20")

    truth_subset = [ex["label"] for ex in consistency_subset]

    zero_run_results = run_experiment_consistency(
        consistency_subset,
        prompt_type="zero-shot",
        few_shot_examples=None,
        n_runs=20
    )
    zero_consistency = compute_consistency(zero_run_results)

    few_run_results = run_experiment_consistency(
        consistency_subset,
        prompt_type="few-shot",
        few_shot_examples=few_shot_examples,
        n_runs=20
    )
    few_consistency = compute_consistency(few_run_results)

    print("\nZero-shot consistency")
    print("Exact consistency:", round(zero_consistency["exact_consistency"], 4))
    print("Pairwise agreement:", round(zero_consistency["pairwise_agreement"], 4))
    print(
        "Majority-vote accuracy (subset 20):",
        round(accuracy_score_simple(truth_subset, zero_consistency["majority_vote_predictions"]), 4)
    )

    print("\nFew-shot consistency")
    print("Exact consistency:", round(few_consistency["exact_consistency"], 4))
    print("Pairwise agreement:", round(few_consistency["pairwise_agreement"], 4))
    print(
        "Majority-vote accuracy (subset 20):",
        round(accuracy_score_simple(truth_subset, few_consistency["majority_vote_predictions"]), 4)
    )

    print_metrics_with_evaluate(
        "LLM Zero-shot - Majority vote no subset de 20",
        truth_subset,
        zero_consistency["majority_vote_predictions"]
    )
    print_metrics_with_evaluate(
        "LLM Few-shot - Majority vote no subset de 20",
        truth_subset,
        few_consistency["majority_vote_predictions"]
    )

if __name__ == "__main__":
    main()