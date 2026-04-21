import argparse
import os
import random
import re
import time
from collections import Counter
from typing import Optional, Tuple

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from sentiment_metrics import InferenceRecord, compute_all_metrics

# ======================
# CONSTANTS
# ======================
VALID_LABELS = {"positive", "negative", "neutral"}

LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

SYSTEM_PROMPT = (
    "You are a financial sentiment classifier.\n"
    "Output ONLY one word: positive, negative, or neutral.\n"
    "Do NOT output anything else."
)

AGREEMENT_TO_CONFIG = {
    "allagree": "sentences_allagree",
    "75agree": "sentences_75agree",
    "66agree": "sentences_66agree",
    "50agree": "sentences_50agree",
}

DATASET_CANDIDATES = [
    "takala/financial_phrasebank",
    "financial_phrasebank",
]


# ======================
# UTILS
# ======================
def normalize_label(text: str) -> str:
    cleaned = text.strip().lower()
    for token in [",", ".", ":", ";", "!", "?", "\"", "'", "`", "(", ")"]:
        cleaned = cleaned.replace(token, " ")
    words = cleaned.split()
    for w in words:
        if w in VALID_LABELS:
            return w
    if cleaned in VALID_LABELS:
        return cleaned
    return "invalid"


def get_true_label(row):
    if isinstance(row["label"], int):
        return LABEL_MAP[row["label"]]
    return normalize_label(str(row["label"]))


# ======================
# MODEL CALL
# ======================
def predict_label(client: OpenAI, sentence: str, model: str, debug=False) -> Tuple[str, InferenceRecord]:
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Sentence: "
                    f"{sentence}\n\n"
                    "Return only one word."
                ),
            },
        ],
    )
    end = time.perf_counter()

    text = response.choices[0].message.content or ""

    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)

    record = InferenceRecord(
        latency_seconds=max(0.0, end - start),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )

    if debug:
        print("RAW OUTPUT:", text)

    return normalize_label(text), record


# ======================
# DATA LOADING
# ======================
def load_financial_phrasebank(agreement_level: str, allow_fallback: bool = True):
    if agreement_level not in AGREEMENT_TO_CONFIG:
        valid = ", ".join(sorted(AGREEMENT_TO_CONFIG.keys()))
        raise ValueError(f"Unsupported agreement level: {agreement_level}. Valid values: {valid}")

    config_name = AGREEMENT_TO_CONFIG[agreement_level]

    # trust_remote_code has been removed in recent datasets versions.
    # Try both hub IDs to handle renamed/migrated dataset repos.
    errors = []
    for dataset_id in DATASET_CANDIDATES:
        try:
            ds = load_dataset(dataset_id, config_name)
            return ds["train"], config_name, dataset_id
        except Exception as e:
            errors.append((dataset_id, str(e)))

    # If only allagree exists in local cache, fall back with a warning.
    requested = config_name
    fallback = AGREEMENT_TO_CONFIG["allagree"]
    cache_config_error = next(
        (
            msg
            for _, msg in errors
            if "Available configs in the cache" in msg and requested in msg
        ),
        None,
    )
    if allow_fallback and cache_config_error:
        match = re.search(r"Available configs in the cache:\s*\[(.*)\]", cache_config_error)
        available_text = match.group(1) if match else ""
        if fallback in available_text:
            print(
                f"[WARN] Requested config '{requested}' is not in local cache. "
                f"Falling back to '{fallback}'."
            )
            for dataset_id in DATASET_CANDIDATES:
                try:
                    ds = load_dataset(dataset_id, fallback)
                    return ds["train"], fallback, dataset_id
                except Exception:
                    pass

    detail = "\n\n".join([f"- {dataset_id}: {msg}" for dataset_id, msg in errors])
    raise RuntimeError(
        "Failed to load Financial PhraseBank with the requested agreement level.\n"
        "Tips:\n"
        "1. Remove trust_remote_code (already handled in this script).\n"
        "2. If you are offline and cache only has allagree, run: --agreement-level allagree\n"
        "3. If you need 75/66/50agree, reconnect network and retry once to download cache.\n"
        "4. To fail fast instead of fallback, use: --no-fallback\n"
        f"Requested config: {requested}\n"
        f"Tried dataset IDs: {', '.join(DATASET_CANDIDATES)}\n"
        f"Underlying errors:\n{detail}"
    )


# ======================
# EVALUATION
# ======================
def evaluate(
    model: str,
    agreement_level: str,
    max_samples: Optional[int],
    seed: int,
    sleep_seconds: float,
    allow_fallback: bool,
    debug=False,
) -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found.")

    client = OpenAI(base_url="https://aihubmix.com/v1",api_key=api_key,)

    dataset, loaded_config, loaded_dataset_id = load_financial_phrasebank(
        agreement_level=agreement_level,
        allow_fallback=allow_fallback,
    )
    rows = list(dataset)

    if max_samples is not None and max_samples < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, max_samples)

    y_true = []
    y_pred = []
    valid_records = []
    inference_records = []
    invalid_count = 0
    errors = 0

    for row in tqdm(rows, desc="Evaluating"):
        sentence = row["sentence"]
        true_label = get_true_label(row)  # ✅ 修复点


        try:
            pred, info = predict_label(client, sentence, model=model, debug=debug)
        except Exception as e:
            if debug:
                print("ERROR:", e)
            pred = "invalid"
            info = InferenceRecord(latency_seconds=0.0)
            errors += 1

        if pred == "invalid":
            invalid_count += 1
            continue

        y_true.append(true_label)
        y_pred.append(pred)
        valid_records.append((sentence, true_label, pred))
        inference_records.append(info)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if not y_true:
        raise RuntimeError("No valid predictions collected; cannot compute accuracy.")

    metrics = compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        # GPT path: pass inference records to enable latency/token statistics.
        inference_records=inference_records,
    )

    print("\n=== Evaluation Result ===")
    print(f"Model: {model}")
    print(f"PhraseBank split requested: {AGREEMENT_TO_CONFIG[agreement_level]}")
    print(f"PhraseBank split loaded: {loaded_config}")
    print(f"Dataset ID used: {loaded_dataset_id}")
    print(f"Total samples requested: {len(rows)}")
    print(f"Valid predictions: {len(y_true)}")
    print(f"Invalid predictions: {invalid_count}")
    print(f"API errors: {errors}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"S-MAE: {metrics['s_mae']:.4f}")

    print("\nPer-class metrics:")
    for label, stats in metrics["per_class"].items():
        print(
            f"  {label:8s} | "
            f"P={stats['precision']:.4f} "
            f"R={stats['recall']:.4f} "
            f"F1={stats['f1']:.4f} "
            f"Support={int(stats['support'])}"
        )

    cm = metrics["confusion_matrix"]
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("  labels:", cm["labels"])
    for row in cm["matrix"]:
        print(" ", row)

    eff = metrics["efficiency"]
    print("\nEfficiency:")
    print(f"  Avg latency (ms): {eff['latency_avg_ms']:.2f}")
    print(f"  P50 latency (ms): {eff['latency_p50_ms']:.2f}")
    print(f"  P95 latency (ms): {eff['latency_p95_ms']:.2f}")
    print(f"  Prompt tokens total: {int(eff['prompt_tokens_total'])}")
    print(f"  Completion tokens total: {int(eff['completion_tokens_total'])}")
    print(f"  Total tokens total: {int(eff['total_tokens_total'])}")

    confusion_like = Counter(zip(y_true, y_pred))
    print("\nTop pair counts (true -> pred):")
    for (t, p), count in confusion_like.most_common(10):
        print(f"  {t:8s} -> {p:8s}: {count}")

    mismatches = []
    for sentence, true_label, pred in valid_records:
        if true_label != pred:
            mismatches.append((sentence, true_label, pred))
        if len(mismatches) >= 5:
            break

    if mismatches:
        print("\nExample mismatches:")
        for sentence, t, p in mismatches:
            print(f"- true={t}, pred={p} | sentence={sentence}")


# ======================
# CLI
# ======================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument(
        "--agreement-level",
        choices=sorted(AGREEMENT_TO_CONFIG.keys()),
        default="allagree",
        help="PhraseBank agreement level to evaluate: allagree, 75agree, 66agree, or 50agree.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable automatic fallback to allagree when requested split is missing in local cache.",
    )
    parser.add_argument("--debug", action="store_true")  # 👈 新增debug模式
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        model=args.model,
        agreement_level=args.agreement_level,
        max_samples=args.max_samples,
        seed=args.seed,
        sleep_seconds=args.sleep_seconds,
        allow_fallback=not args.no_fallback,
        debug=args.debug
    )