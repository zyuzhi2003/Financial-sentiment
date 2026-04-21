from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ==============================
# Label configuration and mapping
# ==============================
DEFAULT_LABEL_ORDER: Tuple[str, ...] = ("negative", "neutral", "positive")
VALID_LABELS = set(DEFAULT_LABEL_ORDER)

# Financial sentiment ordinal scale used by S-MAE.
ORDINAL_MAP: Dict[str, int] = {
    "negative": -1,
    "neutral": 0,
    "positive": 1,
}

PHRASEBANK_INT_TO_LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

ORDINAL_INT_TO_LABEL = {
    -1: "negative",
    0: "neutral",
    1: "positive",
}


# =====================================
# Per-sample inference telemetry payload
# =====================================
@dataclass
class InferenceRecord:
    latency_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: Optional[int] = None


# ====================================
# Label normalization and validation
# ====================================
def _to_sentiment_label(value: Union[str, int], int_label_scheme: str = "phrasebank") -> str:
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in VALID_LABELS:
            return cleaned
        raise ValueError(f"Unsupported label string: {value}")

    if not isinstance(value, int):
        raise ValueError(f"Label must be str or int, got: {type(value)}")

    if int_label_scheme == "phrasebank":
        mapping = PHRASEBANK_INT_TO_LABEL
    elif int_label_scheme == "ordinal":
        mapping = ORDINAL_INT_TO_LABEL
    else:
        raise ValueError("int_label_scheme must be 'phrasebank' or 'ordinal'")

    if value not in mapping:
        raise ValueError(f"Unsupported int label for scheme '{int_label_scheme}': {value}")

    return mapping[value]


def _normalize_labels(
    labels: Sequence[Union[str, int]], int_label_scheme: str = "phrasebank"
) -> List[str]:
    return [_to_sentiment_label(x, int_label_scheme=int_label_scheme) for x in labels]


# ==================
# Core metric blocks
# ==================
def confusion_matrix(
    y_true: Sequence[Union[str, int]],
    y_pred: Sequence[Union[str, int]],
    labels: Sequence[str] = DEFAULT_LABEL_ORDER,
    int_label_scheme: str = "phrasebank",
) -> Dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    true_labels = _normalize_labels(y_true, int_label_scheme=int_label_scheme)
    pred_labels = _normalize_labels(y_pred, int_label_scheme=int_label_scheme)

    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    matrix = [[0 for _ in range(n)] for _ in range(n)]

    for t, p in zip(true_labels, pred_labels):
        if t not in label_to_idx or p not in label_to_idx:
            raise ValueError(f"Found unknown label pair: true={t}, pred={p}")
        matrix[label_to_idx[t]][label_to_idx[p]] += 1

    return {
        "labels": list(labels),
        "matrix": matrix,  # rows=true, cols=pred
    }


def per_class_precision_recall_f1(
    matrix: Sequence[Sequence[int]], labels: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    n = len(labels)
    result: Dict[str, Dict[str, float]] = {}

    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(n) if r != i)
        fn = sum(matrix[i][c] for c in range(n) if c != i)
        support = sum(matrix[i])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        result[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }

    return result


def sentiment_mae(
    y_true: Sequence[Union[str, int]],
    y_pred: Sequence[Union[str, int]],
    int_label_scheme: str = "phrasebank",
) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    true_labels = _normalize_labels(y_true, int_label_scheme=int_label_scheme)
    pred_labels = _normalize_labels(y_pred, int_label_scheme=int_label_scheme)

    diffs = [abs(ORDINAL_MAP[t] - ORDINAL_MAP[p]) for t, p in zip(true_labels, pred_labels)]
    return mean(diffs) if diffs else 0.0


# =================================
# Efficiency (latency + token usage)
# =================================
def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 100:
        return float(max(values))

    sorted_vals = sorted(values)
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    left = int(pos)
    right = min(left + 1, len(sorted_vals) - 1)
    frac = pos - left
    return sorted_vals[left] * (1 - frac) + sorted_vals[right] * frac


def summarize_inference_efficiency(
    records: Sequence[InferenceRecord],
) -> Dict[str, float]:
    if not records:
        return {
            "sample_count": 0.0,
            "latency_avg_ms": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_min_ms": 0.0,
            "latency_max_ms": 0.0,
            "prompt_tokens_total": 0.0,
            "completion_tokens_total": 0.0,
            "total_tokens_total": 0.0,
        }

    latencies_ms = [max(0.0, r.latency_seconds) * 1000.0 for r in records]

    prompt_tokens_total = sum(max(0, r.prompt_tokens) for r in records)
    completion_tokens_total = sum(max(0, r.completion_tokens) for r in records)
    total_tokens_total = sum(
        max(0, r.total_tokens) if r.total_tokens is not None else max(0, r.prompt_tokens) + max(0, r.completion_tokens)
        for r in records
    )

    return {
        "sample_count": float(len(records)),
        "latency_avg_ms": mean(latencies_ms),
        "latency_p50_ms": _percentile(latencies_ms, 50),
        "latency_p95_ms": _percentile(latencies_ms, 95),
        "latency_min_ms": min(latencies_ms),
        "latency_max_ms": max(latencies_ms),
        "prompt_tokens_total": float(prompt_tokens_total),
        "completion_tokens_total": float(completion_tokens_total),
        "total_tokens_total": float(total_tokens_total),
    }


# =====================
# Unified public facade
# =====================
def compute_all_metrics(
    y_true: Sequence[Union[str, int]],
    y_pred: Sequence[Union[str, int]],
    labels: Sequence[str] = DEFAULT_LABEL_ORDER,
    int_label_scheme: str = "phrasebank",
    inference_records: Optional[Sequence[InferenceRecord]] = None,
) -> Dict[str, Any]:
    cm = confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        int_label_scheme=int_label_scheme,
    )

    matrix = cm["matrix"]
    per_class = per_class_precision_recall_f1(matrix=matrix, labels=cm["labels"])

    correct = sum(matrix[i][i] for i in range(len(matrix)))
    total = sum(sum(row) for row in matrix)
    accuracy = (correct / total) if total > 0 else 0.0

    macro_f1 = mean(v["f1"] for v in per_class.values()) if per_class else 0.0

    metrics: Dict[str, Any] = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "s_mae": sentiment_mae(
            y_true=y_true,
            y_pred=y_pred,
            int_label_scheme=int_label_scheme,
        ),
        # For non-GPT models, pass latency records with token fields set to zero.
        "efficiency": summarize_inference_efficiency(
            records=inference_records or [],
        ),
    }

    return metrics
