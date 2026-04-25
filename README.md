# Financial Sentiment Evaluation (Unified Metrics)

This project evaluates financial sentiment classifiers on Financial PhraseBank with a unified metric pipeline.

Current model path:

- GPT API based classifier (`gpt_based_method.py`)

Future model paths (recommended):

- Traditional ML (e.g., SVM)
- Transformer models (e.g., FinBERT)

All model types should connect to the same metric module: `sentiment_metrics.py`.

## Codebase Overview

- `gpt_based_method.py`: Main GPT evaluation entry point. Supports `allagree/75agree/66agree/50agree`, optional fallback to `allagree`, and prints requested vs loaded split.
- `sentiment_metrics.py`: Shared metric engine. Computes confusion matrix, per-class precision/recall/F1, macro-F1, S-MAE, and efficiency statistics.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation and integration guidance.
- `traditional_approaches.ipynb`: Applying traditional ML methods, namely Naive Bayes, logistic regression and SVM to the dataset. Computes confusion matrix, per-class precision/recall/F1
- `lightgbm xgboost.py`: Implements LightGBM and XGBoost models using TF-IDF features, and evaluates them with the shared `compute_all_metrics` framework (accuracy, macro-F1, per-class metrics, confusion matrix, S-MAE).

## Unified Metrics (for all models)

The shared metric module computes:

- Per-class Precision / Recall / F1
- Confusion Matrix (rows=true label, cols=pred label)
- Sentiment Mean Absolute Error (S-MAE)

S-MAE uses an ordinal scale:

- negative = -1
- neutral = 0
- positive = 1

Formula:

`S-MAE = (1/N) * sum(|y_i - y_hat_i|)`

## Efficiency Metrics Policy

- All methods: compute latency metrics.
- GPT method: compute both latency and token usage.
- Non-GPT methods (SVM/FinBERT/local models): set token fields to zero.
- API price/cost is intentionally not calculated.

Recommended average latency calculation:

- `latency_avg_ms = (total_inference_time_seconds / sample_count) * 1000`

Efficiency fields in output:

- `latency_avg_ms`, `latency_p50_ms`, `latency_p95_ms`, `latency_min_ms`, `latency_max_ms`
- `prompt_tokens_total`, `completion_tokens_total`, `total_tokens_total`

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

If you previously installed `datasets>=4`, downgrade to the project-pinned version:

```bash
pip install datasets==2.19.0
```

3. Set API key:

```bash
copy .env.example .env
# then edit .env and set OPENAI_API_KEY
```

## Run

Evaluate all samples:

```bash
python gpt_based_method.py
```

Choose PhraseBank agreement level (default is `allagree`):

```bash
python gpt_based_method.py --agreement-level allagree
python gpt_based_method.py --agreement-level 75agree
python gpt_based_method.py --agreement-level 66agree
python gpt_based_method.py --agreement-level 50agree
```

Evaluate a quick subset (for cheaper/faster testing):

```bash
python gpt_based_method.py --agreement-level allagree --max-samples 100
```

Fail fast when the requested split is unavailable (instead of auto fallback):

```bash
python gpt_based_method.py --agreement-level 75agree --no-fallback
```

## Cache PhraseBank Splits (75/66/50)

If your machine only has `allagree` in cache, pre-download the other splits once:

```bash
python -c "from datasets import load_dataset; [load_dataset('financial_phrasebank', c) for c in ['sentences_75agree','sentences_66agree','sentences_50agree']]"
```

Expected train sizes:

- `sentences_75agree`: 3453
- `sentences_66agree`: 4217
- `sentences_50agree`: 4846

## Output

The GPT evaluation script prints:

- Dataset and validity stats
- Accuracy, Macro-F1, S-MAE
- Per-class Precision / Recall / F1
- Confusion matrix
- GPT efficiency metrics (latency + token totals)

## How current GPT code is connected to unified metrics

`gpt_based_method.py` now imports from `sentiment_metrics.py`:

- `InferenceRecord`
- `compute_all_metrics`

Flow:

1. Predict each sample with GPT.
2. Collect `y_true` and `y_pred`.
3. Collect one `InferenceRecord` per valid GPT prediction (latency + token usage).
4. Call:

```python
metrics = compute_all_metrics(
	y_true=y_true,
	y_pred=y_pred,
	inference_records=inference_records,
)
```

5. Print classification + efficiency metrics from `metrics`.

## How to connect future models (SVM, FinBERT, etc.)

Use this checklist to connect any non-GPT model to the same metric pipeline.

1. Prepare dataset rows and labels

- Load one PhraseBank split (`sentences_allagree/75agree/66agree/50agree`).
- Build `inputs` from `row["sentence"]`.
- Build `y_true` from `row["label"]`.
- Keep `inputs[i]` aligned with `y_true[i]`.

2. Implement model inference adapter

- Create one function that maps one sentence to one label in `{negative, neutral, positive}`.
- If your model outputs logits or IDs, convert to the canonical labels before evaluation.
- For binary models, add an explicit rule to map to three-way sentiment or skip unsupported samples.

3. Measure per-sample latency

- Time each prediction with `time.perf_counter()`.
- Append one `InferenceRecord` for each successful prediction.
- For non-GPT models, set token fields to zero.

4. Handle invalid predictions

- If output is outside `{negative, neutral, positive}`, convert to `invalid` and skip from `y_pred/y_true` pair lists.
- Log invalid count separately so final metrics stay interpretable.

5. Compute unified metrics

- Call `compute_all_metrics(...)` once after inference.
- Pass `inference_records` to enable latency summary.
- Keep `int_label_scheme="phrasebank"` if labels are `0/1/2`.

6. Report results consistently

- Always report: sample counts, accuracy, macro-F1, S-MAE, per-class table, confusion matrix, efficiency.
- For non-GPT methods, token totals should remain zero by design.

### Minimal Adapter Template

```python
import time
from typing import List

from sentiment_metrics import InferenceRecord, compute_all_metrics

VALID_LABELS = {"negative", "neutral", "positive"}


def normalize_pred(raw: str) -> str:
	pred = raw.strip().lower()
	return pred if pred in VALID_LABELS else "invalid"


def evaluate_local_model(inputs: List[str], y_true: List[str], model_predict):
	y_true_valid = []
	y_pred = []
	inference_records = []
	invalid_count = 0

	for sentence, true_label in zip(inputs, y_true):
		start = time.perf_counter()
		raw_pred = model_predict(sentence)
		end = time.perf_counter()

		pred = normalize_pred(str(raw_pred))
		if pred == "invalid":
			invalid_count += 1
			continue

		y_true_valid.append(true_label)
		y_pred.append(pred)
		inference_records.append(
			InferenceRecord(
				latency_seconds=max(0.0, end - start),
				prompt_tokens=0,
				completion_tokens=0,
				total_tokens=0,
			)
		)

	metrics = compute_all_metrics(
		y_true=y_true_valid,
		y_pred=y_pred,
		int_label_scheme="phrasebank",
		inference_records=inference_records,
	)
	return metrics, invalid_count
```

### Notes For SVM / FinBERT

- SVM (e.g., scikit-learn): map class IDs to labels once (for example with a dictionary), then reuse `normalize_pred`.
- FinBERT (Hugging Face Transformers): map model labels (`positive/neutral/negative`) directly if names match; otherwise map from `LABEL_0/LABEL_1/LABEL_2` explicitly.
- Batch inference is fine, but latency should still be recorded in a way that is comparable across methods (per sample average is recommended).

## Label Compatibility

The metric module supports labels as:

- strings: `negative`, `neutral`, `positive`
- integers in PhraseBank style: `0, 1, 2` (`int_label_scheme="phrasebank"`)
- integers in ordinal style: `-1, 0, 1` (`int_label_scheme="ordinal"`)

Example:

```python
metrics = compute_all_metrics(
	y_true=[0, 1, 2],
	y_pred=[0, 2, 2],
	int_label_scheme="phrasebank",
)
```
