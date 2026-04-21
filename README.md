# Financial Sentiment Evaluation (Unified Metrics)

This project evaluates financial sentiment classifiers on Financial PhraseBank with a unified metric pipeline.

Current model path:

- GPT API based classifier (`gpt_based_method.py`)

Future model paths (recommended):

- Traditional ML (e.g., SVM)
- Transformer models (e.g., FinBERT)

All model types should connect to the same metric module: `sentiment_metrics.py`.

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

For non-GPT models, collect real per-sample latency and set token fields to zero.

Important:

- In real evaluation, latency must be measured at runtime for each sample.

```python
import time

from sentiment_metrics import InferenceRecord, compute_all_metrics

# y_true and y_pred are aligned lists of labels
# Example: measure per-sample latency for a non-GPT model
inference_records = []
for x in inputs:
	start = time.perf_counter()
	pred = model_predict(x)
	end = time.perf_counter()

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
	y_true=y_true,
	y_pred=y_pred,
	inference_records=inference_records,
)
```

This keeps latency metrics meaningful for all models and keeps token metrics as zero for non-GPT models.

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
