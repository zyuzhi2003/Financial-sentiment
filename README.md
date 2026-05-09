# Financial Sentiment Evaluation (Unified Metrics)

This project evaluates financial sentiment classifiers on Financial PhraseBank with a shared metrics pipeline. It currently includes two main model paths: a GPT API classifier and a FinBERT transformer baseline, both reporting the same accuracy and efficiency metrics.

## What is implemented

- GPT API based classifier: [gpt_based_method.py](gpt_based_method.py)
- FinBERT evaluation pipeline: [FinBERT.py](FinBERT.py)
- Unified metrics engine: [sentiment_metrics.py](sentiment_metrics.py)

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

3. Set API key (GPT only):

```bash
copy .env.example .env
# then edit .env and set at least one provider key:
# OPENAI_API_KEY or DEEPSEEK_API_KEY
```

Provider env vars:

- `OPENAI_API_KEY`, optional `OPENAI_BASE_URL` (default: `https://aihubmix.com/v1`)
- `DEEPSEEK_API_KEY`, optional `DEEPSEEK_BASE_URL` (default: `https://api.deepseek.com/v1`)

## Run

### GPT-based method

This script loads a PhraseBank split, prompts a chat model to output exactly one of `positive/negative/neutral`, retries once with a stricter prompt if needed, and evaluates valid predictions with unified metrics. It also records per-sample latency and token usage.

Evaluate all samples (default split is `allagree`):

```bash
python gpt_based_method.py
```

Select provider/model:

```bash
# OpenAI
python gpt_based_method.py --provider openai --model gpt-3.5-turbo

# DeepSeek
python gpt_based_method.py --provider deepseek --model deepseek-v4-flash
```

Choose PhraseBank agreement level:

```bash
python gpt_based_method.py --agreement-level allagree
python gpt_based_method.py --agreement-level 75agree
python gpt_based_method.py --agreement-level 66agree
python gpt_based_method.py --agreement-level 50agree
```

Evaluate a quick subset:

```bash
python gpt_based_method.py --provider openai --model gpt-3.5-turbo --agreement-level allagree --max-samples 100
```

Fail fast when a requested split is unavailable (instead of auto fallback):

```bash
python gpt_based_method.py --agreement-level 75agree --no-fallback
```

Notes:

- `--max-output-tokens` default is provider-aware: `openai=32`, `deepseek=64`.
- When output is `invalid`, the script retries once using a stricter prompt.

### FinBERT method

This script uses the Transformers `pipeline("sentiment-analysis")` with `ProsusAI/finbert`. It maps the pipeline labels to `negative/neutral/positive`, records per-sample latency (token fields are zero), and evaluates with the same unified metrics.

Run FinBERT evaluation (current split is `sentences_50agree` in code):

```bash
python FinBERT.py
```

Notes:

- If a CUDA GPU is available, the script will use it automatically; otherwise it runs on CPU.
- To change the split, edit the dataset config in `FinBERT.py`.

## Cache PhraseBank splits (75/66/50)

If your machine only has `allagree` in cache, pre-download the other splits once:

```bash
python -c "from datasets import load_dataset; [load_dataset('financial_phrasebank', c) for c in ['sentences_75agree','sentences_66agree','sentences_50agree']]"
```

Expected train sizes:

- `sentences_75agree`: 3453
- `sentences_66agree`: 4217
- `sentences_50agree`: 4846

## Metrics summary

All methods report:

- Accuracy and macro-F1
- Per-class precision/recall/F1
- Confusion matrix (rows = true, cols = pred)
- S-MAE using the ordinal mapping: negative = -1, neutral = 0, positive = 1
- Latency statistics (token totals are only non-zero for GPT)
