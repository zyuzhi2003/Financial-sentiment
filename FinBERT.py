import time
from datasets import load_dataset
from transformers import pipeline

# 从你的 metrics 模块导入所有需要的工具
from sentiment_metrics import (
    compute_all_metrics,
    InferenceRecord,
    DEFAULT_LABEL_ORDER,
)

# -------------------- 加载数据 --------------------
print("Loading Financial Phrasebank (all agree)...")
dataset = load_dataset("financial_phrasebank", "sentences_50agree")
data = dataset["train"]
sentences = data["sentence"]
dataset_labels_int = data["label"]          # 0: negative, 1: neutral, 2: positive

# 映射数据集整数标签到语义标签字符串
PHRASEBANK_INT_TO_LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}
true_labels = [PHRASEBANK_INT_TO_LABEL[i] for i in dataset_labels_int]

# -------------------- 加载 FinBERT 推理管道 --------------------
print("Loading FinBERT pipeline...")
finbert = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
    device=0 if __import__("torch").cuda.is_available() else -1  # 自动使用 GPU
)

# -------------------- 逐样本预测并记录延迟 --------------------
print(f"Running inference on {len(sentences)} sentences...")
pred_labels = []
inference_records = []

for i, sentence in enumerate(sentences):
    start = time.perf_counter()
    # 预测单条，返回 [{"label": "POSITIVE"/"NEGATIVE"/"NEUTRAL", "score": ...}]
    result = finbert(sentence)[0]
    elapsed = time.perf_counter() - start

    # 转为小写，与我们的标签体系一致
    pred_label = result["label"].lower()
    pred_labels.append(pred_label)

    # 记录延迟（FinBERT 没有 token 概念，token 字段填 0）
    inference_records.append(
        InferenceRecord(
            latency_seconds=elapsed,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
    )

    if (i + 1) % 200 == 0:
        print(f"  Processed {i + 1}/{len(sentences)}...")

# -------------------- 计算并展示所有指标 --------------------
metrics = compute_all_metrics(
    y_true=true_labels,          # 真实标签（字符串）
    y_pred=pred_labels,          # 预测标签（字符串）
    labels=DEFAULT_LABEL_ORDER,  # ["negative", "neutral", "positive"]
    int_label_scheme="phrasebank",
    inference_records=inference_records,
)

print("\n" + "=" * 50)
print("FINBERT EVALUATION ON FINANCIAL PHRASEBANK (all agree)")
print("=" * 50)
print(f"Accuracy         : {metrics['accuracy']:.4f}")
print(f"Macro F1          : {metrics['macro_f1']:.4f}")
print(f"S‑MAE (lower better): {metrics['s_mae']:.4f}\n")

print("--- Per‑Class Metrics ---")
for label, stats in metrics["per_class"].items():
    print(f"  {label:8s}  "
          f"Precision: {stats['precision']:.4f}  "
          f"Recall: {stats['recall']:.4f}  "
          f"F1: {stats['f1']:.4f}  "
          f"Support: {int(stats['support'])}")

print("\n--- Confusion Matrix (rows=true, cols=pred) ---")
cm = metrics["confusion_matrix"]
print(f"Labels: {cm['labels']}")
for i, row in enumerate(cm["matrix"]):
    print(f"  {cm['labels'][i]:8s}: {row}")

print("\n--- Inference Efficiency ---")
eff = metrics["efficiency"]
print(f"  Samples         : {int(eff['sample_count'])}")
print(f"  Latency (avg)   : {eff['latency_avg_ms']:.1f} ms")
print(f"  Latency (p50)   : {eff['latency_p50_ms']:.1f} ms")
print(f"  Latency (p95)   : {eff['latency_p95_ms']:.1f} ms")
print(f"  Latency (min)   : {eff['latency_min_ms']:.1f} ms")
print(f"  Latency (max)   : {eff['latency_max_ms']:.1f} ms")
print(f"  Token totals    : prompt={int(eff['prompt_tokens_total'])}, "
      f"completion={int(eff['completion_tokens_total'])}, "
      f"total={int(eff['total_tokens_total'])}")