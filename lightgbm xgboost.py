#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sentiment_metrics import compute_all_metrics


# In[13]:


df = pd.read_csv("all-data.csv", header=None, encoding="latin-1")
df.columns = ["label", "text"]

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)


# In[14]:


tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF shape:", X_train_tfidf.shape)


# In[15]:


lgb_model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=-1,
    random_state=42
)

lgb_model.fit(X_train_tfidf, y_train)

y_pred_lgb = lgb_model.predict(X_test_tfidf)


# In[16]:


label_to_int = {"negative":0, "neutral":1, "positive":2}
int_to_label = {0:"negative", 1:"neutral", 2:"positive"}

y_train_num = [label_to_int[y] for y in y_train]

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)

xgb_model.fit(X_train_tfidf, y_train_num)

y_pred_xgb = xgb_model.predict(X_test_tfidf)

y_pred_xgb = [int_to_label[y] for y in y_pred_xgb]


# In[17]:


def print_metrics(model_name, y_true, y_pred):
    metrics = compute_all_metrics(
        y_true=y_true.tolist(),
        y_pred=y_pred,
        inference_records=None
    )

    print("\n" + "="*50)
    print(f"  {model_name}")
    print("="*50)

    # Overall
    print("\n[Overall Metrics]")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Macro F1 : {metrics['macro_f1']:.4f}")
    print(f"S-MAE    : {metrics['s_mae']:.4f}")

    # Per-class
    print("\n[Per-class breakdown]")
    for cls, vals in metrics['per_class'].items():
        print(
            f"  {cls:10s}  "
            f"P={vals['precision']:.3f}  "
            f"R={vals['recall']:.3f}  "
            f"F1={vals['f1']:.3f}  "
            f"support={int(vals['support'])}"
        )

    # Confusion Matrix
    print("\n[Confusion Matrix]")
    labels = metrics['confusion_matrix']['labels']
    matrix = metrics['confusion_matrix']['matrix']

    print("Labels:", labels)
    print("Matrix (rows=true, cols=pred):")
    for row in matrix:
        print(row)

    # Efficiency
    eff = metrics['efficiency']
    print("\n[Efficiency]")
    print(f"Samples        : {int(eff['sample_count'])}")
    print(f"Latency avg(ms): {eff['latency_avg_ms']:.2f}")
    print(f"P50 latency(ms): {eff['latency_p50_ms']:.2f}")
    print(f"P95 latency(ms): {eff['latency_p95_ms']:.2f}")


# In[18]:


print_metrics("LightGBM", y_test, y_pred_lgb)
print_metrics("XGBoost", y_test, y_pred_xgb)


# In[ ]:




