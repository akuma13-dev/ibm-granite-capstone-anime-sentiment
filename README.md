# Anime Review Sentiment Classification & One‑Sentence Summarization (Granite 3.3 Instruct)

> **Capstone Project — IBM Skills Build × Hactive8.** Zero‑shot sentiment classification (Positive / Neutral / Negative) plus one‑sentence summaries for anime reviews. The analysis runs on **Google Colab (Tesla T4)** with **4‑bit quantization** for efficiency.

---

## Executive Summary

* **Model**: `ibm‑granite/granite‑3.3‑8b‑instruct` (4‑bit) with a strict **JSON‑only** prompt and balanced few‑shots.
* **Evaluation set**: stratified **sample ≈ 200** reviews derived from the dataset.
* **Headline results** (sample ≈ 200): **Accuracy ≈ 0.77**, **Macro‑F1 ≈ 0.55**. Largest error pattern: **Neutral → Positive**.
* **Summaries**: concise and on‑topic; **median = 14 words**, **95th percentile = 24 words** (target ≤ 25 words).

---

## Dataset & Labeling

* **Unit**: 1 row = 1 user review. Text cleaned (trim, drop NaN).
* **Weak labels** from `score_user` using tighter thresholds: **Positive ≥ 8**, **Negative ≤ 3**, otherwise **Neutral**.
* **Label distribution (full dataset)**: Negative **14,608** • Neutral **35,271** • Positive **135,079**  → **skewed Positive**.
* **Sample (≈200) distributions**:

  * **Weak labels**: Negative **16** • Neutral **39** • Positive **145**.
  * **Model predictions**: Negative **52** • Neutral **13** • Positive **134** (shift away from Neutral).

---

## Methodology (Notebook‑based)

* **Prompting**: English‑only; class rules:

  * *Positive*: positive emotions/praise dominate.
  * *Negative*: negative emotions/criticism dominate.
  * *Neutral*: balanced / descriptive / ambiguous.
* **Inference**: deterministic (no sampling), batched generation with **max\_input\_len = 640** and **max\_new\_tokens = 48**; robust JSON parsing with regex fallback.
* **Metrics & Visuals**: `classification_report`, **confusion matrix** (counts + row‑normalized), **label distributions**, **error examples**, and **summary length histogram**.

---

## Evaluation (Sample ≈ 200)

**Overall**

* **Accuracy** ≈ **0.77**
* **Macro‑F1** ≈ **0.55**

**Per‑class performance (typical from the latest run)**

* **Negative** — Precision ≈ **0.29**, Recall ≈ **0.94**, F1 ≈ **0.44**
* **Neutral** — Precision ≈ **0.54**, Recall ≈ **0.18**, F1 ≈ **0.27**
* **Positive** — Precision ≈ **0.98**, Recall ≈ **0.90**, F1 ≈ **0.94**

**Error patterns (from confusion matrix & examples)**

* Dominant confusion: **Neutral → Positive** (descriptive or mildly positive text pulled into Positive).
* Secondary confusions exist (e.g., Neutral ↔ Negative) but are smaller.

---

## Summary Quality

* **Length control**: **median 14 words**, **p95 24 words** → within the ≤ 25‑word target.
* **Content**: summaries stay on topic and reflect review stance; occasional over‑positivity mirrors the label bias.

---

## Limitations

* **Weak‑label noise** around the 3/8 thresholds impacts ground truth quality.
* **Class imbalance** (Positive‑skewed) inflates accuracy while **Macro‑F1** exposes minority‑class drop, especially **Neutral recall**.
* **LLM cost/latency**: running an 8B LLM across 46k+ reviews is inefficient without a selective strategy.

---

## Recommendations

1. **Two‑stage pipeline**:

   * **Fast classifier** (TF‑IDF + LogisticRegression or a small encoder) for full‑corpus sentiment.
   * Use **Granite** for **one‑sentence summaries** and **ambiguous cases** (or periodic audits), not for every record.
2. **Prompt & policy**: keep English‑only JSON prompt; enrich **Neutral** few‑shots; formalize Neutral as *balanced/ambiguous/descriptive*.
3. **Calibration**: simple rule (if top‑2 confidences are close → set **Neutral**) or small **voting (k=3)** on critical subsets.
4. **Future work**: light supervised fine‑tuning on weak labels for sentiment; Granite retained for summarization.

---

## Artifacts & Links

* **Notebook (Colab)**: [https://colab.research.google.com/drive/1fyh0LSphvANVr-ltksHTFCKtmUeoksD4?usp=sharing](https://colab.research.google.com/drive/1fyh0LSphvANVr-ltksHTFCKtmUeoksD4?usp=sharing)
* **Notebook file**: `notebooks/granite_classification_summarization.ipynb`
* **Outputs**: prediction CSVs saved under `outputs/` (checkpoint & final) as shown in the notebook.
**Dataset** [https://drive.google.com/file/d/17nYUXlQItz_1JNOq6jtb_Bb75QNdukP2/view?usp=sharing]


---

## Acknowledgements

Prepared as part of the **IBM Skills Build × Hactive8** capstone. Thanks to IBM Granite and Google Colab.
