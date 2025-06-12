# 🧠 LLM-Based Subprocess Prediction in Data Warehouse Activities

This project uses a Large Language Model (LLM) to predict the next subprocess in a data warehouse activity flow using zero-shot classification.

> ✅ This was part of my Master's coursework. I was responsible for implementing and evaluating the LLM pipeline.

---

## 📌 Objective

To predict the next subprocess in a business activity sequence without supervised training, by leveraging the semantic understanding of a pre-trained LLM (`facebook/bart-large-mnli`).

---

## 🛠️ Technologies Used

- Python
- Hugging Face Transformers
- BART-Large-MNLI (zero-shot classification)
- Pandas, Scikit-learn
- Google Colab
- Matplotlib, Seaborn

---

## 🚀 Pipeline Overview

1. **Data Loading:** Read multiple CSV files with activity logs.
2. **Sequence Generation:** Create 5-step activity sequences.
3. **Model Inference:** Use `pipeline("zero-shot-classification")` to classify the next subprocess.
4. **Evaluation:** Measure prediction accuracy and print a detailed classification report.

---

## 🧪 Accuracy

```text
✅ LLM Prediction Accuracy: 82.37%

---

📂 File Structure
