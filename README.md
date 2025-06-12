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

✅ LLM Prediction Accuracy: 82.37%

---

## 📂 File Structure
<code>
  <pre>
  llm-datawarehouse-prediction/
  ├── llm_model_pipeline.py      # Full LLM implementation
  ├── README.md
  └── sample_data/ (optional)    # If data is sharable, include samples
</pre>
</code>


---

#3 📎 Sample Code Snippet

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
predictions = classifier(list(test_sequences_df["Activity_Sequence"]), labels, batch_size=8)

---

## 📌 Notes


*Model: facebook/bart-large-mnli from Hugging Face

*Code was developed and tested in Google Colab

*Dataset not included due to academic restrictions

---

## 👩‍💻 My Contribution

*Implemented the LLM pipeline for sequence prediction

*Preprocessed CSV logs and formatted sequences

*Tuned the zero-shot classifier and evaluated performance







