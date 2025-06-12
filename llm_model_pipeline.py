import pandas as pd
from transformers import pipeline
from google.colab import files
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os

### 1️⃣ Load Processed Data ###

# Upload files manually
uploaded = files.upload()

# Display uploaded file names
print("Uploaded Files:", uploaded.keys())

# Define file paths for training and testing
train_files = ["processed_S01.csv", "processed_S03.csv", "processed_S04.csv"]
test_files = ["processed_S07.csv", "processed_S15.csv"]

# Load training datasets and concatenate them
train_dfs = [pd.read_csv(file) for file in train_files]
train_df = pd.concat(train_dfs, ignore_index=True)

# Load testing datasets and concatenate them
test_dfs = [pd.read_csv(file) for file in test_files]
test_df = pd.concat(test_dfs, ignore_index=True)

print(f"✅ Training dataset size: {train_df.shape}")
print(f"✅ Testing dataset size: {test_df.shape}")

### 2️⃣ Generate Activity Sequences ###

def create_sequences(df, sequence_length=5):
    """Generate activity sequences from activity locations and the next subprocess."""
    data = []
    for i in range(len(df) - sequence_length):
        activity_sequence = " -> ".join(df['Activity_Main_Area'].iloc[i:i+sequence_length].tolist())
        next_subprocess = df['Sub-Process'].iloc[i + sequence_length]
        data.append((activity_sequence, next_subprocess))
    return pd.DataFrame(data, columns=["Activity_Sequence", "Next_Sub-Process"])

# Create structured datasets
train_sequences_df = create_sequences(train_df)
test_sequences_df = create_sequences(test_df)

### 3️⃣ Load Data for LLM Prediction ###

# Load the zero-shot classification model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Extract possible labels (unique subprocesses)
labels = train_sequences_df["Next_Sub-Process"].unique().tolist()

# Batch processing for efficiency
predictions = classifier(list(test_sequences_df["Activity_Sequence"]), labels, batch_size=8)

# Extract top predicted labels from the results
predicted_labels = [result["labels"][0] for result in predictions]

# Ensure both test labels and predictions are strings
test_labels = test_sequences_df["Next_Sub-Process"].astype(str)
predicted_labels = [str(label) for label in predicted_labels]

# Compute accuracy
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"✅ LLM Prediction Accuracy: {accuracy * 100:.2f}%")

# Compute classification report
report = classification_report(test_labels, predicted_labels, digits=2)

# Print the formatted classification report
print("Classification Report:\n")
print(report)

print("\nPredicted subprocess for new sequence:", predicted_labels[:1])
