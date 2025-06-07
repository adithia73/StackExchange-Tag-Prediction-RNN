# 🧠 StackExchange Tag Prediction with RNN

Automatically predict relevant tags for a StackExchange question using the question's body text with a Recurrent Neural Network (RNN).

---

## 📌 Problem Statement

Build a machine learning model to predict appropriate tags for StackExchange questions using their content. This multi-label classification problem helps categorize user questions effectively, enabling better content search and topic indexing.

---

## 📂 Dataset

- **Source:** [Kaggle - StackOverflow Stats Questions](https://www.kaggle.com/datasets/stackoverflow/statsquestions#Questions.csv)
- **Files Used:** `Questions.csv`, `Tags.csv`
- **License:** CC-BY-SA 3.0 (Attribution required)

---

## 🧪 Approach

### 1. Data Preparation

- Loaded and merged `Questions.csv` with `Tags.csv` on `Id`.
- Cleaned the `Body` field using:
  - HTML tag removal with BeautifulSoup
  - Non-alphabetic character removal
  - Lowercasing and whitespace normalization

### 2. Tag Processing

- Selected top 10 most frequent tags.
- Filtered records containing at least 2 of these tags.
- Used `MultiLabelBinarizer` to encode tags into a binary matrix.

### 3. Text Vectorization

- Tokenized cleaned text using Keras' `Tokenizer`.
- Limited vocabulary to words appearing more than 3 times.
- Converted text into padded sequences of max length 100.

### 4. Model Architecture

A Sequential Keras model with the following layers:
- `Embedding` layer
- `SimpleRNN` layer with 128 units
- Fully connected `Dense` layer with ReLU
- Output `Dense` layer with sigmoid activation (for multi-label classification)

### 5. Training

- Used `binary_crossentropy` loss and `Adam` optimizer.
- Trained for 10 epochs with checkpointing on validation loss.

### 6. Prediction & Evaluation

- Predicted tag probabilities on validation data.
- Converted probabilities to binary outputs using a threshold.
- Optimized F1-score over a range of thresholds (0 to 0.5 in 0.01 steps).
- Selected the best threshold based on macro F1-score.

---

## 🔧 Dependencies

- Python
- Pandas, NumPy
- BeautifulSoup
- Scikit-learn
- TensorFlow / Keras
- Matplotlib

---

## 🧠 Key Metrics

   precision    recall  f1-score   support

           0       0.91      0.82      0.86     17520
           1       0.50      0.69      0.58      4700

    accuracy                           0.79     22220
   macro avg       0.71      0.75      0.72     22220
weighted avg       0.82      0.79      0.80     22220

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/StackExchange-Tag-Prediction-RNN.git
   cd StackExchange-Tag-Prediction-RNN
