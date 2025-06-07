# 🧠 StackExchange Tag Prediction with RNN, LSTM & GRU

Automatically predict relevant tags for a StackExchange question using the body of the question. This project explores deep learning models (RNN, LSTM, GRU) for multi-label text classification.

---

## 📌 Problem Statement

Build a machine learning model to predict appropriate tags for StackExchange questions using their content. This is a **multi-label classification** task where each question can have multiple tags. Automating this process helps improve searchability, categorization, and content indexing.

---

## 📂 Dataset

- **Source:** [Kaggle - StackOverflow Stats Questions](https://www.kaggle.com/datasets/stackoverflow/statsquestions)
- **Files Used:**
  - `Questions.csv` – contains question text and metadata
  - `Tags.csv` – contains associated tags
- **License:** CC-BY-SA 3.0 (Attribution required)

---

## 🧪 Approach

### 🔹 1. Data Preparation
- Loaded `Questions.csv` and `Tags.csv` and merged them using `Id`.
- Cleaned the `Body` field:
  - Removed HTML using BeautifulSoup
  - Removed non-alphabetic characters
  - Lowercased text and removed stopwords

### 🔹 2. Tag Processing
- Extracted the top 10 most frequent tags.
- Filtered to include questions with at least 2 of those top 10 tags.
- Encoded tags using `MultiLabelBinarizer`.

### 🔹 3. Text Vectorization
- Tokenized the cleaned question body using Keras' `Tokenizer`.
- Limited vocabulary to top 20,000 words.
- Converted text into padded sequences with a max length of 100.

---

## 🧠 Model Architectures

Implemented three types of deep learning models using Keras' Sequential API:

### ✅ RNN Model
Embedding -> SimpleRNN -> Dense -> Output(sigmoid)

### ✅ LSTM Model
Embedding -> LSTM -> Dense -> Output (sigmoid)

### ✅ GRU Model
Embedding -> GRU -> Dense -> Output (sigmoid)

- All models use `binary_crossentropy` loss and `Adam` optimizer.
- Training monitored using validation loss with model checkpointing.

---

## 📊 Evaluation

- Predictions are multi-label probabilities.
- Optimized threshold from 0.01 to 0.5 to convert probabilities to binary predictions.
- Selected the best threshold based on **macro F1-score**.

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **RNN**  | 0.7940   | 0.84      | 0.79   | 0.81     |
| **LSTM** | 0.8051   | 0.85      | 0.80   | 0.82     |
| **GRU**  | 0.8112   | 0.86      | 0.81   | 0.83     |

✅ **GRU** performed the best in terms of F1-score and accuracy.

---

## 🔧 Dependencies

- Python 3.x
- Pandas
- NumPy
- BeautifulSoup4
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Jupyter Notebook


