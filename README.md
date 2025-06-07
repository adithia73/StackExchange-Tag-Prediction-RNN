# 🧠 StackExchange Tag Prediction with RNN

Automatically predict relevant tags for a StackExchange question using its text content with a Recurrent Neural Network (RNN).

---

## 📌 Problem Statement

The goal of this project is to build an NLP-based deep learning model to predict tags for a given StackExchange question using only its title and body text. This automation can improve content categorization and discoverability on community platforms like Stack Overflow.

---

## 📂 Dataset

- **Source:** [Kaggle - StackOverflow Stats Questions](https://www.kaggle.com/datasets/stackoverflow/statsquestions#Questions.csv)
- **License:** CC-BY-SA 3.0 (Attribution required)

The dataset contains StackExchange questions related to statistics and includes:
- `Id`
- `Title`
- `Body`
- `Tags`
- `CreationDate`, etc.

---

## 🧪 Approach

1. **Data Cleaning & Preprocessing**
   - HTML tag removal
   - Stopword removal
   - Lemmatization

2. **Text Tokenization**
   - Tokenized question titles and bodies
   - Padded sequences for input to RNN

3. **Model Architecture**
   - Embedding Layer
   - Bidirectional LSTM
   - Dense layers with sigmoid activation for multi-label classification

4. **Evaluation Metrics**
   - Precision, Recall, F1-Score
   - Custom threshold optimization for tag prediction

---

## 📊 Performance

The model is evaluated using macro/micro F1-score on a held-out validation set. Multi-label classification threshold tuning was applied for optimal performance.

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- Numpy, Pandas
- Scikit-learn
- NLTK / BeautifulSoup (for preprocessing)

---

## 🚀 Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/StackExchange-Tag-Prediction-RNN.git
   cd StackExchange-Tag-Prediction-RNN
