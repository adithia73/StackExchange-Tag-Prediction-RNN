🧠 StackExchange Tag Prediction with RNN, LSTM & GRU
Automatically predict relevant tags for a StackExchange question using the body of the question. This project explores deep learning models (RNN, LSTM, GRU) for multi-label text classification.

📌 Problem Statement
Build a machine learning model to predict appropriate tags for StackExchange questions using their content. This is a multi-label classification task where each question can have multiple tags.

📂 Dataset
Source: Kaggle - StackOverflow Stats Questions

Files Used: Questions.csv, Tags.csv

License: CC-BY-SA 3.0 (Attribution required)

🧪 Approach
🔹 1. Data Preparation
Loaded both Questions.csv and Tags.csv.

Merged datasets on the Id field.

Cleaned the Body of each question using:

BeautifulSoup for HTML removal

Lowercasing, special character filtering, stopword removal

🔹 2. Tag Processing
Selected top 10 most frequent tags.

Filtered rows with at least 2 of those top 10 tags.

Applied MultiLabelBinarizer to encode tags into a binary matrix.

🔹 3. Text Preprocessing
Tokenized and vectorized question body using Tokenizer from Keras.

Limited vocabulary size and applied padding to a max length (max_len=100).

🧠 Model Architectures
Three deep learning models were implemented and compared:

✅ RNN Model
python
Copy
Edit
Embedding -> SimpleRNN -> Dense -> Output(sigmoid)
✅ LSTM Model
python
Copy
Edit
Embedding -> LSTM -> Dense -> Output(sigmoid)
✅ GRU Model
python
Copy
Edit
Embedding -> GRU -> Dense -> Output(sigmoid)
All models used binary_crossentropy loss and Adam optimizer.

Models were trained with checkpointing on validation loss.

📊 Evaluation Strategy
Predicted tag probabilities for validation samples.

Tuned the classification threshold from 0.01 to 0.50.

Chose the best threshold based on macro F1-score.

📈 Model Performance
Model	Accuracy	Precision	Recall	F1 Score
RNN	0.7940	0.84	0.79	0.81
LSTM	0.8051	0.85	0.80	0.82
GRU	0.8112	0.86	0.81	0.83

✅ GRU yielded the best macro F1-score and general accuracy.

🔧 Dependencies
Python 3.x

Pandas, NumPy

BeautifulSoup

Scikit-learn

TensorFlow / Keras

Matplotlib

🚀 How to Run
Clone this repository:

bash
Copy
Edit
git clone https://github.com/<your-username>/StackExchange-Tag-Prediction-RNN.git
cd StackExchange-Tag-Prediction-RNN
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the notebook:

bash
Copy
Edit
jupyter notebook Predict_Tags_With_RNN__LSTM_GRU.ipynb
