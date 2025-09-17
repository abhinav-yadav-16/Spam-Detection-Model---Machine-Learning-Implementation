# Spam-Detection-Model---Machine-Learning-Implementation

# 📧 Spam Email Detection using Machine Learning

## 📌 Project Overview
This project implements a **Spam Email Detection System** using **scikit-learn**.  
The model classifies SMS/email messages as **Spam** or **Ham (Not Spam)** based on their content.

We use the **SMS Spam Collection Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).

---

## 🚀 Features
- Preprocesses text data using **TF-IDF Vectorization**.
- Trains a **Naive Bayes Classifier** for text classification.
- Evaluates the model with:
  - Accuracy
  - Confusion Matrix
  - Precision, Recall, and F1-score
- Achieves high accuracy on test data.

---

## 🛠️ Tech Stack
- **Python 3**
- **pandas** → Data handling
- **scikit-learn** → ML model, train-test split, evaluation
- **numpy** → Numerical operations

---

## 📂 Dataset
- Source: [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip)
- File used: `SMSSpamCollection`
- Format: Tab-separated file with two columns:
  - `label` → spam or ham
  - `message` → SMS/email text

---

## 📊 Steps Implemented
1. **Load Dataset** → Read the SMS Spam Collection dataset.  
2. **Preprocessing** → Encoded labels (ham = 0, spam = 1).  
3. **Train-Test Split** → 80% training, 20% testing.  
4. **Text Vectorization** → Used `TfidfVectorizer` to convert text into numerical features.  
5. **Model Training** → Trained a `MultinomialNB` classifier.  
6. **Model Evaluation** → Measured accuracy, confusion matrix, precision, recall, and F1-score.  

---

## 📈 Results
- The model achieved **high accuracy** on test data.
- Example metrics (may vary depending on train-test split):
  - Accuracy: ~97%
  - Precision, Recall, F1: Very high for both ham and spam classes.

---

## ▶️ How to Run
1. Clone this repository or copy the files.
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
