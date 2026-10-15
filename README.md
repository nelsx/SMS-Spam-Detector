# 📬 SMS Spam Detector ML System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/status-complete-success.svg)]()

An end-to-end machine learning pipeline and interactive application designed to analyze and classify SMS messages as either spam or legitimate (ham), leveraging Natural Language Processing (NLP) and TF-IDF feature extraction.

---

## 📌 Features

- **Automated Data Processing Pipeline:** Structured ingestion, noise reduction, and data cleaning transforming raw SMS texts into analytics-ready data.
- **TF-IDF Feature Extraction:** Serialized vectorizer mapping raw text documents into weighted numerical feature vectors for consistent inference.
- **Supervised Classification Model:** Optimized Logistic Regression model trained to accurately separate spam messages from normal correspondence.
- **Interactive Application Interface:** Dedicated deployment script for running real-time SMS classification queries.

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Data Manipulation & Processing:** Pandas, NumPy
- **Machine Learning & NLP:** Scikit-Learn (TF-IDF Vectorizer, Logistic Regression), Joblib
- **Interface / Deployment:** Python Scripting

---

## 📁 Repository Structure

| Component | Path / File | Description |
| :--- | :--- | :--- |
| **Raw Dataset** | `SMS Spam Detector/Data/sms_spam_dataset.csv` | Raw SMS message dataset containing uncleaned text inputs[cite: 1]. |
| **Cleaned Dataset** | `SMS Spam Detector/Data/cleaned_sms_spam_dataset.csv` | Preprocessed text data normalized for machine learning[cite: 1]. |
| **Encoded Dataset** | `SMS Spam Detector/Data/encoded_sms_spam_dataset.csv` | Fully structured dataset ready for downstream model ingestion[cite: 1]. |
| **Data Cleaning Notebook** | `SMS Spam Detector/Data/Data Cleaning.ipynb` | Jupyter Notebook handling missing data, noise reduction, and text normalization[cite: 1]. |
| **Data Encoding Notebook** | `SMS Spam Detector/Data/Data Encoding.ipynb` | Jupyter Notebook managing target encoding and feature structuring steps[cite: 1]. |
| **Model Training Notebook** | `SMS Spam Detector/Models/Logistric Regression.ipynb` | Experimental notebook dedicated to training and tuning the classification model[cite: 1]. |
| **Classification Model** | `SMS Spam Detector/Artifacts/Logistic_Regression_SMS_Spam.joblib` | Serialized Logistic Regression model weights and configuration[cite: 1]. |
| **TF-IDF Vectorizer** | `SMS Spam Detector/Artifacts/TfidfVectorizer_SMS_Spam.joblib` | Serialized text vectorizer ensuring consistent feature space mapping[cite: 1]. |
| **Application Script** | `SMS Spam Detector/App/Model_App.py` | Core application script running inference and serving the detection utility[cite: 1]. |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/sms-spam-detector.git](https://github.com/your-username/sms-spam-detector.git)
cd sms-spam-detector
