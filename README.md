# 🩺 Heart Disease Prediction App

A machine learning project to predict the likelihood of heart disease based on clinical parameters. Built using Python, Scikit-learn, and Streamlit.

---

## 📌 Project Overview

This project aims to build a predictive model that determines whether a person is at risk of developing heart disease. It uses classification algorithms to analyze key health indicators like age, cholesterol, blood pressure, and more.

---

## 💡 Key Features

- ✅ Cleaned and visualized real-world medical data  
- ✅ Trained multiple ML classifiers (Logistic Regression, Random Forest)  
- ✅ Evaluated using accuracy, precision, recall, F1-score  
- ✅ Built an interactive web app with **Streamlit**  
- ✅ Model deployed for real-time predictions

---

## 🛠️ Tech Stack

| Tool/Library       | Purpose                          |
|--------------------|----------------------------------|
| Python             | Programming language             |
| Pandas, NumPy      | Data handling & preprocessing    |
| Matplotlib, Seaborn| Exploratory Data Analysis (EDA)  |
| Scikit-learn       | ML model building & evaluation   |
| Streamlit          | Web application deployment       |

---

## 📂 Dataset

- **Source:** [Kaggle – Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Size:** ~1000 rows
- **Target variable:** `output` (1 = Heart disease, 0 = No heart disease)

---

## 📈 Workflow

1. **Load dataset**
2. **Data cleaning & EDA**
3. **Feature selection**
4. **Model training & testing**
5. **Model evaluation**
6. **Streamlit deployment**

---

## 🎯 ML Algorithms Used

- Logistic Regression  
- Random Forest Classifier  
- (Optional) Support Vector Machine (SVM)  
- Hyperparameter tuning with GridSearchCV

---

## 📊 Model Evaluation

- Accuracy: ~85% (Random Forest)
- Precision & Recall analysis
- Confusion matrix visualization
- ROC-AUC curve

---

## 🖥️ Web App Demo

Built with **Streamlit**, the web app allows users to enter clinical data and receive instant predictions.

### ▶️ To Run Locally:
```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt
streamlit run app/streamlit_app.py