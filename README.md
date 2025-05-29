# 📉 Customer Churn Prediction

A machine learning project to predict customer churn using classification models based on structured customer data. This project demonstrates model training, evaluation, and deployment using Streamlit.

## 📌 Project Overview

The goal of this project is to predict whether a customer will churn (i.e., leave the service) based on various features such as credit score, age, geography, tenure, balance, etc. This is a crucial problem in industries like banking and telecom where retaining customers is more cost-effective than acquiring new ones.

## 💡 Key Features

- Cleaned and preprocessed data
- Feature engineering and scaling
- Model training using classification algorithms (Random Forest, etc.)
- Model evaluation using accuracy, confusion matrix, etc.
- Streamlit-based interactive web application for live predictions

## 🚀 Technologies Used

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Matplotlib & Seaborn**
- **Streamlit** for deployment
- **Pickle** for model serialization

## 🧪 Dataset

- The dataset used contains information about bank customers, including demographics, account information, and churn status.
- Features include: `Credit Score`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `Number of Products`, `Has Credit Card`, `Is Active Member`, `Estimated Salary`.

## 📊 Model Building & Evaluation

- Categorical data encoded using label encoding and one-hot encoding.
- Feature scaling performed using `StandardScaler`.
- Trained models: `Random Forest Classifier`, `Logistic Regression`, etc.
- Evaluation Metrics: Accuracy, Precision, Recall, Confusion Matrix.

## 🌐 Web Application (Deployed)

The project includes a **Streamlit-based web app** allowing users to input customer details and receive a real-time churn prediction.

# Navigate to the folder containing churn.py
cd C:\\Users\\hp\\Documents

# Run the Streamlit app
streamlit run churn.py

I have added glimpse of my deployment project

# Thank you!




