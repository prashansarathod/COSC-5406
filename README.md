# COSC-5406


---

# ✈️ **Airfare Price Prediction using Machine Learning**

### *COSC-5406 – Final Project (Algoma University)*

This project builds a machine learning system to **predict airline ticket prices** using supervised learning techniques. The workflow includes data preprocessing, feature engineering, exploratory data analysis (EDA), model training, hyperparameter tuning, evaluation, and deployment using **Streamlit**.

---

## 📌 **Project Overview**

Airfare prices are influenced by several factors such as airline, source/destination city, travel duration, stops, departure/arrival times, and seasonal variations.
This project uses machine learning models to analyze these factors and predict flight prices accurately.

The final system integrates:

* Cleaned & engineered dataset
* Trained Random Forest model (tuned)
* Interactive Streamlit Web Application
* Deployment-ready structure

---

## 🧠 **Machine Learning Models Used**

The following models were developed and evaluated:

| **Model**                              | **MAE**     | **MSE**          | **RMSE**    | **R² Score**                  |
| -------------------------------------- | ----------- | ---------------- | ----------- | ----------------------------- |
| **🌟 Random Forest Regressor (Tuned)** | **1311.81** | **4,877,900.02** | **2208.60** | **0.7695 — Best Performance** |
| Gradient Boosting Regressor            | 2077.56     | 9,249,490.86     | 3041.30     | 0.5630                        |
| XGBoost Regressor                      | 2077.18     | 9,258,664.00     | 3042.81     | 0.5625                        |


The **tuned Random Forest model** provided the best predictive accuracy.

---

## 📊 **Key Features of the Project**

### ✔ **Feature Engineering**

* Extracted Journey Day, Month
* Converted Duration to Total Minutes
* Extracted Dep & Arrival hour/minute
* Encoded categorical features (Airline, Source, Destination)

### ✔ **EDA & Insights**

* Correlation matrix
* Distribution plots
* Price variation across airlines & total stops
* Feature importance ranking

### ✔ **Model Training & Tuning**

* Train-test split
* Hyperparameter tuning (manual + trial-based)
* Evaluation using MAE, MSE, RMSE, R²

### ✔ **Deployment**

Developed an interactive **Streamlit web application** that allows users to input flight details and receive predicted fare instantly.

---

## 🚀 **Technologies Used**

### **Programming Language**

* Python 3.10+

### **Libraries**

* numpy
* pandas
* matplotlib / seaborn
* scikit-learn
* xgboost
* streamlit
* pickle

---

## 📂 **Project Structure**

```
COSC-5406/
│
├── app.py                     # Streamlit web application
├── 1.ipynb                    # Full Jupyter Notebook (EDA + ML pipeline)
├── cleaned_data.csv           # Cleaned dataset after preprocessing
├── final_rf_model.pkl         # Tuned Random Forest model
├── model_columns.pkl          # Column transformer for inference
├── Prediction.csv             # Sample predictions
│
├── README.md                  # Project documentation
└── requirements.txt           # Dependencies for running the project
```

---

## ▶️ **How to Run the Project**

### **1️⃣ Clone the repository**

```
git clone https://github.com/prashansarathod/COSC-5406.git
cd COSC-5406
```

### **2️⃣ Install required packages**

```
pip install -r requirements.txt
```

### **3️⃣ Run the Streamlit app**

```
streamlit run app.py
```

You will now see the **Airfare Price Prediction App** in your browser.

---

## 📈 **Results Summary**

* **Random Forest achieved the best accuracy** after tuning
* Feature engineering significantly improved performance
* Duration, Airline, and Total Stops were top predictors
* Final R² Score: **~0.56–0.76 range** (consistent with real-world airfare complexity)

---

## 🛠️ **Future Improvements**

* Try deep learning models
* Use attention-based models for sequential pricing
* Integrate live airline API for real-time updates
* Deploy on cloud platform (Azure / AWS / Streamlit Cloud)

---

## 🔗 **Dataset Source**

Kaggle Dataset — Shubham Sarafo
🔗 [https://www.kaggle.com/datasets/shubhamsarafo/flight-price](https://www.kaggle.com/datasets/shubhamsarafo/flight-price)

---

## 👨‍💻 **Author**

**Prashansa Rathod**
Master's in Computer Science
Algoma University
COSC-5406 (Research Project)

---


