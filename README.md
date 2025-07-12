# 🏡 House Price Prediction using Machine Learning

A machine learning project that predicts house prices based on 79 features using the Random Forest algorithm. Built as part of my academic learning in the field of Data Science and Regression Modeling.

---

## 📌 Project Objective

To build a predictive model that accurately estimates the selling prices of residential homes using detailed property data. This project aims to apply data preprocessing, feature engineering, model training, and hyperparameter tuning to produce reliable predictions.

---

## 🛠️ Technologies & Libraries Used

- **Language**: Python  
- **Libraries**:  
  - `pandas` – Data manipulation  
  - `numpy` – Numerical operations  
  - `scikit-learn` – Machine learning models and tools  
  - `matplotlib` / `seaborn` – Data visualization  
- **Model**: Random Forest Regressor  
- **Tuning**: GridSearchCV  
- **Evaluation**: RMSE (Root Mean Squared Error)

---

## ⚙️ Project Workflow

### 1. 📂 Data Collection
- Dataset sourced from [Kaggle – House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- Includes 79 features: area, room count, material quality, location, etc.

### 2. 🧹 Data Preprocessing
- Handled missing values using statistical imputation (mean/median/mode)
- Categorical variables encoded using One-Hot or Label Encoding
- Normalized features to ensure consistency across numerical scales

### 3. 🌲 Model Building – Random Forest Regressor
- Ensemble learning method using multiple decision trees
- Handles non-linear data and reduces overfitting

### 4. 🔧 Hyperparameter Tuning – GridSearchCV
- Optimized `n_estimators`, `max_depth`, and other parameters
- Cross-validation ensures generalization and prevents overfitting

### 5. 📊 Evaluation
- Model evaluated using RMSE (Root Mean Squared Error)
- Lower RMSE = better prediction accuracy

---

## 🎯 Results

- Achieved competitive RMSE score on test data
- Ranked in the top 2500+ on the Kaggle leaderboard with a score of **0.15035**

---

## 💡 What I Learned

- Complete end-to-end machine learning pipeline
- Feature engineering and data preprocessing best practices
- Power of ensemble models like Random Forest
- Tuning models using GridSearchCV and evaluating them effectively

---

## 🚀 Future Improvements

- Try alternative models like XGBoost, Gradient Boosting
- Add feature selection techniques (like Recursive Feature Elimination)
- Automate pipeline with MLflow or a web dashboard for predictions

