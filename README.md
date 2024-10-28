# Diabetes Prediction using Logistic Regression

This project involves building a predictive model to determine the likelihood of diabetes in individuals based on medical data. Using logistic regression, the project demonstrates data cleaning, feature engineering, model training, hyperparameter tuning, and evaluation steps. 

## Table of Contents
- [Installation](#installation)
- [Project Overview](#project-overview)
- [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
- [Model Training and Hyperparameter Tuning](#model-training-and-hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [File Descriptions](#file-descriptions)
- [Acknowledgments](#acknowledgments)

---

## Installation

To run this project, clone the repository and install the required libraries listed in `requirements.txt` (e.g., `pandas`, `numpy`, `scikit-learn`, `matplotlib`, and `seaborn`).

## Project Overview

The goal of this project is to analyze medical attributes and build a predictive model to classify individuals as diabetic or non-diabetic. Logistic regression is used as it is suitable for binary classification tasks. The project includes:

1. Data Import and Exploration
2. Data Cleaning and Preprocessing
3. Feature Engineering
4. Model Training and Hyperparameter Tuning
5. Model Evaluation

---

## Data Cleaning and Preprocessing

The initial data contains features such as glucose levels, insulin levels, blood pressure, BMI, and skin thickness. Some records have invalid values (e.g., zeros in fields where zeros are biologically implausible). Key preprocessing steps include:

1. **Handling Missing and Erroneous Values:**  
   Invalid zero values in the dataset are replaced with the mean of the respective columns, as these values likely indicate missing data rather than actual measurements.

2. **Feature Scaling:**  
   To improve model performance, features are scaled using standardization, allowing for consistent ranges across features, which is especially important for distance-based models and regularization.

---

## Model Training and Hyperparameter Tuning

1. **Logistic Regression Model:**  
   Logistic regression is selected due to its effectiveness with binary classification problems and its interpretability. This model calculates the probability of diabetes and makes predictions accordingly.

2. **Hyperparameter Tuning with Grid Search:**  
   To enhance accuracy, hyperparameter tuning is conducted. Grid Search with cross-validation tests combinations of parameters (e.g., regularization strength, penalty type) to determine the best configuration for optimal accuracy.

---

## Model Evaluation

Model performance is assessed with various metrics to ensure reliability:

1. **Accuracy:**  
   Measures the proportion of correct predictions over the total predictions.

2. **Confusion Matrix and Derived Metrics:**  
   A confusion matrix provides insights into true positives, false positives, false negatives, and true negatives, leading to the calculation of:
   - **Precision**: The ratio of true positives to all predicted positives, indicating how many predicted cases are actually positive.
   - **Recall**: The ratio of true positives to all actual positives, indicating the model’s ability to detect positive cases.
   - **F1 Score**: The harmonic mean of Precision and Recall, providing a single metric to balance the two.

3. **Saving the Model:**  
   The final model is saved as a pickle file, enabling future use in deployment or additional testing without retraining.

---

## File Descriptions

- **`diabetes.csv`**: The dataset used for training and testing.
- **`standardScalar.pkl`**: The saved scaler for preprocessing new data during deployment.
- **`modelForPrediction.pkl`**: The final trained model, ready for predictions on new data.

---

## Acknowledgments

The dataset and concept are based on publicly available resources in medical diagnostics, demonstrating the application of logistic regression for binary classification in healthcare predictions.
