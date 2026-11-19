# Predictive Modeling Using Machine Learning — Project Report

## 1. Introduction
Predictive modeling uses historical data and machine-learning algorithms to estimate outcomes for unseen records. This project demonstrates supervised classification by predicting employee attrition.

## 2. Objective
Build, train, test, compare, and evaluate classification models using Python and scikit-learn.

## 3. Dataset
The dataset contains 600 synthetic records and the following features: Age, MonthlyIncome, YearsAtCompany, JobSatisfaction, WorkLifeBalance, OverTime, Department, and DistanceFromHome_km. The target is Attrition (0 = No, 1 = Yes). The dataset is synthetic and created for demonstration.

## 4. Methodology
1. Load the CSV dataset using Pandas.
2. Separate features and target.
3. Convert categorical variables using one-hot encoding.
4. Split into training and testing sets (80/20), stratified by target.
5. Train Logistic Regression, Decision Tree, and Random Forest models.
6. Evaluate using accuracy, confusion matrix, classification report, and ROC curve.
7. Compare models.

## 5. Algorithms
- **Logistic Regression:** a linear classifier that estimates the probability of a binary outcome.
- **Decision Tree:** learns decision rules through recursive feature splits.
- **Random Forest:** combines multiple decision trees to improve generalization.

## 6. Results
              Model  Accuracy
Logistic Regression     0.950
      Decision Tree     0.925
      Random Forest     0.950

The highest accuracy in this run was achieved by **Logistic Regression** with **95.00%** accuracy. This result is only illustrative because the dataset is synthetic.

## 7. Visualizations
The project includes:
- Model accuracy comparison
- Confusion matrix for the best-performing model
- ROC curve

## 8. Conclusion
This project demonstrates a basic end-to-end supervised learning workflow, including preprocessing, training, testing, comparison, and evaluation. The interactive Streamlit app allows users to select a model and inspect its performance.

## 9. Limitations
The synthetic dataset does not represent real employees or a real organization. Accuracy and other metrics cannot be used for real HR decisions. A real, appropriately licensed dataset should be used if required by the internship instructions.
