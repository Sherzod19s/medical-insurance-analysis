# medical-insurance-analysis
Medical Insurance Cost Prediction using Linear and Ridge Regression

🏥 Medical Insurance Cost Analysis – Project Summary
📌 Objectives
This project analyzes an insurance dataset to identify how different factors (like age, BMI, smoking status, region, etc.) affect medical insurance charges. The goal is to:

Perform data cleaning and preprocessing

Conduct exploratory data analysis (EDA)

Build various regression models (linear, polynomial, ridge)

Libraries used:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


📊 Exploratory Data Analysis (EDA)
Converted categorical variables like smoker, region, and gender using get_dummies().

Noticed that smoking status had a strong influence on charges.

Visualizations like scatter plots and heatmaps supported correlation findings.

🔍 Modeling & Results
✅ 1. Linear Regression (Single Feature)
Feature Used: smoker

R² Score: 0.622

✅ 2. Linear Regression (All Features)
Features Used: All except charges, encoded with pd.get_dummies()

R² Score: 0.752

✅ 3. Polynomial Regression Pipeline
Pipeline: StandardScaler + PolynomialFeatures + LinearRegression

R² Score: 0.852

✅ 4. Train-Test Split + Ridge Regression
Train/Test Split: 80/20

Ridge α=0.1, Degree=2

R² Score (test set): 0.841

🔎 Key Insights
Smoking status alone explains 62% of the variation in insurance charges.

Using all attributes improves the model significantly.

Polynomial transformation adds complexity and improves performance.

Ridge regression helps in controlling overfitting, especially when using polynomial features.
