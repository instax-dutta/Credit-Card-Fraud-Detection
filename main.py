import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import scipy.sparse as sp
import warnings

# Disable warnings
warnings.filterwarnings("ignore")

# Load the dataset
fraud_train = pd.read_csv('fraudtrain.csv')
fraud_test = pd.read_csv('fraudtest.csv')

# Preprocess the data
categorical_cols = ['merchant', 'category', 'gender', 'job']
numerical_cols = [col for col in fraud_train.columns if col not in categorical_cols + ['is_fraud', 'trans_date_trans_time', 'trans_num', 'unix_time']]

# Replace string values in numerical columns with NaN
X_train_numerical = fraud_train[numerical_cols].apply(pd.to_numeric, errors='coerce')
X_train_categorical = fraud_train[categorical_cols]
y_train = fraud_train['is_fraud']

X_test_numerical = fraud_test[numerical_cols].apply(pd.to_numeric, errors='coerce')
X_test_categorical = fraud_test[categorical_cols]
y_test = fraud_test['is_fraud']

# Fill missing values in categorical columns with random strings
X_train_categorical = X_train_categorical.apply(lambda x: x.fillna(pd.Series(np.random.choice(['A', 'B', 'C', 'D', 'E'], size=len(x))), axis=0))
X_test_categorical = X_test_categorical.apply(lambda x: x.fillna(pd.Series(np.random.choice(['A', 'B', 'C', 'D', 'E'], size=len(x))), axis=0))

# One-hot encode categorical columns
encoder = OneHotEncoder(handle_unknown='ignore', sparse=True)
X_train_categorical = encoder.fit_transform(X_train_categorical)
X_test_categorical = encoder.transform(X_test_categorical)

# Impute missing values in numerical columns
imputer = SimpleImputer(strategy='mean')
X_train_numerical = imputer.fit_transform(X_train_numerical)
X_test_numerical = imputer.transform(X_test_numerical)

# Combine numerical and one-hot encoded categorical columns
X_train = sp.hstack((X_train_numerical, X_train_categorical))
X_test = sp.hstack((X_test_numerical, X_test_categorical))

# Split the training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_val)
print('Logistic Regression:')
print('Accuracy:', accuracy_score(y_val, y_pred_lr))
print('Precision:', precision_score(y_val, y_pred_lr))
print('Recall:', recall_score(y_val, y_pred_lr))
print('F1-score:', f1_score(y_val, y_pred_lr))
print()

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_val)
print('Decision Tree:')
print('Accuracy:', accuracy_score(y_val, y_pred_dt))
print('Precision:', precision_score(y_val, y_pred_dt))
print('Recall:', recall_score(y_val, y_pred_dt))
print('F1-score:', f1_score(y_val, y_pred_dt))
print()

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)
print('Random Forest:')
print('Accuracy:', accuracy_score(y_val, y_pred_rf))
print('Precision:', precision_score(y_val, y_pred_rf))
print('Recall:', recall_score(y_val, y_pred_rf))
print('F1-score:', f1_score(y_val, y_pred_rf))