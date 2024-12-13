
# Web App for Automated ML Model Training

## Overview

The Web App for Automated ML Model Training simplifies the process of training machine learning models by providing a user-friendly interface. With this application, users can upload datasets, select target columns, preprocess data, choose models, and evaluate performance with ease. The app is built using Streamlit and supports several ML models and preprocessing techniques.

## Features

1.Upload datasets in CSV or Excel format.

2.Automatic handling of missing values.

3.Support for numerical and categorical data preprocessing.

4.Choose from a variety of scalers (StandardScaler, MinMaxScaler).

5.Train models like:

  Logistic Regression

  Support Vector Classifier

  Random Forest Classifier

  XGBoost Classifier

6.Save trained models for future use.

7.Evaluate models with test accuracy.

## Technologies Used

Python

Streamlit

Scikit-learn

XGBoost

Pandas

NumPy

## How to Use

1)Upload your dataset (CSV or Excel format).

2)Select the target column for prediction.

3)Choose a scaler and model from the dropdown menus.

4)Provide a name for your model.

5)Click Train the Model to start training.

6)View the test accuracy of the trained model.


## Example Dataset

Make sure your dataset:

  1)Contains a clear target column for prediction.

  2) Has no unsupported formats (like images or nested data).
  

## Future Enhancements

1)Add support for regression models.

2)Implement hyperparameter tuning.

3)Provide detailed evaluation metrics (e.g., precision, recall, F1-score).

4)Include visualization for dataset and model performance.
