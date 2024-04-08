# Breast Cancer Detection using Logistic Regression

![Cytecare](https://cytecare.com/wp-content/uploads/2020/06/breast-cancer-detection.jpg)

## Index

1. [Introduction](#introduction)
2. [Tools Used](#tools-used)
3. [Dataset Information](#dataset-information)
4. [Implementation Details](#implementation-details)
    - [Libraries Used](#libraries-used)
    - [Process](#process)
        - [Exploring the Data with SQL](#exploring-the-data-with-sql)
        - [Data Cleaning with SQL](#data-cleaning-with-sql)
        - [Extracting the Clean Dataset and Analyzing Data with Pandas](#extracting-the-clean-dataset-and-analyzing-data-with-pandas)
        - [Model Training](#model-training)
        - [Model Evaluation](#model-evaluation)
        - [Visualization](#visualization)
5. [Author](#author)
6. [License](#license)

## Introduction

In this project, we explore the process of training a logistic regression model for breast cancer detection. We utilize the Breast Cancer Wisconsin (Diagnostic) dataset, which contains detailed measurements of cell nuclei along with their diagnosis as malignant or benign. Our objective is to train a model capable of predicting the likelihood of a cell being malignant based on its measurements.

## Tools Used

- Python for machine learning and visualization.
- SQL Server Management for data cleaning queries.

## Dataset Information

- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Features:** Computed from digitized images of fine needle aspirates (FNA) of breast masses. Features describe characteristics of cell nuclei.
- **Attributes:**
  1. ID number
  2. Diagnosis (M = malignant, B = benign)
  3-32: Ten real-valued features computed for each cell nucleus.
- **Class Distribution:** 357 benign, 212 malignant
- **Missing Attribute Values:** None
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

## Implementation Details

### Libraries Used

- Pandas: Data manipulation
- Seaborn & Matplotlib: Data visualization
- Scikit-learn: Machine learning tools (StandardScaler, train_test_split, LogisticRegression, accuracy_score, classification_report)

### Process

#### Exploring the Data with SQL

Loading and exploring the dataset to understand its structure and attributes.

#### Data Cleaning with SQL

- **Dropped Column:** The 'id' column was dropped from the table using the ALTER TABLE statement.
- **Updated Column Values:** The values in the 'diagnosis' column were updated from string ('M' for malignant and 'B' for benign) to Boolean ('1' for malignant and '0' for benign) using a CASE statement in the UPDATE query.

#### Extracting the Clean Dataset and Analyzing Data with Pandas

Extracting the clean dataset and analyzing data with pandas, including normalization of feature values with StandardScaler and splitting into predictors and target.

#### Model Training

Training a logistic regression model using the training data.

#### Model Evaluation

Evaluating the model's performance using accuracy score and classification report.

#### Visualization

Visualizing the model's predictions and evaluation metrics using plots.

## Author

- Leonardo Lessa

## License

This project is licensed under the MIT License. 

MIT License

Copyright (c) 2024 Leonardo Lessa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
