# Breast Cancer Detection using Logistic Regression


![Cytecare](https://cytecare.com/wp-content/uploads/2020/06/breast-cancer-detection.jpg)


## Introduction

In this project, we explore the process of training a logistic regression model for breast cancer detection. We utilize the Breast Cancer Wisconsin (Diagnostic) dataset, which contains detailed measurements of cell nuclei along with their diagnosis as malignant or benign. Our objective is to train a model capable of predicting the likelihood of a cell being malignant based on its measurements.

### Tools Used
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
1. **Exploring the Data with SQL:** Loading and exploring the dataset to understand its structure and attributes.
2. **Data Cleaning with SQL:**
    - **Dropped Column:** The 'id' column was dropped from the table using the ALTER TABLE statement.
    - **Updated Column Values:** The values in the 'diagnosis' column were updated from string ('M' for malignant and 'B' for benign) to Boolean ('1' for malignant and '0' for benign) using a CASE statement in the UPDATE query.
3. **Extracting the Clean Dataset and Analyzing Data with Pandas:** Extracting the clean dataset and analyzing data with pandas, including normalization of feature values with StandardScaler and splitting into predictors and target.
4. **Model Training:** Training a logistic regression model using the training data.
5. **Model Evaluation:** Evaluating the model's performance using accuracy score and classification report.
6. **Visualization:** Visualizing the model's predictions and evaluation metrics using plots.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author
- Leonardo Lessa



