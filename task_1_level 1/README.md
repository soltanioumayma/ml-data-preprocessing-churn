# ML Data Preprocessing - Customer Churn Dataset

## Overview

This project demonstrates comprehensive data preprocessing techniques for machine learning using a customer churn dataset. The preprocessing pipeline includes handling missing data, encoding categorical variables, standardizing numerical features, and splitting the dataset into training and testing sets.

## Project Objectives

- Handle missing data through appropriate imputation strategies
- Encode categorical variables using one-hot encoding and label encoding
- Normalize numerical features using standardization
- Split dataset into training and testing sets with stratification
- Prepare clean, processed data ready for machine learning model training

## Dataset

The project uses the Telco Customer Churn dataset (`churn-bigml-80.csv`) containing 2,666 records with 20 features including customer demographics, service usage patterns, and churn status.

### Dataset Features

- **State**: Customer's US state (51 unique values)
- **Account length**: Duration of customer account
- **Area code**: Area code of customer location
- **International plan**: Whether customer has international plan (Yes/No)
- **Voice mail plan**: Whether customer has voice mail plan (Yes/No)
- **Number vmail messages**: Number of voice mail messages
- **Total day/eve/night/intl minutes**: Usage minutes for different time periods
- **Total day/eve/night/intl calls**: Number of calls for different time periods
- **Total day/eve/night/intl charge**: Charges for different time periods
- **Customer service calls**: Number of customer service calls
- **Churn**: Target variable (True/False)

## Preprocessing Pipeline

### 1. Missing Data Handling
- Comprehensive check for missing values across all features
- No missing data was found in this dataset
- Pipeline includes imputation strategies for future datasets

### 2. Categorical Variable Encoding
- **Target Variable (Churn)**: Label encoded (False=0, True=1)
- **State**: One-hot encoded (50 binary features)
- **International plan**: One-hot encoded (1 binary feature)
- **Voice mail plan**: One-hot encoded (1 binary feature)

### 3. Numerical Feature Standardization
- Applied StandardScaler to 16 numerical features
- Features scaled to mean=0, standard deviation=1
- Ensures all numerical features contribute equally to model training

### 4. Train/Test Split
- 80% training set (2,132 samples)
- 20% testing set (534 samples)
- Stratified sampling to maintain class distribution
- Preserves churn rate imbalance (14.5% positive class)

## Technologies Used

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning preprocessing utilities
  - StandardScaler
  - LabelEncoder
  - OneHotEncoder
  - train_test_split

## Project Structure

```
task_1_level 1/
├── task1_data_preprocessing.ipynb    # Main preprocessing notebook
├── check_missing_data.py             # Utility script for missing data analysis
├── churn-bigml-80.csv                # Original dataset
├── train_data_preprocessed.csv       # Processed training data
├── test_data_preprocessed.csv        # Processed testing data
└── README.md                         # Project documentation
```

## Usage

### Running the Preprocessing Pipeline

1. Ensure required dependencies are installed:
```bash
pip install pandas numpy scikit-learn
```

2. Open the Jupyter notebook:
```bash
jupyter notebook task1_data_preprocessing.ipynb
```

3. Execute all cells sequentially to run the preprocessing pipeline

### Using Preprocessed Data

The preprocessed datasets are saved as CSV files and can be directly loaded for model training:

```python
import pandas as pd

# Load preprocessed data
train_data = pd.read_csv('train_data_preprocessed.csv')
test_data = pd.read_csv('test_data_preprocessed.csv')

# Separate features and target
X_train = train_data.drop('Churn', axis=1)
y_train = train_data['Churn']
X_test = test_data.drop('Churn', axis=1)
y_test = test_data['Churn']
```

## Key Results

- **Original dataset shape**: (2666, 20)
- **Processed dataset shape**: (2666, 69)
- **Missing values**: 0 (none found)
- **Categorical features encoded**: 4 features
- **Numerical features standardized**: 16 features
- **Training samples**: 2,132
- **Testing samples**: 534
- **Class distribution**: 85.4% non-churn, 14.6% churn

## Future Work

- Apply various machine learning algorithms (Logistic Regression, Random Forest, XGBoost)
- Perform feature selection to identify most predictive features
- Handle class imbalance using techniques like SMOTE or class weighting
- Perform hyperparameter tuning for optimal model performance
- Deploy model as a web service for real-time predictions

## Author

This project was completed as part of a machine learning internship program, demonstrating fundamental data preprocessing skills essential for building robust machine learning models.

## License

This project is for educational purposes. Please refer to the original dataset source for any specific licensing requirements.
