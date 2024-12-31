# Diabetes Prediction Project

This project focuses on predicting the likelihood of diabetes diagnosis using machine learning techniques. It involves preprocessing, exploratory data analysis (EDA), and implementing various classification algorithms.

## Project Overview

- **Objective**: To develop a model that predicts whether a person has diabetes based on key health indicators and lifestyle factors.
- **Dataset**: The dataset contains 128 entries and 11 features, including numerical and categorical variables.
- **Techniques Used**:
  - Data cleaning and preprocessing
  - Exploratory data analysis (EDA)
  - Classification modeling
  - Model evaluation

## Dataset Details

- **Source**: `Diabetes Classification.csv`
- **Features**:
  - `Age`: Age of the individual
  - `Gender`: Gender (Male/Female)
  - `BMI`: Body Mass Index
  - `Blood Pressure`: Blood pressure levels (Normal/High)
  - `FBS`: Fasting Blood Sugar level
  - `HbA1c`: Hemoglobin A1c levels
  - `Family History of Diabetes`: Family history of diabetes (Yes/No)
  - `Smoking`: Smoking habit (Yes/No)
  - `Diet`: Quality of diet (Healthy/Poor)
  - `Exercise`: Exercise frequency (Regular/No)
  - `Diagnosis`: Target variable indicating diabetes status (Yes/No)

## Methodology

### Data Preprocessing
- Handled missing and duplicate values.
- Encoded categorical variables using `LabelEncoder`.
- Scaled numerical features using `StandardScaler`.

### Exploratory Data Analysis
- Visualized distribution and relationships between features using:
  - Histograms, bar charts, and count plots
  - Correlation heatmap
- Identified significant correlations between features and the target variable.

### Machine Learning Models

Implemented the following classification models:

1. **Random Forest Classifier**
   - Used to handle feature importance and classification.
2. **Decision Tree Classifier**
   - Simple and interpretable tree-based model.
3. **Logistic Regression**
   - Baseline model for binary classification.
4. **K-Nearest Neighbors (KNN)**
   - Distance-based classifier for predictions.

### Model Evaluation
- Metrics used:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Visualized confusion matrices for each model.

## Results

| Model                 | Accuracy | Precision | Recall | F1-Score |
|-----------------------|----------|-----------|--------|----------|
| Random Forest         | *Value*  | *Value*   | *Value*| *Value*  |
| Decision Tree         | *Value*  | *Value*   | *Value*| *Value*  |
| Logistic Regression   | *Value*  | *Value*   | *Value*| *Value*  |
| K-Nearest Neighbors   | *Value*  | *Value*   | *Value*| *Value*  |

## Conclusion

This project demonstrates the application of machine learning techniques to predict diabetes. The models were evaluated based on key metrics, and insights from the EDA helped understand the data better. Future work can include:

- Using larger datasets.
- Tuning hyperparameters for better model performance.
- Exploring advanced algorithms like Gradient Boosting or Neural Networks.

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter notebooks and run the cells sequentially.

## Dependencies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn