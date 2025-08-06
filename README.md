# Wine Quality Prediction using Machine Learning

[](https://opensource.org/licenses/MIT)
[](https://www.python.org/downloads/)
[](https://scikit-learn.org/)

A machine learning project to predict wine quality based on physicochemical test data. This model can be configured as a classification problem (e.g., good vs. bad quality) or a regression problem (predicting a quality score).

-----

### 📝 Project Description

This project aims to build a machine learning model that can accurately predict wine quality. By analyzing various physicochemical attributes such as acidity level, sugar content, and alcohol, the model learns to classify or rate the quality of the wine. This project includes exploratory data analysis (EDA), data preprocessing, model training, and performance evaluation to provide insights into which factors most significantly influence wine quality.

### 🎯 Background

Traditionally, wine quality assessment is conducted by expert tasters (sommeliers), which can be subjective and time-consuming. By leveraging machine learning, wine producers can obtain an objective and rapid method to assess the quality of their products during the production process. This can assist in quality control, recipe optimization, and product standardization.

### ✨ Key Features

  - **Comprehensive Data Analysis**: A Jupyter Notebook containing in-depth visualizations and statistical analysis of the red and white wine datasets.
  - **Model Flexibility**: Can be implemented as a classification model (good/bad quality) or a regression model (score 1-10).
  - **Classification Modeling**: Uses a powerful model like **Random Forest Classifier**, which is effective for tabular data.
  - **Performance Evaluation**: Measures model performance using standard metrics such as Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.

### 📊 Dataset

This model is trained using the highly popular "Wine Quality" dataset from the UCI Machine Learning Repository. The dataset is available in two separate files for red and white wine.

  - **Source**: [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
  - **Features Analyzed**:
      - `fixed acidity`
      - `volatile acidity`
      - `citric acid`
      - `residual sugar`
      - `chlorides`
      - `free sulfur dioxide`
      - `total sulfur dioxide`
      - `density`
      - `pH`
      - `sulphates`
      - `alcohol`
  - **Target Variable**: `quality` (a score between 0 and 10).

### 🛠️ Tech Stack

  - **Language**: Python 3.8+
  - **Analysis Libraries**: Pandas, NumPy
  - **Visualization Libraries**: Matplotlib, Seaborn
  - **Machine Learning Library**: Scikit-learn

### 📁 Project Structure

```
.
├── data/
│   ├── winequalityNcsv
├── notebooks/
│   └── wine_quality_analysis.ipynb
├── requirements.txt
└── README.md
```

### 🧠 Methodology and Model

This project is generally focused on being a **binary classification problem**. Due to the imbalanced distribution of the `quality` scores, the target is converted into two categories:

  - **Good Quality**: `quality` \> 6
  - **Bad Quality**: `quality` \<= 6

**Model Used**: **Random Forest Classifier** was chosen for several reasons:

  - It performs well on tabular data.
  - It is robust against overfitting if configured correctly.
  - It can provide insights into feature importance.

**Data Preprocessing**:

  - Combining both datasets (red and white) if desired.
  - Performing feature scaling using `StandardScaler` to normalize the data.

**Evaluation**:
The model's performance is evaluated using the following metrics on the test data:

  - **Accuracy**: The percentage of correct predictions.
  - **Confusion Matrix**: To see the details of True/False Positives/Negatives.
  - **Precision, Recall, F1-Score**: More reliable metrics for imbalanced datasets.

### 🤝 Contributing

Contributions to refine the model or add new analyses are very welcome. Please **Fork** this repository, create a new **Branch**, make your changes, and submit a **Pull Request**.

### 📄 License

This project is licensed under the **MIT License**.
