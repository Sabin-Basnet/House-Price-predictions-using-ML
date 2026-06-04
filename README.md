# 🏠 House Price Prediction using Machine Learning

A comprehensive machine learning project to predict California housing prices using advanced regression models and hyperparameter tuning.

## 📊 Project Overview

This project implements a complete machine learning pipeline for predicting median house prices based on various features including location (latitude/longitude), house characteristics, and demographic information. The project demonstrates best practices in data science including exploratory data analysis, preprocessing, model selection, and hyperparameter optimization.

## 📈 Dataset

**Source:** [California Housing Prices - Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

### Features:

| Feature | Description |
|---------|-------------|
| **longitude** | How far west a house is (higher = farther west) |
| **latitude** | How far north a house is (higher = farther north) |
| **housing_median_age** | Median age of a house within a block (lower = newer) |
| **total_rooms** | Total number of rooms within a block |
| **total_bedrooms** | Total number of bedrooms within a block |
| **population** | Total number of people within a block |
| **households** | Total number of households within a block |
| **median_income** | Median income for households (in tens of thousands USD) |
| **ocean_proximity** | Location relative to ocean/sea *(categorical)* |
| **median_house_value** | **Target variable** (in USD) |

## 🔍 Methodology

The project follows a structured machine learning workflow:

```
EDA → Preprocessing → Baseline Model → Model Selection → Hyperparameter Tuning → Final Evaluation → Inference
```

### 1. **Exploratory Data Analysis (EDA)**
- Statistical analysis of numerical and categorical features
- Correlation analysis and multicollinearity detection
- Distribution analysis and outlier identification
- Target variable analysis (right-skewed, capped at $500K)

### 2. **Data Preprocessing**
- Train-test split (80-20) to prevent data leakage
- Missing value imputation (median for numerical, mode for categorical)
- One-hot encoding for categorical features
- Feature scaling using StandardScaler

### 3. **Baseline Model**
- Linear Regression as baseline for performance comparison

### 4. **Model Selection** (with 5-Fold Cross-Validation)
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- **HistGradientBoosting** ✓ (Best performer)

### 5. **Hyperparameter Tuning**
- GridSearchCV on HistGradientBoostingRegressor
- Tuned parameters:
  - Learning rate, max depth, max leaf nodes
  - Regularization (L2), minimum samples per leaf

### 6. **Evaluation Metrics**
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## 📊 Results

### Best Model: HistGradientBoostingRegressor

| Metric | Train | Test |
|--------|-------|------|
| **RMSE** | $35,861.329 | $46,475.241 |
| **MAE** | $24,370.484 | $30,560.374 |
| **R² Score** | 0.904 | 0.835 |

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/Sabin-Basnet/House-Price-predictions-using-ML.git
cd House-Price-predictions-using-ML

# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

Open and run the Jupyter notebook:

```bash
jupyter notebook prediction.ipynb
```

## 💡 Usage Example

```python
from prediction import predict_house_price

# Predict house price for a property
predicted_price = predict_house_price(
    model=hgb_best,
    longitude=-122.23,
    latitude=37.88,
    housing_median_age=41.0,
    total_rooms=880.0,
    total_bedrooms=129.0,
    population=322.0,
    households=126.0,
    median_income=8.3252,
    ocean_proximity="NEAR BAY"
)

print(f"Predicted house price: ${predicted_price:,.2f}")
```

## 📁 Project Structure

```
House-Price-predictions-using-ML/
├── prediction.ipynb          # Main analysis and model training notebook
├── housing.csv               # Dataset
├── README.md                 # This file
└── venv/                     # Virtual environment
```

## 🔧 Technologies & Libraries

- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn
  - Preprocessing: StandardScaler, OneHotEncoder
  - Models: LinearRegression, Ridge, Lasso, RandomForestRegressor, HistGradientBoostingRegressor
  - Utilities: Pipeline, ColumnTransformer, GridSearchCV, cross_validate

## 📝 Key Insights

1. **Strongest predictor:** Median income has the highest correlation with house prices(Target Column)
2. **Multicollinearity:** High correlation between rooms and population features
3. **Target Distribution:** Right-skewed and capped at $500,000
4. **Best Model:** HistGradientBoosting outperformed other models due to its robustness and ability to capture non-linear relationships

## 🎯 Future Improvements

- Feature engineering and polynomial features
- Ensemble methods combining multiple models
- Handling target variable capping effects
- Cross-validation with different splitting strategies
- Feature importance analysis
