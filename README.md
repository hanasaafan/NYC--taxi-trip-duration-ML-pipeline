# NYC--taxi-trip-duration-ML-pipeline

This project implements a machine learning pipeline to predict taxi trip durations in New York City. The pipeline follows a structured workflow from data loading through modeling and evaluation.

### 1. Data Loading
- Load data using `pandas.read_csv()`
- Explore dataset structure with `df.info()` and `df.describe()`
- Note dataset size (rows, columns)
- Identify target variable: `trip_duration`

### 2. Data Cleaning
- Remove or impute missing values
- Treat outliers using IQR, quantiles, or z-score methods
- Correct data types (e.g., convert date strings to datetime)
- Remove invalid or duplicate rows

### 3. Preprocessing
- Encode categorical variables (one-hot encoding)
- Feature engineering: extract datetime features, create new variables
- Convert units if necessary (e.g., seconds to hours for trip duration)
- Calculate distance in km using Haversine formula on pickup and dropoff coordinates
- Drop redundant or low-correlation features (`vendor_id`, `passenger_count`, `store_and_fwd_flag`, raw coordinates, `pickup_month`)

### 4. Train-Validation-Test Split
- Split data into training and validation sets (e.g., 80/20)
- Keep test set separate for final evaluation
- Perform splitting after cleaning and preprocessing to avoid data leakage


### 5. Modeling
- Train and evaluate models including:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - XGBoost Regressor
  - Neural Networks (simple MLP)
- Monitor for overfitting and underfitting

### 6. Hyperparameter Tuning
- Use GridSearchCV or RandomizedSearchCV with cross-validation 
- Tune parameters for each model:
  - Decision Tree: max_depth, min_samples_split
  - Random Forest: n_estimators, max_features, bootstrap
  - XGBoost: max_depth, gamma, scale_pos_weight,learning rate
  - Neural Networks: number of layers, neurons, learning rate, batch size

### 7. Evaluation and Interpretation
- Evaluate models with metrics such as RMSE, MAE, and R²
- Assess final model performance on test set


---

## Dataset
Source: [Kaggle - NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration)

---

## Author
Hana Saafan 
[LinkedIn](https://www.linkedin.com/in/hana-saafan-269182218/)







