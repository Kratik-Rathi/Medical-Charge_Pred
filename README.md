# Medical-Charge_Pred
"Predict medical insurance charges using Random Forest Regression and Time Series models (SARIMAX, ARIMAX). Explore the dataset, train models, and visualize insights. Python, Scikit-learn, Statsmodels. Contributions welcome! 🚀"

# Insurance Charges Prediction

This repository contains code for predicting medical insurance charges based on various features such as age, sex, BMI, children, smoking status, and region. The prediction is done using both Random Forest Regression and Time Series models like SARIMAX and ARIMAX. Additionally, the repository includes data exploration and visualization to gain insights into the dataset.

## Dataset

The dataset used for this project is `insurance.csv`, which contains information about individuals and their medical charges.

## Data Preprocessing

- Handling duplicates
- Handling missing values
- Label encoding and one-hot encoding for categorical variables
- Train-test split

## Model Training

Random Forest Regression is used for predicting insurance charges, and the model is trained and evaluated using cross-validation.


# Sample code for model training
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

rand_forest_model = RandomForestRegressor(n_estimators=50, n_jobs=2, random_state=42)

The model is evaluated using metrics such as Root Mean Squared Error (RMSE) and R-squared.

from sklearn.metrics import mean_absolute_error, r2_score

predictions = rand_forest_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f'Mean Absolute Error: {mae:.2f}')
print(f'R-squared: {r2:.2f}')

Time Series Models
SARIMAX and ARIMAX models are used for time series predictions. These models take into account the temporal aspect of the data.

Results
The performance metrics and results of both Random Forest and Time Series models are presented and compared.

Visualizations
Various visualizations are included in the repository:

Feature importance plot
Residual plots
Actual vs. Predicted charges scatter plot
Distribution of medical charges
Correlation heatmap
Pairplot and boxplots for data exploration
Contributing
Feel free to contribute to this project by opening issues or submitting pull requests.
