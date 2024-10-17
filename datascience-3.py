import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import GradientBoostingRegressor


#load and read the dataset 
df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")
df3 = pd.read_csv("dataset3.csv")

# Merging the datasets on the 'ID' column
merged_data = pd.merge(df1, df2, on='ID', how='inner')
merged_data = pd.merge(merged_data, df3, on='ID', how='inner')

# Checking the structure and first few rows of the merged dataset
merged_data_head = merged_data.head()
merged_data_info = merged_data.info()

# print(merged_data_head)
# print(merged_data_info)

# Define well-being columns
well_being_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 
                      'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 
                      'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
# Define demographic column
demographic_columns = ['gender', 'minority', 'deprived']

# Compute total well-being score
merged_data['Total_Well_Being_Score'] = merged_data[well_being_columns].sum(axis=1)

# List of screen time columns
screen_time_columns = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']

# Correlation analysis
correlation_data = merged_data[screen_time_columns + ['Total_Well_Being_Score']]
correlation_matrix = correlation_data.corr()

# Visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap of Total Well-Being Score and Screen Time Variables')
plt.show()

# Prepare data for regression analysis
X = merged_data[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']]  # Independent variables (screen time)
y = merged_data['Total_Well_Being_Score']  # Dependent variable (Total well-being score)

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predictions on the test set
y_pred = lin_reg.predict(X_test)

# Calculate R-squared and Adjusted R-squared
r_squared = r2_score(y_test, y_pred)
n = X_test.shape[0]  # Number of observations
p = X_test.shape[1]  # Number of predictors
adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1)) / (n - p - 1)

# Display R-squared and Adjusted R-squared
print(f"r-squared is: ", r_squared)
print(f'Adjusted r-square is: ', adjusted_r_squared)

#Optimization-1
# Adding demographic variables (gender, minority, deprived) to the regression model
X_demographics = merged_data[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk', 'gender', 'minority', 'deprived']]
y = merged_data['Total_Well_Being_Score']  # Dependent variable (Total well-being score)

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_demographics, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model with demographic variables
lin_reg_with_demo = LinearRegression()
lin_reg_with_demo.fit(X_train, y_train)

# Predictions on the test set
y_pred_with_demo = lin_reg_with_demo.predict(X_test)

# Calculate R-squared and Adjusted R-squared
r_squared_with_demo = r2_score(y_test, y_pred_with_demo)
n = X_test.shape[0]  # Number of observations
p = X_test.shape[1]  # Number of predictors
adjusted_r_squared_with_demo = 1 - ((1 - r_squared_with_demo) * (n - 1)) / (n - p - 1)

# Display R-squared and Adjusted R-squared
print("r_squared with demographic", r_squared_with_demo)
print("Adjusted r_squared with demographic", adjusted_r_squared_with_demo)

# Visualization for Optimization-2
# Visualize the distribution of screen time variables and total well-being score
# Screen time and total well-being variables
screen_time_vars = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']
well_being_var = 'Total_Well_Being_Score'

# Plot histograms for screen time variables
plt.figure(figsize=(12, 10))
for i, var in enumerate(screen_time_vars, 1):
    plt.subplot(3, 3, i)
    sns.histplot(merged_data[var], bins=30, kde=True, color='blue')
    plt.title(f'Distribution of {var}')

# Plot histogram for total well-being score
plt.subplot(3, 3, 9)
sns.histplot(merged_data[well_being_var], bins=30, kde=True, color='green')
plt.title('Distribution of Total Well-Being Score')

plt.tight_layout()
plt.show()

# Square Root Transformation
# Apply square root transformation to the screen time variables
X_transformed_sqrt = X_demographics.copy()

# Apply square root transformation (note: for negative or zero values, we add a small constant)
X_transformed_sqrt[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']] = np.sqrt(X_transformed_sqrt[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']] + 1)

# Split the transformed data into training and testing sets
X_train_sqrt, X_test_sqrt, y_train_sqrt, y_test_sqrt = train_test_split(X_transformed_sqrt, y, test_size=0.2, random_state=42)

# Apply linear regression on the square root transformed data
lin_reg_sqrt = LinearRegression()
lin_reg_sqrt.fit(X_train_sqrt, y_train_sqrt)

# Predictions on the test set
y_pred_sqrt = lin_reg_sqrt.predict(X_test_sqrt)

# Calculate R-squared and Adjusted R-squared for the square root transformed model
r_squared_sqrt = r2_score(y_test_sqrt, y_pred_sqrt)
n_sqrt = X_test_sqrt.shape[0]  # Number of observations
p_sqrt = X_test_sqrt.shape[1]  # Number of predictors
adjusted_r_squared_sqrt = 1 - ((1 - r_squared_sqrt) * (n_sqrt - 1)) / (n_sqrt - p_sqrt - 1)

# Display R-squared and Adjusted R-squared for the square root transformed model
print("r_squared with square root", r_squared_sqrt)
print("Adjusted r_squared with square root", adjusted_r_squared_sqrt)



# Prepare data for regression analysis
X = merged_data[screen_time_columns + demographic_columns]  # Independent variables
y = merged_data['Total_Well_Being_Score']  # Dependent variable

# Step 1: Calculate VIF
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display VIF values
print("VIF Data:")
print(vif_data)

# Step 2: Remove features with high VIF
# Example: Drop a feature if its VIF is greater than 10
high_vif_features = vif_data[vif_data["VIF"] > 10]["feature"]
print(f"Features to drop due to high VIF: {high_vif_features.tolist()}")

# Remove high VIF features from the dataset
X_reduced = X.drop(columns=high_vif_features)

# Proceed with regression analysis using X_reduced
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Apply Ridge Regression
ridge_reg = Ridge(alpha=1.0)  # You can adjust alpha to tune regularization strength
ridge_reg.fit(X_train, y_train)

# Predictions and evaluation for Ridge
y_pred_ridge = ridge_reg.predict(X_test)
r_squared_ridge = r2_score(y_test, y_pred_ridge)
print("R-squared (Ridge):", r_squared_ridge)

# Apply Lasso Regression
lasso_reg = Lasso(alpha=0.1)  # Adjust alpha to tune regularization strength
lasso_reg.fit(X_train, y_train)

# Predictions and evaluation for Lasso
y_pred_lasso = lasso_reg.predict(X_test)
r_squared_lasso = r2_score(y_test, y_pred_lasso)
print("R-squared (Lasso):", r_squared_lasso)

#optimization 4 - Gradient Boosting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize and train Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

# Predictions
y_pred_gb = gb_model.predict(X_test)

# Calculate R-squared
r_squared_gb = r2_score(y_test, y_pred_gb)
print("R-squared (Gradient Boosting):", r_squared_gb)

# Calculate Adjusted R-squared
n = X_test.shape[0]  # Number of observations
p = X_test.shape[1]  # Number of predictors
adjusted_r_squared_gb = 1 - ((1 - r_squared_gb) * (n - 1)) / (n - p - 1)
print("Adjusted R-squared (Gradient Boosting):", adjusted_r_squared_gb)

# Optional: Plot feature importances
plt.figure(figsize=(12, 6))
feature_importances = gb_model.feature_importances_
features = X.columns
sns.barplot(x=feature_importances, y=features)
plt.title("Feature Importances (Gradient Boosting)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

#optimization-5 Hyperparamter tuning for ridge and lasso
# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge Regression with hyperparameter tuning
ridge = Ridge()
ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge_cv = GridSearchCV(ridge, ridge_params, cv=5, scoring='r2', n_jobs=-1)
ridge_cv.fit(X_train, y_train)
print(f"Best Ridge alpha: {ridge_cv.best_params_}")
print(f"Ridge R-squared (CV): {ridge_cv.best_score_}")

# Lasso Regression with hyperparameter tuning
lasso = Lasso(max_iter=10000)
lasso_params = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
lasso_cv = GridSearchCV(lasso, lasso_params, cv=5, scoring='r2', n_jobs=-1)
lasso_cv.fit(X_train, y_train)
print(f"Best Lasso alpha: {lasso_cv.best_params_}")
print(f"Lasso R-squared (CV): {lasso_cv.best_score_}")

# Evaluate models on the test set for Ridge and Lasso
ridge_best = ridge_cv.best_estimator_
lasso_best = lasso_cv.best_estimator_

ridge_test_score = r2_score(y_test, ridge_best.predict(X_test))
lasso_test_score = r2_score(y_test, lasso_best.predict(X_test))

print(f"Ridge Test R-squared: {ridge_test_score}")
print(f"Lasso Test R-squared: {lasso_test_score}")