import numpy as np
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


beer_data = pd.read_csv('beer-servings_forWebApp.csv') # read data from input csv file.
#print(beer_data.head())

""" dropping Unnamed: 0 -- As it is an extra feature came from input csv file.
dropping country -- country has all 193 unique value and it is a categorical feature.
There is no point in encoding this feature, as it will confuse the algorithm."""

beer_data.drop(columns=['Unnamed: 0','country'],inplace=True) 

# dropping the record which has a missing value for the target feature.
beer_data = beer_data.dropna(subset=['total_litres_of_pure_alcohol']) 

# filling all the missing values with 'median' to avoid any impact of skewness.
num_cols = ['beer_servings', 'spirit_servings', 'wine_servings']
imputer_num = SimpleImputer(strategy='median')
beer_data[num_cols] = imputer_num.fit_transform(beer_data[num_cols])

# categorical feature imputed usign mode.
imputer_cat = SimpleImputer(strategy='most_frequent')
beer_data[['continent']] = imputer_cat.fit_transform(beer_data[['continent']])

# using OHE encoding the categorical feature.
beer_data = pd.get_dummies(beer_data, columns=['continent'], drop_first=True)

# seperating input from target feature.
X = beer_data.drop('total_litres_of_pure_alcohol', axis=1)
y = beer_data['total_litres_of_pure_alcohol']

# splitting the dataset as Train and Test
X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Training the 1st model- Linear Regression
lr_model = LinearRegression()

# Train the model using the training data
lr_model.fit(X_train, y_train)

# Make predictions on the test set and checking R2 score for Linear Regression.
y_pred_lr = lr_model.predict(X_test)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Linear Regression R2-Score: {r2_lr:.4f}")

# Training with the 2nd model
rf = RandomForestRegressor(random_state=42)

# Hyper parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],  # Number of trees
    'max_depth': [None, 5, 10, 20],   # Depth of trees to prevent overfitting
    'min_samples_split': [2, 5, 10]    # Min samples required to split a node
}

# Fitting the model and finding the R2 score.
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"Random Forest R2-Score: {r2_rf:.4f}")

# Comparing both R2 to pick the best performing model
if r2_rf > r2_lr:
    best_model = best_rf_model
    print("Decision: Random Forest is the best model for deployment.")
else:
    best_model = lr_model
    print("Decision: Linear Regression is the best model for deployment.")

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Save the feature columns to ensure consistency in Flask
with open('columns.pkl', 'wb') as file:
    pickle.dump(list(X.columns), file)

    