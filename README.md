# House-price-prediction-Model-data-scientce-project-
Data Collection → Data Preprocessing → Train-Test Split → Model Selection → Model Training → Model Evaluation → Hyperparameter Tuning → Prediction → Deployment

# House Price Prediction using Linear Regression

# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Step 2: Load Dataset
# Replace "house_prices.csv" with your dataset filename
df = pd.read_csv("house_prices.csv")

# Step 3: View Data
print(df.head())
print(df.info())

# Step 4: Handle Missing Values
df = df.dropna()

# Step 5: Select Features and Target
# Example columns:
# Area, Bedrooms, Bathrooms, Age, Price

X = df[['Area', 'Bedrooms', 'Bathrooms', 'Age']]
y = df['Price']

# Step 6: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Step 7: Create Model
model = LinearRegression()

# Step 8: Train Model
model.fit(X_train, y_train)

# Step 9: Predict
y_pred = model.predict(X_test)

# Step 10: Evaluate Model
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 11: Predict Price for New House
new_house = pd.DataFrame({
    'Area': [1800],
    'Bedrooms': [3],
    'Bathrooms': [2],
    'Age': [5]
})

predicted_price = model.predict(new_house)
print("Predicted House Price:", predicted_price[0])

# Step 12: Model Coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
Machine Learning Workflow
Import libraries
Load dataset
Explore data
Preprocess data
Split into train and test sets
Build the Linear Regression model
Train the model
Make predictions
Evaluate the model (MAE, MSE, RMSE, R²)
Predict the price of a new house
This is a complete beginner-friendly project and is suitable for practicing a common data science interview problem.
