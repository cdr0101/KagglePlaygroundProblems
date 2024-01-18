from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import median_absolute_error
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Read the dataset
train_df = pd.read_csv("/kaggle/input/playground-series-s3e25/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s3e25/test.csv")

# Select features and target variable
X = train_df.drop(columns=['Hardness', 'id'])
y = train_df['Hardness']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

# Evaluate the linear regression model using Median Absolute Error (MedAE)
medae = median_absolute_error(y_test, predictions)
print(f'Linear Regression - Median Absolute Error (MedAE): {medae}')

# Train a Support Vector Regression (SVR) model
svr_model = SVR()
svr_model.fit(X_train_scaled, y_train)
svr_predictions = svr_model.predict(X_test_scaled)

# Evaluate the SVR model using Median Absolute Error (MedAE)
medae_svr = median_absolute_error(y_test, svr_predictions)
print(f'SVR - Median Absolute Error (MedAE): {medae_svr}')

# Train a Ridge Regression model
ridge_model = Ridge()
ridge_model.fit(X_train_scaled, y_train)
ridge_predictions = ridge_model.predict(X_test_scaled)

# Evaluate the Ridge model using Median Absolute Error (MedAE)
medae_ridge = median_absolute_error(y_test, ridge_predictions)
print(f'Ridge - Median Absolute Error (MedAE): {medae_ridge}')

# Train a Lasso Regression model
lasso_model = Lasso()
lasso_model.fit(X_train_scaled, y_train)
lasso_predictions = lasso_model.predict(X_test_scaled)

# Evaluate the Lasso model using Median Absolute Error (MedAE)
medae_lasso = median_absolute_error(y_test, lasso_predictions)
print(f'Lasso - Median Absolute Error (MedAE): {medae_lasso}')
