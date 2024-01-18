import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# Load the datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s4e1/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s4e1/test.csv')

# Data Cleaning
# Assuming you want to fill missing values in numerical columns with the mean
imputer = SimpleImputer(strategy='mean')
train_data.iloc[:, [6, 8, 11]] = imputer.fit_transform(train_data.iloc[:, [6, 8, 11]])
test_data.iloc[:, [6, 8, 11]] = imputer.transform(test_data.iloc[:, [6, 8, 11]])

# Preprocess the data
label_encoder = LabelEncoder()

for column in ['Geography', 'Gender']:
    train_data[column] = label_encoder.fit_transform(train_data[column])
    test_data[column] = label_encoder.transform(test_data[column])

# Feature Engineering
train_data['BalanceSalaryRatio'] = train_data['Balance'] / train_data['EstimatedSalary']
test_data['BalanceSalaryRatio'] = test_data['Balance'] / test_data['EstimatedSalary']

# Select features and target variable
X = train_data.drop(['id', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = train_data['Exited']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
random_forest = RandomForestClassifier(random_state=42)
gradient_boosting = GradientBoostingClassifier(random_state=42)

# Hyperparameter Tuning using Grid Search
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5],
}

param_grid_gb = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
}

grid_search_rf = GridSearchCV(random_forest, param_grid_rf, cv=3, scoring='roc_auc', n_jobs=-1)
grid_search_gb = GridSearchCV(gradient_boosting, param_grid_gb, cv=3, scoring='roc_auc', n_jobs=-1)

grid_search_rf.fit(X_train, y_train)
grid_search_gb.fit(X_train, y_train)

# Print best hyperparameters
print("Best Random Forest Hyperparameters:", grid_search_rf.best_params_)
print("Best Gradient Boosting Hyperparameters:", grid_search_gb.best_params_)

# Ensemble Voting Classifier with best hyperparameters
best_rf = grid_search_rf.best_estimator_
best_gb = grid_search_gb.best_estimator_

ensemble_model = VotingClassifier(estimators=[('RandomForest', best_rf), ('GradientBoosting', best_gb)],
                                  voting='soft')

# Train the ensemble model
ensemble_model.fit(X_train, y_train)

# Make predictions on the validation set
val_predictions = ensemble_model.predict_proba(X_val)[:, 1]

# Calculate AUC-ROC
roc_auc = roc_auc_score(y_val, val_predictions)
print(f'AUC-ROC on Validation Set (Ensemble): {roc_auc}')

# Make predictions on the test set
test_features = test_data.drop(['id', 'CustomerId', 'Surname'], axis=1)
test_features_scaled = scaler.transform(test_features)
test_predictions = ensemble_model.predict_proba(test_features_scaled)[:, 1]
