import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import os
import joblib
import logging
from itertools import product
from tqdm import tqdm
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

directory = os.getcwd()
file_path = os.path.join(directory, 'cleaned_develops.csv')

df = pd.read_csv(file_path)
logger.info(f"Data loaded successfully with shape: {df.shape}")
#df = df[df['will_develop_diabetes'] != 0]
df.columns = df.columns.str.strip()

df.dropna(subset=['will_develop_diabetes', 'when_develops_diabetes'], inplace=True)
df.fillna(df.mean(), inplace=True)

X = df.drop(columns=['will_develop_diabetes'])
y = df['will_develop_diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data loaded successfully with shape: {df.shape}")

xgb_model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.6, 1.0],
    'colsample_bytree': [0.6, 1.0],
    'gamma': [0.1],
    'reg_alpha': [0.1],
    'reg_lambda': [1],
}

rf = RandomForestClassifier(random_state=42)

logger.info("Starting custom GridSearch")

param_combinations = list(product(param_grid['n_estimators'],
                                  param_grid['max_depth'],
                                  param_grid['learning_rate'],
                                  param_grid['subsample'],
                                  param_grid['colsample_bytree'],
                                  param_grid['gamma'],
                                  param_grid['reg_alpha'],
                                  param_grid['reg_lambda']))

total_runs = len(param_combinations)
best_accuracy = 0
best_params = None

logger.info("Starting custom GridSearch for XGBoost")

# Corrected loop to unpack all parameters
for i, (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda) in enumerate(tqdm(param_combinations, total=total_runs), 1):
    xgb_model.set_params(n_estimators=n_estimators,
                         max_depth=max_depth,
                         learning_rate=learning_rate,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         gamma=gamma,
                         reg_alpha=reg_alpha,
                         reg_lambda=reg_lambda)

    # Perform cross-validation
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=3, n_jobs=-1)
    mean_cv_score = np.mean(cv_scores)

    # Update best parameters if the current model is better
    if mean_cv_score > best_accuracy:
        best_accuracy = mean_cv_score
        best_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda
        }
        logger.info(f'NEW BEST ACCURACY ACHIEVED: {best_accuracy:.4f}')
        logger.info(f"Hyperparameters: {best_params}")

    logger.info(f"Run {i} / {total_runs} complete")

logger.info("Training best model with best parameters")
xgb_model.set_params(**best_params)
xgb_model.fit(X_train, y_train)


y_pred = xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Hyperparameters: {best_params}")

xgb_model.save_model('best_xgb_model.json')

# Train a Random Forest classifier to get feature importances
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Get feature importances from the Random Forest model
importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

# Save the feature importances to a CSV file
importances_df = pd.DataFrame({
    'Feature': df.columns[sorted_idx],  # Use data.feature_names for the original features
    'Importance': importances[sorted_idx]
})


top_50 = importances_df.head(50)
important_features = top_50['Feature'].tolist()
print(f"Most important features = {important_features}")