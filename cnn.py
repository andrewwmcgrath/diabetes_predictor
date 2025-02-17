import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, Dense
from tensorflow.keras import regularizers
import numpy as np
import os
import sys

learning_rate = 4e-4
filters1 = 64
filters2 = 128
kernel_size = 3
dropout_rate = 0.25
l2_reg = 1e-3
epochs = 25
batch_size = 32

# Load the data
directory = os.getcwd()
file_path = os.path.join(directory, 'cleaned.csv')  

df = pd.read_csv(file_path)
print(f"Data loaded successfully with shape: {df.shape}")
df.columns = df.columns.str.strip()

df.dropna(subset=['2443-0.0_Diabetes diagnosed by doctor'], inplace=True)
df.fillna(df.mean(), inplace=True)  

# Extract features and target variable
X = df.drop(columns=['2443-0.0_Diabetes diagnosed by doctor'])
y = df['2443-0.0_Diabetes diagnosed by doctor']

y_bar = y.mean()
print(f"Mean of y = {y_bar:.4f}")

if y_bar == 0:
    print("Error: Mean of y = 0, meaning everyone is defined as not developing diabetes.")
    print("Stopping program due to error.")
    sys.exit(1)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape the data for Conv1D
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
print('Data ready for ML')

model = Sequential([
    Conv1D(filters=filters1, kernel_size=kernel_size, activation='relu', input_shape=(X_train.shape[1], 1), kernel_regularizer=regularizers.l2(l2_reg)),
    Conv1D(filters=filters2, kernel_size=kernel_size, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
    Dropout(dropout_rate),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
    Dropout(dropout_rate),  # Additional dropout layer
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks = [early_stopping])

# Make predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the CNN model
model.save('is_diabetic_classifier.keras')
print("Model saved as 'is_diabetic_classifier.keras'.")

# Train a Random Forest classifier to get feature importances
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_scaled, y)

# Get feature importances from the Random Forest model
importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

# Save the feature importances to a CSV file
importances_df = pd.DataFrame({
    'Feature': X.columns[sorted_idx],
    'Importance': importances[sorted_idx]
})

importances_df.to_csv('feature_importances_rf.csv', index=False)
print("Feature importances saved to 'feature_importances_rf.csv'.")

# Print the top 200 features by importance as a list
top_50 = importances_df.head(50)
important_features = top_50['Feature'].tolist()
print(f"Most important features = {important_features}")