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
import time
import os
import sys

learning_rate = [1e-3, 4e-4, 1e-4]
filters1 = [32, 64]
#filters2 = 2*filters1
kernel_size = [3, 5]
dropout_rate = [0.25, 0.5]
l2_reg = [1e-3, 1e-4]
epochs = 25
batch_size = [32, 64, 128]
total_runs = len(learning_rate) * len(filters1) * len(kernel_size) * len(dropout_rate) * len(l2_reg) * len(epochs) * len(batch_size)
print(f'{total_runs} models to run')

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


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
print('Data ready for ML')

best_accuracy = 0
best_accuracy_classification_report = None
best_params = None
print(f'{total_runs} models to run')

runs = 0
start_time = time.time()
ep = 25

for lr in learning_rate:
    for f1 in filters1:
        for ks in kernel_size:
            for dr in dropout_rate:
                for l2 in l2_reg:
                    for bs in batch_size:
                        runs += 1
                        print(f'Run {runs} / {total_runs}')
                        f2 = 2 * f1

                        model = Sequential([
                            Conv1D(filters=f1, kernel_size=ks, activation='relu', input_shape=(X_train.shape[1], 1), kernel_regularizer=regularizers.l2(l2)),
                            Conv1D(filters=f2, kernel_size=ks, activation='relu', kernel_regularizer=regularizers.l2(l2)),
                            Dropout(dr),
                            Flatten(),
                            Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2)),
                            Dropout(dr),  # Additional dropout layer
                            Dense(1, activation='sigmoid')
                        ])

                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
                        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                        history = model.fit(X_train, y_train, epochs=ep, batch_size=bs, validation_split=0.2, callbacks=[early_stopping])
                        y_pred = (model.predict(X_test) > 0.5).astype("int32")

                        accuracy = accuracy_score(y_test, y_pred)
                        report = classification_report(y_test, y_pred)

                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_accuracy_classification_report = report
                            best_params = {'learning_rate': lr, 'filters1': f1, 'filters2': f2, 'kernel_size': ks, 'dropout_rate': dr, 'l2_reg': l2, 'epochs': ep, 'batch_size': bs}
                            print('NEW BEST ACCURACY ACHIEVED!!!')
                            print(f'accuracy = {accuracy}')
                            print(f"Hyperparameters are lr = {lr}, f1 = {f1}, f2 = {f2}, ks = {ks}, dr = {dr}, l2 = {l2}, bs = {bs}")
                            model.save('is_diabetic_classifier.keras')
                            print("Model saved as 'is_diabetic_classifier.keras'.")

                        print(f'{100 * runs / total_runs:.2f}% complete')
                        current_time = time.time()
                        duration = current_time - start_time
                        duration_hours, rem = divmod(duration, 3600)
                        duration_mins, duration_secs = divmod(rem, 60)
                        print(f'Model has been running for {int(duration_hours)} hours, {int(duration_mins)} minutes, {duration_secs:.0f} seconds.')

print(f'Best parameters are:', best_params)
print(f'Best accuracy = {best_accuracy:.4g}')
print('Classification report:')
print(best_accuracy_classification_report)