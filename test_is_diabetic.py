import pandas as pd
import xgboost as xgb
import time

begin = time.time()

# Load the trained XGBoost models
xgb_model_is_diabetic = xgb.Booster()
xgb_model_is_diabetic.load_model('is_diabetic_classifier.json')

xgb_model_develops_diabetes = xgb.Booster()
xgb_model_develops_diabetes.load_model('develops_diabetes_classifier.json')

# Load the CSV file containing test data
data = pd.read_csv('form_results_test.csv')

# Rename columns as needed
renaming = {
    '6152-0.0_Blood clot DVT bronchitis emphysema asthma rhinitis eczema allergy diagnosed by doctor': '6152-0.0_Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor',
    '21003-0.0_Age when attended assessment center': '21003-0.0_Age when attended assessment centre',
    '24018-0.0_Nitrogen dioxide air pollution': '24018-0.0_Nitrogen dioxide air pollution; 2007',
    '6144-0.0_Never eat eggs dairy wheat sugar': '6144-0.0_Never eat eggs, dairy, wheat, sugar',
    '6150-0.0_Vascular heart problems diagnosed by doctor':  '6150-0.0_Vascular/heart problems diagnosed by doctor',
    '6154-0.0_Medication for pain relief constipation heartburn': '6154-0.0_Medication for pain relief, constipation, heartburn',
    '1558-0.0_Alcohol intake frequency': '1558-0.0_Alcohol intake frequency.'
}

data.rename(columns=renaming, inplace=True)

# Remove the target column from test data
data = data.drop(columns=['2443-0.0_Diabetes diagnosed by doctor'], errors='ignore')

# Load the original training data to get the correct column order
training_data = pd.read_csv('filtered_cleaned_titles.csv')

# Get the training column names (excluding the target column)
training_columns = training_data.drop(columns=['2443-0.0_Diabetes diagnosed by doctor']).columns.tolist()

# Ensure the test data has the same columns and order as the training data
data = data[training_columns]

# Convert the data to a format suitable for XGBoost
dmatrix = xgb.DMatrix(data.values, feature_names=training_columns)

# Make predictions with the first XGBoost model (is_diabetic)
xgb_prediction_is_diabetic = xgb_model_is_diabetic.predict(dmatrix)
xgb_percentage_chance_is_diabetic = xgb_prediction_is_diabetic[0] * 100

print()
# Print the result for the is_diabetic model
if xgb_prediction_is_diabetic[0] > 0.5:
    print(f"The model predicts that the person is a type 2 diabetic with a {xgb_percentage_chance_is_diabetic:.2f}% chance of being diabetic.")
else:
    print(f"The model predicts that the person is not a type 2 diabetic with a {xgb_percentage_chance_is_diabetic:.2f}% chance of being diabetic.")

# Make predictions with the second XGBoost model (develops_diabetes)
xgb_prediction_develops_diabetes = xgb_model_develops_diabetes.predict(dmatrix)
xgb_percentage_chance_develops_diabetes = xgb_prediction_develops_diabetes[0] * 100
print()
# Print the result for the develops_diabetes model
if xgb_prediction_develops_diabetes[0] > 0.5:
    print(f"The model predicts that the person will develop type 2 diabetes with a {xgb_percentage_chance_develops_diabetes:.2f}%")
    print("chance of developing diabetes within the next 15 years.")
else:
    print(f"The model predicts that the person will not develop type 2 diabetes with a {xgb_percentage_chance_develops_diabetes:.2f}%")
    print("chance of developing diabetes within the next 15 years.")

end = time.time()

s = end - begin
print()
print(f'Model took {s:.3f} seconds to calculate')