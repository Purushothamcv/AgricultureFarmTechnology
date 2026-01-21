import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load the crop dataset
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
crop_df = pd.read_csv(os.path.join(base_path, 'data', 'crop.csv'))

# Debug: Check for missing values
print("Missing values per column:\n", crop_df.isnull().sum())

# Debug: Check unique labels
print("Unique crop labels:", crop_df['label'].unique())

# Features and target
X = crop_df[['N','P','K','temperature','humidity','ph','rainfall','ozone']]
y = crop_df['label']

# Debug: Show feature sample
print("Feature sample:")
print(X.head())

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
model_path = os.path.join(base_path, 'model', 'crop_model.pkl')
joblib.dump(model, model_path)

print('Crop recommendation model trained and saved as crop_model.pkl')
