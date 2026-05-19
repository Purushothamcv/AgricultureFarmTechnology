import joblib
import json
from pathlib import Path

model_dir = Path(__file__).parent

encoders_path = model_dir / 'fertilizer_encoders.pkl'
label_encoder_path = model_dir / 'fertilizer_label_encoder.pkl'
model_path = model_dir / 'fertilizer_model.pkl'
feature_info_path = model_dir / 'fertilizer_feature_info.json'

print('--- Inspecting fertilizer model artifacts ---')

# Load feature info
with open(feature_info_path, 'r') as f:
    feature_info = json.load(f)

print('\nFeature info:')
print('feature_columns:', feature_info.get('feature_columns'))
print('original_features:', feature_info.get('original_features'))
print('numerical_features:', feature_info.get('numerical_features'))
print('fertilizer_classes:', feature_info.get('fertilizer_classes'))

# Load encoders
encoders = joblib.load(encoders_path)
print('\nEncoders (categorical features and their classes):')
for feat, enc in encoders.items():
    try:
        classes = enc.classes_.tolist()
    except Exception as e:
        classes = str(e)
    print(f"- {feat}: {classes}")

# Load label encoder
label_enc = joblib.load(label_encoder_path)
try:
    label_classes = label_enc.classes_.tolist()
except Exception as e:
    label_classes = str(e)
print('\nLabel encoder classes:', label_classes)

# Load model
model = joblib.load(model_path)
print('\nModel type:', type(model))
# Print model classes_ if available
if hasattr(model, 'classes_'):
    print('model.classes_:', model.classes_.tolist())
else:
    print('model.classes_ not present')

# Print feature names expected by model if available
if hasattr(model, 'feature_names_in_'):
    print('model.feature_names_in_:', model.feature_names_in_.tolist())
else:
    print('model.feature_names_in_ attribute not found')

print('\n--- End of inspection ---')
