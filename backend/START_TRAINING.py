"""
START TRAINING - Fruit Disease Detection Model
==============================================
Simple script to start training immediately

This will:
1. Verify dataset structure
2. Train EfficientNet-B0 model
3. Generate fruit_disease_model.h5
4. Create evaluation reports
5. Save everything for deployment

Expected time: 1-3 hours (GPU) or 6-12 hours (CPU)

Usage:
    python START_TRAINING.py
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*70)
print(" "*10 + "🚀 STARTING FRUIT DISEASE MODEL TRAINING 🚀")
print("="*70)

# Check if dataset exists
dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'archive')
if not os.path.exists(dataset_path):
    print("\n❌ ERROR: Dataset not found!")
    print(f"Expected path: {dataset_path}")
    print("\nPlease ensure your dataset is in backend/data/archive/")
    sys.exit(1)

print(f"\n✓ Dataset found: {dataset_path}")

# Import and run training
try:
    print("\n📦 Loading training modules...")
    from model import train_fruit_disease_model
    
    print("✓ Modules loaded successfully")
    print("\n" + "="*70)
    print("STARTING TRAINING - This will take 1-3 hours")
    print("="*70 + "\n")
    
    # Run training
    train_fruit_disease_model.main()
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\n✓ Model saved to: backend/model/fruit_disease_model.h5")
    print("✓ Ready for deployment!")
    print("\nNext step: Start your API server")
    print("  uvicorn main_fastapi:app --reload")
    print("\n" + "="*70 + "\n")
    
except ImportError as e:
    print("\n❌ ERROR: Missing dependencies!")
    print(f"Error: {e}")
    print("\nInstall required packages:")
    print("  pip install tensorflow keras pillow matplotlib seaborn scikit-learn")
    sys.exit(1)
    
except Exception as e:
    print(f"\n❌ ERROR during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
