"""
RESTART TRAINING - CLEAN SLATE
===============================
This script safely removes old checkpoints and starts fresh training.
Use this when you want to completely restart training from scratch.

Author: SmartAgri-AI Team
Date: January 22, 2026
"""

import os
import shutil
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# Files to backup and remove
FILES_TO_CLEAN = [
    'fruit_disease_model.h5',
    'training_history.json',
    'training_history.png',
    'classification_report.txt',
    'confusion_matrix.png'
]

def restart_training():
    """Remove old checkpoints and start fresh training"""
    
    print("\n" + "="*70)
    print("RESTARTING TRAINING - CLEAN SLATE")
    print("="*70)
    
    # Create backup directory
    backup_dir = os.path.join(MODEL_DIR, f'backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(backup_dir, exist_ok=True)
    print(f"\n✓ Created backup directory: {backup_dir}")
    
    # Backup and remove old files
    print("\n" + "-"*70)
    print("BACKING UP AND REMOVING OLD FILES:")
    print("-"*70)
    
    for filename in FILES_TO_CLEAN:
        filepath = os.path.join(MODEL_DIR, filename)
        
        if os.path.exists(filepath):
            # Backup
            backup_path = os.path.join(backup_dir, filename)
            shutil.copy2(filepath, backup_path)
            print(f"✓ Backed up: {filename}")
            
            # Remove
            os.remove(filepath)
            print(f"✓ Removed: {filename}")
        else:
            print(f"⊘ Not found: {filename}")
    
    print("\n" + "="*70)
    print("CLEANUP COMPLETE")
    print("="*70)
    print(f"✓ Old files backed up to: {backup_dir}")
    print("✓ Training environment is clean")
    print("\nYou can now run fresh training with:")
    print("  python backend/model/train_fruit_disease_optimized.py")
    print("\nor:")
    print("  python backend/model/train_fruit_disease_model.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Safety confirmation
    print("\n⚠️  WARNING: This will remove all existing checkpoints and training history!")
    print("Old files will be backed up before removal.")
    
    response = input("\nAre you sure you want to restart training? (yes/no): ")
    
    if response.lower() in ['yes', 'y']:
        restart_training()
    else:
        print("\n✓ Restart cancelled. No files were modified.\n")
