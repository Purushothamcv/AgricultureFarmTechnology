"""
Fix Dataset Structure
====================
Reorganizes the dataset from nested structure to flat structure

Current:  archive/APPLE/Blotch_Apple/
Fixed:    archive/Blotch_Apple/

This script will move all disease folders to the root level.
"""

import os
import shutil
from pathlib import Path


def reorganize_dataset():
    """Flatten the dataset structure"""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    archive_path = os.path.join(base_dir, 'data', 'archive')
    
    print("\n" + "="*70)
    print(" "*20 + "REORGANIZING DATASET")
    print("="*70 + "\n")
    
    if not os.path.exists(archive_path):
        print(f"❌ Archive path not found: {archive_path}")
        return False
    
    # Fruit folders to process
    fruit_folders = ['APPLE', 'GUAVA', 'MANGO', 'POMEGRANATE']
    
    moved_count = 0
    
    for fruit in fruit_folders:
        fruit_path = os.path.join(archive_path, fruit)
        
        if not os.path.exists(fruit_path):
            print(f"⚠️  Skipping {fruit} (not found)")
            continue
        
        print(f"📁 Processing {fruit}/")
        
        # Get all disease folders inside fruit folder
        disease_folders = [d for d in os.listdir(fruit_path)
                          if os.path.isdir(os.path.join(fruit_path, d))]
        
        for disease_folder in disease_folders:
            source = os.path.join(fruit_path, disease_folder)
            destination = os.path.join(archive_path, disease_folder)
            
            # Check if destination already exists
            if os.path.exists(destination):
                print(f"  ⚠️  {disease_folder} already exists at root, skipping...")
                continue
            
            # Move the folder
            try:
                shutil.move(source, destination)
                print(f"  ✓ Moved: {disease_folder}")
                moved_count += 1
            except Exception as e:
                print(f"  ❌ Error moving {disease_folder}: {e}")
        
        # Remove empty fruit folder
        try:
            if not os.listdir(fruit_path):  # Check if empty
                os.rmdir(fruit_path)
                print(f"  ✓ Removed empty folder: {fruit}")
        except Exception as e:
            print(f"  ⚠️  Could not remove {fruit} folder: {e}")
    
    print(f"\n" + "="*70)
    print(f"✅ Reorganization complete!")
    print(f"   Moved {moved_count} disease folders to root level")
    print("="*70 + "\n")
    
    # Verify new structure
    print("New structure:")
    disease_folders = [d for d in os.listdir(archive_path)
                      if os.path.isdir(os.path.join(archive_path, d))]
    
    for folder in sorted(disease_folders):
        image_count = len([f for f in os.listdir(os.path.join(archive_path, folder))
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        print(f"  ✓ {folder:50s}: {image_count:4d} images")
    
    print(f"\n📊 Total classes: {len(disease_folders)}")
    print("\n✅ Dataset is now ready for training!")
    print("   Run: python verify_and_train.py\n")
    
    return True


if __name__ == "__main__":
    try:
        reorganize_dataset()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
