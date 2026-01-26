"""
Quick Start - Fruit Disease CNN Training
=========================================
This script provides an interactive guide to restart training from scratch
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    print("\n" + "=" * 70)
    print("🍎 FRUIT DISEASE CNN TRAINING - CLEAN RESTART")
    print("=" * 70)
    print()
    print("This script will guide you through restarting the training process")
    print("completely from scratch (no checkpoints, no old optimizer state).")
    print()

def print_step(step_num, title):
    """Print step header"""
    print("\n" + "-" * 70)
    print(f"STEP {step_num}: {title}")
    print("-" * 70)

def confirm(message):
    """Get user confirmation"""
    while True:
        response = input(f"{message} (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'")

def run_verification():
    """Run pre-training verification"""
    print_step(1, "PRE-TRAINING VERIFICATION")
    print("\nRunning verification checks...")
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, 'verify_before_training.py'],
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running verification: {e}")
        return False

def show_training_info():
    """Show training information"""
    print_step(2, "TRAINING CONFIGURATION")
    print()
    print("📋 TRAINING DETAILS:")
    print("   • Architecture: EfficientNet-B0 (Transfer Learning)")
    print("   • Dataset: data/archive/ (17 classes)")
    print("   • Training Strategy: Two-Phase")
    print()
    print("   PHASE 1: Feature Extraction")
    print("   • Epochs: 20")
    print("   • Learning Rate: 1e-3")
    print("   • Base Model: FROZEN")
    print()
    print("   PHASE 2: Fine-Tuning")
    print("   • Epochs: 30")
    print("   • Learning Rate: 1e-5")
    print("   • Base Model: Top 30 layers UNFROZEN")
    print()
    print("⏱️  ESTIMATED TIME:")
    print("   • With GPU: 1-2 hours")
    print("   • With CPU: 3-5 hours")
    print()
    print("💾 OUTPUT FILES:")
    print("   • fruit_disease_model.h5 (trained model)")
    print("   • fruit_disease_labels.json (class labels)")
    print("   • training_history.json (metrics)")
    print("   • training_history.png (visualization)")
    print()

def check_old_files():
    """Check for old training files and offer to remove them"""
    print_step(3, "CLEAN OLD FILES")
    print()
    
    old_files = [
        'fruit_disease_model.h5',
        'phase1_best_model.h5',
        'training_history.json',
        'training_history.png'
    ]
    
    found_old = [f for f in old_files if os.path.exists(f)]
    
    if found_old:
        print(f"⚠️  Found {len(found_old)} old training file(s):")
        for f in found_old:
            file_size = os.path.getsize(f) / (1024 * 1024)  # MB
            print(f"   • {f} ({file_size:.1f} MB)")
        print()
        
        if confirm("Do you want to remove these files before training?"):
            for f in found_old:
                try:
                    os.remove(f)
                    print(f"   ✓ Removed: {f}")
                except Exception as e:
                    print(f"   ✗ Error removing {f}: {e}")
            print()
        else:
            print("   ℹ Files will be overwritten during training")
            print()
    else:
        print("✓ No old training files found (clean start)")
        print()

def start_training():
    """Start the training process"""
    print_step(4, "START TRAINING")
    print()
    print("🚀 Ready to start training!")
    print()
    print("⚠️  IMPORTANT:")
    print("   • Do NOT interrupt training once started")
    print("   • Keep your computer powered on")
    print("   • Close unnecessary applications")
    print("   • Training progress will be displayed in real-time")
    print()
    
    if not confirm("Start training now?"):
        print("\n❌ Training cancelled by user")
        return False
    
    print("\n" + "=" * 70)
    print("TRAINING STARTED")
    print("=" * 70)
    print()
    
    try:
        # Run training script
        subprocess.run(
            [sys.executable, 'train_fruit_disease_clean.py'],
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def show_next_steps():
    """Show next steps after training"""
    print("\n" + "=" * 70)
    print("📚 NEXT STEPS")
    print("=" * 70)
    print()
    print("1. Test the Model:")
    print("   • Load: model = keras.models.load_model('fruit_disease_model.h5')")
    print("   • Test with sample images")
    print()
    print("2. Deploy to FastAPI:")
    print("   • Integrate with backend/fruit_disease_service.py")
    print("   • Test API endpoints")
    print()
    print("3. Connect to Frontend:")
    print("   • Update React components")
    print("   • Test end-to-end workflow")
    print()
    print("4. Monitor Performance:")
    print("   • Check prediction accuracy")
    print("   • Measure inference time")
    print()
    print("📖 For deployment instructions, see:")
    print("   • backend/model/PRODUCTION_DEPLOYMENT.md")
    print("   • backend/FRUIT_DISEASE_IMPLEMENTATION.md")
    print()

def main():
    """Main quick start flow"""
    print_banner()
    
    # Step 1: Verification
    print("Before starting, we'll verify that everything is set up correctly.")
    print()
    
    if not confirm("Run pre-training verification?"):
        print("\n⚠️  Skipping verification (not recommended)")
        print()
    else:
        run_verification()
        print()
        if not confirm("Verification complete. Continue to training?"):
            print("\n❌ Setup cancelled by user")
            return
    
    # Step 2: Show training info
    show_training_info()
    
    if not confirm("Review the training configuration. Continue?"):
        print("\n❌ Setup cancelled by user")
        return
    
    # Step 3: Clean old files
    check_old_files()
    
    # Step 4: Start training
    success = start_training()
    
    # Show results
    if success:
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print()
        print("📦 Generated Files:")
        
        output_files = [
            ('fruit_disease_model.h5', 'Trained model (deploy this)'),
            ('fruit_disease_labels.json', 'Class label mapping'),
            ('training_history.json', 'Training metrics'),
            ('training_history.png', 'Training visualization')
        ]
        
        for filename, description in output_files:
            if os.path.exists(filename):
                size = os.path.getsize(filename) / (1024 * 1024)  # MB
                print(f"   ✓ {filename:30s} ({size:6.1f} MB) - {description}")
            else:
                print(f"   ✗ {filename:30s} - NOT FOUND")
        
        show_next_steps()
    else:
        print("\n" + "=" * 70)
        print("❌ TRAINING FAILED OR INTERRUPTED")
        print("=" * 70)
        print()
        print("Troubleshooting:")
        print("   1. Check error messages above")
        print("   2. Review RESTART_TRAINING_GUIDE.md")
        print("   3. Run verification: python verify_before_training.py")
        print("   4. Check dataset: data/archive/")
        print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
