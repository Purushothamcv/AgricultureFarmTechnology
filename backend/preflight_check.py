"""
Pre-Flight Check for Fruit Disease Detection Module
===================================================
Verifies that all components are properly set up before training

Author: SmartAgri-AI Team
Date: 2026-01-21
"""

import os
import sys
from pathlib import Path
import importlib.util


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
    print(f"{text:^70}")
    print(f"{'='*70}{Colors.END}\n")


def print_check(name, status, message=""):
    """Print check result"""
    symbol = f"{Colors.GREEN}✓{Colors.END}" if status else f"{Colors.RED}✗{Colors.END}"
    status_text = f"{Colors.GREEN}PASS{Colors.END}" if status else f"{Colors.RED}FAIL{Colors.END}"
    print(f"{symbol} {name:50s} [{status_text}]")
    if message:
        print(f"  → {message}")


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    is_valid = version.major == 3 and version.minor >= 8
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    message = f"Python {version_str}" if is_valid else f"Python {version_str} (Need 3.8+)"
    return is_valid, message


def check_package(package_name, import_name=None):
    """Check if package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True, f"{package_name} installed"
    except ImportError:
        return False, f"{package_name} not installed"


def check_directory_structure():
    """Check if required directories exist"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    required_dirs = {
        'data': os.path.join(base_dir, 'data'),
        'data/archive': os.path.join(base_dir, 'data', 'archive'),
        'model': os.path.join(base_dir, 'model'),
    }
    
    results = {}
    for name, path in required_dirs.items():
        exists = os.path.exists(path) and os.path.isdir(path)
        results[name] = (exists, path)
    
    return results


def check_dataset():
    """Check if dataset is properly structured"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'data', 'archive')
    
    if not os.path.exists(dataset_path):
        return False, "Dataset directory not found"
    
    # Expected classes
    expected_classes = [
        'Blotch_Apple', 'Rot_Apple', 'Scab_Apple', 'Healthy_Apple',
        'Anthracnose_Guava', 'Fruitfly_Guava', 'Healthy_Guava',
        'Alternaria_Mango', 'Anthracnose_Mango', 
        'Black Mould Rot (Aspergillus)_Mango',
        'Stem and Rot (Lasiodiplodia)_Mango', 'Healthy_Mango',
        'Alternaria_Pomegranate', 'Anthracnose_Pomegranate',
        'Bacterial_Blight_Pomegranate', 'Cercospora_Pomegranate',
        'Healthy_Pomegranate'
    ]
    
    # Check subdirectories
    subdirs = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    if len(subdirs) == 0:
        return False, "No class directories found"
    
    # Count images
    total_images = 0
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    for subdir in subdirs:
        subdir_path = os.path.join(dataset_path, subdir)
        images = [f for f in os.listdir(subdir_path)
                 if Path(f).suffix.lower() in image_extensions]
        total_images += len(images)
    
    message = f"{len(subdirs)} classes, {total_images} images"
    return True, message


def check_scripts():
    """Check if required scripts exist"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    required_scripts = {
        'Training Script': os.path.join(base_dir, 'model', 'train_fruit_disease_model.py'),
        'Inference Script': os.path.join(base_dir, 'model', 'fruit_disease_inference.py'),
        'Dataset Analyzer': os.path.join(base_dir, 'model', 'dataset_analyzer.py'),
        'FastAPI Service': os.path.join(base_dir, 'fruit_disease_service.py'),
        'Quick Start': os.path.join(base_dir, 'quick_start.py'),
    }
    
    results = {}
    for name, path in required_scripts.items():
        exists = os.path.exists(path) and os.path.isfile(path)
        results[name] = exists
    
    return results


def check_gpu():
    """Check if GPU is available for TensorFlow"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return True, f"{len(gpus)} GPU(s) available"
        else:
            return True, "No GPU (will use CPU)"
    except Exception as e:
        return False, f"TensorFlow check failed: {str(e)}"


def main():
    """Run all pre-flight checks"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("="*70)
    print(" "*15 + "FRUIT DISEASE DETECTION MODULE")
    print(" "*20 + "PRE-FLIGHT CHECK")
    print("="*70)
    print(Colors.END)
    
    all_passed = True
    
    # Check 1: Python Version
    print_header("PYTHON ENVIRONMENT")
    status, message = check_python_version()
    print_check("Python Version", status, message)
    all_passed &= status
    
    # Check 2: Required Packages
    print_header("REQUIRED PACKAGES")
    packages = [
        ('tensorflow', 'tensorflow'),
        ('keras', 'keras'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('pillow', 'PIL'),
        ('scikit-learn', 'sklearn'),
        ('seaborn', 'seaborn'),
        ('fastapi', 'fastapi'),
    ]
    
    for package_name, import_name in packages:
        status, message = check_package(package_name, import_name)
        print_check(package_name, status, message)
        all_passed &= status
    
    # Check 3: Directory Structure
    print_header("DIRECTORY STRUCTURE")
    dir_results = check_directory_structure()
    for name, (exists, path) in dir_results.items():
        print_check(f"{name}/", exists, path if exists else f"Missing: {path}")
        all_passed &= exists
    
    # Check 4: Dataset
    print_header("DATASET")
    status, message = check_dataset()
    print_check("Dataset Structure", status, message)
    all_passed &= status
    
    # Check 5: Required Scripts
    print_header("REQUIRED SCRIPTS")
    script_results = check_scripts()
    for name, exists in script_results.items():
        print_check(name, exists)
        all_passed &= exists
    
    # Check 6: GPU Availability
    print_header("HARDWARE")
    status, message = check_gpu()
    print_check("GPU/CPU Support", status, message)
    # Don't fail on GPU check
    
    # Final Summary
    print("\n" + "="*70)
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}")
        print("✅ ALL CHECKS PASSED - READY TO TRAIN!")
        print(Colors.END)
        print("\nNext steps:")
        print("1. Analyze dataset:  python quick_start.py --analyze")
        print("2. Train model:      python quick_start.py --train")
        print("3. Test model:       python quick_start.py --test <image_path>")
        print("4. Full workflow:    python quick_start.py --full")
    else:
        print(f"{Colors.RED}{Colors.BOLD}")
        print("❌ SOME CHECKS FAILED - PLEASE FIX ISSUES ABOVE")
        print(Colors.END)
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
    
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
