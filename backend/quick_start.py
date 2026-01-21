"""
Fruit Disease Detection - Quick Start Script
============================================
Automated workflow for dataset analysis, training, and testing

Usage:
    python quick_start.py --analyze        # Analyze dataset only
    python quick_start.py --train          # Train model only
    python quick_start.py --test <image>   # Test with image
    python quick_start.py --full           # Complete workflow

Author: SmartAgri-AI Team
Date: 2026-01-21
"""

import os
import sys
import argparse
from pathlib import Path

# Add model directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'model'))


def print_banner(text):
    """Print formatted banner"""
    print("\n" + "="*70)
    print(f"{text:^70}")
    print("="*70 + "\n")


def analyze_dataset():
    """Run dataset analysis"""
    print_banner("STEP 1: DATASET ANALYSIS")
    
    try:
        from dataset_analyzer import DatasetAnalyzer
        
        dataset_path = os.path.join(BASE_DIR, 'data', 'archive')
        
        analyzer = DatasetAnalyzer(dataset_path)
        
        # Validate structure
        if not analyzer.validate_structure():
            print("❌ Dataset validation failed!")
            return False
        
        # Analyze
        analyzer.analyze()
        
        # Check balance
        analyzer.check_balance()
        
        # Visualize
        plot_path = os.path.join(BASE_DIR, 'model', 'dataset_distribution.png')
        analyzer.visualize_distribution(save_path=plot_path)
        
        # Export stats
        stats_path = os.path.join(BASE_DIR, 'model', 'dataset_stats.json')
        analyzer.export_stats(stats_path)
        
        print("✅ Dataset analysis completed successfully!\n")
        return True
        
    except Exception as e:
        print(f"❌ Dataset analysis failed: {e}\n")
        return False


def train_model():
    """Run model training"""
    print_banner("STEP 2: MODEL TRAINING")
    
    try:
        # Import and run training
        import train_fruit_disease_model
        train_fruit_disease_model.main()
        
        print("\n✅ Model training completed successfully!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Model training failed: {e}\n")
        return False


def test_model(image_path):
    """Test model with image"""
    print_banner("STEP 3: MODEL TESTING")
    
    try:
        from fruit_disease_inference import FruitDiseasePredictor
        
        # Check if model exists
        model_path = os.path.join(BASE_DIR, 'model', 'fruit_disease_model.h5')
        if not os.path.exists(model_path):
            print("❌ Model not found! Please train the model first.")
            return False
        
        # Initialize predictor
        print("Loading model...")
        predictor = FruitDiseasePredictor()
        
        # Make prediction
        print(f"Predicting: {image_path}\n")
        result = predictor.predict_with_recommendations(image_path)
        
        if result['success']:
            print("="*70)
            print("PREDICTION RESULTS")
            print("="*70)
            print(f"\n✅ Predicted Class:  {result['predicted_class']}")
            print(f"🍎 Fruit Type:       {result['fruit_type']}")
            print(f"🦠 Disease:          {result['disease']}")
            print(f"📊 Confidence:       {result['confidence_percentage']}")
            print(f"\n💊 Treatment Recommendation:")
            print(f"   {result['treatment']}")
            
            print("\n📈 Top 3 Predictions:")
            for i, pred in enumerate(result['top_predictions'], 1):
                print(f"   {i}. {pred['class']:50s} {pred['percentage']:>8s}")
            
            print("\n" + "="*70 + "\n")
            return True
        else:
            print(f"❌ Prediction failed: {result['error']}\n")
            return False
        
    except Exception as e:
        print(f"❌ Model testing failed: {e}\n")
        return False


def run_full_workflow():
    """Run complete workflow"""
    print("\n" + "="*70)
    print("FRUIT DISEASE DETECTION - COMPLETE WORKFLOW")
    print("="*70)
    
    # Step 1: Analyze dataset
    if not analyze_dataset():
        print("❌ Workflow stopped: Dataset analysis failed")
        return
    
    # Step 2: Train model
    proceed = input("Dataset analysis complete. Proceed with training? (y/n): ")
    if proceed.lower() != 'y':
        print("Workflow stopped by user.")
        return
    
    if not train_model():
        print("❌ Workflow stopped: Model training failed")
        return
    
    # Step 3: Test with sample
    print("\n" + "="*70)
    print("Model training complete!")
    test_img = input("\nEnter image path for testing (or press Enter to skip): ").strip()
    
    if test_img and os.path.exists(test_img):
        test_model(test_img)
    
    print("\n" + "="*70)
    print("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review generated plots and metrics")
    print("2. Test with more images")
    print("3. Integrate with FastAPI")
    print("4. Deploy to production\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Fruit Disease Detection - Quick Start',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_start.py --analyze                    # Analyze dataset
  python quick_start.py --train                      # Train model
  python quick_start.py --test path/to/image.jpg     # Test model
  python quick_start.py --full                       # Complete workflow
        """
    )
    
    parser.add_argument('--analyze', action='store_true',
                       help='Run dataset analysis only')
    parser.add_argument('--train', action='store_true',
                       help='Train model only')
    parser.add_argument('--test', type=str, metavar='IMAGE_PATH',
                       help='Test model with image')
    parser.add_argument('--full', action='store_true',
                       help='Run complete workflow')
    
    args = parser.parse_args()
    
    # Show help if no arguments
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Execute based on arguments
    if args.full:
        run_full_workflow()
    elif args.analyze:
        analyze_dataset()
    elif args.train:
        train_model()
    elif args.test:
        if not os.path.exists(args.test):
            print(f"❌ Image not found: {args.test}")
        else:
            test_model(args.test)


if __name__ == "__main__":
    main()
