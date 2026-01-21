"""
Fruit Disease Dataset Analysis Utility
======================================
Helper functions for dataset exploration, validation, and visualization

Author: SmartAgri-AI Team
Date: 2026-01-21
"""

import os
import json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class DatasetAnalyzer:
    """Analyze and validate fruit disease dataset"""
    
    def __init__(self, dataset_path):
        """
        Initialize analyzer
        
        Args:
            dataset_path: Path to dataset root directory
        """
        self.dataset_path = dataset_path
        self.stats = None
    
    def analyze(self):
        """
        Perform comprehensive dataset analysis
        
        Returns:
            Dictionary with dataset statistics
        """
        print("\n" + "="*70)
        print(" "*20 + "DATASET ANALYSIS")
        print("="*70)
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        stats = {
            'total_images': 0,
            'total_classes': 0,
            'classes': {},
            'fruits': defaultdict(lambda: {'classes': [], 'count': 0}),
            'min_samples': float('inf'),
            'max_samples': 0,
            'avg_samples': 0
        }
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        # Scan dataset directory
        print(f"\n📂 Scanning dataset: {self.dataset_path}\n")
        
        for fruit_dir in os.listdir(self.dataset_path):
            fruit_path = os.path.join(self.dataset_path, fruit_dir)
            
            # Skip if not a directory
            if not os.path.isdir(fruit_path):
                continue
            
            # Extract fruit name (last part after underscore)
            class_name = fruit_dir
            parts = class_name.split('_')
            fruit_name = parts[-1] if parts else "Unknown"
            disease_name = '_'.join(parts[:-1]) if len(parts) > 1 else "Unknown"
            
            # Count images in this class
            image_files = [
                f for f in os.listdir(fruit_path)
                if os.path.isfile(os.path.join(fruit_path, f)) and
                Path(f).suffix.lower() in image_extensions
            ]
            
            num_images = len(image_files)
            
            # Update stats
            stats['classes'][class_name] = {
                'count': num_images,
                'fruit': fruit_name,
                'disease': disease_name,
                'path': fruit_path
            }
            
            stats['fruits'][fruit_name]['classes'].append(class_name)
            stats['fruits'][fruit_name]['count'] += num_images
            
            stats['total_images'] += num_images
            stats['min_samples'] = min(stats['min_samples'], num_images)
            stats['max_samples'] = max(stats['max_samples'], num_images)
            
            print(f"✓ {class_name:50s} : {num_images:5d} images")
        
        stats['total_classes'] = len(stats['classes'])
        stats['avg_samples'] = stats['total_images'] / stats['total_classes'] if stats['total_classes'] > 0 else 0
        
        self.stats = stats
        
        # Print summary
        print("\n" + "="*70)
        print("DATASET SUMMARY")
        print("="*70)
        print(f"Total Images:     {stats['total_images']:,}")
        print(f"Total Classes:    {stats['total_classes']}")
        print(f"Fruit Types:      {len(stats['fruits'])}")
        print(f"Min Samples:      {stats['min_samples']}")
        print(f"Max Samples:      {stats['max_samples']}")
        print(f"Avg Samples:      {stats['avg_samples']:.2f}")
        
        print("\n" + "-"*70)
        print("FRUIT-WISE BREAKDOWN")
        print("-"*70)
        for fruit, info in sorted(stats['fruits'].items()):
            print(f"{fruit:20s}: {info['count']:5d} images across {len(info['classes'])} classes")
        
        print("\n" + "="*70 + "\n")
        
        return stats
    
    def check_balance(self, threshold=0.5):
        """
        Check dataset balance
        
        Args:
            threshold: Imbalance threshold (0-1)
            
        Returns:
            Boolean indicating if dataset is balanced
        """
        if self.stats is None:
            self.analyze()
        
        print("\n" + "="*70)
        print("DATASET BALANCE CHECK")
        print("="*70)
        
        min_samples = self.stats['min_samples']
        max_samples = self.stats['max_samples']
        
        imbalance_ratio = min_samples / max_samples if max_samples > 0 else 0
        is_balanced = imbalance_ratio >= threshold
        
        print(f"\nImbalance Ratio: {imbalance_ratio:.3f}")
        print(f"Threshold:       {threshold}")
        print(f"Status:          {'✅ BALANCED' if is_balanced else '⚠️  IMBALANCED'}")
        
        if not is_balanced:
            print("\n💡 Recommendations:")
            print("   - Consider data augmentation for underrepresented classes")
            print("   - Use class weights during training")
            print("   - Collect more samples for smaller classes")
        
        print("\n" + "="*70 + "\n")
        
        return is_balanced
    
    def visualize_distribution(self, save_path=None):
        """
        Visualize class distribution
        
        Args:
            save_path: Optional path to save plot
        """
        if self.stats is None:
            self.analyze()
        
        # Prepare data
        classes = list(self.stats['classes'].keys())
        counts = [self.stats['classes'][c]['count'] for c in classes]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Plot 1: Bar chart
        colors = plt.cm.Set3(range(len(classes)))
        bars = ax1.barh(classes, counts, color=colors, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Number of Images', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Class Name', fontsize=12, fontweight='bold')
        ax1.set_title('Class Distribution - Fruit Disease Dataset', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax1.text(count + 10, i, str(count), va='center', fontsize=9)
        
        # Plot 2: Fruit-wise pie chart
        fruit_names = list(self.stats['fruits'].keys())
        fruit_counts = [self.stats['fruits'][f]['count'] for f in fruit_names]
        
        wedges, texts, autotexts = ax2.pie(
            fruit_counts, 
            labels=fruit_names,
            autopct='%1.1f%%',
            startangle=90,
            colors=plt.cm.Pastel1(range(len(fruit_names))),
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        ax2.set_title('Fruit-wise Distribution', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Distribution plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def validate_structure(self):
        """
        Validate dataset structure
        
        Returns:
            Boolean indicating if structure is valid
        """
        print("\n" + "="*70)
        print("DATASET STRUCTURE VALIDATION")
        print("="*70)
        
        if not os.path.exists(self.dataset_path):
            print(f"❌ Dataset path does not exist: {self.dataset_path}")
            return False
        
        if not os.path.isdir(self.dataset_path):
            print(f"❌ Dataset path is not a directory: {self.dataset_path}")
            return False
        
        # Check for class directories
        subdirs = [d for d in os.listdir(self.dataset_path) 
                   if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        if len(subdirs) == 0:
            print(f"❌ No class directories found in: {self.dataset_path}")
            return False
        
        print(f"\n✅ Dataset structure is valid")
        print(f"✓ Found {len(subdirs)} class directories")
        
        # Check for empty directories
        empty_dirs = []
        for subdir in subdirs:
            subdir_path = os.path.join(self.dataset_path, subdir)
            files = os.listdir(subdir_path)
            if len(files) == 0:
                empty_dirs.append(subdir)
        
        if empty_dirs:
            print(f"\n⚠️  Warning: {len(empty_dirs)} empty directories found:")
            for d in empty_dirs:
                print(f"   - {d}")
        
        print("\n" + "="*70 + "\n")
        
        return True
    
    def export_stats(self, output_path):
        """
        Export statistics to JSON
        
        Args:
            output_path: Path to save JSON file
        """
        if self.stats is None:
            self.analyze()
        
        # Convert defaultdict to dict for JSON serialization
        export_data = {
            'total_images': self.stats['total_images'],
            'total_classes': self.stats['total_classes'],
            'classes': self.stats['classes'],
            'fruits': dict(self.stats['fruits']),
            'min_samples': self.stats['min_samples'],
            'max_samples': self.stats['max_samples'],
            'avg_samples': self.stats['avg_samples']
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=4)
        
        print(f"✓ Statistics exported to: {output_path}")


def main():
    """Main function for dataset analysis"""
    import sys
    
    # Get dataset path
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = os.path.join(base_dir, 'data', 'archive')
    
    print("\n" + "="*70)
    print(" "*15 + "FRUIT DISEASE DATASET ANALYZER")
    print("="*70)
    print(f"\nDataset Path: {dataset_path}\n")
    
    try:
        # Initialize analyzer
        analyzer = DatasetAnalyzer(dataset_path)
        
        # Validate structure
        if not analyzer.validate_structure():
            print("❌ Dataset structure validation failed!")
            return
        
        # Analyze dataset
        stats = analyzer.analyze()
        
        # Check balance
        analyzer.check_balance(threshold=0.5)
        
        # Visualize distribution
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        plot_path = os.path.join(base_dir, 'model', 'dataset_distribution.png')
        analyzer.visualize_distribution(save_path=plot_path)
        
        # Export stats
        stats_path = os.path.join(base_dir, 'model', 'dataset_stats.json')
        analyzer.export_stats(stats_path)
        
        print("\n" + "="*70)
        print("✅ Dataset analysis completed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}\n")


if __name__ == "__main__":
    main()
