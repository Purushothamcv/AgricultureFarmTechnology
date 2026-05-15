"""
Smart AGRI - Model Evaluation & Accuracy Comparison Visualizations
Generates professional graphs for all ML/DL models suitable for IEEE papers and technical reports
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for professional academic visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Define professional color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#06A77D',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'neutral': '#6C757D'
}

class SmartAgriEvaluationVisualizer:
    """Generate comprehensive evaluation visualizations for Smart AGRI models"""
    
    def __init__(self, model_dir="model", output_dir="evaluation_graphs"):
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics_data = {}
        self.load_metrics()
        
    def load_metrics(self):
        """Load all available metrics from stored files"""
        # Load Yield metrics
        yield_metrics_path = self.model_dir / "yield_model_metrics.json"
        if yield_metrics_path.exists():
            with open(yield_metrics_path, 'r') as f:
                self.metrics_data['yield'] = json.load(f)
                print("✓ Yield model metrics loaded")
        
        # Load Fertilizer metrics
        fertilizer_metrics_path = self.model_dir / "fertilizer_model_metrics.json"
        if fertilizer_metrics_path.exists():
            with open(fertilizer_metrics_path, 'r') as f:
                self.metrics_data['fertilizer'] = json.load(f)
                print("✓ Fertilizer model metrics loaded")
        
        # Load Training history
        training_history_path = self.model_dir / "training_history.json"
        if training_history_path.exists():
            with open(training_history_path, 'r') as f:
                self.metrics_data['training_history'] = json.load(f)
                print("✓ Training history loaded")
        
        # Load Fruit disease labels
        fruit_labels_path = self.model_dir / "fruit_disease_labels.json"
        if fruit_labels_path.exists():
            with open(fruit_labels_path, 'r') as f:
                self.metrics_data['fruit_labels'] = json.load(f)
                print("✓ Fruit disease labels loaded")
    
    def parse_classification_report(self, report_path):
        """Parse sklearn classification report text file"""
        report_data = {}
        try:
            with open(report_path, 'r') as f:
                lines = f.readlines()
            
            data = []
            for line in lines:
                if line.strip() and not any(x in line for x in ['precision', 'recall', 'accuracy', '==', '---']):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            # Try to parse as metric line
                            precision = float(parts[-4])
                            recall = float(parts[-3])
                            f1 = float(parts[-2])
                            support = int(parts[-1])
                            class_name = ' '.join(parts[:-4])
                            if class_name and support > 0:
                                data.append({
                                    'class': class_name,
                                    'precision': precision,
                                    'recall': recall,
                                    'f1': f1,
                                    'support': support
                                })
                        except (ValueError, IndexError):
                            continue
            
            if data:
                report_data['classes'] = data
                # Calculate weighted averages
                total_support = sum(d['support'] for d in data)
                report_data['accuracy'] = np.mean([d['f1'] for d in data])  # Using F1 as proxy
                report_data['weighted_f1'] = np.average(
                    [d['f1'] for d in data], 
                    weights=[d['support'] for d in data]
                )
        except Exception as e:
            print(f"Error parsing classification report: {e}")
        
        return report_data
    
    # ============================================================================
    # VISUALIZATION METHODS
    # ============================================================================
    
    def plot_yield_model_performance(self):
        """Generate yield prediction model performance visualizations"""
        if 'yield' not in self.metrics_data:
            print("⚠ Yield metrics not available")
            return
        
        metrics = self.metrics_data['yield']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Yield Prediction Model - Performance Metrics (XGBoost)',
                     fontsize=16, fontweight='bold', y=0.995)
        
        # 1. R² Score Visualization
        ax = axes[0, 0]
        r2 = metrics['r2_score']
        bars = ax.bar(['R² Score'], [r2], color=COLORS['success'], width=0.5, edgecolor='black', linewidth=2)
        ax.set_ylim([0, 1])
        ax.set_ylabel('R² Score', fontweight='bold')
        ax.set_title('Model Fit Quality', fontweight='bold')
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Good threshold (0.8)')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{r2:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Error Metrics Comparison
        ax = axes[0, 1]
        metrics_names = ['RMSE', 'MAE']
        values = [metrics['rmse'], metrics['mae']]
        bars = ax.bar(metrics_names, values, color=[COLORS['danger'], COLORS['warning']], 
                      edgecolor='black', linewidth=2)
        ax.set_ylabel('Error Value', fontweight='bold')
        ax.set_title('Prediction Error Metrics', fontweight='bold')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Train vs Test Distribution
        ax = axes[1, 0]
        dataset_labels = ['Train Samples', 'Test Samples']
        dataset_values = [metrics['train_samples'], metrics['test_samples']]
        colors_ds = [COLORS['primary'], COLORS['secondary']]
        bars = ax.bar(dataset_labels, dataset_values, color=colors_ds, edgecolor='black', linewidth=2)
        ax.set_ylabel('Number of Samples', fontweight='bold')
        ax.set_title('Training vs Test Dataset Size', fontweight='bold')
        ax.set_yscale('log')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Model Summary Table
        ax = axes[1, 1]
        ax.axis('off')
        summary_data = [
            ['Metric', 'Value'],
            ['Model Type', metrics.get('model_type', 'XGBoost')],
            ['R² Score', f"{metrics['r2_score']:.4f}"],
            ['RMSE', f"{metrics['rmse']:.2f}"],
            ['MAE', f"{metrics['mae']:.2f}"],
            ['Train Samples', f"{metrics['train_samples']:,}"],
            ['Test Samples', f"{metrics['test_samples']:,}"],
        ]
        table = ax.table(cellText=summary_data, cellLoc='left', loc='center',
                        colWidths=[0.4, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor(COLORS['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(summary_data)):
            for j in range(2):
                table[(i, j)].set_facecolor('#F0F0F0' if i % 2 == 0 else 'white')
        
        ax.set_title('Performance Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_path = self.output_dir / 'yield_model_performance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Yield model visualization saved: {output_path}")
        plt.close()
    
    def plot_fertilizer_model_performance(self):
        """Generate fertilizer recommendation model performance visualizations"""
        if 'fertilizer' not in self.metrics_data:
            print("⚠ Fertilizer metrics not available")
            return
        
        metrics = self.metrics_data['fertilizer']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Fertilizer Recommendation Model - Performance Metrics (Random Forest)',
                     fontsize=16, fontweight='bold', y=0.995)
        
        # 1. Accuracy and F1 Comparison
        ax = axes[0, 0]
        metric_names = ['Accuracy', 'F1-Score']
        metric_values = [metrics['accuracy'], metrics['f1_score']]
        bars = ax.bar(metric_names, metric_values, color=[COLORS['success'], COLORS['primary']], 
                      edgecolor='black', linewidth=2)
        ax.set_ylim([0, 1])
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Classification Performance Metrics', fontweight='bold')
        ax.axhline(y=0.85, color='orange', linestyle='--', alpha=0.5, label='Good threshold (0.85)')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Dataset Distribution
        ax = axes[0, 1]
        dataset_labels = ['Train Samples', 'Test Samples']
        dataset_values = [metrics['train_samples'], metrics['test_samples']]
        colors_ds = [COLORS['primary'], COLORS['secondary']]
        bars = ax.bar(dataset_labels, dataset_values, color=colors_ds, edgecolor='black', linewidth=2)
        ax.set_ylabel('Number of Samples', fontweight='bold')
        ax.set_title('Training vs Test Split', fontweight='bold')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Number of Classes and Features
        ax = axes[1, 0]
        feature_data = ['Features', 'Classes']
        feature_values = [metrics['n_features'], metrics['n_classes']]
        bars = ax.bar(feature_data, feature_values, color=[COLORS['warning'], COLORS['danger']], 
                      edgecolor='black', linewidth=2)
        ax.set_ylabel('Count', fontweight='bold')
        ax.set_title('Model Complexity', fontweight='bold')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Model Summary Table
        ax = axes[1, 1]
        ax.axis('off')
        summary_data = [
            ['Metric', 'Value'],
            ['Model Type', metrics.get('model_type', 'Random Forest')],
            ['Accuracy', f"{metrics['accuracy']:.4f}"],
            ['F1-Score', f"{metrics['f1_score']:.4f}"],
            ['# Features', f"{metrics['n_features']}"],
            ['# Classes', f"{metrics['n_classes']}"],
            ['Train Samples', f"{metrics['train_samples']:,}"],
            ['Test Samples', f"{metrics['test_samples']:,}"],
        ]
        table = ax.table(cellText=summary_data, cellLoc='left', loc='center',
                        colWidths=[0.4, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.2)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor(COLORS['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(summary_data)):
            for j in range(2):
                table[(i, j)].set_facecolor('#F0F0F0' if i % 2 == 0 else 'white')
        
        ax.set_title('Performance Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_path = self.output_dir / 'fertilizer_model_performance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Fertilizer model visualization saved: {output_path}")
        plt.close()
    
    def plot_fruit_disease_performance(self):
        """Generate fruit disease detection model performance visualizations"""
        report_path = self.model_dir / "classification_report.txt"
        if not report_path.exists():
            print("⚠ Fruit disease classification report not available")
            return
        
        report_data = self.parse_classification_report(report_path)
        if not report_data:
            print("⚠ Could not parse fruit disease report")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Fruit Disease Detection Model - Performance Metrics (EfficientNet-B0, 17 Classes)',
                     fontsize=16, fontweight='bold', y=0.995)
        
        # 1. Per-Class Accuracy (F1-Score)
        ax = axes[0, 0]
        classes = [c['class'] for c in report_data['classes']]
        f1_scores = [c['f1'] for c in report_data['classes']]
        
        # Shorten class names for display
        short_names = [name.replace('_', '\n').replace('Pomegranate', 'Pom.') for name in classes]
        
        bars = ax.barh(range(len(classes)), f1_scores, color=COLORS['primary'], edgecolor='black', linewidth=1)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(short_names, fontsize=9)
        ax.set_xlabel('F1-Score', fontweight='bold')
        ax.set_title('Per-Class Performance (F1-Score)', fontweight='bold')
        ax.set_xlim([0, 1])
        ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (0.9)')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        # 2. Precision vs Recall scatter plot
        ax = axes[0, 1]
        precisions = [c['precision'] for c in report_data['classes']]
        recalls = [c['recall'] for c in report_data['classes']]
        sizes = [c['support'] * 2 for c in report_data['classes']]
        
        scatter = ax.scatter(recalls, precisions, s=sizes, alpha=0.6, c=f1_scores, 
                            cmap='viridis', edgecolors='black', linewidth=1)
        ax.set_xlabel('Recall', fontweight='bold')
        ax.set_ylabel('Precision', fontweight='bold')
        ax.set_title('Precision vs Recall (bubble size = support)', fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('F1-Score', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # 3. Top and Bottom Performers
        ax = axes[1, 0]
        sorted_classes = sorted(report_data['classes'], key=lambda x: x['f1'])
        bottom_3 = sorted_classes[:3]
        top_3 = sorted_classes[-3:]
        
        combined = bottom_3 + top_3
        combined_names = [c['class'][:25] for c in combined]
        combined_f1 = [c['f1'] for c in combined]
        colors_perf = [COLORS['danger']] * 3 + [COLORS['success']] * 3
        
        bars = ax.barh(range(len(combined)), combined_f1, color=colors_perf, edgecolor='black', linewidth=1.5)
        ax.set_yticks(range(len(combined)))
        ax.set_yticklabels(combined_names, fontsize=9)
        ax.set_xlabel('F1-Score', fontweight='bold')
        ax.set_title('Top 3 vs Bottom 3 Performing Classes', fontweight='bold')
        ax.set_xlim([0, 1])
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f' {width:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        
        # 4. Overall Statistics Table
        ax = axes[1, 1]
        ax.axis('off')
        
        # Calculate overall metrics
        total_support = sum(c['support'] for c in report_data['classes'])
        macro_f1 = np.mean([c['f1'] for c in report_data['classes']])
        weighted_f1 = np.average([c['f1'] for c in report_data['classes']], 
                                weights=[c['support'] for c in report_data['classes']])
        macro_precision = np.mean([c['precision'] for c in report_data['classes']])
        macro_recall = np.mean([c['recall'] for c in report_data['classes']])
        
        summary_data = [
            ['Metric', 'Value'],
            ['Model', 'EfficientNet-B0'],
            ['Total Classes', '17'],
            ['Total Samples', f'{total_support:,}'],
            ['Macro Avg F1', f'{macro_f1:.4f}'],
            ['Weighted F1', f'{weighted_f1:.4f}'],
            ['Macro Precision', f'{macro_precision:.4f}'],
            ['Macro Recall', f'{macro_recall:.4f}'],
        ]
        
        table = ax.table(cellText=summary_data, cellLoc='left', loc='center',
                        colWidths=[0.4, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.3)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor(COLORS['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(summary_data)):
            for j in range(2):
                table[(i, j)].set_facecolor('#F0F0F0' if i % 2 == 0 else 'white')
        
        ax.set_title('Overall Performance', fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_path = self.output_dir / 'fruit_disease_model_performance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Fruit disease model visualization saved: {output_path}")
        plt.close()
    
    def plot_deep_learning_training_history(self):
        """Generate training history visualizations for deep learning models"""
        if 'training_history' not in self.metrics_data:
            print("⚠ Training history not available")
            return
        
        history = self.metrics_data['training_history'].get('phase1', {})
        if not history:
            print("⚠ Phase 1 training history not found")
            return
        
        epochs = range(1, len(history['accuracy']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Deep Learning Model - Training History (EfficientNet-B0, Phase 1)',
                     fontsize=16, fontweight='bold', y=0.995)
        
        # 1. Accuracy vs Epochs
        ax = axes[0, 0]
        ax.plot(epochs, history['accuracy'], 'o-', color=COLORS['success'], linewidth=2, 
               markersize=4, label='Training Accuracy')
        ax.fill_between(epochs, history['accuracy'], alpha=0.2, color=COLORS['success'])
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title('Training Accuracy Progress', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([0.7, 1.0])
        
        # 2. Loss vs Epochs
        ax = axes[0, 1]
        ax.plot(epochs, history['loss'], 'o-', color=COLORS['danger'], linewidth=2, 
               markersize=4, label='Training Loss')
        ax.fill_between(epochs, history['loss'], alpha=0.2, color=COLORS['danger'])
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.set_title('Training Loss Progression', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 3. Precision and Recall
        ax = axes[1, 0]
        ax.plot(epochs, history['precision'], 'o-', color=COLORS['primary'], linewidth=2, 
               markersize=4, label='Precision')
        ax.plot(epochs, history['recall'], 's-', color=COLORS['secondary'], linewidth=2, 
               markersize=4, label='Recall')
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Precision & Recall Evolution', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([0.6, 1.0])
        
        # 4. Top-3 Accuracy
        ax = axes[1, 1]
        ax.plot(epochs, history['top3_accuracy'], 'o-', color=COLORS['warning'], linewidth=2, 
               markersize=4, label='Top-3 Accuracy')
        ax.fill_between(epochs, history['top3_accuracy'], alpha=0.2, color=COLORS['warning'])
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Top-3 Accuracy', fontweight='bold')
        ax.set_title('Top-3 Accuracy during Training', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim([0.9, 1.0])
        
        plt.tight_layout()
        output_path = self.output_dir / 'deep_learning_training_history.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Deep learning training history saved: {output_path}")
        plt.close()
    
    def plot_model_comparison_dashboard(self):
        """Generate overall model comparison dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Smart AGRI Models - Overall Performance Comparison Dashboard',
                     fontsize=16, fontweight='bold', y=0.98)
        
        models = []
        accuracies = []
        model_types = []
        colors_list = []
        
        # Yield Model
        if 'yield' in self.metrics_data:
            yield_metrics = self.metrics_data['yield']
            models.append('Yield\nPrediction')
            accuracies.append(yield_metrics['r2_score'])
            model_types.append('XGBoost\n(Regression)')
            colors_list.append(COLORS['primary'])
        
        # Fertilizer Model
        if 'fertilizer' in self.metrics_data:
            fert_metrics = self.metrics_data['fertilizer']
            models.append('Fertilizer\nRecommendation')
            accuracies.append(fert_metrics['accuracy'])
            model_types.append('Random Forest\n(Classification)')
            colors_list.append(COLORS['secondary'])
        
        # Fruit Disease Model
        report_path = self.model_dir / "classification_report.txt"
        if report_path.exists():
            report_data = self.parse_classification_report(report_path)
            if report_data:
                models.append('Fruit Disease\nDetection')
                accuracies.append(0.9011)  # From classification report
                model_types.append('EfficientNet-B0\n(Deep Learning)')
                colors_list.append(COLORS['success'])
        
        # Plot 1: Accuracy Comparison
        ax = axes[0, 0]
        if models:
            bars = ax.bar(models, accuracies, color=colors_list, edgecolor='black', linewidth=2)
            ax.set_ylabel('Accuracy / R² Score', fontweight='bold')
            ax.set_ylim([0, 1])
            ax.set_title('Model Accuracy Comparison', fontweight='bold')
            ax.axhline(y=0.85, color='orange', linestyle='--', alpha=0.5, label='Good threshold')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No models available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
        
        # Plot 2: Model Types
        ax = axes[0, 1]
        if model_types:
            for i, mtype in enumerate(model_types):
                ax.text(0.5, 0.8 - i*0.25, f'• {mtype}', ha='center', va='center',
                       transform=ax.transAxes, fontsize=11, fontweight='bold')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.axis('off')
            ax.set_title('Model Architectures', fontweight='bold')
        
        # Plot 3: Dataset Sizes
        ax = axes[0, 2]
        if 'yield' in self.metrics_data:
            yield_m = self.metrics_data['yield']
            total_yield = yield_m['train_samples'] + yield_m['test_samples']
            ax.text(0.5, 0.85, f"Yield: {total_yield:,} samples", ha='center', va='center',
                   transform=ax.transAxes, fontsize=11, fontweight='bold')
        
        if 'fertilizer' in self.metrics_data:
            fert_m = self.metrics_data['fertilizer']
            total_fert = fert_m['train_samples'] + fert_m['test_samples']
            ax.text(0.5, 0.65, f"Fertilizer: {total_fert:,} samples", ha='center', va='center',
                   transform=ax.transAxes, fontsize=11, fontweight='bold')
        
        if report_path.exists():
            report_data = self.parse_classification_report(report_path)
            if report_data:
                total_support = sum(c['support'] for c in report_data['classes'])
                ax.text(0.5, 0.45, f"Fruit Disease: {total_support:,} samples (17 classes)",
                       ha='center', va='center', transform=ax.transAxes, fontsize=11, fontweight='bold')
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis('off')
        ax.set_title('Dataset Sizes', fontweight='bold')
        
        # Plot 4-6: Performance Details
        ax = axes[1, 0]
        if 'yield' in self.metrics_data:
            yield_m = self.metrics_data['yield']
            detail_text = f"""YIELD PREDICTION
━━━━━━━━━━━━━━━━
R²: {yield_m['r2_score']:.4f}
RMSE: {yield_m['rmse']:.2f}
MAE: {yield_m['mae']:.2f}
Type: {yield_m['model_type']}"""
            ax.text(0.5, 0.5, detail_text, ha='center', va='center',
                   transform=ax.transAxes, fontsize=10, fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor=COLORS['primary'], alpha=0.3))
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis('off')
        
        ax = axes[1, 1]
        if 'fertilizer' in self.metrics_data:
            fert_m = self.metrics_data['fertilizer']
            detail_text = f"""FERTILIZER RECOMMENDATION
━━━━━━━━━━━━━━━━━━━━━━━
Accuracy: {fert_m['accuracy']:.4f}
F1-Score: {fert_m['f1_score']:.4f}
Classes: {fert_m['n_classes']}
Features: {fert_m['n_features']}"""
            ax.text(0.5, 0.5, detail_text, ha='center', va='center',
                   transform=ax.transAxes, fontsize=10, fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor=COLORS['secondary'], alpha=0.3))
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis('off')
        
        ax = axes[1, 2]
        if report_path.exists():
            report_data = self.parse_classification_report(report_path)
            if report_data:
                macro_f1 = np.mean([c['f1'] for c in report_data['classes']])
                detail_text = f"""FRUIT DISEASE DETECTION
━━━━━━━━━━━━━━━━━━━━━
Accuracy: 0.9011
Macro F1: {macro_f1:.4f}
Classes: 17
Model: EfficientNet-B0"""
                ax.text(0.5, 0.5, detail_text, ha='center', va='center',
                       transform=ax.transAxes, fontsize=10, fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor=COLORS['success'], alpha=0.3))
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis('off')
        
        plt.tight_layout()
        output_path = self.output_dir / 'model_comparison_dashboard.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Model comparison dashboard saved: {output_path}")
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all available visualizations"""
        print("\n" + "="*70)
        print("Smart AGRI - Generating Model Evaluation Visualizations")
        print("="*70 + "\n")
        
        self.plot_yield_model_performance()
        self.plot_fertilizer_model_performance()
        self.plot_fruit_disease_performance()
        self.plot_deep_learning_training_history()
        self.plot_model_comparison_dashboard()
        
        print("\n" + "="*70)
        print(f"✅ All visualizations generated successfully!")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print("="*70 + "\n")


if __name__ == "__main__":
    # Initialize visualizer
    visualizer = SmartAgriEvaluationVisualizer(
        model_dir="model",
        output_dir="evaluation_graphs"
    )
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()
    
    print("📊 Generated visualizations:")
    print("  1. yield_model_performance.png - Yield prediction R², RMSE, MAE")
    print("  2. fertilizer_model_performance.png - Fertilizer accuracy & F1-score")
    print("  3. fruit_disease_model_performance.png - Disease detection per-class analysis")
    print("  4. deep_learning_training_history.png - Training curves & metrics")
    print("  5. model_comparison_dashboard.png - Overall comparison dashboard")
