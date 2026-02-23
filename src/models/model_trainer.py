import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import (
    cross_validate, 
    cross_val_predict,
    StratifiedKFold,
    GridSearchCV,
    train_test_split
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from scipy.sparse import csr_matrix
from typing import Dict, Tuple, Optional, Union, List
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import os
import logging
import pickle
import warnings
from itertools import cycle
import sklearn
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


class EmotionClassifierPipeline:
    """
    Complete pipeline for training and comparing multiple emotion classification models.
    Supports: Random Forest, Logistic Regression, SVM, Naive Bayes
    Compatible with scikit-learn >= 1.0
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the pipeline.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.cv_results = {}
        self.test_results = {}
        self.training_times = {}
        self.cv_times = {}
        self.resource_usage = {}
        
        logger.info(f"EmotionClassifierPipeline initialized (scikit-learn {sklearn.__version__})")
    
    def _get_model(self, model_type: str):
        """
        Get a model instance based on type.
        Compatible with scikit-learn >= 1.0
        
        Args:
            model_type: Type of model
            
        Returns:
            Model instance
        """
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=310,
                criterion='gini',
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1
            )
        elif model_type == 'logistic_regression':
            # Fixed for newer scikit-learn versions
            return LogisticRegression(
                penalty='l2',
                C=1.0,
                solver='lbfgs',
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif model_type == 'svm':
            return SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=self.random_state
            )
        elif model_type == 'naive_bayes':
            return MultinomialNB(alpha=1.0, fit_prior=True)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train_all_models(self,
                        X_train: Union[np.ndarray, csr_matrix],
                        y_train: Union[np.ndarray, pd.Series],
                        models: List[str] = ['random_forest', 'logistic_regression', 'svm', 'naive_bayes']):
        """
        Train all specified models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            models: List of model types to train
        """
        logger.info("="*80)
        logger.info("TRAINING ALL MODELS")
        logger.info("="*80)
        
        for model_type in models:
            logger.info(f"\nTraining {model_type.replace('_', ' ').title()}...")
            
            # Initialize model
            model = self._get_model(model_type)
            
            # Monitor resources
            process = psutil.Process(os.getpid())
            cpu_before = process.cpu_percent(interval=0.1)
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Train
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Monitor resources after
            cpu_after = process.cpu_percent(interval=0.1)
            mem_after = process.memory_info().rss / 1024 / 1024
            
            # Store results
            self.models[model_type] = model
            self.training_times[model_type] = training_time
            self.resource_usage[model_type] = {
                'cpu_percent': (cpu_before + cpu_after) / 2,
                'memory_mb': mem_after - mem_before
            }
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"CPU: {self.resource_usage[model_type]['cpu_percent']:.2f}%")
            logger.info(f"Memory: {self.resource_usage[model_type]['memory_mb']:.2f} MB")
    
    def cross_validate_all(self,
                          X: Union[np.ndarray, csr_matrix],
                          y: Union[np.ndarray, pd.Series],
                          cv: int = 10):
        """
        Perform k-fold cross-validation on all models.
        
        Args:
            X: Feature matrix
            y: Labels
            cv: Number of folds
        """
        logger.info("\n" + "="*80)
        logger.info(f"PERFORMING {cv}-FOLD CROSS-VALIDATION ON ALL MODELS")
        logger.info("="*80)
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'f1': 'f1_weighted'
        }
        
        # Setup cross-validation
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        for model_name, model in self.models.items():
            logger.info(f"\nCross-validating {model_name.replace('_', ' ').title()}...")
            
            # Perform cross-validation
            start_time = time.time()
            cv_results = cross_validate(
                estimator=model,
                X=X,
                y=y,
                cv=cv_splitter,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1,
                verbose=0
            )
            cv_time = time.time() - start_time
            
            # Get predictions for confusion matrix
            predictions = cross_val_predict(model, X, y, cv=cv_splitter, n_jobs=-1)
            
            # Store results
            self.cv_results[model_name] = {
                'accuracy': {
                    'mean': cv_results['test_accuracy'].mean(),
                    'std': cv_results['test_accuracy'].std(),
                    'scores': cv_results['test_accuracy']
                },
                'precision': {
                    'mean': cv_results['test_precision'].mean(),
                    'std': cv_results['test_precision'].std(),
                    'scores': cv_results['test_precision']
                },
                'recall': {
                    'mean': cv_results['test_recall'].mean(),
                    'std': cv_results['test_recall'].std(),
                    'scores': cv_results['test_recall']
                },
                'f1': {
                    'mean': cv_results['test_f1'].mean(),
                    'std': cv_results['test_f1'].std(),
                    'scores': cv_results['test_f1']
                },
                'predictions': predictions,
                'cv_time': cv_time
            }
            
            self.cv_times[model_name] = cv_time
            
            logger.info(f"CV completed in {cv_time:.2f} seconds")
            logger.info(f"Accuracy: {self.cv_results[model_name]['accuracy']['mean']:.4f} "
                       f"(±{self.cv_results[model_name]['accuracy']['std']:.4f})")
            logger.info(f"Precision: {self.cv_results[model_name]['precision']['mean']:.4f} "
                       f"(±{self.cv_results[model_name]['precision']['std']:.4f})")
            logger.info(f"Recall: {self.cv_results[model_name]['recall']['mean']:.4f} "
                       f"(±{self.cv_results[model_name]['recall']['std']:.4f})")
            logger.info(f"F1-Score: {self.cv_results[model_name]['f1']['mean']:.4f} "
                       f"(±{self.cv_results[model_name]['f1']['std']:.4f})")
    
    def evaluate_all_models(self,
                           X_test: Union[np.ndarray, csr_matrix],
                           y_test: Union[np.ndarray, pd.Series],
                           label_names: Optional[List[str]] = None):
        """
        Evaluate all models on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            label_names: Names of class labels
        """
        logger.info("\n" + "="*80)
        logger.info("EVALUATING ALL MODELS ON TEST SET")
        logger.info("="*80)
        
        for model_name, model in self.models.items():
            logger.info(f"\nEvaluating {model_name.replace('_', ' ').title()}...")
            
            # Make predictions
            start_time = time.time()
            y_pred = model.predict(X_test)
            prediction_time = time.time() - start_time
            
            # Get probabilities if available
            try:
                y_pred_proba = model.predict_proba(X_test)
            except AttributeError:
                y_pred_proba = None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Calculate AUC
            auc_score = None
            if y_pred_proba is not None:
                try:
                    if len(np.unique(y_test)) == 2:
                        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
                    else:
                        auc_score = roc_auc_score(y_test, y_pred_proba, 
                                                 multi_class='ovr', average='weighted')
                except Exception as e:
                    logger.warning(f"Could not calculate AUC: {e}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Classification report
            report = classification_report(y_test, y_pred, target_names=label_names,
                                         output_dict=True, zero_division=0)
            
            # Store results
            self.test_results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc_score,
                'confusion_matrix': cm,
                'classification_report': report,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'prediction_time': prediction_time
            }
            
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1-Score: {f1:.4f}")
            if auc_score:
                logger.info(f"AUC: {auc_score:.4f}")
    
    def get_comparison_table(self, include_cv: bool = True, include_test: bool = True) -> pd.DataFrame:
        """
        Generate comparison table for all models.
        
        Args:
            include_cv: Include cross-validation results
            include_test: Include test set results
            
        Returns:
            DataFrame with comparison results
        """
        data = []
        
        for model_name in self.models.keys():
            row = {'Model': model_name.replace('_', ' ').title()}
            
            # CV results
            if include_cv and model_name in self.cv_results:
                cv = self.cv_results[model_name]
                row['CV_Accuracy'] = f"{cv['accuracy']['mean']:.4f} ± {cv['accuracy']['std']:.4f}"
                row['CV_Precision'] = f"{cv['precision']['mean']:.4f}"
                row['CV_Recall'] = f"{cv['recall']['mean']:.4f}"
                row['CV_F1'] = f"{cv['f1']['mean']:.4f}"
            
            # Test results
            if include_test and model_name in self.test_results:
                test = self.test_results[model_name]
                row['Test_Accuracy'] = f"{test['accuracy']:.4f}"
                row['Test_Precision'] = f"{test['precision']:.4f}"
                row['Test_Recall'] = f"{test['recall']:.4f}"
                row['Test_F1'] = f"{test['f1_score']:.4f}"
                if test['auc']:
                    row['AUC'] = f"{test['auc']:.4f}"
            
            # Performance metrics
            if model_name in self.training_times:
                row['Train_Time(s)'] = f"{self.training_times[model_name]:.2f}"
            if model_name in self.cv_times:
                row['CV_Time(s)'] = f"{self.cv_times[model_name]:.2f}"
            if model_name in self.resource_usage:
                row['CPU(%)'] = f"{self.resource_usage[model_name]['cpu_percent']:.2f}"
                row['Memory(MB)'] = f"{self.resource_usage[model_name]['memory_mb']:.2f}"
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def plot_cv_comparison(self, figsize: Tuple[int, int] = (16, 10), save_path: Optional[str] = None):
        """
        Plot comprehensive cross-validation comparison.
        
        Args:
            figsize: Figure size
            save_path: Path to save the plot
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)  # Changed from 3x3 to 3x4
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        # Extract data
        model_names = [name.replace('_', ' ').title() for name in self.models.keys()]
        
        # 1. Bar plots for each metric (top row - 4 plots)
        for idx, (metric, color) in enumerate(zip(metrics, colors)):
            ax = fig.add_subplot(gs[0, idx])
            
            means = [self.cv_results[m][metric]['mean'] for m in self.models.keys()]
            stds = [self.cv_results[m][metric]['std'] for m in self.models.keys()]
            
            bars = ax.bar(model_names, means, yerr=stds, capsize=5, 
                         color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            ax.set_title(f'{metric.title()} (10-Fold CV)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=10)
            ax.set_ylim([0, 1.0])
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, mean, std in zip(bars, means, stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.3f}\n±{std:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        # 2. Box plots showing distribution (middle row - spans all columns)
        ax_box = fig.add_subplot(gs[1, :])
        
        box_data = []
        labels = []
        positions = []
        pos = 1
        
        for model_name in self.models.keys():
            for metric in metrics:
                scores = self.cv_results[model_name][metric]['scores']
                box_data.append(scores)
                labels.append(f"{model_name.replace('_', ' ').title()}\n{metric.title()}")
                positions.append(pos)
                pos += 1
            pos += 1  # Gap between models
        
        bp = ax_box.boxplot(box_data, positions=positions, widths=0.6,
                            patch_artist=True, showmeans=True)
        
        # Color the boxes
        for idx, patch in enumerate(bp['boxes']):
            metric_idx = idx % len(metrics)
            patch.set_facecolor(colors[metric_idx])
            patch.set_alpha(0.7)
        
        ax_box.set_xticks(positions)
        ax_box.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax_box.set_ylabel('Score', fontsize=10)
        ax_box.set_title('Distribution of CV Scores Across 10 Folds', 
                         fontsize=12, fontweight='bold')
        ax_box.grid(axis='y', alpha=0.3)
        ax_box.set_ylim([0, 1.05])
        
        # 3. Performance metrics (bottom row - 4 plots)
        # Training time
        ax_time = fig.add_subplot(gs[2, 0])
        train_times = [self.training_times[m] for m in self.models.keys()]
        ax_time.bar(model_names, train_times, color='#9b59b6', alpha=0.7)
        ax_time.set_title('Training Time', fontsize=11, fontweight='bold')
        ax_time.set_ylabel('Seconds', fontsize=10)
        ax_time.tick_params(axis='x', rotation=45)
        ax_time.grid(axis='y', alpha=0.3)
        
        # CV time
        ax_cv_time = fig.add_subplot(gs[2, 1])
        cv_times = [self.cv_times[m] for m in self.models.keys()]
        ax_cv_time.bar(model_names, cv_times, color='#1abc9c', alpha=0.7)
        ax_cv_time.set_title('CV Time (10-Fold)', fontsize=11, fontweight='bold')
        ax_cv_time.set_ylabel('Seconds', fontsize=10)
        ax_cv_time.tick_params(axis='x', rotation=45)
        ax_cv_time.grid(axis='y', alpha=0.3)
        
        # CPU usage
        ax_cpu = fig.add_subplot(gs[2, 2])
        cpu_usage = [self.resource_usage[m]['cpu_percent'] for m in self.models.keys()]
        ax_cpu.bar(model_names, cpu_usage, color='#e67e22', alpha=0.7)
        ax_cpu.set_title('CPU Usage', fontsize=11, fontweight='bold')
        ax_cpu.set_ylabel('Percentage (%)', fontsize=10)
        ax_cpu.tick_params(axis='x', rotation=45)
        ax_cpu.grid(axis='y', alpha=0.3)
        
        # Memory usage
        ax_memory = fig.add_subplot(gs[2, 3])
        memory_usage = [self.resource_usage[m]['memory_mb'] for m in self.models.keys()]
        ax_memory.bar(model_names, memory_usage, color='#34495e', alpha=0.7)
        ax_memory.set_title('Memory Usage', fontsize=11, fontweight='bold')
        ax_memory.set_ylabel('MB', fontsize=10)
        ax_memory.tick_params(axis='x', rotation=45)
        ax_memory.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Comprehensive Model Comparison - 10-Fold Cross-Validation', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"CV comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrices(self, y_true: np.ndarray, label_names: Optional[List[str]] = None,
                               figsize: Tuple[int, int] = (16, 4), save_path: Optional[str] = None):
        """
        Plot confusion matrices for all models.
        
        Args:
            y_true: True labels
            label_names: Names of class labels
            figsize: Figure size
            save_path: Path to save the plot
        """
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            if model_name in self.cv_results:
                predictions = self.cv_results[model_name]['predictions']
                cm = confusion_matrix(y_true, predictions)
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                           xticklabels=label_names, yticklabels=label_names,
                           cbar_kws={'label': 'Count'})
                
                axes[idx].set_title(f'{model_name.replace("_", " ").title()}',
                                   fontsize=12, fontweight='bold')
                axes[idx].set_ylabel('True Label', fontsize=10)
                axes[idx].set_xlabel('Predicted Label', fontsize=10)
        
        plt.suptitle('Confusion Matrices - 10-Fold CV Predictions', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, y_test: np.ndarray, label_names: Optional[List[str]] = None,
                       figsize: Tuple[int, int] = (14, 10), save_path: Optional[str] = None):
        """
        Plot ROC curves for all models (multi-class).
        
        Args:
            y_test: True test labels
            label_names: Names of class labels
            figsize: Figure size
            save_path: Path to save the plot
        """
        classes = np.unique(y_test)
        n_classes = len(classes)
        
        if label_names is None:
            label_names = [f"Class {i}" for i in classes]
        
        # Create subplots for each model
        n_models = len([m for m in self.test_results.keys() 
                       if self.test_results[m]['probabilities'] is not None])
        
        if n_models == 0:
            logger.warning("No models with probability predictions available for ROC curves")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        model_idx = 0
        for model_name in self.models.keys():
            if (model_name not in self.test_results or 
                self.test_results[model_name]['probabilities'] is None):
                continue
            
            ax = axes[model_idx]
            y_pred_proba = self.test_results[model_name]['probabilities']
            
            # Plot ROC curve for each class
            for i, (cls, label) in enumerate(zip(classes, label_names)):
                y_test_binary = (y_test == cls).astype(int)
                y_score = y_pred_proba[:, i]
                
                fpr, tpr, _ = roc_curve(y_test_binary, y_score)
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, linewidth=2, 
                       label=f'{label} (AUC = {roc_auc:.3f})')
            
            # Plot diagonal
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=10)
            ax.set_ylabel('True Positive Rate', fontsize=10)
            ax.set_title(f'{model_name.replace("_", " ").title()}',
                        fontsize=12, fontweight='bold')
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(alpha=0.3)
            
            model_idx += 1
        
        # Hide unused subplots
        for idx in range(model_idx, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('ROC Curves - Multi-Class Classification', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def plot_metric_radar(self, figsize: Tuple[int, int] = (10, 10), save_path: Optional[str] = None):
        """
        Plot radar chart comparing models across metrics.
        
        Args:
            figsize: Figure size
            save_path: Path to save the plot
        """
        # Prepare data
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for idx, (model_name, color) in enumerate(zip(self.models.keys(), colors)):
            if model_name in self.cv_results:
                values = [
                    self.cv_results[model_name]['accuracy']['mean'],
                    self.cv_results[model_name]['precision']['mean'],
                    self.cv_results[model_name]['recall']['mean'],
                    self.cv_results[model_name]['f1']['mean']
                ]
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, 
                       label=model_name.replace('_', ' ').title(), color=color)
                ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.set_title('Model Performance Comparison - Radar Chart',
                    fontsize=14, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Radar chart saved to {save_path}")
        
        plt.show()
    
    def save_all_models(self, directory: str = 'models'):
        """
        Save all trained models.
        
        Args:
            directory: Directory to save models
        """
        os.makedirs(directory, exist_ok=True)
        
        for model_name, model in self.models.items():
            filepath = os.path.join(directory, f'{model_name}_model.pkl')
            
            save_data = {
                'model': model,
                'model_type': model_name,
                'cv_results': self.cv_results.get(model_name),
                'test_results': self.test_results.get(model_name),
                'training_time': self.training_times.get(model_name),
                'cv_time': self.cv_times.get(model_name),
                'resource_usage': self.resource_usage.get(model_name)
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"Model {model_name} saved to {filepath}")
    
    def generate_full_report(self, save_path: str = 'model_comparison_report.txt'):
        """
    Generate a comprehensive text report.
    
    Args:
        save_path: Path to save the report
    """
        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EMOTION CLASSIFICATION - COMPREHENSIVE MODEL COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
        
            # Model list
            f.write("Models Trained:\n")
            for idx, model_name in enumerate(self.models.keys(), 1):
                f.write(f"  {idx}. {model_name.replace('_', ' ').title()}\n")
            f.write("\n")

            # Cross-validation results
            f.write("="*80 + "\n")
            f.write("10-FOLD CROSS-VALIDATION RESULTS\n")
            f.write("="*80 + "\n\n")
            
            for model_name in self.models.keys():
                if model_name in self.cv_results:
                    cv = self.cv_results[model_name]
                    f.write(f"{model_name.replace('_', ' ').title()}:\n")
                    f.write(f"  Accuracy:  {cv['accuracy']['mean']:.4f} ± {cv['accuracy']['std']:.4f}\n")
                    f.write(f"  Precision: {cv['precision']['mean']:.4f} ± {cv['precision']['std']:.4f}\n")
                    f.write(f"  Recall:    {cv['recall']['mean']:.4f} ± {cv['recall']['std']:.4f}\n")
                    f.write(f"  F1-Score:  {cv['f1']['mean']:.4f} ± {cv['f1']['std']:.4f}\n")
                    f.write(f"  CV Time:   {cv['cv_time']:.2f} seconds\n\n")
            
            # Test set results
            if self.test_results:
                f.write("="*80 + "\n")
                f.write("TEST SET EVALUATION RESULTS\n")
                f.write("="*80 + "\n\n")
            
                for model_name in self.models.keys():
                    if model_name in self.test_results:
                        test = self.test_results[model_name]
                        f.write(f"{model_name.replace('_', ' ').title()}:\n")
                        f.write(f"  Accuracy:  {test['accuracy']:.4f}\n")
                        f.write(f"  Precision: {test['precision']:.4f}\n")
                        f.write(f"  Recall:    {test['recall']:.4f}\n")
                        f.write(f"  F1-Score:  {test['f1_score']:.4f}\n")
                        if test['auc']:
                            f.write(f"  AUC:       {test['auc']:.4f}\n")
                        f.write(f"  Prediction Time: {test['prediction_time']:.4f} seconds\n\n")
        
            # Performance comparison
            f.write("="*80 + "\n")
            f.write("PERFORMANCE METRICS COMPARISON\n")
            f.write("="*80 + "\n\n")
            
            comparison_df = self.get_comparison_table()
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")
        
            # Best model
            f.write("="*80 + "\n")
            f.write("BEST MODEL SELECTION\n")
            f.write("="*80 + "\n\n")
        
            if self.cv_results:
                best_acc = max(self.cv_results.items(), 
                            key=lambda x: x[1]['accuracy']['mean'])
                best_f1 = max(self.cv_results.items(), 
                            key=lambda x: x[1]['f1']['mean'])
            
                f.write(f"Best Accuracy: {best_acc[0].replace('_', ' ').title()} "
                        f"({best_acc[1]['accuracy']['mean']:.4f})\n")
                f.write(f"Best F1-Score: {best_f1[0].replace('_', ' ').title()} "
                     f"({best_f1[1]['f1']['mean']:.4f})\n\n")
        
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
    
        # THIS LINE WAS INCORRECTLY INDENTED - NOW IT'S INSIDE THE METHOD
            logger.info(f"Full report saved to {save_path}")
# ==========================================
# 1. GENERATE SAMPLE DATA
# ==========================================
print("\n1. Generating sample dataset...")

np.random.seed(42)
n_samples = 813  # Same as ML-NLPEmot paper
n_features = 100
n_classes = 9  # Plutchik's 8 emotions + neutral

# Generate synthetic features (simulating BoW + TF-IDF)
X = np.random.rand(n_samples, n_features)

# Generate labels with realistic distribution
label_probs = [0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.05, 0.04, 0.03]
y = np.random.choice(n_classes, n_samples, p=label_probs)

# Convert to sparse matrix (like real text features)
X_sparse = csr_matrix(X)

# Label names (Plutchik's emotions)
label_names = ['Positive', 'Trust', 'Fear', 'Negative', 'Anticipation', 
               'Sadness', 'Anger', 'Surprise', 'Neutral']

print(f"Dataset: {n_samples} samples, {n_features} features, {n_classes} classes")
print(f"Class distribution: {np.bincount(y)}")

# ==========================================
# 2. SPLIT DATA
# ==========================================
print("\n2. Splitting data (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_sparse, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# ==========================================
# 3. INITIALIZE PIPELINE
# ==========================================
print("\n3. Initializing pipeline...")

pipeline = EmotionClassifierPipeline(random_state=42)

# ==========================================
# 4. TRAIN ALL MODELS
# ==========================================
print("\n4. Training all models...")

models_to_train = ['random_forest', 'logistic_regression', 'svm', 'naive_bayes']
pipeline.train_all_models(X_train, y_train, models=models_to_train)

# ==========================================
# 5. PERFORM 10-FOLD CROSS-VALIDATION
# ==========================================
print("\n5. Performing 10-fold cross-validation...")

pipeline.cross_validate_all(X_train, y_train, cv=10)

# ==========================================
# 6. EVALUATE ON TEST SET
# ==========================================
print("\n6. Evaluating on test set...")

pipeline.evaluate_all_models(X_test, y_test, label_names=label_names)

# ==========================================
# 7. GENERATE COMPARISON TABLE
# ==========================================
print("\n7. Generating comparison table...")

comparison_df = pipeline.get_comparison_table()
print("\n" + "="*80)
print("MODEL COMPARISON TABLE")
print("="*80)
print(comparison_df.to_string(index=False))

# ==========================================
# 8. VISUALIZATIONS
# ==========================================
print("\n8. Generating visualizations...")

# CV comparison plot
pipeline.plot_cv_comparison(save_path='cv_comparison_all_models.png')

# Confusion matrices
pipeline.plot_confusion_matrices(y_train, label_names=label_names,
                                save_path='confusion_matrices_all_models.png')

# ROC curves
pipeline.plot_roc_curves(y_test, label_names=label_names,
                        save_path='roc_curves_all_models.png')

# Radar chart
pipeline.plot_metric_radar(save_path='radar_chart_comparison.png')

# ==========================================
# 9. SAVE MODELS
# ==========================================
print("\n9. Saving all models...")

pipeline.save_all_models(directory='trained_models')

# ==========================================
# 10. GENERATE FULL REPORT
# ==========================================
print("\n10. Generating comprehensive report...")

pipeline.generate_full_report(save_path='model_comparison_report.txt')

print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  - cv_comparison_all_models.png")
print("  - confusion_matrices_all_models.png")
print("  - roc_curves_all_models.png")
print("  - radar_chart_comparison.png")
print("  - model_comparison_report.txt")
print("  - trained_models/ directory with all models")
print("="*80)

