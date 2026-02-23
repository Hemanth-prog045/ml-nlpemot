import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    cohen_kappa_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
import pickle
from scipy.sparse import csr_matrix
import logging
from tabulate import tabulate
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation framework for academic research.
    Generates publication-ready tables and visualizations.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.models = {}
        self.results = {}
        self.cv_results = {}
        
        logger.info("ModelEvaluator initialized")
    
    def load_model(self, model_path: str, model_name: str):
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
            model_name: Name to assign to the model
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models[model_name] = model_data['model']
        logger.info(f"Model '{model_name}' loaded from {model_path}")
    
    def add_model(self, model, model_name: str):
        """
        Add a trained model directly.
        
        Args:
            model: Trained sklearn model
            model_name: Name to assign to the model
        """
        self.models[model_name] = model
        logger.info(f"Model '{model_name}' added")
    
    def evaluate_single_model(self,
                             model_name: str,
                             X_test: Union[np.ndarray, csr_matrix],
                             y_test: np.ndarray,
                             label_names: Optional[List[str]] = None,
                             average: str = 'weighted') -> Dict:
        """
        Evaluate a single model on test data.
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test labels
            label_names: Names of class labels
            average: Averaging method for multi-class metrics
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        logger.info(f"Evaluating model: {model_name}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        try:
            y_pred_proba = model.predict_proba(X_test)
        except AttributeError:
            y_pred_proba = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average, zero_division=0)
        recall = recall_score(y_test, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(y_test, y_pred)
        
        # AUC if probabilities available
        auc = None
        if y_pred_proba is not None:
            try:
                n_classes = len(np.unique(y_test))
                if n_classes == 2:
                    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    auc = roc_auc_score(y_test, y_pred_proba, 
                                       multi_class='ovr', average=average)
            except Exception as e:
                logger.warning(f"Could not calculate AUC: {e}")
        
        # Per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=label_names,
                                      output_dict=True, zero_division=0)
        
        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'kappa': kappa,
            'confusion_matrix': cm,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        self.results[model_name] = results
        
        logger.info(f"Evaluation complete for {model_name}")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        
        return results
    
    def evaluate_all_models(self,
                           X_test: Union[np.ndarray, csr_matrix],
                           y_test: np.ndarray,
                           label_names: Optional[List[str]] = None,
                           average: str = 'weighted'):
        """
        Evaluate all loaded models.
        
        Args:
            X_test: Test features
            y_test: Test labels
            label_names: Names of class labels
            average: Averaging method for multi-class metrics
        """
        logger.info("="*80)
        logger.info("EVALUATING ALL MODELS")
        logger.info("="*80)
        
        for model_name in self.models.keys():
            self.evaluate_single_model(model_name, X_test, y_test, 
                                      label_names, average)
            print()  # Blank line between models
    
    def cross_validate_model(self,
                            model_name: str,
                            X: Union[np.ndarray, csr_matrix],
                            y: np.ndarray,
                            cv: int = 10,
                            scoring: List[str] = ['accuracy', 'precision_weighted', 
                                                 'recall_weighted', 'f1_weighted']) -> Dict:
        """
        Perform cross-validation on a model.
        
        Args:
            model_name: Name of the model
            X: Feature matrix
            y: Labels
            cv: Number of folds
            scoring: List of scoring metrics
            
        Returns:
            Dictionary with CV results
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        logger.info(f"Performing {cv}-fold cross-validation on {model_name}")
        
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        cv_results = {}
        for metric in scoring:
            scores = cross_val_score(model, X, y, cv=cv_splitter, 
                                    scoring=metric, n_jobs=-1)
            cv_results[metric] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std()
            }
            logger.info(f"  {metric}: {scores.mean():.4f} (±{scores.std():.4f})")
        
        self.cv_results[model_name] = cv_results
        
        return cv_results
    
    def generate_comparison_table(self, 
                                 include_std: bool = False,
                                 table_format: str = 'latex') -> str:
        """
        Generate a comparison table suitable for academic papers.
        
        Args:
            include_std: Include standard deviation from CV
            table_format: Output format ('latex', 'markdown', 'grid', 'simple')
            
        Returns:
            Formatted table string
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run evaluate_all_models first.")
        
        # Prepare data
        table_data = []
        headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Kappa']
        
        for model_name in sorted(self.results.keys()):
            result = self.results[model_name]
            
            row = [
                model_name.replace('_', ' ').title(),
                f"{result['accuracy']:.4f}",
                f"{result['precision']:.4f}",
                f"{result['recall']:.4f}",
                f"{result['f1_score']:.4f}",
                f"{result['auc']:.4f}" if result['auc'] else 'N/A',
                f"{result['kappa']:.4f}"
            ]
            
            # Add CV std if available and requested
            if include_std and model_name in self.cv_results:
                cv = self.cv_results[model_name]
                row[1] += f" ± {cv['accuracy']['std']:.4f}"
                row[2] += f" ± {cv['precision_weighted']['std']:.4f}"
                row[3] += f" ± {cv['recall_weighted']['std']:.4f}"
                row[4] += f" ± {cv['f1_weighted']['std']:.4f}"
            
            table_data.append(row)
        
        # Generate table
        if table_format == 'latex':
            table_str = self._generate_latex_table(headers, table_data)
        else:
            table_str = tabulate(table_data, headers=headers, tablefmt=table_format)
        
        return table_str
    
    def _generate_latex_table(self, headers: List[str], data: List[List[str]]) -> str:
        """Generate LaTeX table format."""
        n_cols = len(headers)
        
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Model Performance Comparison}\n"
        latex += "\\label{tab:model_comparison}\n"
        latex += "\\begin{tabular}{" + "l" + "c" * (n_cols - 1) + "}\n"
        latex += "\\hline\n"
        latex += " & ".join(headers) + " \\\\\n"
        latex += "\\hline\n"
        
        for row in data:
            latex += " & ".join(row) + " \\\\\n"
        
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def generate_per_class_table(self, 
                                 model_name: str,
                                 label_names: List[str],
                                 table_format: str = 'latex') -> str:
        """
        Generate per-class performance table.
        
        Args:
            model_name: Name of the model
            label_names: Names of class labels
            table_format: Output format
            
        Returns:
            Formatted table string
        """
        if model_name not in self.results:
            raise ValueError(f"No results for model '{model_name}'")
        
        result = self.results[model_name]
        
        headers = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
        table_data = []
        
        report = result['classification_report']
        
        for i, label in enumerate(label_names):
            if label in report:
                row = [
                    label,
                    f"{report[label]['precision']:.4f}",
                    f"{report[label]['recall']:.4f}",
                    f"{report[label]['f1-score']:.4f}",
                    f"{int(report[label]['support'])}"
                ]
                table_data.append(row)
        
        # Add macro and weighted averages
        if 'macro avg' in report:
            table_data.append(['---', '---', '---', '---', '---'])
            table_data.append([
                'Macro Avg',
                f"{report['macro avg']['precision']:.4f}",
                f"{report['macro avg']['recall']:.4f}",
                f"{report['macro avg']['f1-score']:.4f}",
                '-'
            ])
            table_data.append([
                'Weighted Avg',
                f"{report['weighted avg']['precision']:.4f}",
                f"{report['weighted avg']['recall']:.4f}",
                f"{report['weighted avg']['f1-score']:.4f}",
                '-'
            ])
        
        if table_format == 'latex':
            return self._generate_latex_per_class_table(model_name, headers, table_data)
        else:
            return tabulate(table_data, headers=headers, tablefmt=table_format)
    
    def _generate_latex_per_class_table(self, model_name: str, 
                                       headers: List[str], 
                                       data: List[List[str]]) -> str:
        """Generate LaTeX table for per-class metrics."""
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{Per-Class Performance: {model_name.replace('_', ' ').title()}}}\n"
        latex += f"\\label{{tab:perclass_{model_name}}}\n"
        latex += "\\begin{tabular}{lcccc}\n"
        latex += "\\hline\n"
        latex += " & ".join(headers) + " \\\\\n"
        latex += "\\hline\n"
        
        for row in data:
            if row[0] == '---':
                latex += "\\hline\n"
            else:
                latex += " & ".join(row) + " \\\\\n"
        
        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def plot_confusion_matrices(self,
                               label_names: Optional[List[str]] = None,
                               figsize: Tuple[int, int] = (16, 4),
                               save_path: Optional[str] = None,
                               normalize: bool = False):
        """
        Plot confusion matrices for all models.
        
        Args:
            label_names: Names of class labels
            figsize: Figure size
            save_path: Path to save the plot
            normalize: Whether to normalize the confusion matrix
        """
        n_models = len(self.results)
        
        if n_models == 0:
            logger.warning("No results to plot")
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                fmt = '.2f'
                title_suffix = ' (Normalized)'
            else:
                fmt = 'd'
                title_suffix = ''
            
            sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=axes[idx],
                       xticklabels=label_names, yticklabels=label_names,
                       cbar_kws={'label': 'Proportion' if normalize else 'Count'})
            
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}{title_suffix}',
                               fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=10)
            axes[idx].set_xlabel('Predicted Label', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self,
                               metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
                               figsize: Tuple[int, int] = (12, 6),
                               save_path: Optional[str] = None):
        """
        Plot bar chart comparing metrics across models.
        
        Args:
            metrics: List of metrics to plot
            figsize: Figure size
            save_path: Path to save the plot
        """
        if not self.results:
            logger.warning("No results to plot")
            return
        
        # Prepare data
        model_names = []
        metric_data = {metric: [] for metric in metrics}
        
        for model_name, result in self.results.items():
            model_names.append(model_name.replace('_', ' ').title())
            for metric in metrics:
                metric_data[metric].append(result[metric])
        
        # Plot
        x = np.arange(len(model_names))
        width = 0.8 / len(metrics)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for idx, metric in enumerate(metrics):
            offset = width * idx - width * len(metrics) / 2 + width / 2
            bars = ax.bar(x + offset, metric_data[metric], width, 
                         label=metric.replace('_', ' ').title(),
                         color=colors[idx % len(colors)], alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=0)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison plot saved to {save_path}")
        
        plt.show()
    
    def save_all_tables(self, output_dir: str = 'evaluation_tables'):
        """
        Save all tables to files.
        
        Args:
            output_dir: Directory to save tables
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save comparison tables in different formats
        for fmt in ['latex', 'markdown', 'grid']:
            table = self.generate_comparison_table(table_format=fmt)
            filename = f'model_comparison.{fmt}'
            
            with open(output_path / filename, 'w') as f:
                f.write(table)
            
            logger.info(f"Saved comparison table: {filename}")
        
        logger.info(f"All tables saved to {output_dir}")
    
    def generate_summary_report(self, save_path: str = 'evaluation_report.txt'):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            save_path: Path to save the report
        """
        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Models Evaluated: {len(self.results)}\n")
            f.write(f"Models: {', '.join([m.replace('_', ' ').title() for m in self.results.keys()])}\n\n")
            
            # Overall performance
            f.write("="*80 + "\n")
            f.write("OVERALL PERFORMANCE METRICS\n")
            f.write("="*80 + "\n\n")
            
            # Create DataFrame for easy display
            df_data = []
            for model_name, result in self.results.items():
                df_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Accuracy': f"{result['accuracy']:.4f}",
                    'Precision': f"{result['precision']:.4f}",
                    'Recall': f"{result['recall']:.4f}",
                    'F1-Score': f"{result['f1_score']:.4f}",
                    'AUC': f"{result['auc']:.4f}" if result['auc'] else 'N/A',
                    'Kappa': f"{result['kappa']:.4f}"
                })
            
            df = pd.DataFrame(df_data)
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            # Best performing model
            f.write("="*80 + "\n")
            f.write("BEST PERFORMING MODELS\n")
            f.write("="*80 + "\n\n")
            
            best_accuracy = max(self.results.items(), key=lambda x: x[1]['accuracy'])
            best_f1 = max(self.results.items(), key=lambda x: x[1]['f1_score'])
            
            f.write(f"Best Accuracy: {best_accuracy[0].replace('_', ' ').title()} "
                   f"({best_accuracy[1]['accuracy']:.4f})\n")
            f.write(f"Best F1-Score: {best_f1[0].replace('_', ' ').title()} "
                   f"({best_f1[1]['f1_score']:.4f})\n\n")
            
            # Confusion matrices
            f.write("="*80 + "\n")
            f.write("CONFUSION MATRICES\n")
            f.write("="*80 + "\n\n")
            
            for model_name, result in self.results.items():
                f.write(f"{model_name.replace('_', ' ').title()}:\n")
                f.write(str(result['confusion_matrix']))
                f.write("\n\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        logger.info(f"Evaluation report saved to {save_path}")


# Example usage / demonstration
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import train_test_split
    from scipy.sparse import csr_matrix
    
    print("="*80)
    print("MODEL EVALUATION FRAMEWORK - DEMONSTRATION")
    print("="*80)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 813
    n_features = 100
    n_classes = 9
    
    X = np.random.rand(n_samples, n_features)
    label_probs = [0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.05, 0.04, 0.03]
    y = np.random.choice(n_classes, n_samples, p=label_probs)
    X_sparse = csr_matrix(X)
    
    # Label names
    label_names = ['Positive', 'Trust', 'Fear', 'Negative', 'Anticipation', 
                   'Sadness', 'Anger', 'Surprise', 'Neutral']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_sparse, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset: {n_samples} samples, {n_features} features, {n_classes} classes")
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Train models
    print("\nTraining models...")
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': MultinomialNB()
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"  ✓ {name} trained")
    
    # Initialize evaluator
    print("\nInitializing evaluator...")
    evaluator = ModelEvaluator()
    
    # Add models
    for name, model in models.items():
        evaluator.add_model(model, name.lower().replace(' ', '_'))
    
    # Evaluate all models
    print("\nEvaluating models on test set...")
    evaluator.evaluate_all_models(X_test, y_test, label_names=label_names)
    
    # Perform cross-validation
    print("\nPerforming 10-fold cross-validation...")
    for name in models.keys():
        model_key = name.lower().replace(' ', '_')
        evaluator.cross_validate_model(model_key, X_train, y_train, cv=10)
    
    # Generate comparison tables
    print("\n" + "="*80)
    print("COMPARISON TABLE (Grid Format)")
    print("="*80)
    print(evaluator.generate_comparison_table(include_std=False, table_format='grid'))
    
    print("\n" + "="*80)
    print("COMPARISON TABLE (LaTeX Format)")
    print("="*80)
    print(evaluator.generate_comparison_table(include_std=False, table_format='latex'))
    
    print("\n" + "="*80)
    print("COMPARISON TABLE WITH CV STD (Markdown Format)")
    print("="*80)
    print(evaluator.generate_comparison_table(include_std=True, table_format='markdown'))
    
    # Generate per-class table for Random Forest
    print("\n" + "="*80)
    print("PER-CLASS METRICS: Random Forest")
    print("="*80)
    print(evaluator.generate_per_class_table('random_forest', label_names, table_format='grid'))
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    evaluator.plot_confusion_matrices(label_names=label_names,
                                     save_path='evaluation_confusion_matrices.png')
    
    evaluator.plot_metrics_comparison(save_path='evaluation_metrics_comparison.png')
    
    # Save all tables
    print("\nSaving tables to files...")
    evaluator.save_all_tables(output_dir='evaluation_tables')
    
    # Generate summary report
    print("\nGenerating summary report...")
    evaluator.generate_summary_report(save_path='evaluation_summary_report.txt')
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - evaluation_confusion_matrices.png")
    print("  - evaluation_metrics_comparison.png")
    print("  - evaluation_summary_report.txt")
    print("  - evaluation_tables/ directory with LaTeX, Markdown, and Grid tables")
    print("="*80)