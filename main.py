# # 
# """
# ML-NLPEmot Main Execution Script
# """

# import numpy as np
# import pandas as pd
# from pathlib import Path
# import sys
# import os

# # Add src to Python path
# sys.path.insert(0, str(Path(__file__).parent / 'src'))

# from preprocessing.text_normalizer import TextPreprocessor
# from feature_extraction.feature_combiner import FeatureExtractor
# from models.model_trainer import EmotionClassifierPipeline
# from sklearn.model_selection import train_test_split

# def main():
#     print("="*80)
#     print("ML-NLPEMOT: EMOTION CLASSIFICATION")
#     print("="*80)
    
#     # 1. LOAD DATA
#     print("\n1. Loading data...")
#     df = pd.read_csv(
#         r"C:\Users\heman\ml-nlpemot\src\data_collection\Combined_Data.csv",
#         encoding_errors='ignore'

#     )
    
#     print(f"   Loaded {len(df)} samples")
#     print(f"   Columns: {df.columns.tolist()}")
    
#     # DATA VALIDATION AND CLEANING
#     print("\n   Data validation...")
#     print(f"   Initial shape: {df.shape}")
#     print(f"   Missing values:\n{df.isnull().sum()}")
    
#     # Remove rows with missing text or labels
#     initial_rows = len(df)
#     df = df.dropna(subset=['statement', 'status'])
#     print(f"   Removed {initial_rows - len(df)} rows with missing values")
    
#     # Remove duplicates
#     initial_rows = len(df)
#     df = df.drop_duplicates(subset=['statement'])
#     print(f"   Removed {initial_rows - len(df)} duplicate rows")
    
#     # Remove empty or very short statements
#     initial_rows = len(df)
#     df = df[df['statement'].str.len() >= 3]
#     print(f"   Removed {initial_rows - len(df)} rows with very short statement")
    
#     print(f"   Final shape: {df.shape}")
#     print(f"   status distribution:\n{df['status'].value_counts()}")
    
#     if len(df) < 50:
#         print("\n   WARNING: Very small dataset. Results may not be reliable.")
    
#     # 2. PREPROCESS
#     print("\n2. Preprocessing text...")
#     preprocessor = TextPreprocessor(language='english')
    
#     try:
#         df['clean_statement'] = preprocessor.preprocess_batch(
#             df['statement'], 
#             return_string=True, 
#             show_progress=True
#         )
#     except Exception as e:
#         print(f"   Error during preprocessing: {e}")
#         print("   Using original text...")
#         df['clean_statement'] = df['statement']
    
#     # Remove empty preprocessed statements
#     initial_rows = len(df)
#     df = df[df['clean_statement'].str.len() >= 1]
#     print(f"   Removed {initial_rows - len(df)} rows with empty preprocessed statement")
    
#     print(f"   Sample preprocessed statements:")
#     for i in range(min(3, len(df))):
#         print(f"   Original: {df['statement'].iloc[i][:50]}...")
#         print(f"   Cleaned:  {df['clean_statement'].iloc[i][:50]}...")
#         print()
    
#     # 3. EXTRACT FEATURES
#     print("\n3. Extracting features...")
    
#     # Check if we have enough data
#     min_samples_per_class = df['status'].value_counts().min()
#     if min_samples_per_class < 5:
#         print(f"   WARNING: Some classes have < 5 samples. Disabling stratification.")
#         stratify = None
#     else:
#         stratify = df['status']
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         df['clean_statement'], 
#         df['status'], 
#         test_size=0.2, 
#         random_state=42, 
#         stratify=stratify
#     )
    
#     print(f"   Train samples: {len(X_train)}")
#     print(f"   Test samples: {len(X_test)}")
    
#     # Extract features
#     try:
#         extractor = FeatureExtractor(
#             max_features=min(1000, len(X_train)),  # Adjust based on data size
#             min_df=1,
#             max_df=0.95
#         )
        
#         print("   Fitting and transforming training data...")
#         X_train_features = extractor.fit_transform_concatenated(X_train)
        
#         print("   Transforming test data...")
#         X_test_features = extractor.transform_concatenated(X_test)
        
#         print(f"   Feature matrix shape: {X_train_features.shape}")
#         print(f"   Feature density: {X_train_features.nnz / (X_train_features.shape[0] * X_train_features.shape[1]):.4f}")
        
#     except Exception as e:
#         print(f"   ERROR during feature extraction: {e}")
#         print("   Please check your preprocessed statement data.")
#         return
    
#     # 4. TRAIN & EVALUATE MODELS
#     print("\n4. Training and evaluating models...")
#     pipeline = EmotionClassifierPipeline(random_state=42)
    
#     models_to_train = ['random_forest', 'logistic_regression', 'svm', 'naive_bayes']
    
#     try:
#         pipeline.train_all_models(X_train_features, y_train, models=models_to_train)
#     except Exception as e:
#         print(f"   ERROR during model training: {e}")
#         return
    
#     # Determine CV folds based on data size
#     n_folds = min(10, min_samples_per_class) if stratify is not None else min(10, len(X_train) // 10)
#     n_folds = max(2, n_folds)  # At least 2 folds
    
#     print(f"\n   Performing {n_folds}-fold cross-validation...")
#     try:
#         pipeline.cross_validate_all(X_train_features, y_train, cv=n_folds)
#     except Exception as e:
#         print(f"   WARNING: Cross-validation failed: {e}")
#         print("   Continuing with test set evaluation...")
    
#     # Get unique statuss
#     label_names = sorted(df['status'].unique().tolist())
#     print(f"   Label names: {label_names}")
    
#     try:
#         pipeline.evaluate_all_models(X_test_features, y_test, label_names=label_names)
#     except Exception as e:
#         print(f"   ERROR during evaluation: {e}")
#         return
    
#     # 5. GENERATE OUTPUTS
#     print("\n5. Generating outputs...")
    
#     # Create results directory
#     results_dir = Path('results')
#     results_dir.mkdir(exist_ok=True)
    
#     try:
#         comparison_df = pipeline.get_comparison_table()
#         print("\n" + "="*80)
#         print("MODEL COMPARISON TABLE")
#         print("="*80)
#         print(comparison_df.to_string(index=False))
        
#         # Save comparison table
#         comparison_df.to_csv(results_dir / 'model_comparison.csv', index=False)
#         print("\n   ✓ Saved: results/model_comparison.csv")
        
#     except Exception as e:
#         print(f"   WARNING: Could not generate comparison table: {e}")
    
#     # Generate visualizations
#     try:
#         print("\n   Generating visualizations...")
        
#         pipeline.plot_cv_comparison(save_path=str(results_dir / 'cv_comparison.png'))
#         print("   ✓ Saved: results/cv_comparison.png")
        
#         pipeline.plot_confusion_matrices(
#             y_train, 
#             label_names=label_names,
#             save_path=str(results_dir / 'confusion_matrices.png')
#         )
#         print("   ✓ Saved: results/confusion_matrices.png")
        
#         pipeline.plot_roc_curves(
#             y_test, 
#             label_names=label_names,
#             save_path=str(results_dir / 'roc_curves.png')
#         )
#         print("   ✓ Saved: results/roc_curves.png")
        
#         pipeline.plot_metric_radar(save_path=str(results_dir / 'radar_chart.png'))
#         print("   ✓ Saved: results/radar_chart.png")
        
#     except Exception as e:
#         print(f"   WARNING: Could not generate all visualizations: {e}")
    
#     # Save models
#     try:
#         models_dir = results_dir / 'models'
#         pipeline.save_all_models(directory=str(models_dir))
#         print(f"   ✓ Saved models to: {models_dir}")
#     except Exception as e:
#         print(f"   WARNING: Could not save models: {e}")
    
#     # Generate report
#     try:
#         pipeline.generate_full_report(save_path=str(results_dir / 'model_report.txt'))
#         print(f"   ✓ Saved: results/model_report.txt")
#     except Exception as e:
#         print(f"   WARNING: Could not generate report: {e}")
    
#     print("\n" + "="*80)
#     print("PIPELINE COMPLETE!")
#     print("="*80)
#     print(f"\nResults saved to: {results_dir.absolute()}")
#     print("\nGenerated files:")
#     print("  - model_comparison.csv")
#     print("  - cv_comparison.png")
#     print("  - confusion_matrices.png")
#     print("  - roc_curves.png")
#     print("  - radar_chart.png")
#     print("  - model_report.txt")
#     print("  - models/ (directory with trained models)")
#     print("="*80)


# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         print("\n" + "="*80)
#         print("FATAL ERROR!")
#         print("="*80)
#         print(f"Error: {e}")
#         import traceback
#         traceback.print_exc()
#         print("="*80)
"""
ML-NLPEmot Main Execution Script
Uses Combined_Data.csv for both training and testing
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessing.text_normalizer import TextPreprocessor
from feature_extraction.feature_combiner import FeatureExtractor
from models.model_trainer import EmotionClassifierPipeline
from sklearn.model_selection import train_test_split
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset path
DATA_PATH = r"C:\Users\heman\ml-nlpemot\src\data_collection\Combined_Data.csv"


def load_and_inspect_data() -> pd.DataFrame:
    """
    Load data and inspect its structure before processing.
    
    Returns:
        Raw DataFrame
    """
    print("\n" + "="*80)
    print("STEP 1: LOADING AND INSPECTING DATA")
    print("="*80)

    df = pd.read_csv(DATA_PATH, encoding_errors='ignore')

    print(f"\n   File: {DATA_PATH}")
    print(f"   Shape: {df.shape}")
    print(f"\n   Columns: {df.columns.tolist()}")
    print(f"\n   Data Types:\n{df.dtypes}")
    print(f"\n   First 5 rows:\n{df.head()}")
    print(f"\n   Missing values:\n{df.isnull().sum()}")
    print(f"\n   Duplicated rows: {df.duplicated().sum()}")

    # Auto-detect text and label columns
    print("\n   Inspecting each column...")
    for col in df.columns:
        unique_count = df[col].nunique()
        sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else "EMPTY"
        avg_len = df[col].astype(str).str.len().mean()
        print(f"     {col}: unique={unique_count}, avg_len={avg_len:.1f}, sample='{str(sample)[:60]}'")

    return df


def detect_columns(df: pd.DataFrame) -> tuple:
    """
    Auto-detect text and label columns based on data characteristics.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (text_column, label_column)
    """
    print("\n   Auto-detecting text and label columns...")

    text_col = None
    label_col = None

    for col in df.columns:
        avg_len = df[col].astype(str).str.len().mean()
        unique_ratio = df[col].nunique() / len(df)

        # Text column: long average length, high unique ratio
        if avg_len > 20 and unique_ratio > 0.05:
            if text_col is None or avg_len > df[text_col].astype(str).str.len().mean():
                text_col = col

        # Label column: short average length, low unique ratio
        if avg_len < 30 and unique_ratio < 0.05:
            if label_col is None or unique_ratio < df[label_col].nunique() / len(df):
                label_col = col

    # Fallback: known common column names
    if text_col is None:
        for name in ['statement', 'text', 'tweet', 'message', 'content', 'post']:
            if name in df.columns:
                text_col = name
                break

    if label_col is None:
        for name in ['status', 'label', 'emotion', 'sentiment', 'category']:
            if name in df.columns:
                label_col = name
                break

    print(f"   Detected text column: '{text_col}'")
    print(f"   Detected label column: '{label_col}'")

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not auto-detect columns.\n"
            f"   Text column: {text_col}\n"
            f"   Label column: {label_col}\n"
            f"   Available columns: {df.columns.tolist()}\n"
            f"   Please set TEXT_COL and LABEL_COL manually."
        )

    return text_col, label_col


def clean_data(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    """
    Clean dataset thoroughly.
    
    Args:
        df: Input DataFrame
        text_col: Name of text column
        label_col: Name of label column
        
    Returns:
        Cleaned DataFrame
    """
    print("\n" + "="*80)
    print("STEP 2: DATA CLEANING")
    print("="*80)

    initial_rows = len(df)
    print(f"\n   Initial rows: {initial_rows}")

    # Remove rows with missing text or labels
    df = df.dropna(subset=[text_col, label_col])
    print(f"   After removing NaN: {len(df)} rows (removed {initial_rows - len(df)})")

    # Convert text to string type
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str)

    # Strip whitespace from labels
    df[label_col] = df[label_col].str.strip()

    # Remove empty text
    rows_before = len(df)
    df = df[df[text_col].str.strip().str.len() >= 3]
    print(f"   After removing short text: {len(df)} rows (removed {rows_before - len(df)})")

    # Remove duplicates
    rows_before = len(df)
    df = df.drop_duplicates(subset=[text_col])
    print(f"   After removing duplicates: {len(df)} rows (removed {rows_before - len(df)})")

    # Remove labels that appear less than 5 times
    rows_before = len(df)
    label_counts = df[label_col].value_counts()
    rare_labels = label_counts[label_counts < 5].index.tolist()
    if rare_labels:
        print(f"   Removing rare labels (< 5 samples): {rare_labels}")
        df = df[~df[label_col].isin(rare_labels)]
    print(f"   After removing rare labels: {len(df)} rows (removed {rows_before - len(df)})")

    # Reset index
    df = df.reset_index(drop=True)

    print(f"\n   Final rows: {len(df)}")
    print(f"\n   Label distribution:\n{df[label_col].value_counts()}")

    return df


def preprocess_text(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Preprocess text using TextPreprocessor.
    
    Args:
        df: Input DataFrame
        text_col: Name of text column
        
    Returns:
        DataFrame with added clean_text column
    """
    print("\n" + "="*80)
    print("STEP 3: TEXT PREPROCESSING")
    print("="*80)

    preprocessor = TextPreprocessor(language='english')

    try:
        df['clean_text'] = preprocessor.preprocess_batch(
            df[text_col],
            return_string=True,
            show_progress=True
        )
    except Exception as e:
        print(f"   ERROR during preprocessing: {e}")
        print("   Falling back to basic cleaning...")
        df['clean_text'] = df[text_col].str.lower().str.strip()

    # Remove rows where preprocessing resulted in empty text
    rows_before = len(df)
    df = df[df['clean_text'].str.strip().str.len() >= 1]
    print(f"   Removed {rows_before - len(df)} rows with empty preprocessed text")

    # Show samples
    print(f"\n   Sample preprocessed texts:")
    for i in range(min(5, len(df))):
        print(f"\n     Original:    {df[text_col].iloc[i][:70]}...")
        print(f"     Preprocessed: {df['clean_text'].iloc[i][:70]}...")

    return df


def extract_features(df: pd.DataFrame, label_col: str) -> tuple:
    """
    Extract features and split into train/test.
    
    Args:
        df: Preprocessed DataFrame
        label_col: Name of label column
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, label_names)
    """
    print("\n" + "="*80)
    print("STEP 4: FEATURE EXTRACTION & TRAIN-TEST SPLIT")
    print("="*80)

    # Get label info
    label_names = sorted(df[label_col].unique().tolist())
    min_samples = df[label_col].value_counts().min()
    print(f"\n   Labels: {label_names}")
    print(f"   Min samples per class: {min_samples}")

    # Determine stratification
    stratify = df[label_col] if min_samples >= 2 else None
    if stratify is None:
        print("   WARNING: Disabled stratification due to small class sizes")

    # Split BEFORE feature extraction to avoid data leakage
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['clean_text'],
        df[label_col],
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    print(f"\n   Train samples: {len(X_train_text)}")
    print(f"   Test samples:  {len(X_test_text)}")
    print(f"\n   Train label distribution:\n{y_train.value_counts()}")
    print(f"\n   Test label distribution:\n{y_test.value_counts()}")

    # Feature extraction
    max_feat = min(5000, len(X_train_text))
    print(f"\n   Extracting features (max_features={max_feat})...")

    extractor = FeatureExtractor(
        max_features=max_feat,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2)  # Unigrams + Bigrams
    )

    # Fit on training data only, transform both
    X_train_features = extractor.fit_transform_concatenated(X_train_text)
    X_test_features = extractor.transform_concatenated(X_test_text)

    print(f"\n   Train feature matrix: {X_train_features.shape}")
    print(f"   Test feature matrix:  {X_test_features.shape}")
    print(f"   Feature density: {X_train_features.nnz / (X_train_features.shape[0] * X_train_features.shape[1]):.4f}")

    return X_train_features, X_test_features, y_train, y_test, label_names


def train_and_evaluate(X_train, X_test, y_train, y_test, label_names) -> EmotionClassifierPipeline:
    """
    Train and evaluate all models.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        label_names: List of label names
        
    Returns:
        Trained pipeline
    """
    print("\n" + "="*80)
    print("STEP 5: MODEL TRAINING & EVALUATION")
    print("="*80)

    pipeline = EmotionClassifierPipeline(random_state=42)

    # Train all models
    models_to_train = ['random_forest', 'logistic_regression', 'svm', 'naive_bayes']
    pipeline.train_all_models(X_train, y_train, models=models_to_train)

    # Determine CV folds
    min_class_count = pd.Series(y_train).value_counts().min()
    n_folds = min(10, min_class_count)
    n_folds = max(2, n_folds)
    print(f"\n   Using {n_folds}-fold cross-validation")

    # Cross-validation
    try:
        pipeline.cross_validate_all(X_train, y_train, cv=n_folds)
    except Exception as e:
        print(f"   WARNING: CV failed: {e}")

    # Test evaluation
    try:
        pipeline.evaluate_all_models(X_test, y_test, label_names=label_names)
    except Exception as e:
        print(f"   ERROR during evaluation: {e}")

    return pipeline


def generate_outputs(pipeline: EmotionClassifierPipeline,
                    y_train, y_test, label_names):
    """
    Generate all output files.
    
    Args:
        pipeline: Trained pipeline
        y_train: Training labels
        y_test: Test labels
        label_names: List of label names
    """
    print("\n" + "="*80)
    print("STEP 6: GENERATING OUTPUTS")
    print("="*80)

    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Comparison table
    try:
        comparison_df = pipeline.get_comparison_table()
        print("\n" + "="*80)
        print("MODEL COMPARISON TABLE")
        print("="*80)
        print(comparison_df.to_string(index=False))

        comparison_df.to_csv(results_dir / 'model_comparison.csv', index=False)
        print("\n   ✓ Saved: results/model_comparison.csv")
    except Exception as e:
        print(f"   WARNING: Comparison table failed: {e}")

    # Visualizations
    plots = {
        'cv_comparison.png': lambda: pipeline.plot_cv_comparison(
            save_path=str(results_dir / 'cv_comparison.png')
        ),
        'confusion_matrices.png': lambda: pipeline.plot_confusion_matrices(
            y_train,
            label_names=label_names,
            save_path=str(results_dir / 'confusion_matrices.png')
        ),
        'roc_curves.png': lambda: pipeline.plot_roc_curves(
            y_test,
            label_names=label_names,
            save_path=str(results_dir / 'roc_curves.png')
        ),
        'radar_chart.png': lambda: pipeline.plot_metric_radar(
            save_path=str(results_dir / 'radar_chart.png')
        )
    }

    print("\n   Generating visualizations...")
    for name, plot_fn in plots.items():
        try:
            plot_fn()
            print(f"   ✓ Saved: results/{name}")
        except Exception as e:
            print(f"   ✗ Failed {name}: {e}")

    # Save models
    try:
        pipeline.save_all_models(directory=str(results_dir / 'models'))
        print("   ✓ Saved: results/models/")
    except Exception as e:
        print(f"   ✗ Failed saving models: {e}")

    # Generate report
    try:
        pipeline.generate_full_report(save_path=str(results_dir / 'model_report.txt'))
        print("   ✓ Saved: results/model_report.txt")
    except Exception as e:
        print(f"   ✗ Failed report: {e}")


def main():
    print("="*80)
    print("  ML-NLPEmot: EMOTION DETECTION PIPELINE")
    print(f"  Dataset: Combined_Data.csv")
    print("="*80)

    # Step 1: Load and inspect
    df = load_and_inspect_data()

    # Step 2: Auto-detect columns
    text_col, label_col = detect_columns(df)

    # Step 3: Clean data
    df = clean_data(df, text_col, label_col)

    # Step 4: Preprocess
    df = preprocess_text(df, text_col)

    # Step 5: Feature extraction + split
    X_train, X_test, y_train, y_test, label_names = extract_features(df, label_col)

    # Step 6: Train and evaluate
    pipeline = train_and_evaluate(X_train, X_test, y_train, y_test, label_names)

    # Step 7: Generate outputs
    generate_outputs(pipeline, y_train, y_test, label_names)

    # Final summary
    print("\n" + "="*80)
    print("  PIPELINE COMPLETE!")
    print("="*80)
    print(f"\n  Dataset used: {DATA_PATH}")
    print(f"  Results saved to: {Path('results').absolute()}")
    print("\n  Output files:")
    print("    - model_comparison.csv")
    print("    - cv_comparison.png")
    print("    - confusion_matrices.png")
    print("    - roc_curves.png")
    print("    - radar_chart.png")
    print("    - model_report.txt")
    print("    - models/ (trained models)")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "="*80)
        print("  FATAL ERROR")
        print("="*80)
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()
        print("="*80)