import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from typing import Tuple, Optional, Union, List, Dict
import logging
import pickle
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Feature extraction for text data using:
    1. Bag of Words (BoW)
    2. Term Frequency-Inverse Document Frequency (TF-IDF)
    3. Concatenated BoW + TF-IDF
    
    Based on ML-NLPEmot paper methodology.
    """
    
    def __init__(self,
                 max_features: Optional[int] = None,
                 min_df: Union[int, float] = 1,
                 max_df: Union[int, float] = 1.0,
                 ngram_range: Tuple[int, int] = (1, 1),
                 binary: bool = False,
                 use_idf: bool = True,
                 smooth_idf: bool = True,
                 sublinear_tf: bool = False):
        """
        Initialize the feature extractor.
        
        Args:
            max_features: Maximum number of features to extract
            min_df: Minimum document frequency for a term
            max_df: Maximum document frequency for a term
            ngram_range: Range of n-grams to extract (e.g., (1,1) for unigrams, (1,2) for unigrams+bigrams)
            binary: If True, all non-zero counts are set to 1 for BoW
            use_idf: Enable inverse-document-frequency reweighting
            smooth_idf: Add one to document frequencies (prevents zero divisions)
            sublinear_tf: Apply sublinear tf scaling (replace tf with 1 + log(tf))
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.binary = binary
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        
        # Initialize vectorizers
        self.bow_vectorizer = None
        self.tfidf_vectorizer = None
        self.is_fitted = False
        
        logger.info("FeatureExtractor initialized")
    
    def _initialize_bow_vectorizer(self) -> CountVectorizer:
        """
        Initialize Bag of Words vectorizer.
        
        Returns:
            Configured CountVectorizer
        """
        return CountVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            binary=self.binary,
            token_pattern=r'\b\w+\b'  # Match word boundaries
        )
    
    def _initialize_tfidf_vectorizer(self) -> TfidfVectorizer:
        """
        Initialize TF-IDF vectorizer.
        
        Returns:
            Configured TfidfVectorizer
        """
        return TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf,
            token_pattern=r'\b\w+\b'
        )
    
    def fit_bow(self, texts: Union[List[str], pd.Series]) -> 'FeatureExtractor':
        """
        Fit Bag of Words vectorizer on training texts.
        
        Args:
            texts: List or Series of preprocessed texts
            
        Returns:
            Self (for method chaining)
        """
        logger.info("Fitting Bag of Words vectorizer...")
        
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        self.bow_vectorizer = self._initialize_bow_vectorizer()
        self.bow_vectorizer.fit(texts)
        
        vocab_size = len(self.bow_vectorizer.vocabulary_)
        logger.info(f"BoW vocabulary size: {vocab_size}")
        
        return self
    
    def fit_tfidf(self, texts: Union[List[str], pd.Series]) -> 'FeatureExtractor':
        """
        Fit TF-IDF vectorizer on training texts.
        
        Args:
            texts: List or Series of preprocessed texts
            
        Returns:
            Self (for method chaining)
        """
        logger.info("Fitting TF-IDF vectorizer...")
        
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        self.tfidf_vectorizer = self._initialize_tfidf_vectorizer()
        self.tfidf_vectorizer.fit(texts)
        
        vocab_size = len(self.tfidf_vectorizer.vocabulary_)
        logger.info(f"TF-IDF vocabulary size: {vocab_size}")
        
        return self
    
    def fit(self, texts: Union[List[str], pd.Series]) -> 'FeatureExtractor':
        """
        Fit both BoW and TF-IDF vectorizers on training texts.
        
        Args:
            texts: List or Series of preprocessed texts
            
        Returns:
            Self (for method chaining)
        """
        logger.info("Fitting all vectorizers...")
        
        self.fit_bow(texts)
        self.fit_tfidf(texts)
        self.is_fitted = True
        
        logger.info("All vectorizers fitted successfully")
        return self
    
    def transform_bow(self, texts: Union[List[str], pd.Series]) -> csr_matrix:
        """
        Transform texts using Bag of Words.
        
        Args:
            texts: List or Series of preprocessed texts
            
        Returns:
            Sparse matrix of BoW features
        """
        if self.bow_vectorizer is None:
            raise ValueError("BoW vectorizer not fitted. Call fit_bow() or fit() first.")
        
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        features = self.bow_vectorizer.transform(texts)
        logger.info(f"BoW features shape: {features.shape}")
        
        return features
    
    def transform_tfidf(self, texts: Union[List[str], pd.Series]) -> csr_matrix:
        """
        Transform texts using TF-IDF.
        
        Args:
            texts: List or Series of preprocessed texts
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf() or fit() first.")
        
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        features = self.tfidf_vectorizer.transform(texts)
        logger.info(f"TF-IDF features shape: {features.shape}")
        
        return features
    
    def transform_concatenated(self, texts: Union[List[str], pd.Series]) -> csr_matrix:
        """
        Transform texts using concatenated BoW + TF-IDF features.
        This is the method used in the ML-NLPEmot paper.
        
        Args:
            texts: List or Series of preprocessed texts
            
        Returns:
            Sparse matrix of concatenated features
        """
        if not self.is_fitted:
            raise ValueError("Vectorizers not fitted. Call fit() first.")
        
        logger.info("Extracting concatenated BoW + TF-IDF features...")
        
        # Get BoW features
        bow_features = self.transform_bow(texts)
        
        # Get TF-IDF features
        tfidf_features = self.transform_tfidf(texts)
        
        # Concatenate horizontally (column-wise)
        concatenated_features = hstack([bow_features, tfidf_features])
        
        logger.info(f"Concatenated features shape: {concatenated_features.shape}")
        logger.info(f"  BoW features: {bow_features.shape[1]}")
        logger.info(f"  TF-IDF features: {tfidf_features.shape[1]}")
        logger.info(f"  Total features: {concatenated_features.shape[1]}")
        
        return concatenated_features
    
    def fit_transform_bow(self, texts: Union[List[str], pd.Series]) -> csr_matrix:
        """
        Fit and transform texts using BoW in one step.
        
        Args:
            texts: List or Series of preprocessed texts
            
        Returns:
            Sparse matrix of BoW features
        """
        self.fit_bow(texts)
        return self.transform_bow(texts)
    
    def fit_transform_tfidf(self, texts: Union[List[str], pd.Series]) -> csr_matrix:
        """
        Fit and transform texts using TF-IDF in one step.
        
        Args:
            texts: List or Series of preprocessed texts
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        self.fit_tfidf(texts)
        return self.transform_tfidf(texts)
    
    def fit_transform_concatenated(self, texts: Union[List[str], pd.Series]) -> csr_matrix:
        """
        Fit and transform texts using concatenated features in one step.
        
        Args:
            texts: List or Series of preprocessed texts
            
        Returns:
            Sparse matrix of concatenated features
        """
        self.fit(texts)
        return self.transform_concatenated(texts)
    
    def get_feature_names_bow(self) -> List[str]:
        """
        Get feature names from BoW vectorizer.
        
        Returns:
            List of feature names
        """
        if self.bow_vectorizer is None:
            raise ValueError("BoW vectorizer not fitted.")
        
        return self.bow_vectorizer.get_feature_names_out().tolist()
    
    def get_feature_names_tfidf(self) -> List[str]:
        """
        Get feature names from TF-IDF vectorizer.
        
        Returns:
            List of feature names
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted.")
        
        return self.tfidf_vectorizer.get_feature_names_out().tolist()
    
    def get_feature_names_concatenated(self) -> List[str]:
        """
        Get feature names from concatenated features.
        BoW features are prefixed with 'bow_' and TF-IDF with 'tfidf_'.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Vectorizers not fitted.")
        
        bow_features = [f"bow_{name}" for name in self.get_feature_names_bow()]
        tfidf_features = [f"tfidf_{name}" for name in self.get_feature_names_tfidf()]
        
        return bow_features + tfidf_features
    
    def get_top_features_bow(self, X: csr_matrix, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N features by average BoW score.
        
        Args:
            X: BoW feature matrix
            top_n: Number of top features to return
            
        Returns:
            List of (feature_name, avg_score) tuples
        """
        feature_names = self.get_feature_names_bow()
        avg_scores = np.asarray(X.mean(axis=0)).flatten()
        top_indices = avg_scores.argsort()[-top_n:][::-1]
        
        return [(feature_names[i], avg_scores[i]) for i in top_indices]
    
    def get_top_features_tfidf(self, X: csr_matrix, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N features by average TF-IDF score.
        
        Args:
            X: TF-IDF feature matrix
            top_n: Number of top features to return
            
        Returns:
            List of (feature_name, avg_score) tuples
        """
        feature_names = self.get_feature_names_tfidf()
        avg_scores = np.asarray(X.mean(axis=0)).flatten()
        top_indices = avg_scores.argsort()[-top_n:][::-1]
        
        return [(feature_names[i], avg_scores[i]) for i in top_indices]
    
    def save_vectorizers(self, filepath: str):
        """
        Save fitted vectorizers to disk.
        
        Args:
            filepath: Path to save the vectorizers
        """
        if not self.is_fitted:
            raise ValueError("Vectorizers not fitted. Nothing to save.")
        
        save_data = {
            'bow_vectorizer': self.bow_vectorizer,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'config': {
                'max_features': self.max_features,
                'min_df': self.min_df,
                'max_df': self.max_df,
                'ngram_range': self.ngram_range,
                'binary': self.binary,
                'use_idf': self.use_idf,
                'smooth_idf': self.smooth_idf,
                'sublinear_tf': self.sublinear_tf
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Vectorizers saved to {filepath}")
    
    def load_vectorizers(self, filepath: str):
        """
        Load fitted vectorizers from disk.
        
        Args:
            filepath: Path to load the vectorizers from
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.bow_vectorizer = save_data['bow_vectorizer']
        self.tfidf_vectorizer = save_data['tfidf_vectorizer']
        
        # Restore config
        config = save_data['config']
        self.max_features = config['max_features']
        self.min_df = config['min_df']
        self.max_df = config['max_df']
        self.ngram_range = config['ngram_range']
        self.binary = config['binary']
        self.use_idf = config['use_idf']
        self.smooth_idf = config['smooth_idf']
        self.sublinear_tf = config['sublinear_tf']
        
        self.is_fitted = True
        logger.info(f"Vectorizers loaded from {filepath}")
    
    def get_statistics(self, X_bow: csr_matrix = None, 
                      X_tfidf: csr_matrix = None,
                      X_concat: csr_matrix = None) -> Dict:
        """
        Get statistics about the feature matrices.
        
        Args:
            X_bow: BoW feature matrix
            X_tfidf: TF-IDF feature matrix
            X_concat: Concatenated feature matrix
            
        Returns:
            Dictionary containing statistics
        """
        stats = {}
        
        if X_bow is not None:
            stats['bow'] = {
                'shape': X_bow.shape,
                'n_samples': X_bow.shape[0],
                'n_features': X_bow.shape[1],
                'density': X_bow.nnz / (X_bow.shape[0] * X_bow.shape[1]),
                'avg_features_per_sample': X_bow.nnz / X_bow.shape[0]
            }
        
        if X_tfidf is not None:
            stats['tfidf'] = {
                'shape': X_tfidf.shape,
                'n_samples': X_tfidf.shape[0],
                'n_features': X_tfidf.shape[1],
                'density': X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]),
                'avg_features_per_sample': X_tfidf.nnz / X_tfidf.shape[0]
            }
        
        if X_concat is not None:
            stats['concatenated'] = {
                'shape': X_concat.shape,
                'n_samples': X_concat.shape[0],
                'n_features': X_concat.shape[1],
                'density': X_concat.nnz / (X_concat.shape[0] * X_concat.shape[1]),
                'avg_features_per_sample': X_concat.nnz / X_concat.shape[0]
            }
        
        return stats


def extract_features(texts_train: Union[List[str], pd.Series],
                    texts_test: Union[List[str], pd.Series],
                    method: str = 'concatenated',
                    max_features: Optional[int] = None,
                    ngram_range: Tuple[int, int] = (1, 1)) -> Tuple[csr_matrix, csr_matrix]:
    """
    Convenience function to extract features from train and test texts.
    
    Args:
        texts_train: Training texts (preprocessed)
        texts_test: Test texts (preprocessed)
        method: Feature extraction method ('bow', 'tfidf', or 'concatenated')
        max_features: Maximum number of features
        ngram_range: N-gram range
        
    Returns:
        Tuple of (train_features, test_features)
    """
    extractor = FeatureExtractor(
        max_features=max_features,
        ngram_range=ngram_range
    )
    
    if method == 'bow':
        X_train = extractor.fit_transform_bow(texts_train)
        X_test = extractor.transform_bow(texts_test)
    elif method == 'tfidf':
        X_train = extractor.fit_transform_tfidf(texts_train)
        X_test = extractor.transform_tfidf(texts_test)
    elif method == 'concatenated':
        X_train = extractor.fit_transform_concatenated(texts_train)
        X_test = extractor.transform_concatenated(texts_test)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bow', 'tfidf', or 'concatenated'")
    
    return X_train, X_test


# Demonstration and testing
if __name__ == "__main__":
    # Sample preprocessed texts (after applying TextPreprocessor)
    train_texts = [
        "getting vaccinated best way protect others",
        "free vaccination residents great news",
        "pfizer vaccine dubai feeling relieved hopeful",
        "china sinopharm vaccine available abu dhabi",
        "worried vaccine side effects anyone experienced issues",
        "uae amazing job vaccination campaign",
        "second dose vaccination centers open forget complete schedule",
        "registration approved pfizer vaccine elderly",
        "clinical trial russian vaccine sputnik",
        "health ministry dedicated vaccination centers elderly chronic disease"
    ]
    
    test_texts = [
        "received first dose feeling good",
        "booster shot available booking appointment",
        "vaccine effectiveness high protection rate"
    ]
    
    print("="*70)
    print("FEATURE EXTRACTION DEMONSTRATION")
    print("="*70)
    
    # Initialize feature extractor
    extractor = FeatureExtractor(
        max_features=50,  # Limit for demonstration
        min_df=1,
        max_df=0.8,
        ngram_range=(1, 1),  # Unigrams only
        binary=False
    )
    
    # ==========================================
    # 1. BAG OF WORDS (BoW)
    # ==========================================
    print("\n" + "="*70)
    print("1. BAG OF WORDS (BoW)")
    print("="*70)
    
    # Fit and transform
    X_train_bow = extractor.fit_transform_bow(train_texts)
    X_test_bow = extractor.transform_bow(test_texts)
    
    print(f"\nTrain BoW shape: {X_train_bow.shape}")
    print(f"Test BoW shape: {X_test_bow.shape}")
    print(f"Vocabulary size: {len(extractor.get_feature_names_bow())}")
    
    # Show some feature names
    feature_names_bow = extractor.get_feature_names_bow()
    print(f"\nFirst 10 BoW features: {feature_names_bow[:10]}")
    
    # Show dense representation of first sample
    print(f"\nFirst training sample (dense):")
    print(f"Text: {train_texts[0]}")
    first_sample_bow = X_train_bow[0].toarray().flatten()
    non_zero_indices = np.where(first_sample_bow > 0)[0]
    for idx in non_zero_indices[:10]:  # Show first 10 non-zero features
        print(f"  {feature_names_bow[idx]}: {first_sample_bow[idx]}")
    
    # Top features by average count
    top_bow = extractor.get_top_features_bow(X_train_bow, top_n=10)
    print(f"\nTop 10 BoW features by average count:")
    for feature, score in top_bow:
        print(f"  {feature}: {score:.4f}")
    
    # ==========================================
    # 2. TF-IDF
    # ==========================================
    print("\n" + "="*70)
    print("2. TF-IDF")
    print("="*70)
    
    # Fit and transform
    X_train_tfidf = extractor.fit_transform_tfidf(train_texts)
    X_test_tfidf = extractor.transform_tfidf(test_texts)
    
    print(f"\nTrain TF-IDF shape: {X_train_tfidf.shape}")
    print(f"Test TF-IDF shape: {X_test_tfidf.shape}")
    print(f"Vocabulary size: {len(extractor.get_feature_names_tfidf())}")
    
    # Show some feature names
    feature_names_tfidf = extractor.get_feature_names_tfidf()
    print(f"\nFirst 10 TF-IDF features: {feature_names_tfidf[:10]}")
    
    # Show dense representation of first sample
    print(f"\nFirst training sample (dense):")
    print(f"Text: {train_texts[0]}")
    first_sample_tfidf = X_train_tfidf[0].toarray().flatten()
    non_zero_indices = np.where(first_sample_tfidf > 0)[0]
    for idx in non_zero_indices[:10]:
        print(f"  {feature_names_tfidf[idx]}: {first_sample_tfidf[idx]:.4f}")
    
    # Top features by average TF-IDF score
    top_tfidf = extractor.get_top_features_tfidf(X_train_tfidf, top_n=10)
    print(f"\nTop 10 TF-IDF features by average score:")
    for feature, score in top_tfidf:
        print(f"  {feature}: {score:.4f}")
    
    # ==========================================
    # 3. CONCATENATED BoW + TF-IDF (Paper Method)
    # ==========================================
    print("\n" + "="*70)
    print("3. CONCATENATED BoW + TF-IDF (ML-NLPEmot Paper Method)")
    print("="*70)
    
    # Re-initialize to fit both
    extractor2 = FeatureExtractor(
        max_features=50,
        min_df=1,
        max_df=0.8,
        ngram_range=(1, 1)
    )
    
    # Fit and transform with concatenated features
    X_train_concat = extractor2.fit_transform_concatenated(train_texts)
    X_test_concat = extractor2.transform_concatenated(test_texts)
    
    print(f"\nTrain concatenated shape: {X_train_concat.shape}")
    print(f"Test concatenated shape: {X_test_concat.shape}")
    print(f"Total features: {X_train_concat.shape[1]}")
    print(f"  BoW features: {len(extractor2.get_feature_names_bow())}")
    print(f"  TF-IDF features: {len(extractor2.get_feature_names_tfidf())}")
    
    # Show concatenated feature names (sample)
    concat_features = extractor2.get_feature_names_concatenated()
    print(f"\nFirst 5 concatenated features: {concat_features[:5]}")
    print(f"Last 5 concatenated features: {concat_features[-5:]}")
    
    # Show dense representation of first sample
    print(f"\nFirst training sample (concatenated - first 10 non-zero):")
    print(f"Text: {train_texts[0]}")
    first_sample_concat = X_train_concat[0].toarray().flatten()
    non_zero_indices = np.where(first_sample_concat > 0)[0]
    for idx in non_zero_indices[:10]:
        print(f"  {concat_features[idx]}: {first_sample_concat[idx]:.4f}")
    
    # ==========================================
    # 4. STATISTICS
    # ==========================================
    print("\n" + "="*70)
    print("4. FEATURE MATRIX STATISTICS")
    print("="*70)
    
    stats = extractor2.get_statistics(
        X_bow=X_train_bow,
        X_tfidf=X_train_tfidf,
        X_concat=X_train_concat
    )
    
    for method, method_stats in stats.items():
        print(f"\n{method.upper()}:")
        for key, value in method_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    # ==========================================
    # 5. COMPARISON TABLE
    # ==========================================
    print("\n" + "="*70)
    print("5. FEATURE EXTRACTION COMPARISON")
    print("="*70)
    
    comparison_df = pd.DataFrame({
        'Method': ['BoW', 'TF-IDF', 'Concatenated'],
        'Train Shape': [X_train_bow.shape, X_train_tfidf.shape, X_train_concat.shape],
        'Test Shape': [X_test_bow.shape, X_test_tfidf.shape, X_test_concat.shape],
        'N Features': [X_train_bow.shape[1], X_train_tfidf.shape[1], X_train_concat.shape[1]],
        'Density': [
            f"{stats['bow']['density']:.4f}",
            f"{stats['tfidf']['density']:.4f}",
            f"{stats['concatenated']['density']:.4f}"
        ]
    })
    
    print(f"\n{comparison_df.to_string(index=False)}")
    
    # ==========================================
    # 6. SAVE AND LOAD VECTORIZERS
    # ==========================================
    print("\n" + "="*70)
    print("6. SAVE AND LOAD VECTORIZERS")
    print("="*70)
    
    # Save
    save_path = "feature_extractors.pkl"
    extractor2.save_vectorizers(save_path)
    print(f"Vectorizers saved to: {save_path}")
    
    # Load
    new_extractor = FeatureExtractor()
    new_extractor.load_vectorizers(save_path)
    print(f"Vectorizers loaded from: {save_path}")
    
    # Verify it works
    X_test_loaded = new_extractor.transform_concatenated(test_texts)
    print(f"Loaded extractor - Test shape: {X_test_loaded.shape}")
    print(f"Original extractor - Test shape: {X_test_concat.shape}")
    print(f"Shapes match: {X_test_loaded.shape == X_test_concat.shape}")
    
    # ==========================================
    # 7. USING CONVENIENCE FUNCTION
    # ==========================================
    print("\n" + "="*70)
    print("7. USING CONVENIENCE FUNCTION")
    print("="*70)
    
    # Extract features using convenience function
    X_tr_conv, X_te_conv = extract_features(
        train_texts,
        test_texts,
        method='concatenated',
        max_features=50,
        ngram_range=(1, 1)
    )
    
    print(f"Train features: {X_tr_conv.shape}")
    print(f"Test features: {X_te_conv.shape}")
    
    # ==========================================
    # 8. DIFFERENT N-GRAM CONFIGURATIONS
    # ==========================================
    print("\n" + "="*70)
    print("8. DIFFERENT N-GRAM CONFIGURATIONS")
    print("="*70)
    
    ngram_configs = [
        (1, 1),  # Unigrams only
        (1, 2),  # Unigrams + Bigrams
        (1, 3),  # Unigrams + Bigrams + Trigrams
    ]
    
    for ngram_range in ngram_configs:
        ext = FeatureExtractor(ngram_range=ngram_range, max_features=100)
        X_tr = ext.fit_transform_concatenated(train_texts)
        print(f"\nN-gram range {ngram_range}: {X_tr.shape[1]} features")
        
        # Show sample features
        features = ext.get_feature_names_bow()[:5]
        print(f"  Sample BoW features: {features}")
    
    # ==========================================
    # 9. INTEGRATION WITH PANDAS
    # ==========================================
    print("\n" + "="*70)
    print("9. INTEGRATION WITH PANDAS DATAFRAME")
    print("="*70)
    
    # Create DataFrames
    train_df = pd.DataFrame({
        'text': train_texts,
        'label': ['positive', 'positive', 'positive', 'neutral', 'negative',
                 'positive', 'neutral', 'positive', 'neutral', 'neutral']
    })
    
    test_df = pd.DataFrame({
        'text': test_texts,
        'label': ['positive', 'neutral', 'positive']
    })
    
    print("\nTrain DataFrame:")
    print(train_df.head())
    
    # Extract features from DataFrame
    extractor3 = FeatureExtractor(max_features=50)
    X_train_df = extractor3.fit_transform_concatenated(train_df['text'])
    X_test_df = extractor3.transform_concatenated(test_df['text'])
    
    print(f"\nExtracted features from DataFrame:")
    print(f"Train: {X_train_df.shape}")
    print(f"Test: {X_test_df.shape}")
    
    # Convert sparse matrix to DataFrame for inspection
    feature_names = extractor.get_feature_names_concatenated()
    train_features_df = pd.DataFrame(
    X_train_df.toarray(),
    columns=feature_names
    )

    print(f"\nFeature matrix as DataFrame (first 5 rows, first 5 columns):")
    print(train_features_df.iloc[:5, :5])

    print("\n" + "="*70)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*70)