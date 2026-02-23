# import re
# import string
# import nltk
# from typing import List, Optional, Union
# import pandas as pd
# import numpy as np
# from collections import Counter
# import logging

# # Download required NLTK data
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('stopwords')

# try:
#     nltk.data.find('corpora/wordnet')
# except LookupError:
#     nltk.download('wordnet')

# try:
#     nltk.data.find('corpora/omw-1.4')
# except LookupError:
#     nltk.download('omw-1.4')

# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)


# class TextPreprocessor:
#     """
#     Text preprocessing pipeline based on ML-NLPEmot paper methodology.
    
#     Preprocessing steps:
#     1. Remove duplicates
#     2. Remove usernames (@username)
#     3. Remove URLs/links
#     4. Remove punctuation
#     5. Tokenization
#     6. Case folding (lowercase)
#     7. Remove stop words
#     8. Lemmatization
#     """
    
#     def __init__(self, 
#                  language: str = 'english',
#                  custom_stopwords: Optional[List[str]] = None,
#                  keep_stopwords: Optional[List[str]] = None,
#                  remove_numbers: bool = False,
#                  min_token_length: int = 2):
#         """
#         Initialize the text preprocessor.
        
#         Args:
#             language: Language for stopwords (default: 'english')
#             custom_stopwords: Additional stopwords to remove
#             keep_stopwords: Stopwords to keep (won't be removed)
#             remove_numbers: Whether to remove numeric tokens
#             min_token_length: Minimum length for tokens to keep
#         """
#         self.language = language
#         self.remove_numbers = remove_numbers
#         self.min_token_length = min_token_length
        
#         # Initialize lemmatizer
#         self.lemmatizer = WordNetLemmatizer()
        
#         # Setup stopwords
#         self.stopwords = set(stopwords.words(language))
        
#         if custom_stopwords:
#             self.stopwords.update(custom_stopwords)
        
#         if keep_stopwords:
#             self.stopwords -= set(keep_stopwords)
        
#         logger.info(f"TextPreprocessor initialized with {len(self.stopwords)} stopwords")
    
#     def remove_urls(self, text: str) -> str:
#         """
#         Remove URLs and hyperlinks from text.
        
#         Args:
#             text: Input text
            
#         Returns:
#             Text with URLs removed
#         """
#         # Pattern to match http://, https://, www., and common TLDs
#         url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\\\(\\\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
#         text = re.sub(url_pattern, '', text)
        
#         # Match www. URLs
#         www_pattern = r'www\\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\\\(\\\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
#         text = re.sub(www_pattern, '', text)
        
#         # Match domain.com patterns
#         domain_pattern = r'\\b[a-zA-Z0-9-]+\\.(com|org|net|edu|gov|mil|int|co|io|ai)\\b'
#         text = re.sub(domain_pattern, '', text)
        
#         return text.strip()
    
#     def remove_usernames(self, text: str) -> str:
#         """
#         Remove usernames (e.g., @username) from text.
        
#         Args:
#             text: Input text
            
#         Returns:
#             Text with usernames removed
#         """
#         # Remove @username mentions
#         text = re.sub(r'@\\w+', '', text)
#         return text.strip()
    
#     def remove_retweet_marker(self, text: str) -> str:
#         """
#         Remove RT (retweet) marker from text.
        
#         Args:
#             text: Input text
            
#         Returns:
#             Text with RT removed
#         """
#         text = re.sub(r'\\bRT\\b', '', text, flags=re.IGNORECASE)
#         return text.strip()
    
#     def remove_punctuation(self, text: str) -> str:
#         """
#         Remove punctuation marks from text.
        
#         Args:
#             text: Input text
            
#         Returns:
#             Text with punctuation removed
#         """
#         # Remove all punctuation
#         translator = str.maketrans('', '', string.punctuation)
#         text = text.translate(translator)
#         return text.strip()
    
#     def remove_extra_whitespace(self, text: str) -> str:
#         """
#         Remove extra whitespace and normalize spacing.
        
#         Args:
#             text: Input text
            
#         Returns:
#             Text with normalized whitespace
#         """
#         # Replace multiple spaces with single space
#         text = re.sub(r'\\s+', ' ', text)
#         return text.strip()
    
#     def case_fold(self, text: str) -> str:
#         """
#         Convert text to lowercase (case folding).
        
#         Args:
#             text: Input text
            
#         Returns:
#             Lowercase text
#         """
#         return text.lower()
    
#     def tokenize(self, text: str) -> List[str]:
#         """
#         Tokenize text into words.
        
#         Args:
#             text: Input text
            
#         Returns:
#             List of tokens
#         """
#         tokens = word_tokenize(text)
#         return tokens
    
#     def remove_stopwords(self, tokens: List[str]) -> List[str]:
#         """
#         Remove stopwords from token list.
        
#         Args:
#             tokens: List of tokens
            
#         Returns:
#             Filtered list of tokens
#         """
#         filtered_tokens = [token for token in tokens if token.lower() not in self.stopwords]
#         return filtered_tokens
    
#     def lemmatize(self, tokens: List[str]) -> List[str]:
#         """
#         Lemmatize tokens to their root form.
        
#         Args:
#             tokens: List of tokens
            
#         Returns:
#             List of lemmatized tokens
#         """
#         lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
#         return lemmatized_tokens
    
#     def filter_tokens(self, tokens: List[str]) -> List[str]:
#         """
#         Filter tokens based on length and numeric content.
        
#         Args:
#             tokens: List of tokens
            
#         Returns:
#             Filtered list of tokens
#         """
#         filtered = []
#         for token in tokens:
#             # Check minimum length
#             if len(token) < self.min_token_length:
#                 continue
            
#             # Remove numbers if specified
#             if self.remove_numbers and token.isdigit():
#                 continue
            
#             filtered.append(token)
        
#         return filtered
    
#     def preprocess(self, text: str, return_string: bool = True) -> Union[str, List[str]]:
#         """
#         Apply complete preprocessing pipeline to text.
        
#         Pipeline:
#         1. Remove URLs
#         2. Remove usernames
#         3. Remove RT markers
#         4. Case folding (lowercase)
#         5. Remove punctuation
#         6. Remove extra whitespace
#         7. Tokenization
#         8. Remove stopwords
#         9. Lemmatization
#         10. Filter tokens
        
#         Args:
#             text: Input text
#             return_string: If True, return preprocessed text as string,
#                           otherwise return list of tokens
            
#         Returns:
#             Preprocessed text (string or list of tokens)
#         """
#         if not isinstance(text, str):
#             text = str(text)
        
#         # Step 1: Remove URLs
#         text = self.remove_urls(text)
        
#         # Step 2: Remove usernames
#         text = self.remove_usernames(text)
        
#         # Step 3: Remove RT markers
#         text = self.remove_retweet_marker(text)
        
#         # Step 4: Case folding
#         text = self.case_fold(text)
        
#         # Step 5: Remove punctuation
#         text = self.remove_punctuation(text)
        
#         # Step 6: Remove extra whitespace
#         text = self.remove_extra_whitespace(text)
        
#         # Step 7: Tokenization
#         tokens = self.tokenize(text)
        
#         # Step 8: Remove stopwords
#         tokens = self.remove_stopwords(tokens)
        
#         # Step 9: Lemmatization
#         tokens = self.lemmatize(tokens)
        
#         # Step 10: Filter tokens
#         tokens = self.filter_tokens(tokens)
        
#         if return_string:
#             return ' '.join(tokens)
#         else:
#             return tokens
    
#     def preprocess_batch(self, 
#                         texts: Union[List[str], pd.Series],
#                         return_string: bool = True,
#                         show_progress: bool = True) -> List[Union[str, List[str]]]:
#         """
#         Preprocess a batch of texts.
        
#         Args:
#             texts: List or Series of texts
#             return_string: If True, return strings, otherwise return token lists
#             show_progress: Whether to show progress logging
            
#         Returns:
#             List of preprocessed texts
#         """
#         if isinstance(texts, pd.Series):
#             texts = texts.tolist()
        
#         preprocessed = []
#         total = len(texts)
        
#         for idx, text in enumerate(texts):
#             preprocessed_text = self.preprocess(text, return_string=return_string)
#             preprocessed.append(preprocessed_text)
            
#             if show_progress and (idx + 1) % 100 == 0:
#                 logger.info(f"Processed {idx + 1}/{total} texts")
        
#         if show_progress:
#             logger.info(f"Batch preprocessing complete: {total} texts")
        
#         return preprocessed
    
#     def get_statistics(self, texts: List[str]) -> dict:
#         """
#         Get preprocessing statistics for a list of texts.
        
#         Args:
#             texts: List of preprocessed texts
            
#         Returns:
#             Dictionary containing statistics
#         """
#         all_tokens = []
#         text_lengths = []
        
#         for text in texts:
#             if isinstance(text, str):
#                 tokens = text.split()
#             else:
#                 tokens = text
            
#             all_tokens.extend(tokens)
#             text_lengths.append(len(tokens))
        
#         token_counts = Counter(all_tokens)
        
#         stats = {
#             'total_texts': len(texts),
#             'total_tokens': len(all_tokens),
#             'unique_tokens': len(token_counts),
#             'avg_tokens_per_text': np.mean(text_lengths) if text_lengths else 0,
#             'median_tokens_per_text': np.median(text_lengths) if text_lengths else 0,
#             'min_tokens_per_text': min(text_lengths) if text_lengths else 0,
#             'max_tokens_per_text': max(text_lengths) if text_lengths else 0,
#             'most_common_tokens': token_counts.most_common(10)
#         }
        
#         return stats


# def create_preprocessing_function(language: str = 'english',
#                                   custom_stopwords: Optional[List[str]] = None,
#                                   keep_stopwords: Optional[List[str]] = None,
#                                   remove_numbers: bool = False,
#                                   min_token_length: int = 2) -> callable:
#     """
#     Factory function to create a reusable preprocessing function.
    
#     Args:
#         language: Language for stopwords
#         custom_stopwords: Additional stopwords to remove
#         keep_stopwords: Stopwords to keep
#         remove_numbers: Whether to remove numbers
#         min_token_length: Minimum token length
        
#     Returns:
#         Preprocessing function
#     """
#     preprocessor = TextPreprocessor(
#         language=language,
#         custom_stopwords=custom_stopwords,
#         keep_stopwords=keep_stopwords,
#         remove_numbers=remove_numbers,
#         min_token_length=min_token_length
#     )
    
#     def preprocess_text(text: str, return_string: bool = True) -> Union[str, List[str]]:
#         """
#         Preprocess a single text.
        
#         Args:
#             text: Input text
#             return_string: Return as string or token list
            
#         Returns:
#             Preprocessed text
#         """
#         return preprocessor.preprocess(text, return_string=return_string)
    
#     return preprocess_text


# # Demonstration and testing
# if __name__ == "__main__":
#     # Example texts (COVID-19 vaccine related tweets from UAE)
#     sample_texts = [
#         "RT @WHO: Getting vaccinated is the best way to protect yourself and others from COVID-19. Visit https://covid19.who.int for more info #VaccineSavesLives",
#         "@UAEGov announced free COVID-19 vaccination for all residents! Great news! 🎉",
#         "Just got my Pfizer vaccine in Dubai today. Feeling relieved and hopeful! #CovidVaccine #UAE",
#         "China's Sinopharm vaccine available in Abu Dhabi. Book your appointment at www.uaehealth.gov.ae",
#         "Worried about vaccine side effects... Has anyone experienced any issues? #COVID19",
#         "The UAE has been doing an AMAZING job with the vaccination campaign!!!",
#         "RT @DubaiHealth: Second dose vaccination centers now open. Don't forget to complete your vaccination schedule.",
#         "   Too many spaces    and @mentions @here    with URLs http://example.com   ",
#     ]
    
#     print("="*70)
#     print("TEXT PREPROCESSING DEMONSTRATION")
#     print("="*70)
    
#     # Initialize preprocessor
#     preprocessor = TextPreprocessor(
#         language='english',
#         custom_stopwords=['covid', 'vaccine', 'uae'],  # Domain-specific stopwords
#         remove_numbers=False,
#         min_token_length=2
#     )
    
#     # Demonstrate step-by-step preprocessing
#     print("\\n" + "="*70)
#     print("STEP-BY-STEP PREPROCESSING EXAMPLE")
#     print("="*70)
    
#     example_text = sample_texts[0]
#     print(f"\\nOriginal text:\\n{example_text}")
    
#     step1 = preprocessor.remove_urls(example_text)
#     print(f"\\n1. After removing URLs:\\n{step1}")
    
#     step2 = preprocessor.remove_usernames(step1)
#     print(f"\\n2. After removing usernames:\\n{step2}")
    
#     step3 = preprocessor.remove_retweet_marker(step2)
#     print(f"\\n3. After removing RT:\\n{step3}")
    
#     step4 = preprocessor.case_fold(step3)
#     print(f"\\n4. After case folding:\\n{step4}")
    
#     step5 = preprocessor.remove_punctuation(step4)
#     print(f"\\n5. After removing punctuation:\\n{step5}")
    
#     step6 = preprocessor.remove_extra_whitespace(step5)
#     print(f"\\n6. After removing extra whitespace:\\n{step6}")
    
#     step7 = preprocessor.tokenize(step6)
#     print(f"\\n7. After tokenization:\\n{step7}")
    
#     step8 = preprocessor.remove_stopwords(step7)
#     print(f"\\n8. After removing stopwords:\\n{step8}")
    
#     step9 = preprocessor.lemmatize(step8)
#     print(f"\\n9. After lemmatization:\\n{step9}")
    
#     step10 = preprocessor.filter_tokens(step9)
#     print(f"\\n10. After filtering:\\n{step10}")
    
#     final = ' '.join(step10)
#     print(f"\\nFinal preprocessed text:\\n{final}")
    
#     # Demonstrate complete preprocessing on all samples
#     print("\\n" + "="*70)
#     print("COMPLETE PREPROCESSING - ALL SAMPLES")
#     print("="*70)
    
#     for idx, text in enumerate(sample_texts, 1):
#         preprocessed = preprocessor.preprocess(text, return_string=True)
#         print(f"\\n--- Sample {idx} ---")
#         print(f"Original: {text}")
#         print(f"Preprocessed: {preprocessed}")
    
#     # Demonstrate batch preprocessing
#     print("\\n" + "="*70)
#     print("BATCH PREPROCESSING")
#     print("="*70)
    
#     preprocessed_batch = preprocessor.preprocess_batch(
#         sample_texts,
#         return_string=True,
#         show_progress=True
#     )
    
#     print(f"\\nPreprocessed {len(preprocessed_batch)} texts")
    
#     # Get statistics
#     print("\\n" + "="*70)
#     print("PREPROCESSING STATISTICS")
#     print("="*70)
    
#     stats = preprocessor.get_statistics(preprocessed_batch)
#     print(f"\\nTotal texts: {stats['total_texts']}")
#     print(f"Total tokens: {stats['total_tokens']}")
#     print(f"Unique tokens: {stats['unique_tokens']}")
#     print(f"Avg tokens per text: {stats['avg_tokens_per_text']:.2f}")
#     print(f"Median tokens per text: {stats['median_tokens_per_text']:.2f}")
#     print(f"Min tokens per text: {stats['min_tokens_per_text']}")
#     print(f"Max tokens per text: {stats['max_tokens_per_text']}")
#     print(f"\\nMost common tokens:")
#     for token, count in stats['most_common_tokens']:
#         print(f"  {token}: {count}")
    
#     # Demonstrate factory function
#     print("\\n" + "="*70)
#     print("USING FACTORY FUNCTION")
#     print("="*70)
    
#     # Create a reusable preprocessing function
#     preprocess = create_preprocessing_function(
#         language='english',
#         custom_stopwords=['covid', 'vaccine', 'coronavirus'],
#         remove_numbers=False,
#         min_token_length=2
#     )
    
#     test_text = "RT @UAEGov: COVID-19 vaccination update! Visit https://uae.gov.ae for details"
#     result_string = preprocess(test_text, return_string=True)
#     result_tokens = preprocess(test_text, return_string=False)
    
#     print(f"\\nOriginal: {test_text}")
#     print(f"As string: {result_string}")
#     print(f"As tokens: {result_tokens}")
    
#     # Use with pandas DataFrame
#     print("\\n" + "="*70)
#     print("INTEGRATION WITH PANDAS DATAFRAME")
#     print("="*70)
    
#     df = pd.DataFrame({
#         'text': sample_texts,
#         'label': ['positive', 'positive', 'positive', 'neutral', 
#                   'negative', 'positive', 'neutral', 'neutral']
#     })
    
#     print("\\nOriginal DataFrame:")
#     print(df.head())
    
#     # Apply preprocessing
#     df['preprocessed_text'] = preprocessor.preprocess_batch(
#         df['text'],
#         return_string=True,
#         show_progress=False
#     )
    
#     # Get token counts
#     df['token_count'] = df['preprocessed_text'].apply(lambda x: len(x.split()))
    
#     print("\\nPreprocessed DataFrame:")
#     print(df[['text', 'preprocessed_text', 'token_count']].head())
    
#     print("\\n" + "="*70)
#     print("PREPROCESSING COMPLETE")
#     print("="*70)