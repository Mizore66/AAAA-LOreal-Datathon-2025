#!/usr/bin/env python3
"""
Feature Text Processor for L'Oréal Datathon 2025
Processes feature files to fix spelling mistakes and translate non-English terms.

Features:
1. Load feature files (parquet, CSV, JSON) containing extracted terms
2. Fix spelling mistakes using spell checkers and AI models
3. Detect language of terms and translate non-English to English
4. Handle beauty/fashion specific terminology
5. Export cleaned and translated feature datasets
6. Generate processing reports and statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
import json
import os
import re
from typing import Dict, List, Tuple, Optional, Set, Union
from tqdm import tqdm
import warnings
from datetime import datetime

# Spell checking libraries
try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
    print("✅ PySpellChecker loaded successfully")
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    print("❌ PySpellChecker not available. Install with: pip install pyspellchecker")

try:
    from autocorrect import Speller
    AUTOCORRECT_AVAILABLE = True
    print("✅ Autocorrect loaded successfully")
except ImportError:
    AUTOCORRECT_AVAILABLE = False
    print("❌ Autocorrect not available. Install with: pip install autocorrect")

# Language detection and translation
try:
    from langdetect import detect, LangDetectException
    from googletrans import Translator
    TRANSLATION_AVAILABLE = True
    print("✅ Translation libraries loaded successfully")
except ImportError:
    TRANSLATION_AVAILABLE = False
    print("❌ Translation libraries not available. Install with: pip install langdetect googletrans==4.0.0-rc1")

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
PROCESSED_FEATURES_DIR = DATA_DIR / "processed_features" 
REPORTS_DIR = DATA_DIR / "feature_processing_reports"

# Create directories
for dir_path in [FEATURES_DIR, PROCESSED_FEATURES_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Beauty/Fashion specific corrections dictionary
BEAUTY_CORRECTIONS = {
    'perfune': 'perfume',
    'maskara': 'mascara',
    'foundaton': 'foundation',
    'contouring': 'contouring',
    'highliter': 'highlighter',
    'blending': 'blending',
    'moisturiser': 'moisturizer',
    'cleanser': 'cleanser',
    'exfoliant': 'exfoliant',
    'serium': 'serum',
    'retinol': 'retinol',
    'niacinamide': 'niacinamide',
    'hyaluronic': 'hyaluronic',
    'glycolic': 'glycolic',
    'salicylic': 'salicylic',
    'vitaminc': 'vitamin c',
    'spf': 'spf',
    'sunscreen': 'sunscreen',
    'skincare': 'skincare',
    'haircare': 'hair care',
    'shampoo': 'shampoo',
    'conditioner': 'conditioner',
    'styling': 'styling',
    'hairmask': 'hair mask',
    'lipstick': 'lipstick',
    'lipgloss': 'lip gloss',
    'eyeliner': 'eyeliner',
    'eyeshadow': 'eyeshadow',
    'bronzer': 'bronzer',
    'concealer': 'concealer',
    'primer': 'primer',
    'blush': 'blush',
    'rouge': 'blush',
    'polish': 'nail polish',
    'manicure': 'manicure',
    'pedicure': 'pedicure'
}

class FeatureTextProcessor:
    """Main class for processing feature text data."""
    
    def __init__(self, confidence_threshold=0.7, use_beauty_dict=True):
        """
        Initialize the feature text processor.
        
        Args:
            confidence_threshold: Minimum confidence for language detection
            use_beauty_dict: Whether to use beauty-specific corrections
        """
        self.confidence_threshold = confidence_threshold
        self.use_beauty_dict = use_beauty_dict
        
        # Initialize spell checkers
        self.spell_checker = None
        self.auto_corrector = None
        self.translator = None
        
        if SPELLCHECKER_AVAILABLE:
            self.spell_checker = SpellChecker()
            # Add beauty terms to dictionary
            if use_beauty_dict:
                beauty_terms = set(BEAUTY_CORRECTIONS.values())
                self.spell_checker.word_frequency.load_words(beauty_terms)
        
        if AUTOCORRECT_AVAILABLE:
            self.auto_corrector = Speller(lang='en')
            
        if TRANSLATION_AVAILABLE:
            self.translator = Translator()
            
        # Statistics
        self.stats = {
            'total_terms_processed': 0,
            'spelling_corrections': 0,
            'translations': 0,
            'english_terms': 0,
            'failed_corrections': 0,
            'failed_translations': 0,
            'beauty_specific_corrections': 0,
            'language_distribution': {},
            'processing_time': 0
        }
        
        logger.info("FeatureTextProcessor initialized")
    
    def correct_spelling(self, term: str) -> Tuple[str, bool, str]:
        """
        Correct spelling of a term using multiple methods.
        
        Args:
            term: Input term to correct
            
        Returns:
            Tuple of (corrected_term, was_corrected, correction_method)
        """
        if not isinstance(term, str) or not term.strip():
            return term, False, 'invalid_input'
        
        original_term = term.strip().lower()
        
        # Check beauty-specific corrections first
        if self.use_beauty_dict and original_term in BEAUTY_CORRECTIONS:
            corrected = BEAUTY_CORRECTIONS[original_term]
            self.stats['beauty_specific_corrections'] += 1
            return corrected, True, 'beauty_dict'
        
        # Use PySpellChecker
        if self.spell_checker:
            if original_term not in self.spell_checker:
                # Get suggestions
                suggestions = self.spell_checker.candidates(original_term)
                if suggestions:
                    corrected = list(suggestions)[0]  # Take first suggestion
                    if corrected != original_term:
                        return corrected, True, 'pyspellchecker'
        
        # Use Autocorrect as fallback
        if self.auto_corrector:
            try:
                corrected = self.auto_corrector(original_term)
                if corrected != original_term:
                    return corrected, True, 'autocorrect'
            except Exception as e:
                logger.debug(f"Autocorrect failed for '{original_term}': {e}")
        
        # No correction needed or available
        return original_term, False, 'no_correction'
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language of text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if not isinstance(text, str) or not text.strip():
            return 'unknown', 0.0
        
        try:
            if TRANSLATION_AVAILABLE:
                lang = detect(text)
                # langdetect doesn't provide confidence directly, 
                # so we estimate based on text length and detection success
                confidence = min(0.9, max(0.5, len(text) / 20))
                return lang, confidence
            else:
                return 'unknown', 0.0
        except (LangDetectException, Exception) as e:
            logger.debug(f"Language detection failed for '{text}': {e}")
            return 'unknown', 0.0
    
    def translate_text(self, text: str, source_lang: str = 'auto') -> Tuple[str, bool, str]:
        """
        Translate text to English.
        
        Args:
            text: Text to translate
            source_lang: Source language code ('auto' for detection)
            
        Returns:
            Tuple of (translated_text, was_translated, target_language)
        """
        if not isinstance(text, str) or not text.strip():
            return text, False, 'invalid_input'
        
        try:
            if not TRANSLATION_AVAILABLE:
                return text, False, 'translation_unavailable'
            
            # Skip if already English
            detected_lang, confidence = self.detect_language(text)
            if detected_lang == 'en' and confidence >= self.confidence_threshold:
                return text, False, 'already_english'
            
            # Translate to English
            if self.translator:
                result = self.translator.translate(text, dest='en', src=source_lang)
                if result and result.text and result.text.strip() != text.strip():
                    return result.text.strip(), True, detected_lang
            
            return text, False, 'translation_failed'
            
        except Exception as e:
            logger.debug(f"Translation failed for '{text}': {e}")
            return text, False, 'translation_error'
    
    def process_term(self, term: str) -> Dict[str, Union[str, bool, float]]:
        """
        Process a single term: spell check and translate if needed.
        
        Args:
            term: Input term
            
        Returns:
            Dictionary with processing results
        """
        result = {
            'original_term': term,
            'processed_term': term,
            'spelling_corrected': False,
            'spelling_method': 'none',
            'translated': False,
            'source_language': 'unknown',
            'language_confidence': 0.0,
            'is_english': False,
            'processing_steps': []
        }
        
        if not isinstance(term, str) or not term.strip():
            return result
        
        current_term = term.strip()
        
        # Step 1: Spelling correction
        corrected_term, was_corrected, correction_method = self.correct_spelling(current_term)
        if was_corrected:
            result['spelling_corrected'] = True
            result['spelling_method'] = correction_method
            result['processing_steps'].append(f"spell_correction_{correction_method}")
            current_term = corrected_term
            self.stats['spelling_corrections'] += 1
        
        # Step 2: Language detection
        detected_lang, confidence = self.detect_language(current_term)
        result['source_language'] = detected_lang
        result['language_confidence'] = confidence
        
        # Update language distribution stats
        if detected_lang in self.stats['language_distribution']:
            self.stats['language_distribution'][detected_lang] += 1
        else:
            self.stats['language_distribution'][detected_lang] = 1
        
        # Step 3: Translation if not English
        if detected_lang != 'en' and confidence >= self.confidence_threshold:
            translated_term, was_translated, source_lang = self.translate_text(current_term)
            if was_translated:
                result['translated'] = True
                result['processing_steps'].append(f"translation_from_{source_lang}")
                current_term = translated_term
                self.stats['translations'] += 1
        elif detected_lang == 'en':
            result['is_english'] = True
            self.stats['english_terms'] += 1
        
        result['processed_term'] = current_term
        self.stats['total_terms_processed'] += 1
        
        return result
    
    def process_feature_file(self, file_path: Union[str, Path], 
                           text_columns: List[str] = None,
                           output_dir: str = None) -> Dict:
        """
        Process a feature file containing terms to be corrected and translated.
        
        Args:
            file_path: Path to the feature file
            text_columns: List of column names containing text to process
            output_dir: Directory to save processed results
            
        Returns:
            Dictionary with processing results and statistics
        """
        file_path = Path(file_path)
        logger.info(f"🔤 Processing feature file: {file_path}")
        
        results = {
            'input_path': str(file_path),
            'output_path': None,
            'correction_stats': {},
            'translation_stats': {},
            'processing_time': None
        }
        
        start_time = time.time()
        
        # Step 1: Load file with progress tracking
        with tqdm(total=6, desc=f"Processing {file_path.name}", unit="step") as main_pbar:
            try:
                main_pbar.set_postfix(step="loading file")
                # Load file based on extension
                if file_path.suffix.lower() == '.parquet':
                    df = pd.read_parquet(file_path)
                elif file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path)
                elif file_path.suffix.lower() == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    else:
                        df = pd.json_normalize(data)
                else:
                    logger.error(f"Unsupported file format: {file_path.suffix}")
                    return results
                main_pbar.update(1)
                
                if df.empty:
                    logger.warning(f"Empty file: {file_path}")
                    return results
                
                # Step 2: Auto-detect text columns if not provided
                main_pbar.set_postfix(step="detecting columns")
                if text_columns is None:
                    text_columns = []
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            # Check if column contains mostly string data
                            if df[col].dropna().apply(lambda x: isinstance(x, str)).mean() > 0.8:
                                text_columns.append(col)
                main_pbar.update(1)
                
                logger.info(f"Processing text columns: {text_columns}")
                
                # Step 3: Initialize tracking
                main_pbar.set_postfix(step="initializing")
                total_corrections = 0
                total_translations = 0
                main_pbar.update(1)
                
                # Step 4: Process each text column
                main_pbar.set_postfix(step="processing columns")
                with tqdm(text_columns, desc="Processing columns", leave=False) as col_pbar:
                    for col in col_pbar:
                        if col not in df.columns:
                            logger.warning(f"Column '{col}' not found in file")
                            continue
                        
                        col_pbar.set_postfix(column=col)
                        logger.info(f"🔤 Processing column: {col}")
                        
                        # Process each term in the column
                        processed_results = []
                        terms = df[col].dropna().unique()
                        
                        # Term-level progress tracking
                        corrections_this_col = 0
                        translations_this_col = 0
                        
                        with tqdm(terms, desc=f"Processing {col}", leave=False, unit="term") as term_pbar:
                            for term in term_pbar:
                                result = self.process_term(term)
                                processed_results.append(result)
                                
                                # Count corrections and translations
                                if result.get('was_corrected', False):
                                    corrections_this_col += 1
                                if result.get('was_translated', False):
                                    translations_this_col += 1
                                
                                # Update progress bar with stats
                                term_pbar.set_postfix({
                                    'corrections': corrections_this_col,
                                    'translations': translations_this_col
                                })
                        
                        total_corrections += corrections_this_col
                        total_translations += translations_this_col
                        
                        # Create mapping of original to processed terms
                        term_mapping = {r['original_term']: r['processed_term'] for r in processed_results}
                        
                        # Apply mapping to create new columns
                        df[f"{col}_original"] = df[col]
                        df[f"{col}_processed"] = df[col].map(term_mapping).fillna(df[col])
                        df[f"{col}_was_corrected"] = df[col].apply(
                            lambda x: term_mapping.get(x, x) != x if pd.notna(x) else False
                        )
                        
                        # Save detailed processing info for unique terms
                        if output_dir:
                            detailed_df = pd.DataFrame(processed_results)
                            detail_path = Path(output_dir) / f"{file_path.stem}_{col}_processing_details.parquet"
                            detail_path.parent.mkdir(parents=True, exist_ok=True)
                            detailed_df.to_parquet(detail_path, index=False)
                        
                        # Update column stats
                        results['correction_stats'][col] = corrections_this_col
                        results['translation_stats'][col] = translations_this_col
                
                main_pbar.update(1)
                
                # Step 5: Save processed file
                main_pbar.set_postfix(step="saving results")
                if output_dir:
                    output_path = Path(output_dir) / f"{file_path.stem}_processed.parquet"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(output_path, index=False)
                    results['output_path'] = str(output_path)
                    logger.info(f"✅ Processed file saved to: {output_path}")
                main_pbar.update(1)
                
                # Step 6: Finalize
                main_pbar.set_postfix(step="finalizing")
                processing_time = time.time() - start_time
                results['processing_time'] = processing_time
                
                logger.info(f"✅ Completed processing {file_path.name}: {total_corrections} corrections, {total_translations} translations in {processing_time:.2f}s")
                main_pbar.update(1)
                
                return results
                
            except Exception as e:
                logger.error(f"❌ Error processing file {file_path}: {e}")
                return results
    
    def process_directory(self, directory_path: Union[str, Path], 
                         file_patterns: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Process all feature files in a directory.
        
        Args:
            directory_path: Path to directory containing feature files
            file_patterns: List of file patterns to match (e.g., ['*.parquet', '*.csv'])
            
        Returns:
            Dictionary mapping file names to processed DataFrames
        """
        directory_path = Path(directory_path)
        logger.info(f"Processing directory: {directory_path}")
        
        if file_patterns is None:
            file_patterns = ['*.parquet', '*.csv', '*.json']
        
        # Find all matching files
        files_to_process = []
        for pattern in file_patterns:
            files_to_process.extend(directory_path.glob(pattern))
        
        if not files_to_process:
            logger.warning(f"No files found matching patterns {file_patterns} in {directory_path}")
            return {}
        
        logger.info(f"Found {len(files_to_process)} files to process")
        
        results = {}
        start_time = time.time()
        
        for file_path in files_to_process:
            processed_df = self.process_feature_file(file_path)
            if not processed_df.empty:
                results[file_path.name] = processed_df
                
                # Save processed file
                output_path = PROCESSED_FEATURES_DIR / f"processed_{file_path.name}"
                if file_path.suffix.lower() == '.parquet':
                    processed_df.to_parquet(output_path, index=False)
                else:
                    # Convert to parquet for consistency
                    output_path = output_path.with_suffix('.parquet')
                    processed_df.to_parquet(output_path, index=False)
                
                logger.info(f"Saved processed file: {output_path}")
        
        self.stats['processing_time'] = time.time() - start_time
        logger.info(f"Directory processing completed in {self.stats['processing_time']:.2f} seconds")
        
        return results
    
    def generate_report(self) -> Dict:
        """Generate a comprehensive processing report."""
        report = {
            'processing_timestamp': datetime.now().isoformat(),
            'statistics': self.stats.copy(),
            'configuration': {
                'confidence_threshold': self.confidence_threshold,
                'use_beauty_dict': self.use_beauty_dict,
                'spellchecker_available': SPELLCHECKER_AVAILABLE,
                'autocorrect_available': AUTOCORRECT_AVAILABLE,
                'translation_available': TRANSLATION_AVAILABLE
            },
            'beauty_corrections_used': BEAUTY_CORRECTIONS if self.use_beauty_dict else {},
            'summary': {
                'correction_rate': (self.stats['spelling_corrections'] / max(1, self.stats['total_terms_processed'])) * 100,
                'translation_rate': (self.stats['translations'] / max(1, self.stats['total_terms_processed'])) * 100,
                'english_rate': (self.stats['english_terms'] / max(1, self.stats['total_terms_processed'])) * 100
            }
        }
        
        return report
    
    def save_report(self, report: Dict = None) -> Path:
        """Save processing report to file."""
        if report is None:
            report = self.generate_report()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"feature_processing_report_{timestamp}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processing report saved to: {report_path}")
        return report_path

def main():
    """Main function to run feature text processing."""
    print("🎨 Feature Text Processor for L'Oréal Datathon 2025")
    print("=" * 60)
    
    # Check if required libraries are available
    if not any([SPELLCHECKER_AVAILABLE, AUTOCORRECT_AVAILABLE]):
        print("❌ No spell checking libraries available!")
        print("Install with: pip install pyspellchecker autocorrect")
        return
    
    if not TRANSLATION_AVAILABLE:
        print("⚠️  Translation libraries not available - only spell checking will be performed")
        print("For full functionality, install: pip install langdetect googletrans==4.0.0-rc1")
    
    # Initialize processor
    processor = FeatureTextProcessor(confidence_threshold=0.7, use_beauty_dict=True)
    
    # Look for feature files to process
    data_directories = [
        ROOT / "data" / "processed" / "dataset",
        ROOT / "data" / "interim",
        ROOT / "data" / "features",
        ROOT / "data"
    ]
    
    files_found = []
    for directory in data_directories:
        if directory.exists():
            files_found.extend(list(directory.glob("*.parquet")))
            files_found.extend(list(directory.glob("*.csv")))
            files_found.extend(list(directory.glob("*.json")))
    
    if not files_found:
        print(f"No feature files found in standard directories")
        print("Creating sample data for demonstration...")
        
        # Create sample feature data
        sample_data = pd.DataFrame({
            'hashtag': ['#perfune', '#maskara', '#beautifull', '#skincare', '#maquillage', '#schönheit'],
            'keyword': ['foundaton', 'lipstick', 'concealer', 'moisturiser', 'rouge', 'parfum'],
            'category': ['beauty', 'makeup', 'cosmetics', 'skincare', 'makeup', 'fragrance'],
            'count': [100, 250, 150, 300, 80, 120]
        })
        
        sample_path = FEATURES_DIR / "sample_features.parquet"
        sample_data.to_parquet(sample_path, index=False)
        print(f"Created sample file: {sample_path}")
        files_found = [sample_path]
    
    # Process files
    print(f"\nProcessing {len(files_found)} feature files...")
    for file_path in files_found:
        print(f"\n📁 Processing: {file_path}")
        processed_df = processor.process_feature_file(file_path)
        
        if not processed_df.empty:
            print(f"✅ Processed {len(processed_df)} rows")
        else:
            print("❌ Processing failed")
    
    # Generate and save report
    report = processor.generate_report()
    report_path = processor.save_report(report)
    
    print(f"\n📊 Processing Summary:")
    print(f"   Total terms processed: {report['statistics']['total_terms_processed']:,}")
    print(f"   Spelling corrections: {report['statistics']['spelling_corrections']:,}")
    print(f"   Translations: {report['statistics']['translations']:,}")
    print(f"   English terms: {report['statistics']['english_terms']:,}")
    print(f"   Correction rate: {report['summary']['correction_rate']:.1f}%")
    print(f"   Translation rate: {report['summary']['translation_rate']:.1f}%")
    print(f"   Processing time: {report['statistics']['processing_time']:.2f}s")
    
    print(f"\n📄 Full report saved to: {report_path}")
    print(f"📁 Processed files saved to: {PROCESSED_FEATURES_DIR}")

if __name__ == "__main__":
    main()