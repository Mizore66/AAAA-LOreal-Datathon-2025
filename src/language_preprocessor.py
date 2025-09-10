#!/usr/bin/env python3
"""
Feature Translation Pipeline for L'Oréal Datathon 2025
Translates extracted features, keywords, hashtags, and emerging terms to English.

This is much more efficient than translating the entire raw dataset.
We translate only the relevant extracted features and terms.

Features:
1. Translate hashtags and keywords from feature files
2. Translate emerging terms and trend words
3. Batch processing for efficiency
4. Preserve original terms alongside translations
5. Smart filtering for beauty/fashion relevance
6. Export translated feature sets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
import json
from typing import Dict, List, Tuple, Optional, Set
from tqdm import tqdm
import warnings
import re
from concurrent.futures import ThreadPoolExecutor

# Language detection and translation libraries
try:
    from langdetect import detect, LangDetectException
    from googletrans import Translator
    TRANSLATION_AVAILABLE = True
    print("✅ Language translation libraries loaded successfully")
except ImportError as e:
    TRANSLATION_AVAILABLE = False
    print(f"❌ Translation libraries not available: {e}")
    print("Install with: pip install langdetect googletrans==4.0.0-rc1")

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureTranslator:
    """Handles translation of extracted features and terms."""
    
    def __init__(self, confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold
        self.translator = Translator() if TRANSLATION_AVAILABLE else None
        self.translation_cache = {}
        self.beauty_keywords = {
            'beauty', 'fashion', 'skincare', 'makeup', 'hair', 'cosmetics', 
            'beauty', 'style', 'skin', 'face', 'lips', 'eyes', 'nail', 'polish',
            'foundation', 'mascara', 'lipstick', 'eyeliner', 'blush', 'powder',
            'serum', 'moisturizer', 'cleanser', 'toner', 'cream', 'lotion',
            'shampoo', 'conditioner', 'styling', 'color', 'highlight', 'contour'
        }
        self.stats = {
            'total_terms': 0,
            'english_terms': 0,
            'translated_terms': 0,
            'failed_translations': 0,
            'beauty_relevant': 0,
            'detected_languages': {}
        }
    
    def is_beauty_relevant(self, text: str) -> bool:
        """Check if text is relevant to beauty/fashion industry."""
        text_lower = text.lower()
        
        # Direct keyword matches
        if any(keyword in text_lower for keyword in self.beauty_keywords):
            return True
        
        # Beauty-related patterns
        beauty_patterns = [
            r'\b(makeup|beauty|skincare|cosmetic|fashion|style)\b',
            r'\b(hair|skin|face|lips|eyes|nail)\b',
            r'\b(foundation|mascara|lipstick|eyeliner|blush)\b',
            r'\b(serum|moisturizer|cleanser|cream|lotion)\b',
            r'\b(shampoo|conditioner|styling|color)\b'
        ]
        
        for pattern in beauty_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language of text."""
        if not TRANSLATION_AVAILABLE or not text or len(text.strip()) < 2:
            return 'en', 1.0
        
        try:
            detected_lang = detect(text)
            confidence = 0.8  # Assume reasonable confidence
            return detected_lang, confidence
        except (LangDetectException, Exception):
            return 'en', 1.0
    
    def translate_term(self, term: str) -> Tuple[str, bool, str]:
        """Translate a single term to English."""
        if not TRANSLATION_AVAILABLE or not term:
            return term, False, 'en'
        
        # Cache check
        if term in self.translation_cache:
            return self.translation_cache[term]
        
        # Clean the term (remove hashtags, special chars for detection)
        clean_term = re.sub(r'[#@]', '', term).strip()
        
        if not clean_term or len(clean_term) < 2:
            result = (term, False, 'en')
            self.translation_cache[term] = result
            return result
        
        try:
            # Detect language
            detected_lang, confidence = self.detect_language(clean_term)
            
            # Update stats
            if detected_lang not in self.stats['detected_languages']:
                self.stats['detected_languages'][detected_lang] = 0
            self.stats['detected_languages'][detected_lang] += 1
            
            # If already English or low confidence, keep original
            if detected_lang == 'en' or confidence < self.confidence_threshold:
                result = (term, False, detected_lang)
                self.translation_cache[term] = result
                return result
            
            # Translate to English
            translation_result = self.translator.translate(clean_term, src=detected_lang, dest='en')
            translated_text = translation_result.text
            
            # For hashtags, preserve the # symbol
            if term.startswith('#') and not translated_text.startswith('#'):
                translated_text = '#' + translated_text
            
            result = (translated_text, True, detected_lang)
            self.translation_cache[term] = result
            return result
            
        except Exception as e:
            logger.debug(f"Translation failed for '{term}': {e}")
            result = (term, False, 'unknown')
            self.translation_cache[term] = result
            return result
    
    def translate_term_list(self, terms: List[str]) -> pd.DataFrame:
        """Translate a list of terms and return a DataFrame with results."""
        results = []
        
        logger.info(f"Translating {len(terms)} terms...")
        
        for term in tqdm(terms, desc="Translating terms"):
            translated_term, was_translated, detected_lang = self.translate_term(term)
            
            # Check beauty relevance (on both original and translated)
            is_beauty_orig = self.is_beauty_relevant(term)
            is_beauty_trans = self.is_beauty_relevant(translated_term)
            is_beauty_relevant = is_beauty_orig or is_beauty_trans
            
            result = {
                'original_term': term,
                'translated_term': translated_term,
                'detected_language': detected_lang,
                'was_translated': was_translated,
                'is_beauty_relevant': is_beauty_relevant,
                'length': len(term)
            }
            
            results.append(result)
            
            # Update stats
            self.stats['total_terms'] += 1
            if was_translated:
                self.stats['translated_terms'] += 1
            else:
                self.stats['english_terms'] += 1
            
            if is_beauty_relevant:
                self.stats['beauty_relevant'] += 1
        
        return pd.DataFrame(results)
    
    def translate_feature_file(self, input_file: Path, output_file: Path, 
                              term_column: str = 'term') -> bool:
        """Translate terms in a feature file."""
        try:
            logger.info(f"Processing feature file: {input_file}")
            
            # Read the feature file
            if input_file.suffix.lower() == '.csv':
                df = pd.read_csv(input_file)
            elif input_file.suffix.lower() == '.parquet':
                df = pd.read_parquet(input_file)
            else:
                logger.error(f"Unsupported file format: {input_file.suffix}")
                return False
            
            logger.info(f"Loaded {len(df)} features from {input_file}")
            
            # Check available columns
            logger.info(f"Available columns: {list(df.columns)}")
            
            # Determine the correct column name based on actual data structure
            possible_columns = ['feature', term_column, 'hashtag', 'keyword', 'term', 'emerging_term']
            actual_column = None
            
            for col in possible_columns:
                if col in df.columns:
                    actual_column = col
                    break
            
            if actual_column is None:
                logger.error(f"Could not find term column in {input_file}. Available columns: {list(df.columns)}")
                return False
            
            logger.info(f"Using column '{actual_column}' for terms")
            
            # Extract unique terms
            unique_terms = df[actual_column].dropna().unique().tolist()
            logger.info(f"Found {len(unique_terms)} unique terms to translate")
            
            if len(unique_terms) == 0:
                logger.warning(f"No terms found in {input_file}")
                return False
            
            # Translate terms
            translation_df = self.translate_term_list(unique_terms)
            
            # Merge back with original data
            df_merged = df.merge(
                translation_df.rename(columns={'original_term': actual_column}),
                on=actual_column,
                how='left'
            )
            
            # Save translated features
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if output_file.suffix.lower() == '.csv':
                df_merged.to_csv(output_file, index=False)
            else:
                df_merged.to_parquet(output_file, index=False)
            
            logger.info(f"Translated features saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error translating feature file {input_file}: {e}")
            return False
    
    def save_translation_report(self, output_path: Path):
        """Save translation statistics."""
        report = {
            'translation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'statistics': self.stats,
            'beauty_keywords': list(self.beauty_keywords),
            'configuration': {
                'confidence_threshold': self.confidence_threshold,
                'translation_available': TRANSLATION_AVAILABLE
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Translation report saved to {output_path}")

def translate_feature_datasets():
    """Main function to translate all feature datasets based on actual directory structure."""
    
    if not TRANSLATION_AVAILABLE:
        logger.error("Translation libraries not available!")
        print("Please install them with: pip install langdetect googletrans==4.0.0-rc1")
        return
    
    # Initialize translator
    translator = FeatureTranslator()
    
    # Define paths based on actual directory structure
    features_dir = Path("data/processed/dataset")
    translated_dir = Path("data/translated_features")
    translated_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all feature files in the actual directory
    feature_files = []
    
    # Look for feature files that contain hashtags, keywords, or emerging terms
    for file_path in features_dir.glob("features_*.parquet"):
        if any(term in file_path.name for term in ['hashtag', 'keyword', 'emerging']):
            feature_files.append(file_path)
    
    logger.info(f"Found {len(feature_files)} feature files to translate:")
    for f in feature_files:
        logger.info(f"  - {f.name}")
    
    if len(feature_files) == 0:
        logger.warning("No feature files found in data/processed/dataset/")
        logger.info("Available files in directory:")
        for f in features_dir.glob("*.parquet"):
            logger.info(f"  - {f.name}")
        return
    
    # Process each feature file
    success_count = 0
    for feature_file in feature_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {feature_file.name}")
        logger.info(f"{'='*60}")
        
        # Determine output filename
        output_name = f"{feature_file.stem}_translated{feature_file.suffix}"
        output_file = translated_dir / output_name
        
        # Skip if already processed
        if output_file.exists():
            logger.info(f"Skipping {feature_file.name} - already translated")
            continue
        
        # Determine the term column name based on file type and actual structure
        if 'hashtag' in feature_file.name.lower():
            term_column = 'feature'  # Based on actual structure
        elif 'keyword' in feature_file.name.lower():
            term_column = 'feature'  # Based on actual structure
        elif 'emerging' in feature_file.name.lower():
            term_column = 'feature'  # Based on actual structure
        else:
            term_column = 'feature'  # Default to 'feature' based on your data structure
        
        # Translate the file
        success = translator.translate_feature_file(
            feature_file, 
            output_file, 
            term_column
        )
        
        if success:
            success_count += 1
            logger.info(f"✅ Successfully translated {feature_file.name}")
        else:
            logger.error(f"❌ Failed to translate {feature_file.name}")
    
    # Save overall translation report
    report_file = translated_dir / "translation_report.json"
    translator.save_translation_report(report_file)
    
    # Print summary
    logger.info(f"\n🎉 Feature translation completed!")
    logger.info(f"Successfully translated: {success_count}/{len(feature_files)} files")
    logger.info(f"Results saved in: {translated_dir}")
    logger.info(f"Translation statistics:")
    logger.info(f"  - Total terms processed: {translator.stats['total_terms']}")
    logger.info(f"  - Terms translated: {translator.stats['translated_terms']}")
    logger.info(f"  - Beauty-relevant terms: {translator.stats['beauty_relevant']}")
    logger.info(f"  - Languages detected: {list(translator.stats['detected_languages'].keys())}")

def translate_emerging_terms_only():
    """Quick function to translate just emerging terms from reports."""
    
    if not TRANSLATION_AVAILABLE:
        logger.error("Translation libraries not available!")
        return
    
    translator = FeatureTranslator()
    
    # Look for report files with emerging terms in the actual directory structure
    reports_dir = Path("data/interim")
    translated_dir = Path("data/translated_features")
    translated_dir.mkdir(parents=True, exist_ok=True)
    
    # Find emerging terms in reports
    emerging_terms = set()
    
    # Check for JSON reports first
    for report_file in reports_dir.glob("*.json"):
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            logger.info(f"Processing report: {report_file.name}")
            
            # Extract emerging terms from different sections
            if 'emerging_hashtags' in report_data:
                for timeframe_data in report_data['emerging_hashtags'].values():
                    if isinstance(timeframe_data, list):
                        emerging_terms.update([item['hashtag'] for item in timeframe_data if 'hashtag' in item])
            
            if 'emerging_keywords' in report_data:
                for timeframe_data in report_data['emerging_keywords'].values():
                    if isinstance(timeframe_data, list):
                        emerging_terms.update([item['keyword'] for item in timeframe_data if 'keyword' in item])
            
            # Handle Phase 3 report structure (both simple and comprehensive)
            if 'phase3_emerging_trends' in report_data:
                phase3_data = report_data['phase3_emerging_trends']
                if 'all_emerging_terms' in phase3_data:
                    emerging_terms.update(phase3_data['all_emerging_terms'])
                    logger.info(f"Added {len(phase3_data['all_emerging_terms'])} terms from Phase 3 report")
            
            # Handle comprehensive Phase 3 report structure
            if 'phase3_emerging_trends_detailed' in report_data:
                phase3_detailed = report_data['phase3_emerging_trends_detailed']
                if 'all_emerging_terms_with_data' in phase3_detailed:
                    detailed_terms = [term['feature'] for term in phase3_detailed['all_emerging_terms_with_data']]
                    emerging_terms.update(detailed_terms)
                    logger.info(f"Added {len(detailed_terms)} terms from comprehensive Phase 3 report")
                    
                    # Also extract terms from detailed timeframes
                    if 'detailed_timeframes' in phase3_detailed:
                        for timeframe, tf_data in phase3_detailed['detailed_timeframes'].items():
                            tf_terms = [term['feature'] for term in tf_data.get('emerging_terms', [])]
                            emerging_terms.update(tf_terms)
                        logger.info(f"Added terms from {len(phase3_detailed['detailed_timeframes'])} timeframes")
            
            # Handle Phase 2 enhanced features report structure
            if 'report_type' in report_data and report_data['report_type'] == 'phase2_enhanced_features':
                # Extract all features from the comprehensive Phase 2 report
                if 'all_features' in report_data:
                    phase2_terms = [feature['feature'] for feature in report_data['all_features']]
                    emerging_terms.update(phase2_terms)
                    logger.info(f"Added {len(phase2_terms)} terms from comprehensive Phase 2 report")
                
                # Extract beauty-relevant features specifically
                if 'beauty_relevant_features' in report_data:
                    beauty_terms = [feature['feature'] for feature in report_data['beauty_relevant_features']]
                    emerging_terms.update(beauty_terms)
                    logger.info(f"Added {len(beauty_terms)} beauty-relevant terms from Phase 2 report")
        
        except Exception as e:
            logger.warning(f"Could not process report {report_file}: {e}")
    
    # Also check emerging terms feature files directly
    features_dir = Path("data/processed/dataset")
    for emerging_file in features_dir.glob("features_emerging_terms_*.parquet"):
        try:
            df = pd.read_parquet(emerging_file)
            logger.info(f"Processing emerging terms file: {emerging_file.name}")
            
            # Try different possible column names (based on actual structure)
            term_col = None
            for col in ['feature', 'emerging_term', 'term', 'hashtag', 'keyword']:
                if col in df.columns:
                    term_col = col
                    break
            
            if term_col:
                terms = df[term_col].dropna().unique().tolist()
                emerging_terms.update(terms)
                logger.info(f"Added {len(terms)} terms from {emerging_file.name}")
            else:
                logger.warning(f"No term column found in {emerging_file.name}. Columns: {list(df.columns)}")
        
        except Exception as e:
            logger.warning(f"Could not process emerging terms file {emerging_file}: {e}")
    
    if emerging_terms:
        logger.info(f"Found {len(emerging_terms)} unique emerging terms to translate")
        
        # Translate emerging terms
        terms_list = list(emerging_terms)
        translation_df = translator.translate_term_list(terms_list)
        
        # Save translated emerging terms
        output_file = translated_dir / "emerging_terms_translated.csv"
        translation_df.to_csv(output_file, index=False)
        
        # Save beauty-relevant terms separately
        beauty_terms = translation_df[translation_df['is_beauty_relevant']]
        beauty_output = translated_dir / "beauty_emerging_terms_translated.csv"
        beauty_terms.to_csv(beauty_output, index=False)
        
        logger.info(f"✅ Emerging terms translated and saved to {output_file}")
        logger.info(f"✅ Beauty-relevant terms saved to {beauty_output}")
        logger.info(f"Beauty-relevant terms: {len(beauty_terms)}/{len(translation_df)}")
        
        # Save report
        report_file = translated_dir / "emerging_terms_translation_report.json"
        translator.save_translation_report(report_file)
    
    else:
        logger.warning("No emerging terms found in reports or feature files to translate")

if __name__ == "__main__":
    print("Feature Translation Options:")
    print("1. Translate all feature files")
    print("2. Translate only emerging terms from reports")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("This process is too expensive to run by default. Currently disabled.")
        # translate_feature_datasets()
    elif choice == "2":
        translate_emerging_terms_only()
    else:
        print("Invalid choice. Translating emerging terms only...")
        translate_emerging_terms_only()