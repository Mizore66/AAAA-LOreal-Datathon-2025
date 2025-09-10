#!/usr/bin/env python3
"""
Unified Translation Pipeline for L'Oréal Datathon 2025
Translates both feature .parquet files and comprehensive JSON reports to English.

Features:
1. Translate feature files (hashtags, keywords, emerging terms)
2. Translate comprehensive JSON reports (Phase 2 & Phase 3)
3. Batch processing for efficiency
4. Preserve original terms alongside translations
5. Smart filtering for beauty/fashion relevance
6. Export translated datasets and reports
7. Comprehensive translation statistics and reports
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
import json
import os
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

class UnifiedTranslator:
    """Handles translation of both feature files and comprehensive reports."""
    
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
            'detected_languages': {},
            'files_processed': {
                'feature_files': 0,
                'json_reports': 0
            }
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
    
    def translate_text(self, text: str) -> Tuple[str, bool, str]:
        """
        Translate text to English if needed.
        
        Returns:
            (translated_text, was_translated, detected_language)
        """
        if not text or not self.translator:
            return text, False, 'unknown'
        
        # Check cache first
        if text in self.translation_cache:
            return self.translation_cache[text]
        
        # Clean the text (remove hashtags, special chars for detection)
        clean_text = re.sub(r'[#@]', '', text).strip()
        
        if not clean_text or len(clean_text) < 2:
            result = (text, False, 'en')
            self.translation_cache[text] = result
            return result
        
        try:
            # Detect language
            detected_lang, confidence = self.detect_language(clean_text)
            
            # Update stats
            if detected_lang not in self.stats['detected_languages']:
                self.stats['detected_languages'][detected_lang] = 0
            self.stats['detected_languages'][detected_lang] += 1
            
            # If already English or low confidence, keep original
            if detected_lang == 'en' or confidence < self.confidence_threshold:
                result = (text, False, detected_lang)
                self.translation_cache[text] = result
                return result
            
            # Translate to English
            translation_result = self.translator.translate(clean_text, src=detected_lang, dest='en')
            translated_text = translation_result.text
            
            # For hashtags, preserve the # symbol
            if text.startswith('#') and not translated_text.startswith('#'):
                translated_text = '#' + translated_text
            
            result = (translated_text, True, detected_lang)
            self.translation_cache[text] = result
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
            
            return result
            
        except Exception as e:
            logger.debug(f"Translation failed for '{text}': {e}")
            result = (text, False, 'error')
            self.translation_cache[text] = result
            return result
    
    def translate_feature_file(self, input_file: Path, output_file: Path, 
                              term_column: str = 'feature') -> bool:
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
            translation_results = []
            for term in tqdm(unique_terms, desc=f"Translating {input_file.name}"):
                translated_term, was_translated, detected_lang = self.translate_text(term)
                
                # Check beauty relevance
                is_beauty_orig = self.is_beauty_relevant(term)
                is_beauty_trans = self.is_beauty_relevant(translated_term)
                is_beauty_relevant = is_beauty_orig or is_beauty_trans
                
                translation_results.append({
                    actual_column: term,
                    f'{actual_column}_translated': translated_term,
                    f'{actual_column}_original': term,
                    'was_translated': was_translated,
                    'detected_language': detected_lang,
                    'is_beauty_relevant': is_beauty_relevant
                })
                
                # Update stats
                self.stats['total_terms'] += 1
                if was_translated:
                    self.stats['translated_terms'] += 1
                else:
                    self.stats['english_terms'] += 1
                
                if is_beauty_relevant:
                    self.stats['beauty_relevant'] += 1
            
            # Create translation DataFrame
            translation_df = pd.DataFrame(translation_results)
            
            # Merge back with original data
            df_merged = df.merge(translation_df, on=actual_column, how='left')
            
            # Replace the original column with translated version
            df_merged[actual_column] = df_merged[f'{actual_column}_translated']
            
            # Save translated features
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if output_file.suffix.lower() == '.csv':
                df_merged.to_csv(output_file, index=False)
            else:
                df_merged.to_parquet(output_file, index=False)
            
            logger.info(f"✅ Translated feature file saved to {output_file}")
            self.stats['files_processed']['feature_files'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error translating feature file {input_file}: {e}")
            return False
    
    def translate_phase2_report(self, input_file: str, output_file: str) -> bool:
        """Translate Phase 2 comprehensive report"""
        try:
            logger.info(f"Loading Phase 2 report: {input_file}")
            
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create translation tracking
            translation_log = []
            
            # Translate all_features
            if 'all_features' in data:
                logger.info("Translating all_features...")
                for feature_data in tqdm(data['all_features'], desc="All features"):
                    original_feature = feature_data['feature']
                    translated, was_translated, lang = self.translate_text(original_feature)
                    
                    # Update feature name
                    feature_data['feature_original'] = original_feature
                    feature_data['feature'] = translated
                    feature_data['was_translated'] = was_translated
                    feature_data['detected_language'] = lang
                    
                    # Track translation
                    translation_log.append({
                        'original': original_feature,
                        'translated': translated,
                        'was_translated': was_translated,
                        'language': lang
                    })
                    
                    self.stats['total_terms'] += 1
                    if was_translated:
                        self.stats['translated_terms'] += 1
                    else:
                        self.stats['english_terms'] += 1
            
            # Translate beauty_relevant_features
            if 'beauty_relevant_features' in data:
                logger.info("Translating beauty_relevant_features...")
                for feature_data in tqdm(data['beauty_relevant_features'], desc="Beauty features"):
                    original_feature = feature_data['feature']
                    translated, was_translated, lang = self.translate_text(original_feature)
                    
                    # Update feature name
                    feature_data['feature_original'] = original_feature
                    feature_data['feature'] = translated
                    feature_data['was_translated'] = was_translated
                    feature_data['detected_language'] = lang
            
            # Translate top_performers_by_timeframe
            if 'top_performers_by_timeframe' in data:
                logger.info("Translating top performers by timeframe...")
                for timeframe, tf_data in data['top_performers_by_timeframe'].items():
                    if 'features' in tf_data:
                        for feature_data in tqdm(tf_data['features'], desc=f"Timeframe {timeframe}"):
                            original_feature = feature_data['feature']
                            translated, was_translated, lang = self.translate_text(original_feature)
                            
                            # Update feature name
                            feature_data['feature_original'] = original_feature
                            feature_data['feature'] = translated
                            feature_data['was_translated'] = was_translated
                            feature_data['detected_language'] = lang
            
            # Add translation metadata
            data['translation_metadata'] = {
                'translated_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'translation_stats': dict(self.stats),
                'translation_log': translation_log
            }
            
            # Save translated report
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Phase 2 translated report saved: {output_file}")
            self.stats['files_processed']['json_reports'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to translate Phase 2 report: {e}")
            return False
    
    def translate_phase3_report(self, input_file: str, output_file: str) -> bool:
        """Translate Phase 3 comprehensive report"""
        try:
            logger.info(f"Loading Phase 3 report: {input_file}")
            
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            translation_log = []
            
            # Navigate to the Phase 3 data structure
            if 'phase3_emerging_trends_detailed' in data:
                phase3_data = data['phase3_emerging_trends_detailed']
                
                # Translate detailed_timeframes
                if 'detailed_timeframes' in phase3_data:
                    logger.info("Translating detailed timeframes...")
                    for timeframe, tf_data in phase3_data['detailed_timeframes'].items():
                        if 'emerging_terms' in tf_data:
                            for term_data in tqdm(tf_data['emerging_terms'], desc=f"Timeframe {timeframe}"):
                                if 'feature' in term_data and term_data['feature']:
                                    original_feature = term_data['feature']
                                    translated, was_translated, lang = self.translate_text(original_feature)
                                    
                                    # Update feature name
                                    term_data['feature_original'] = original_feature
                                    term_data['feature'] = translated
                                    term_data['was_translated'] = was_translated
                                    term_data['detected_language'] = lang
                
                # Translate all_emerging_terms_with_data
                if 'all_emerging_terms_with_data' in phase3_data:
                    logger.info("Translating all emerging terms with data...")
                    for term_data in tqdm(phase3_data['all_emerging_terms_with_data'], desc="All terms"):
                        if 'feature' in term_data and term_data['feature']:
                            original_feature = term_data['feature']
                            translated, was_translated, lang = self.translate_text(original_feature)
                            
                            # Update feature name
                            term_data['feature_original'] = original_feature
                            term_data['feature'] = translated
                            term_data['was_translated'] = was_translated
                            term_data['detected_language'] = lang
                            
                            # Track translation
                            translation_log.append({
                                'original': original_feature,
                                'translated': translated,
                                'was_translated': was_translated,
                                'language': lang
                            })
                            
                            self.stats['total_terms'] += 1
                            if was_translated:
                                self.stats['translated_terms'] += 1
                            else:
                                self.stats['english_terms'] += 1
                
                # Translate category_analysis
                if 'category_analysis' in phase3_data:
                    logger.info("Translating category analysis...")
                    for category, cat_data in phase3_data['category_analysis'].items():
                        if 'terms' in cat_data:
                            translated_terms = []
                            for term in cat_data['terms']:
                                translated, was_translated, lang = self.translate_text(term)
                                translated_terms.append({
                                    'original': term,
                                    'translated': translated,
                                    'was_translated': was_translated,
                                    'detected_language': lang
                                })
                            cat_data['terms_translated'] = translated_terms
                
                # Add translation metadata
                phase3_data['translation_metadata'] = {
                    'translated_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'translation_stats': dict(self.stats),
                    'translation_log': translation_log
                }
            
            # Save translated report
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Phase 3 translated report saved: {output_file}")
            self.stats['files_processed']['json_reports'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to translate Phase 3 report: {e}")
            return False
    
    def translate_all_feature_files(self) -> int:
        """Translate only smaller feature files (emerging terms, not hashtags/keywords)"""
        features_dir = Path("data/processed/dataset")
        translated_dir = Path("data/translated_features")
        translated_dir.mkdir(parents=True, exist_ok=True)
        
        # Find only smaller feature files - skip large hashtags/keywords files
        feature_files = []
        for file_path in features_dir.glob("features_*.parquet"):
            # Only include emerging terms files (smaller and more manageable)
            if 'emerging' in file_path.name:
                feature_files.append(file_path)
            elif any(term in file_path.name for term in ['hashtag', 'keyword']):
                logger.info(f"Skipping large file: {file_path.name} (too large for translation)")
        
        logger.info(f"Found {len(feature_files)} manageable feature files to translate")
        
        if len(feature_files) == 0:
            logger.warning("No suitable feature files found for translation")
            return 0
        
        success_count = 0
        for feature_file in feature_files:
            output_name = f"{feature_file.stem}_translated{feature_file.suffix}"
            output_file = translated_dir / output_name
            
            # Skip if already processed
            if output_file.exists():
                logger.info(f"Skipping {feature_file.name} - already translated")
                continue
            
            success = self.translate_feature_file(feature_file, output_file, 'feature')
            if success:
                success_count += 1
        
        return success_count
    
    def translate_emerging_terms_from_reports(self) -> bool:
        """Extract and translate emerging terms from reports (fast and efficient)"""
        reports_dir = Path("data/interim")
        translated_dir = Path("data/translated_features")
        translated_dir.mkdir(parents=True, exist_ok=True)
        
        # Find emerging terms in reports
        emerging_terms = set()
        
        # Check for JSON reports
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
                
                # Handle Phase 3 report structure
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
                
                # Handle Phase 2 enhanced features report structure
                if 'report_type' in report_data and report_data['report_type'] == 'phase2_enhanced_features':
                    if 'all_features' in report_data:
                        phase2_terms = [feature['feature'] for feature in report_data['all_features']]
                        emerging_terms.update(phase2_terms)
                        logger.info(f"Added {len(phase2_terms)} terms from comprehensive Phase 2 report")
                    
                    if 'beauty_relevant_features' in report_data:
                        beauty_terms = [feature['feature'] for feature in report_data['beauty_relevant_features']]
                        emerging_terms.update(beauty_terms)
                        logger.info(f"Added {len(beauty_terms)} beauty-relevant terms from Phase 2 report")
            
            except Exception as e:
                logger.warning(f"Could not process report {report_file}: {e}")
        
        # Also check small emerging terms feature files
        features_dir = Path("data/processed/dataset")
        for emerging_file in features_dir.glob("features_emerging_terms_*.parquet"):
            try:
                df = pd.read_parquet(emerging_file)
                logger.info(f"Processing emerging terms file: {emerging_file.name}")
                
                # Try different possible column names
                term_col = None
                for col in ['feature', 'emerging_term', 'term', 'hashtag', 'keyword']:
                    if col in df.columns:
                        term_col = col
                        break
                
                if term_col:
                    terms = df[term_col].dropna().unique().tolist()
                    emerging_terms.update(terms)
                    logger.info(f"Added {len(terms)} terms from {emerging_file.name}")
            
            except Exception as e:
                logger.warning(f"Could not process emerging terms file {emerging_file}: {e}")
        
        if emerging_terms:
            logger.info(f"Found {len(emerging_terms)} unique emerging terms to translate")
            
            # Translate emerging terms
            terms_list = list(emerging_terms)
            translation_results = []
            
            for term in tqdm(terms_list, desc="Translating emerging terms"):
                translated_term, was_translated, detected_lang = self.translate_text(term)
                
                # Check beauty relevance
                is_beauty_orig = self.is_beauty_relevant(term)
                is_beauty_trans = self.is_beauty_relevant(translated_term)
                is_beauty_relevant = is_beauty_orig or is_beauty_trans
                
                translation_results.append({
                    'original_term': term,
                    'translated_term': translated_term,
                    'detected_language': detected_lang,
                    'was_translated': was_translated,
                    'is_beauty_relevant': is_beauty_relevant,
                    'length': len(term)
                })
                
                # Update stats
                self.stats['total_terms'] += 1
                if was_translated:
                    self.stats['translated_terms'] += 1
                else:
                    self.stats['english_terms'] += 1
                
                if is_beauty_relevant:
                    self.stats['beauty_relevant'] += 1
            
            # Create DataFrame and save results
            translation_df = pd.DataFrame(translation_results)
            
            # Save all translated emerging terms
            output_file = translated_dir / "emerging_terms_translated.csv"
            translation_df.to_csv(output_file, index=False)
            
            # Save beauty-relevant terms separately
            beauty_terms = translation_df[translation_df['is_beauty_relevant']]
            beauty_output = translated_dir / "beauty_emerging_terms_translated.csv"
            beauty_terms.to_csv(beauty_output, index=False)
            
            logger.info(f"✅ Emerging terms translated and saved to {output_file}")
            logger.info(f"✅ Beauty-relevant terms saved to {beauty_output}")
            logger.info(f"Beauty-relevant terms: {len(beauty_terms)}/{len(translation_df)}")
            
            return True
        else:
            logger.warning("No emerging terms found in reports or feature files to translate")
            return False
    
    def translate_all_json_reports(self) -> int:
        """Translate all comprehensive JSON reports"""
        interim_dir = "data/interim"
        
        # Define file mappings
        reports_to_translate = [
            {
                'input': f"{interim_dir}/phase2_enhanced_features_comprehensive.json",
                'output': f"{interim_dir}/phase2_enhanced_features_comprehensive_translated.json",
                'type': 'phase2'
            },
            {
                'input': f"{interim_dir}/phase3_emerging_trends_comprehensive.json",
                'output': f"{interim_dir}/phase3_emerging_trends_comprehensive_translated.json",
                'type': 'phase3'
            }
        ]
        
        success_count = 0
        for report in reports_to_translate:
            if os.path.exists(report['input']):
                if report['type'] == 'phase2':
                    success = self.translate_phase2_report(report['input'], report['output'])
                else:  # phase3
                    success = self.translate_phase3_report(report['input'], report['output'])
                
                if success:
                    success_count += 1
            else:
                logger.warning(f"Report file not found: {report['input']}")
        
        return success_count
    
    def save_unified_translation_report(self, output_path: Path):
        """Save comprehensive translation statistics."""
        report = {
            'translation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'translation_type': 'unified_features_and_reports',
            'statistics': self.stats,
            'beauty_keywords': list(self.beauty_keywords),
            'configuration': {
                'confidence_threshold': self.confidence_threshold,
                'translation_available': TRANSLATION_AVAILABLE
            },
            'summary': {
                'total_files_processed': sum(self.stats['files_processed'].values()),
                'feature_files_processed': self.stats['files_processed']['feature_files'],
                'json_reports_processed': self.stats['files_processed']['json_reports'],
                'beauty_relevance_rate': (
                    self.stats['beauty_relevant'] / self.stats['total_terms'] 
                    if self.stats['total_terms'] > 0 else 0
                ),
                'translation_rate': (
                    self.stats['translated_terms'] / self.stats['total_terms'] 
                    if self.stats['total_terms'] > 0 else 0
                )
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Unified translation report saved to {output_path}")

def main():
    """Main execution function for unified translation pipeline"""
    
    if not TRANSLATION_AVAILABLE:
        logger.error("Translation libraries not available!")
        print("Please install them with: pip install langdetract googletrans==4.0.0-rc1")
        return
    
    print("🌍 UNIFIED TRANSLATION PIPELINE")
    print("=" * 60)
    print("Choose what to translate:")
    print("1. 📊 Emerging terms from reports (FAST - recommended)")
    print("2. 📄 JSON comprehensive reports only")
    print("3. 🔄 Both emerging terms AND JSON reports")
    print("4. ⚠️  All files including large feature files (SLOW)")
    print("=" * 60)
    
    choice = input("Enter choice (1-4): ").strip()
    
    # Initialize unified translator
    translator = UnifiedTranslator()
    
    # Create output directory
    translated_dir = Path("data/translated_features")
    translated_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    feature_success = 0
    report_success = 0
    emerging_success = False
    
    if choice == "1":
        # Fast option - just emerging terms from reports
        print("\n🔄 Translating Emerging Terms from Reports")
        print("-" * 50)
        emerging_success = translator.translate_emerging_terms_from_reports()
        
    elif choice == "2":
        # Just JSON reports
        print("\n📄 Translating JSON Reports")
        print("-" * 30)
        report_success = translator.translate_all_json_reports()
        
    elif choice == "3":
        # Both emerging terms and JSON reports (recommended)
        print("\n🔄 Step 1: Translating Emerging Terms from Reports")
        print("-" * 50)
        emerging_success = translator.translate_emerging_terms_from_reports()
        
        print("\n📄 Step 2: Translating JSON Reports")
        print("-" * 35)
        report_success = translator.translate_all_json_reports()
        
    elif choice == "4":
        # Full pipeline including large files (slow)
        print("\n⚠️  WARNING: This will process large files and may take a long time!")
        confirm = input("Are you sure? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Operation cancelled.")
            return
            
        print("\n🔄 Step 1: Translating Emerging Terms")
        print("-" * 40)
        emerging_success = translator.translate_emerging_terms_from_reports()
        
        print("\n📊 Step 2: Translating Feature Files")
        print("-" * 35)
        feature_success = translator.translate_all_feature_files()
        
        print("\n📄 Step 3: Translating JSON Reports")
        print("-" * 35)
        report_success = translator.translate_all_json_reports()
    else:
        print("Invalid choice. Defaulting to emerging terms translation...")
        emerging_success = translator.translate_emerging_terms_from_reports()
    
    # Save unified translation report
    report_file = translated_dir / "unified_translation_report.json"
    translator.save_unified_translation_report(report_file)
    
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Print comprehensive summary
    print("\n🎉 TRANSLATION COMPLETED!")
    print("=" * 50)
    print(f"⏱️  Processing time: {processing_time:.2f} seconds")
    
    if emerging_success:
        print("✅ Emerging terms translated successfully")
    if feature_success > 0:
        print(f"📊 Feature files translated: {feature_success}")
    if report_success > 0:
        print(f"📄 JSON reports translated: {report_success}")
    
    print(f"🔢 Total terms processed: {translator.stats['total_terms']}")
    print(f"🔄 Terms translated: {translator.stats['translated_terms']}")
    print(f"🇺🇸 Already in English: {translator.stats['english_terms']}")
    print(f"💄 Beauty-relevant terms: {translator.stats['beauty_relevant']}")
    print(f"🌐 Languages detected: {len(translator.stats['detected_languages'])}")
    
    if translator.stats['total_terms'] > 0:
        beauty_rate = (translator.stats['beauty_relevant'] / translator.stats['total_terms']) * 100
        translation_rate = (translator.stats['translated_terms'] / translator.stats['total_terms']) * 100
        print(f"📈 Beauty relevance rate: {beauty_rate:.1f}%")
        print(f"🔄 Translation rate: {translation_rate:.1f}%")
    
    print(f"\n📁 Results saved in: {translated_dir}")
    print(f"📋 Detailed report: {report_file}")
    
    print("\n✅ RECOMMENDED NEXT STEPS:")
    if choice in ["1", "3"]:
        print("   • Check: data/translated_features/emerging_terms_translated.csv")
        print("   • Check: data/translated_features/beauty_emerging_terms_translated.csv")
    if choice in ["2", "3", "4"]:
        print("   • Check: data/interim/*_translated.json files")
    if choice == "4":
        print("   • Check: data/translated_features/*_translated.parquet files")

if __name__ == "__main__":
    main()
