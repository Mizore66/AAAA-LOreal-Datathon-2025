#!/usr/bin/env python3
"""
🎭 L'Oréal Datathon 2025 - Model Loading and Usage Demo
======================================================

Demonstrates how to load and use the saved models for:
1. Term trend analysis with TF-IDF vectorizers
2. Category classification
3. Sentiment analysis
4. Semantic similarity validation
"""

import sys
sys.path.append('src')

import pickle
import json
import pandas as pd
from pathlib import Path
from modeling_optimized import (
    CategoryClassifier, 
    SentimentAnalyzer,
    SemanticValidator
)

def load_saved_models():
    """Load all previously saved models."""
    
    print("🎭 L'Oréal Datathon 2025 - Model Demo")
    print("=" * 50)
    
    # Load model registry
    registry_path = "models/loreal_datathon_20250911_091826_enhanced_model_registry.json"
    
    if not Path(registry_path).exists():
        print("❌ Model registry not found. Run enhanced modeling pipeline first.")
        return None
    
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    saved_models = registry['saved_models']
    models = {}
    
    print(f"📦 Loading models created at: {registry['created_at']}")
    print()
    
    # Load TF-IDF vectorizers
    if 'tfidf_vectorizers' in saved_models:
        try:
            with open(saved_models['tfidf_vectorizers'], 'rb') as f:
                models['tfidf_vectorizers'] = pickle.load(f)
            print(f"✅ Loaded TF-IDF vectorizers for {len(models['tfidf_vectorizers'])} categories")
        except Exception as e:
            print(f"⚠️ Could not load TF-IDF vectorizers: {e}")
    
    # Load category classifier
    if 'category_classifier' in saved_models:
        try:
            with open(saved_models['category_classifier'], 'rb') as f:
                models['category_classifier'] = pickle.load(f)
            print("✅ Loaded category classifier model")
        except Exception as e:
            print(f"⚠️ Could not load category classifier: {e}")
    
    # Initialize sentiment analyzer (pre-trained model)
    try:
        models['sentiment_analyzer'] = SentimentAnalyzer()
        print("✅ Initialized sentiment analyzer")
    except Exception as e:
        print(f"⚠️ Could not initialize sentiment analyzer: {e}")
    
    # Initialize semantic validator (pre-trained model)
    try:
        models['semantic_validator'] = SemanticValidator()
        print("✅ Initialized semantic validator")
    except Exception as e:
        print(f"⚠️ Could not initialize semantic validator: {e}")
    
    return models

def demo_trend_analysis(models):
    """Demonstrate term trend analysis using TF-IDF."""
    
    print("\n🔥 TERM TREND ANALYSIS DEMO")
    print("=" * 50)
    
    if 'tfidf_vectorizers' not in models:
        print("❌ TF-IDF vectorizers not available")
        return
    
    # Sample beauty texts for analysis
    sample_texts = [
        "I love this new foundation! Perfect coverage and long-lasting wear.",
        "My skincare routine includes vitamin C serum and retinol moisturizer.",
        "This curly hair styling cream gives amazing definition and shine.",
        "Exploring sustainable fashion brands for eco-friendly clothing options.",
        "Natural beauty trends are focusing on minimal makeup and glowing skin."
    ]
    
    print("📝 Analyzing sample beauty content...")
    print()
    
    # Analyze each text with available vectorizers
    for category, vectorizer in models['tfidf_vectorizers'].items():
        print(f"🏷️ {category.title()} Analysis:")
        
        try:
            # Transform text and get feature scores
            tfidf_matrix = vectorizer.transform(sample_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get mean scores across all texts
            mean_scores = tfidf_matrix.mean(axis=0).A1
            
            # Get top 5 terms for this category
            top_indices = mean_scores.argsort()[-5:][::-1]
            
            for i, idx in enumerate(top_indices, 1):
                if mean_scores[idx] > 0:
                    print(f"   {i}. {feature_names[idx]} (score: {mean_scores[idx]:.3f})")
            
        except Exception as e:
            print(f"   ⚠️ Analysis failed: {e}")
        
        print()

def demo_sentiment_analysis(models):
    """Demonstrate sentiment analysis."""
    
    print("😊 SENTIMENT ANALYSIS DEMO")
    print("=" * 50)
    
    if 'sentiment_analyzer' not in models:
        print("❌ Sentiment analyzer not available")
        return
    
    analyzer = models['sentiment_analyzer']
    
    # Sample texts with different sentiments
    test_texts = [
        "This makeup tutorial is absolutely amazing! Love the final look!",
        "The product didn't work as expected. Not worth the money.",
        "Simple everyday makeup routine using basic products.",
        "Terrible customer service experience. Very disappointed.",
        "Beautiful natural makeup that enhances your features perfectly!"
    ]
    
    print("🎭 Analyzing sentiment for beauty-related content:")
    print()
    
    for i, text in enumerate(test_texts, 1):
        try:
            sentiment = analyzer.analyze_sentiment(text)
            print(f"{i}. \"{text[:50]}...\"")
            print(f"   Sentiment: {sentiment}")
            print()
        except Exception as e:
            print(f"   ⚠️ Sentiment analysis failed: {e}")

def demo_semantic_similarity(models):
    """Demonstrate semantic similarity validation."""
    
    print("🔍 SEMANTIC SIMILARITY DEMO")
    print("=" * 50)
    
    if 'semantic_validator' not in models:
        print("❌ Semantic validator not available")
        return
    
    validator = models['semantic_validator']
    
    # Test semantic similarity between beauty terms
    term_pairs = [
        ("lipstick", "lip color"),
        ("foundation", "base makeup"),
        ("skincare", "skin health"),
        ("hairstyle", "hair fashion"),
        ("mascara", "eye makeup")
    ]
    
    print("🔗 Calculating semantic similarity for beauty terms:")
    print()
    
    for term1, term2 in term_pairs:
        try:
            # Use the generate_embeddings method and calculate similarity manually
            embeddings = validator.generate_embeddings([term1, term2])
            if embeddings is not None and len(embeddings) == 2:
                # Calculate cosine similarity manually
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                print(f"'{term1}' ↔ '{term2}': {similarity:.3f} similarity")
            else:
                print(f"'{term1}' ↔ '{term2}': Unable to generate embeddings")
        except Exception as e:
            print(f"⚠️ Similarity calculation failed for '{term1}' ↔ '{term2}': {e}")
    
    print()

def demo_model_usage():
    """Complete demonstration of saved model usage."""
    
    # Load all models
    models = load_saved_models()
    
    if not models:
        return
    
    # Run demonstrations
    demo_trend_analysis(models)
    demo_sentiment_analysis(models)
    demo_semantic_similarity(models)
    
    print("🎉 MODEL DEMO COMPLETE!")
    print("=" * 50)
    print("💡 Key Capabilities:")
    print("   • TF-IDF term trend analysis across beauty categories")
    print("   • Real-time sentiment analysis for user content") 
    print("   • Semantic similarity for content recommendation")
    print("   • Category classification for content organization")
    print()
    print("🚀 Your models are ready for production use!")

if __name__ == "__main__":
    demo_model_usage()
