#!/usr/bin/env python3
"""
Test script to verify the fixes for the ASL gloss processing
"""

import asyncio
import re
from vectorizer import Vectorizer


def clean_gloss_token(token):
    """
    Clean a gloss token by removing brackets, newlines, and extra whitespace
    """
    # Remove brackets and newlines
    cleaned = re.sub(r'[\[\]\n\r]', '', token)
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned.lower()


def test_gloss_parsing():
    """Test the gloss parsing functionality"""
    # Sample gloss output from the notebook
    sample_gloss = ("ASL [BEAR] [NAME] [OSKI] [LOVE] [HONEY] [BUT] [ALWAYS] "
                    "[GET-STUCK] [TREE]\n\n[ONE_DAY] [HE] [DISCOVER] [LADDER]\n\n"
                    "[PROBLEM] [SOLVE] [FINISH]")
    
    print("Original gloss:")
    print(sample_gloss)
    print("\n" + "="*50 + "\n")
    
    # Split by spaces and clean each token
    gloss_tokens = sample_gloss.split()
    cleaned_tokens = []
    
    for token in gloss_tokens:
        cleaned = clean_gloss_token(token)
        if cleaned:  # Only add non-empty tokens
            cleaned_tokens.append(cleaned)
    
    print("Cleaned tokens:")
    for i, token in enumerate(cleaned_tokens):
        print(f"{i+1:2d}. {token}")
    
    return cleaned_tokens


async def test_vectorizer():
    """Test the vectorizer functionality"""
    try:
        vectorizer = Vectorizer()
        
        # Test with a simple word that should be in the vocabulary
        test_words = ["BEAR", "LOVE", "TREE", "HE", "FINISH"]
        
        for word in test_words:
            print(f"\nTesting word: {word}")
            result = await vectorizer.vector_query_from_supabase(word)
            print(f"Result: {result}")
            
    except Exception as e:
        print(f"Error testing vectorizer: {e}")


async def main():
    """Main test function"""
    print("Testing ASL Gloss Processing Fixes")
    print("=" * 50)
    
    # Test 1: Gloss parsing
    print("\n1. Testing gloss parsing...")
    cleaned_tokens = test_gloss_parsing()
    print(f"Total cleaned tokens: {len(cleaned_tokens)}")
    
    # Test 2: Vectorizer (if environment is set up)
    print("\n2. Testing vectorizer...")
    await test_vectorizer()
    
    print("\n" + "=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    asyncio.run(main()) 