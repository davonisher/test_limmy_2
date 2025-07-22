#!/usr/bin/env python3
"""
Performance test script for the optimized affiliation extractor.
Tests processing speed on a small sample to estimate total time.
"""

import asyncio
import pandas as pd
import time
from extract_companies_universities import ScientistAffiliationExtractor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_performance():
    """Test the performance of the optimized extractor"""
    
    # Load data
    csv_path = 'ai_scientists_multilabel_20250717_111542.csv'
    extractor = ScientistAffiliationExtractor(csv_path, model="llama3.3:latest")
    
    df = extractor.load_data()
    if df is None:
        logger.error("Failed to load data")
        return
    
    # Test on a smaller sample first (50 scientists)
    test_size = 50
    test_df = df.head(test_size)
    
    logger.info(f"🧪 Performance Test: Processing {test_size} scientists...")
    logger.info(f"📊 Total dataset size: {len(df)} scientists")
    
    # Time the extraction
    start_time = time.time()
    results = await extractor.extract_affiliations(test_df)
    end_time = time.time()
    
    # Calculate performance metrics
    elapsed_time = end_time - start_time
    scientists_per_second = test_size / elapsed_time
    estimated_total_time = len(df) / scientists_per_second
    
    # Display results
    logger.info(f"🎯 Performance Results:")
    logger.info(f"   ⏱️  Test time: {elapsed_time:.2f} seconds")
    logger.info(f"   🚀 Speed: {scientists_per_second:.2f} scientists/second")
    logger.info(f"   📈 Estimated total time: {estimated_total_time/60:.1f} minutes")
    logger.info(f"   💫 Expected completion: {estimated_total_time/3600:.1f} hours")
    
    # Accuracy check
    successful_extractions = sum(1 for r in results if r['companies'] or r['universities'])
    success_rate = successful_extractions / len(results) * 100
    logger.info(f"   ✅ Success rate: {success_rate:.1f}%")
    
    # Ask user if they want to proceed with full dataset
    print(f"\n🤔 Proceed with full dataset of {len(df)} scientists?")
    print(f"   Estimated time: {estimated_total_time/60:.1f} minutes")
    response = input("   Continue? (y/N): ").strip().lower()
    
    if response == 'y':
        logger.info("🚀 Starting full dataset processing...")
        start_time = time.time()
        full_results = await extractor.extract_affiliations(df)
        end_time = time.time()
        
        total_time = end_time - start_time
        final_speed = len(df) / total_time
        
        # Save results
        output_file = extractor.save_results(full_results)
        
        logger.info(f"🎉 Full Processing Complete!")
        logger.info(f"   ⏱️  Total time: {total_time/60:.1f} minutes")
        logger.info(f"   🚀 Final speed: {final_speed:.2f} scientists/second")
        logger.info(f"   💾 Results saved to: {output_file}")
    else:
        logger.info("👋 Performance test completed. Run the full script when ready!")

if __name__ == "__main__":
    asyncio.run(test_performance()) 