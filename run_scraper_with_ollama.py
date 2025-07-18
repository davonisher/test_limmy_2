#!/usr/bin/env python3
"""
Script to run the Bing scraper with Ollama classification
"""

import asyncio
import logging
import sys
from bing_scraper_gpu_v4 import GPUAcceleratedScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

async def main():
    """Main function to run the scraper"""
    try:
        logger.info("Starting Bing scraper with Ollama classification...")
        
        # Initialize scraper
        scraper = GPUAcceleratedScraper(max_concurrent=20, gpu_batch_size=1000)
        
        # Test Ollama connection first
        logger.info(f"Testing Ollama connection with model: {scraper.ollama.model}")
        test_title = "OpenAI launches new GPT-5 model"
        test_tool_name = "OpenAI"
        test_category = await scraper.ollama.classify_title(test_title, test_tool_name)
        logger.info(f"Ollama test successful: '{test_title}' -> {test_category.category} (relevant: {test_category.company_relevance})")
        
        # Run the full scraper
        results = await scraper.scrape_all_tools()
        
        logger.info(f"Scraping completed successfully! Total articles: {len(results)}")
        
        # Show some statistics
        if results:
            df = scraper.process_data_gpu(results)
            if 'ollama_category' in df.columns:
                category_counts = df['ollama_category'].value_counts()
                logger.info("Article categories:")
                for category, count in category_counts.items():
                    logger.info(f"  {category}: {count}")
        
    except Exception as e:
        logger.error(f"Error running scraper: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 