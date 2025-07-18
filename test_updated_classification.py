#!/usr/bin/env python3
"""
Test script to verify the updated classification system works correctly
"""

import asyncio
import logging
from bing_scraper_gpu_v4 import OllamaClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_updated_classification():
    """Test the updated classification system"""
    
    ollama = OllamaClient(model="llama3.3:latest")
    
    # Test cases
    test_cases = [
        ("OpenAI", "OpenAI launches new GPT-5 model with enhanced capabilities"),
        ("Tricentis", "GTCR Makes $1.33 Billion Investment in Tricentis"),
        ("Intercom", "Top 7 business access control systems companies should consider"),
        ("Pond5", "'Bird Box' Included Real Footage of a Quebec Tragedy That Killed 47 People"),
    ]
    
    logger.info("Testing updated classification system...")
    
    for tool_name, title in test_cases:
        try:
            result = await ollama.classify_title(title, tool_name)
            logger.info(f"Tool: {tool_name}")
            logger.info(f"Title: {title}")
            logger.info(f"Result: {result.category} (relevant: {result.company_relevance})")
            logger.info(f"Reasoning: {result.reasoning}")
            logger.info("-" * 80)
        except Exception as e:
            logger.error(f"Error testing {tool_name}: {e}")
        
        await asyncio.sleep(1)  # Small delay between requests

if __name__ == "__main__":
    asyncio.run(test_updated_classification()) 