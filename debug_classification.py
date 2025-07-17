#!/usr/bin/env python3
"""
Debug script to test article classification
"""

import asyncio
import httpx
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, model="llama3.2"):
        self.base_url = "http://localhost:11434"
        self.model = model

    async def classify_title(self, title, client=None):
        """Classify article title with detailed logging"""
        if len(title.strip()) < 10:
            return "news article"
            
        prompt = (
            f"You are a news classifier. Classify this article title into exactly one of these three categories:\n"
            f"- 'feature update': for new features, product releases, updates, launches\n"
            f"- 'funding': for funding rounds, investments, fundraising, venture capital\n"
            f"- 'news article': for general news, partnerships, research, analysis\n\n"
            f"Title: \"{title}\"\n"
            f"Category (respond with only the category name):"
        )
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9
            }
        }
        
        try:
            if client is None:
                async with httpx.AsyncClient(timeout=30) as ac:
                    response = await ac.post(f"{self.base_url}/api/generate", json=data)
            else:
                response = await client.post(f"{self.base_url}/api/generate", json=data)
            
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "").strip()
            
            logger.info(f"Title: '{title}'")
            logger.info(f"Raw response: '{answer}'")
            
            # Parse the answer
            answer_lower = answer.lower().strip()
            
            # Check for exact matches first
            if answer_lower in ["feature update", "feature", "update"]:
                category = "feature update"
            elif answer_lower in ["funding", "fund"]:
                category = "funding"
            elif answer_lower in ["news article", "news", "article"]:
                category = "news article"
            # Check for funding-related keywords
            elif any(keyword in answer_lower for keyword in ["funding", "fundraise", "fund raise", "investment", "raise", "funded", "series", "round", "venture", "capital"]):
                category = "funding"
            # Check for feature update keywords
            elif any(keyword in answer_lower for keyword in ["feature", "update", "release", "launch", "new feature", "announces", "introduces"]):
                category = "feature update"
            else:
                category = "news article"
            
            logger.info(f"Parsed category: {category}")
            logger.info("-" * 50)
            
            return category
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return "news article"

async def test_classification():
    """Test classification with various sample titles"""
    ollama = OllamaClient(model="llama3.2")
    
    test_titles = [
        "OpenAI launches new GPT-5 model with enhanced capabilities",
        "Anthropic raises $2 billion in Series C funding round",
        "Google announces partnership with Microsoft for AI research",
        "ChatGPT introduces new voice feature for mobile users",
        "Startup receives $50 million investment for AI development",
        "Microsoft releases Windows 11 update with AI features",
        "Tesla announces new autonomous driving features",
        "Apple partners with OpenAI for iOS 18 integration",
        "Meta invests $10 billion in AI infrastructure",
        "Amazon launches new AI-powered shopping assistant"
    ]
    
    logger.info("Testing classification with llama3.2 model...")
    
    async with httpx.AsyncClient(timeout=30) as client:
        for title in test_titles:
            category = await ollama.classify_title(title, client=client)
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(test_classification()) 