import asyncio
import httpx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Simple Ollama client for local LLM inference.
    Assumes Ollama is running locally on port 11434.
    """
    def __init__(self, model="qwen2.5-coder:latest"):
        self.base_url = "http://localhost:11435"
        self.model = model

    async def classify_title(self, title, client=None):
        """
        Classify the article title as 'feature update', 'funding', or 'news article'.
        Returns one of: 'feature update', 'funding', 'news article'
        """
        # Skip very short titles
        if len(title.strip()) < 10:
            return "news article"
            
        prompt = (
            f"Classify this article title into exactly one category: 'feature update', 'funding', or 'news article'. "
            f"Respond with only the category name, nothing else.\n\n"
            f"Title: \"{title}\"\n"
            f"Category:"
        )
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for more consistent classification
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
            # The response is expected to have a 'response' field with the model's answer
            answer = result.get("response", "").strip().lower()
            
            # More robust answer parsing
            if any(word in answer for word in ["feature", "update", "release", "launch", "new"]):
                return "feature update"
            elif any(word in answer for word in ["funding", "fundraise", "fund raise", "investment", "raise", "funded"]):
                return "funding"
            else:
                return "news article"
        except Exception as e:
            logger.warning(f"Ollama classification failed for title '{title}': {e}")
            return "news article"

async def test_ollama():
    """Test Ollama classification with sample titles"""
    ollama = OllamaClient(model="gemma3n")
    
    test_titles = [
        "OpenAI launches new GPT-5 model with enhanced capabilities",
        "Anthropic raises $2 billion in Series C funding round",
        "Google announces partnership with Microsoft for AI research",
        "ChatGPT introduces new voice feature for mobile users",
        "Startup receives $50 million investment for AI development"
    ]
    
    logger.info("Testing Ollama classification with gemma3n model...")
    
    async with httpx.AsyncClient(timeout=30) as client:
        for title in test_titles:
            category = await ollama.classify_title(title, client=client)
            logger.info(f"Title: '{title}' -> Category: {category}")
            await asyncio.sleep(1)  # Small delay between requests

if __name__ == "__main__":
    asyncio.run(test_ollama()) 