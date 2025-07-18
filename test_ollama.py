import asyncio
import httpx
import logging
from pydantic import BaseModel
from typing import Literal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Structured output schema for classification
class ArticleClassification(BaseModel):
    category: Literal["feature update", "funding", "news article", "irrelevant"]
    company_relevance: bool
    reasoning: str

class OllamaClient:
    """
    Simple Ollama client for local LLM inference using structured outputs.
    Assumes Ollama is running locally on port 11434.
    """
    def __init__(self, model="llama3.2:latest"):
        self.base_url = "http://localhost:11434"
        self.model = model

    async def classify_title(self, title, tool_name, client=None):
        """
        Classify the article title using structured outputs.
        Returns an ArticleClassification object.
        """
        # Skip very short titles
        if len(title.strip()) < 10:
            return ArticleClassification(
                category="news article",
                company_relevance=False,
                reasoning="Title too short"
            )
            
        # Create a detailed prompt
        prompt = f"""You are an expert news classifier. Analyze this article title and determine:

1. CATEGORY: Classify into exactly one category:
   - "feature update": ONLY if the article is about the tool's new features, product releases, updates, launches, or announcements
   - "funding": ONLY if the article is about the tool's funding rounds, investments, fundraising, venture capital, or financial news
   - "news article": for general news, partnerships, research, analysis, or other tool-related news
   - "irrelevant": if the article is NOT about the tool at all

2. COMPANY RELEVANCE: Determine if the article is actually about the tool being searched for.

TOOL BEING SEARCHED: {tool_name}
ARTICLE TITLE: "{title}"

IMPORTANT RULES:
- Only classify as "feature update" or "funding" if the article is DIRECTLY about {tool_name}
- If the article mentions {tool_name} but is not about their features/funding, use "news article"
- If the article doesn't mention {tool_name} at all, use "irrelevant"
- Be very strict about tool relevance

Return as JSON with the specified structure."""

        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "format": ArticleClassification.model_json_schema(),
            "options": {
                "temperature": 0.1,  # Low temperature for more consistent classification
                "top_p": 0.9
            }
        }
        
        try:
            if client is None:
                async with httpx.AsyncClient(timeout=30) as ac:
                    response = await ac.post(f"{self.base_url}/api/chat", json=data)
            else:
                response = await client.post(f"{self.base_url}/api/chat", json=data)
            response.raise_for_status()
            result = response.json()
            
            # Parse the structured response
            content = result.get("message", {}).get("content", "")
            if content:
                classification = ArticleClassification.model_validate_json(content)
                return classification
            else:
                raise ValueError("No content in response")
                
        except Exception as e:
            logger.warning(f"Ollama classification failed for title '{title}': {e}")
            # Fallback to basic classification
            tool_name_lower = tool_name.lower()
            title_lower = title.lower()
            company_relevance = tool_name_lower in title_lower
            
            if not company_relevance:
                return ArticleClassification(
                    category="irrelevant",
                    company_relevance=False,
                    reasoning="Tool name not found in title"
                )
            elif any(word in title_lower for word in ["funding", "fund", "investment", "raise", "series", "round"]):
                return ArticleClassification(
                    category="funding",
                    company_relevance=True,
                    reasoning="Funding-related keywords found"
                )
            elif any(word in title_lower for word in ["feature", "update", "release", "launch", "announce"]):
                return ArticleClassification(
                    category="feature update",
                    company_relevance=True,
                    reasoning="Feature update keywords found"
                )
            else:
                return ArticleClassification(
                    category="news article",
                    company_relevance=True,
                    reasoning="General tool news"
                )

async def test_ollama():
    """Test Ollama classification with sample titles"""
    ollama = OllamaClient(model="llama3.2:latest")
    
    test_cases = [
        ("OpenAI", "OpenAI launches new GPT-5 model with enhanced capabilities"),
        ("Anthropic", "Anthropic raises $2 billion in Series C funding round"),
        ("Google", "Google announces partnership with Microsoft for AI research"),
        ("ChatGPT", "ChatGPT introduces new voice feature for mobile users"),
        ("Startup", "Startup receives $50 million investment for AI development")
    ]
    
    logger.info("Testing Ollama classification with llama3.2 model...")
    
    async with httpx.AsyncClient(timeout=30) as client:
        for tool_name, title in test_cases:
            result = await ollama.classify_title(title, tool_name, client=client)
            logger.info(f"Title: '{title}' -> Category: {result.category} (relevant: {result.company_relevance})")
            await asyncio.sleep(1)  # Small delay between requests

if __name__ == "__main__":
    asyncio.run(test_ollama()) 