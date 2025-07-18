import asyncio
import pandas as pd
import httpx
import logging
from pydantic import BaseModel
from typing import List
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Structured output schema for extracting company and university information
class AffiliationExtraction(BaseModel):
    companies: List[str]
    universities: List[str]
    primary_affiliation: str
    reasoning: str

class OllamaClient:
    def __init__(self, model="llama3.3:latest"):
        self.base_url = "http://localhost:11434"
        self.model = model

    async def extract_affiliation_info(self, name, affiliation, client=None):
        """Extract company names and universities from affiliation text using structured outputs."""
        if not affiliation or affiliation.strip() == '':
            return AffiliationExtraction(
                companies=[],
                universities=[],
                primary_affiliation="No affiliation provided",
                reasoning="Empty affiliation field"
            )
            
        prompt = f"""You are an expert at extracting company names and university names from academic/professional affiliations.

Analyze this person's affiliation and extract:

1. COMPANIES: List all company names (tech companies, research labs, startups, etc.)
2. UNIVERSITIES: List all university/educational institution names
3. PRIMARY AFFILIATION: The main/current affiliation (usually the first one mentioned)
4. REASONING: Brief explanation of your extraction

PERSON: {name}
AFFILIATION: "{affiliation}"

IMPORTANT RULES:
- Extract actual company names (e.g., "Google", "Microsoft", "OpenAI", "DeepMind")
- Extract actual university names (e.g., "Stanford University", "MIT", "UC Berkeley")
- For universities, include the full name when possible
- For companies, use the official company name
- If someone has multiple affiliations, list them all
- Primary affiliation is usually the first one mentioned or the most prominent

Return as JSON with the specified structure."""

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": AffiliationExtraction.model_json_schema(),
            "options": {"temperature": 0.1, "top_p": 0.9}
        }
        
        try:
            if client is None:
                async with httpx.AsyncClient(timeout=30) as ac:
                    response = await ac.post(f"{self.base_url}/api/chat", json=data)
            else:
                response = await client.post(f"{self.base_url}/api/chat", json=data)
            response.raise_for_status()
            result = response.json()
            
            content = result.get("message", {}).get("content", "")
            if content:
                extraction = AffiliationExtraction.model_validate_json(content)
                return extraction
            else:
                raise ValueError("No content in response")
                
        except Exception as e:
            logger.warning(f"Ollama extraction failed for {name}: {e}")
            return self._fallback_extraction(name, affiliation)

    def _fallback_extraction(self, name, affiliation):
        """Fallback extraction when Ollama fails"""
        affiliation_lower = affiliation.lower()
        
        company_keywords = ['google', 'microsoft', 'openai', 'deepmind', 'meta', 'facebook', 'apple', 'amazon', 'nvidia', 'intel', 'ibm', 'anthropic', 'cohere', 'waymo', 'tesla']
        university_keywords = ['university', 'college', 'institute', 'mit', 'stanford', 'berkeley', 'harvard', 'yale', 'princeton', 'columbia', 'cornell', 'penn', 'caltech', 'cmu', 'gatech', 'ucla', 'ucsd', 'oxford', 'cambridge', 'imperial', 'ucl', 'toronto', 'waterloo', 'mcgill']
        
        companies = [kw.title() for kw in company_keywords if kw in affiliation_lower]
        universities = [kw.title() for kw in university_keywords if kw in affiliation_lower]
        
        return AffiliationExtraction(
            companies=list(set(companies)),
            universities=list(set(universities)),
            primary_affiliation=affiliation,
            reasoning="Fallback keyword matching"
        )

async def test_extraction():
    """Test the extraction with a few sample scientists"""
    
    # Sample data for testing
    test_data = [
        {
            "name": "Yann LeCun",
            "affiliation": "Chief AI Scientist at Facebook & JT Schwarz Professor at the Courant Institute, New York"
        },
        {
            "name": "Pieter Abbeel",
            "affiliation": "UC Berkeley | Covariant"
        },
        {
            "name": "Vincent Vanhoucke",
            "affiliation": "Distinguished Engineer, Waymo"
        },
        {
            "name": "Sergey Levine",
            "affiliation": "UC Berkeley, Physical Intelligence"
        },
        {
            "name": "Ilya Sutskever",
            "affiliation": "Co-Founder and Chief Scientist at Safe Superintelligence Inc"
        }
    ]
    
    ollama = OllamaClient(model="llama3.3:latest")
    results = []
    
    logger.info("Testing affiliation extraction with sample data...")
    
    async with httpx.AsyncClient(timeout=30) as client:
        for scientist in test_data:
            logger.info(f"Processing: {scientist['name']}")
            result = await ollama.extract_affiliation_info(
                scientist['name'], 
                scientist['affiliation'], 
                client=client
            )
            
            results.append({
                'name': scientist['name'],
                'original_affiliation': scientist['affiliation'],
                'companies': result.companies,
                'universities': result.universities,
                'primary_affiliation': result.primary_affiliation,
                'reasoning': result.reasoning
            })
            
            # Print result
            logger.info(f"  Companies: {result.companies}")
            logger.info(f"  Universities: {result.universities}")
            logger.info(f"  Primary: {result.primary_affiliation}")
            logger.info(f"  Reasoning: {result.reasoning}")
            logger.info("---")
    
    # Save test results
    df_results = pd.DataFrame(results)
    output_filename = f'test_extraction_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df_results.to_csv(output_filename, index=False, encoding='utf-8')
    logger.info(f"Test results saved to {output_filename}")
    
    return results

if __name__ == "__main__":
    asyncio.run(test_extraction()) 