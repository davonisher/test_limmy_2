import asyncio
import pandas as pd
import httpx
import logging
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os

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
    """
    Simple Ollama client for local LLM inference using structured outputs.
    Assumes Ollama is running locally on port 11434.
    """
    def __init__(self, model="llama3.3:latest"):
        self.base_url = "http://localhost:11434"
        self.model = model

    async def extract_affiliation_info(self, name, affiliation, client=None):
        """
        Extract company names and universities from affiliation text using structured outputs.
        Returns an AffiliationExtraction object.
        """
        # Skip empty affiliations
        if not affiliation or affiliation.strip() == '':
            return AffiliationExtraction(
                companies=[],
                universities=[],
                primary_affiliation="No affiliation provided",
                reasoning="Empty affiliation field"
            )
            
        # Create a detailed prompt
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
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "format": AffiliationExtraction.model_json_schema(),
            "options": {
                "temperature": 0.1,  # Low temperature for more consistent extraction
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
                extraction = AffiliationExtraction.model_validate_json(content)
                return extraction
            else:
                raise ValueError("No content in response")
                
        except Exception as e:
            logger.warning(f"Ollama extraction failed for {name} with affiliation '{affiliation}': {e}")
            # Fallback to basic extraction
            return self._fallback_extraction(name, affiliation)

    def _fallback_extraction(self, name, affiliation):
        """Fallback extraction when Ollama fails"""
        affiliation_lower = affiliation.lower()
        
        # Common company keywords
        company_keywords = [
            'google', 'microsoft', 'openai', 'deepmind', 'meta', 'facebook', 'apple', 'amazon',
            'nvidia', 'intel', 'ibm', 'anthropic', 'cohere', 'hugging face', 'stability ai',
            'waymo', 'tesla', 'uber', 'lyft', 'airbnb', 'netflix', 'spotify', 'twitter',
            'linkedin', 'salesforce', 'oracle', 'adobe', 'autodesk', 'cisco', 'dell',
            'research', 'lab', 'labs', 'inc', 'corp', 'llc', 'startup', 'foundation'
        ]
        
        # Common university keywords
        university_keywords = [
            'university', 'college', 'institute', 'school', 'academy', 'polytechnic',
            'mit', 'stanford', 'berkeley', 'harvard', 'yale', 'princeton', 'columbia',
            'cornell', 'penn', 'brown', 'dartmouth', 'duke', 'northwestern', 'chicago',
            'caltech', 'cmu', 'gatech', 'georgia tech', 'ucla', 'ucsd', 'ucsb', 'ucdavis',
            'uc irvine', 'uc riverside', 'uc merced', 'uc santa cruz', 'uc santa barbara',
            'oxford', 'cambridge', 'imperial', 'ucl', 'kcl', 'lse', 'warwick', 'bristol',
            'manchester', 'edinburgh', 'glasgow', 'birmingham', 'leeds', 'sheffield',
            'toronto', 'waterloo', 'mcgill', 'ubc', 'montreal', 'alberta', 'calgary'
        ]
        
        companies = []
        universities = []
        
        # Simple keyword matching
        for keyword in company_keywords:
            if keyword in affiliation_lower:
                # Extract the actual company name (this is simplified)
                companies.append(keyword.title())
        
        for keyword in university_keywords:
            if keyword in affiliation_lower:
                # Extract the actual university name (this is simplified)
                universities.append(keyword.title())
        
        return AffiliationExtraction(
            companies=list(set(companies)),
            universities=list(set(universities)),
            primary_affiliation=affiliation,
            reasoning="Fallback keyword matching"
        )

class ScientistAffiliationExtractor:
    def __init__(self, csv_path, model="llama3.3:latest"):
        self.csv_path = csv_path
        self.ollama = OllamaClient(model=model)
        self.results = []
    
    def load_data(self):
        """Load the scientists dataset"""
        try:
            df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(df)} records from {self.csv_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return None
    
    async def extract_affiliations(self, df):
        """
        Extract company and university information from all scientists using Ollama.
        """
        logger.info(f"Starting affiliation extraction for {len(df)} scientists...")
        
        # Use a single httpx.AsyncClient for efficiency
        async with httpx.AsyncClient(timeout=30) as client:
            # Process in smaller batches to avoid overwhelming Ollama
            batch_size = 20
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1} ({len(batch_df)} scientists)")
                
                tasks = []
                for _, row in batch_df.iterrows():
                    name = row['name']
                    affiliation = row['affiliation']
                    tasks.append(self.ollama.extract_affiliation_info(name, affiliation, client=client))
                
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.warning(f"Extraction failed for scientist {batch_df.iloc[j]['name']}: {result}")
                        # Add empty result
                        self.results.append({
                            'name': batch_df.iloc[j]['name'],
                            'original_affiliation': batch_df.iloc[j]['affiliation'],
                            'companies': [],
                            'universities': [],
                            'primary_affiliation': 'Extraction failed',
                            'reasoning': str(result)
                        })
                    else:
                        # Add successful result
                        self.results.append({
                            'name': batch_df.iloc[j]['name'],
                            'original_affiliation': batch_df.iloc[j]['affiliation'],
                            'companies': result.companies,
                            'universities': result.universities,
                            'primary_affiliation': result.primary_affiliation,
                            'reasoning': result.reasoning
                        })
                
                # Small delay between batches
                await asyncio.sleep(1)
        
        logger.info(f"Completed affiliation extraction for {len(self.results)} scientists")
        return self.results
    
    def save_results(self, results, output_filename=None):
        """Save results to CSV file"""
        if not results:
            logger.warning("No results to save")
            return
        
        if output_filename is None:
            today_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f'scientists_affiliations_extracted_{today_str}.csv'
        
        # Convert to DataFrame
        df_results = pd.DataFrame(results)
        
        # Save to CSV
        df_results.to_csv(output_filename, index=False, encoding='utf-8')
        logger.info(f"Results saved to {output_filename}")
        
        # Print summary statistics
        total_companies = sum(len(r['companies']) for r in results)
        total_universities = sum(len(r['universities']) for r in results)
        
        logger.info(f"Summary:")
        logger.info(f"- Total scientists processed: {len(results)}")
        logger.info(f"- Total companies extracted: {total_companies}")
        logger.info(f"- Total universities extracted: {total_universities}")
        logger.info(f"- Average companies per scientist: {total_companies/len(results):.2f}")
        logger.info(f"- Average universities per scientist: {total_universities/len(results):.2f}")
        
        return output_filename

async def main():
    # Initialize extractor
    csv_path = 'ai_scientists_multilabel_20250717_122610.csv'
    extractor = ScientistAffiliationExtractor(csv_path, model="llama3.3:latest")
    
    # Load data
    df = extractor.load_data()
    if df is None:
        logger.error("Failed to load data")
        return
    
    # Extract affiliations
    results = await extractor.extract_affiliations(df)
    
    # Save results
    output_file = extractor.save_results(results)
    
    logger.info(f"Affiliation extraction completed. Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main()) 