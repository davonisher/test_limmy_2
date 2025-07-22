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

Return ONLY a valid JSON object with this exact structure:
{{
  "companies": ["company1", "company2"],
  "universities": ["university1", "university2"],
  "primary_affiliation": "main affiliation",
  "reasoning": "explanation"
}}"""

        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for more consistent extraction
                "top_p": 0.9
            }
        }
        
        # Add retry logic for failed requests
        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                if client is None:
                    async with httpx.AsyncClient(timeout=60) as ac:
                        response = await ac.post(f"{self.base_url}/api/chat", json=data)
                else:
                    response = await client.post(f"{self.base_url}/api/chat", json=data)
                    
                response.raise_for_status()
                result = response.json()
                
                # Parse the structured response
                content = result.get("message", {}).get("content", "")
                if content:
                    # Try to extract JSON from the response
                    try:
                        # First try direct parsing
                        extraction = AffiliationExtraction.model_validate_json(content)
                        return extraction
                    except Exception as json_error:
                        # Try to extract JSON from the text if it's wrapped in markdown
                        import re
                        import json
                        
                        # Look for JSON code blocks
                        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                        if json_match:
                            try:
                                json_str = json_match.group(1)
                                extraction = AffiliationExtraction.model_validate_json(json_str)
                                return extraction
                            except Exception:
                                pass
                        
                        # Try to find JSON object in the text
                        json_match = re.search(r'\{[^{}]*"companies"[^{}]*"universities"[^{}]*\}', content, re.DOTALL)
                        if json_match:
                            try:
                                json_str = json_match.group(0)
                                extraction = AffiliationExtraction.model_validate_json(json_str)
                                return extraction
                            except Exception:
                                pass
                        
                        # Log the actual response for debugging
                        logger.debug(f"Failed to parse JSON for {name}. Response: {content[:200]}...")
                        logger.debug(f"JSON parsing error: {json_error}")
                        
                        # Try manual extraction as last resort
                        return self._manual_extraction(name, affiliation, content)
                else:
                    raise ValueError("No content in response")
                    
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.debug(f"Attempt {attempt + 1} failed for {name}, retrying... Error: {e}")
                    await asyncio.sleep(1)  # Wait before retry
                    continue
                else:
                    # All attempts failed
                    logger.warning(f"Ollama extraction failed for {name} with affiliation '{affiliation}' after {max_retries + 1} attempts: {e}")
                    break
        
        # Return empty result when all attempts fail
        return AffiliationExtraction(
            companies=[],
            universities=[],
            primary_affiliation="LLM extraction failed",
            reasoning=f"Ollama request failed after {max_retries + 1} attempts: {str(last_error)}"
        )

    def _manual_extraction(self, name, affiliation, content):
        """Manual extraction when JSON parsing fails but we have content"""
        try:
            # Try to extract information from the content using regex
            import re
            
            companies = []
            universities = []
            primary_affiliation = affiliation
            reasoning = "Manual extraction from failed JSON response"
            
            # Look for company mentions
            company_patterns = [
                r'companies?["\s]*:["\s]*\[([^\]]*)\]',
                r'COMPANIES?["\s]*:["\s]*\[([^\]]*)\]',
                r'company["\s]*:["\s]*\[([^\]]*)\]'
            ]
            
            for pattern in company_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    companies_str = match.group(1)
                    # Extract quoted strings
                    company_matches = re.findall(r'"([^"]*)"', companies_str)
                    companies.extend(company_matches)
                    break
            
            # Look for university mentions
            university_patterns = [
                r'universities?["\s]*:["\s]*\[([^\]]*)\]',
                r'UNIVERSITIES?["\s]*:["\s]*\[([^\]]*)\]',
                r'university["\s]*:["\s]*\[([^\]]*)\]'
            ]
            
            for pattern in university_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    universities_str = match.group(1)
                    # Extract quoted strings
                    university_matches = re.findall(r'"([^"]*)"', universities_str)
                    universities.extend(university_matches)
                    break
            
            # Look for primary affiliation
            primary_patterns = [
                r'primary["\s]*:["\s]*"([^"]*)"',
                r'PRIMARY["\s]*:["\s]*"([^"]*)"'
            ]
            
            for pattern in primary_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    primary_affiliation = match.group(1)
                    break
            
            return AffiliationExtraction(
                companies=list(set(companies)),
                universities=list(set(universities)),
                primary_affiliation=primary_affiliation,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.debug(f"Manual extraction also failed for {name}: {e}")
            # Return empty result when manual extraction fails
            return AffiliationExtraction(
                companies=[],
                universities=[],
                primary_affiliation="Manual extraction failed",
                reasoning=f"Manual extraction from LLM response failed: {str(e)}"
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
        Saves progress every 50 scientists.
        """
        logger.info(f"Starting affiliation extraction for {len(df)} scientists...")
        
        # Use a single httpx.AsyncClient for efficiency
        async with httpx.AsyncClient(timeout=60) as client:  # Increased timeout
            # Process in smaller batches to avoid overwhelming Ollama
            batch_size = 20
            save_interval = 50  # Save every 50 scientists
            
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
                        error_msg = str(result) if str(result) else "Unknown error"
                        logger.warning(f"Extraction failed for scientist {batch_df.iloc[j]['name']}: {error_msg}")
                        # Add empty result
                        self.results.append({
                            'name': batch_df.iloc[j]['name'],
                            'original_affiliation': batch_df.iloc[j]['affiliation'],
                            'companies': [],
                            'universities': [],
                            'primary_affiliation': 'Extraction failed',
                            'reasoning': error_msg
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
                
                # Save intermediate results every 50 scientists
                if len(self.results) % save_interval == 0 or len(self.results) >= len(df):
                    today_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                    intermediate_filename = f'scientists_affiliations_intermediate_{len(self.results)}_{today_str}.csv'
                    self.save_results(self.results, intermediate_filename)
                    logger.info(f"Saved intermediate results: {len(self.results)} scientists processed")
                
                # Small delay between batches
                await asyncio.sleep(2)  # Increased delay
        
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