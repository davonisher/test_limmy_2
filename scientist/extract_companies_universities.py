import asyncio
import pandas as pd
import httpx
import logging
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os
import json
import re

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
    Optimized Ollama client for high-throughput local LLM inference.
    Assumes Ollama is running locally on port 11434.
    """
    def __init__(self, model="llama3.3:latest"):
        self.base_url = "http://localhost:11434"
        self.model = model

    async def extract_affiliation_info(self, name, affiliation, client=None):
        """
        Extract company names and universities from affiliation text using structured outputs.
        Optimized for speed with simplified prompt and reduced error handling.
        """
        # Skip empty affiliations
        if not affiliation or affiliation.strip() == '':
            return AffiliationExtraction(
                companies=[],
                universities=[],
                primary_affiliation="No affiliation provided",
                reasoning="Empty affiliation field"
            )
            
        # Simplified, more direct prompt for faster processing
        prompt = f"""Extract from this affiliation:
PERSON: {name}
AFFILIATION: "{affiliation}"

Return JSON only:
{{
  "companies": ["list company names"],
  "universities": ["list university names"], 
  "primary_affiliation": "main affiliation",
  "reasoning": "brief explanation"
}}"""

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_ctx": 2048  # Reduced context window for faster processing
            }
        }
        
        try:
            if client is None:
                async with httpx.AsyncClient(timeout=30) as ac:  # Reduced timeout
                    response = await ac.post(f"{self.base_url}/api/chat", json=data)
            else:
                response = await client.post(f"{self.base_url}/api/chat", json=data)
                
            response.raise_for_status()
            result = response.json()
            
            # Parse the structured response
            content = result.get("message", {}).get("content", "")
            if content:
                # Fast JSON extraction - try direct parsing first
                try:
                    extraction = AffiliationExtraction.model_validate_json(content)
                    return extraction
                except Exception:
                    # Quick regex extraction for JSON blocks
                    json_match = re.search(r'\{[^{}]*"companies"[^{}]*"universities"[^{}]*\}', content, re.DOTALL)
                    if json_match:
                        try:
                            extraction = AffiliationExtraction.model_validate_json(json_match.group(0))
                            return extraction
                        except Exception:
                            pass
                    
                    # Quick fallback - basic extraction
                    return self._fast_fallback_extraction(name, affiliation)
            else:
                return self._fast_fallback_extraction(name, affiliation)
                    
        except Exception as e:
            logger.debug(f"Extraction failed for {name}: {e}")
            return self._fast_fallback_extraction(name, affiliation)

    def _fast_fallback_extraction(self, name, affiliation):
        """Fast fallback when LLM extraction fails"""
        return AffiliationExtraction(
            companies=[],
            universities=[],
            primary_affiliation=affiliation,
            reasoning="Fast fallback - LLM extraction failed"
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
        Optimized for high-throughput processing on strong GPU.
        """
        logger.info(f"Starting high-speed affiliation extraction for {len(df)} scientists...")
        
        # Optimized settings for strong GPU
        batch_size = 100  # Increased from 20 to 100
        max_concurrent = 150  # Maximum concurrent requests
        save_interval = 200  # Save every 200 scientists (less I/O overhead)
        
        # Create connection pool with higher limits
        limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
        timeout = httpx.Timeout(30.0)  # Reduced timeout for faster processing
        
        async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
            # Process in larger batches with controlled concurrency
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_single_scientist(row):
                async with semaphore:
                    name = row['name']
                    affiliation = row['affiliation']
                    return await self.ollama.extract_affiliation_info(name, affiliation, client=client)
            
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1} ({len(batch_df)} scientists)")
                
                # Create all tasks for the batch
                tasks = [process_single_scientist(row) for _, row in batch_df.iterrows()]
                
                # Execute all tasks concurrently
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results quickly
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        error_msg = str(result)[:100]  # Truncate long error messages
                        self.results.append({
                            'name': batch_df.iloc[j]['name'],
                            'original_affiliation': batch_df.iloc[j]['affiliation'],
                            'companies': [],
                            'universities': [],
                            'primary_affiliation': 'Extraction failed',
                            'reasoning': error_msg
                        })
                    else:
                        self.results.append({
                            'name': batch_df.iloc[j]['name'],
                            'original_affiliation': batch_df.iloc[j]['affiliation'],
                            'companies': result.companies,
                            'universities': result.universities,
                            'primary_affiliation': result.primary_affiliation,
                            'reasoning': result.reasoning
                        })
                
                # Save intermediate results less frequently
                if len(self.results) % save_interval == 0 or len(self.results) >= len(df):
                    today_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                    intermediate_filename = f'scientists_affiliations_intermediate_{len(self.results)}_{today_str}.csv'
                    self.save_results(self.results, intermediate_filename)
                    logger.info(f"Saved intermediate results: {len(self.results)} scientists processed")
                
                # No sleep delay - let the GPU work at full speed!
        
        logger.info(f"Completed high-speed affiliation extraction for {len(self.results)} scientists")
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
    csv_path = 'ai_scientists_multilabel_20250717_111542.csv'
    extractor = ScientistAffiliationExtractor(csv_path, model="llama3.3:latest")
    
    # Load data
    df = extractor.load_data()
    if df is None:
        logger.error("Failed to load data")
        return
    
    # Extract affiliations at high speed
    results = await extractor.extract_affiliations(df)
    
    # Save results
    output_file = extractor.save_results(results)
    
    logger.info(f"High-speed affiliation extraction completed. Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main()) 