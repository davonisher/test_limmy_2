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
            logger.warning(f"Ollama extraction failed for {name} with affiliation '{affiliation}': {e}")
            # Fallback to basic extraction
            return self._fallback_extraction(name, affiliation)

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
            return self._fallback_extraction(name, affiliation)

    def _fallback_extraction(self, name, affiliation):
        """Fallback extraction when Ollama fails"""
        affiliation_lower = affiliation.lower()
        
        # Common company keywords with their full names
        company_mappings = {
            'google': 'Google',
            'microsoft': 'Microsoft', 
            'openai': 'OpenAI',
            'deepmind': 'DeepMind',
            'meta': 'Meta',
            'facebook': 'Facebook',
            'apple': 'Apple',
            'amazon': 'Amazon',
            'nvidia': 'NVIDIA',
            'intel': 'Intel',
            'ibm': 'IBM',
            'anthropic': 'Anthropic',
            'cohere': 'Cohere',
            'hugging face': 'Hugging Face',
            'stability ai': 'Stability AI',
            'waymo': 'Waymo',
            'tesla': 'Tesla',
            'uber': 'Uber',
            'lyft': 'Lyft',
            'airbnb': 'Airbnb',
            'netflix': 'Netflix',
            'spotify': 'Spotify',
            'twitter': 'Twitter',
            'linkedin': 'LinkedIn',
            'salesforce': 'Salesforce',
            'oracle': 'Oracle',
            'adobe': 'Adobe',
            'autodesk': 'Autodesk',
            'cisco': 'Cisco',
            'dell': 'Dell',
            'safe superintelligence': 'Safe Superintelligence Inc',
            'covariant': 'Covariant',
            'vercept': 'Vercept',
            'research': 'Research',
            'lab': 'Lab',
            'labs': 'Labs',
            'inc': 'Inc',
            'corp': 'Corp',
            'llc': 'LLC',
            'startup': 'Startup',
            'foundation': 'Foundation'
        }
        
        # Common university keywords with their full names
        university_mappings = {
            'university': 'University',
            'college': 'College',
            'institute': 'Institute',
            'school': 'School',
            'academy': 'Academy',
            'polytechnic': 'Polytechnic',
            'mit': 'MIT',
            'stanford': 'Stanford University',
            'berkeley': 'University of California, Berkeley',
            'uc berkeley': 'University of California, Berkeley',
            'harvard': 'Harvard University',
            'yale': 'Yale University',
            'princeton': 'Princeton University',
            'columbia': 'Columbia University',
            'cornell': 'Cornell University',
            'penn': 'University of Pennsylvania',
            'brown': 'Brown University',
            'dartmouth': 'Dartmouth College',
            'duke': 'Duke University',
            'northwestern': 'Northwestern University',
            'chicago': 'University of Chicago',
            'caltech': 'Caltech',
            'cmu': 'Carnegie Mellon University',
            'gatech': 'Georgia Institute of Technology',
            'georgia tech': 'Georgia Institute of Technology',
            'ucla': 'University of California, Los Angeles',
            'ucsd': 'University of California, San Diego',
            'ucsb': 'University of California, Santa Barbara',
            'ucdavis': 'University of California, Davis',
            'uc irvine': 'University of California, Irvine',
            'uc riverside': 'University of California, Riverside',
            'uc merced': 'University of California, Merced',
            'uc santa cruz': 'University of California, Santa Cruz',
            'uc santa barbara': 'University of California, Santa Barbara',
            'oxford': 'University of Oxford',
            'cambridge': 'University of Cambridge',
            'imperial': 'Imperial College London',
            'ucl': 'University College London',
            'kcl': 'King\'s College London',
            'lse': 'London School of Economics',
            'warwick': 'University of Warwick',
            'bristol': 'University of Bristol',
            'manchester': 'University of Manchester',
            'edinburgh': 'University of Edinburgh',
            'glasgow': 'University of Glasgow',
            'birmingham': 'University of Birmingham',
            'leeds': 'University of Leeds',
            'sheffield': 'University of Sheffield',
            'toronto': 'University of Toronto',
            'waterloo': 'University of Waterloo',
            'mcgill': 'McGill University',
            'ubc': 'University of British Columbia',
            'montreal': 'University of Montreal',
            'alberta': 'University of Alberta',
            'calgary': 'University of Calgary',
            'freiburg': 'University of Freiburg',
            'courant': 'Courant Institute',
            'new york university': 'New York University',
            'nyu': 'New York University'
        }
        
        companies = []
        universities = []
        
        # Extract companies using keyword matching
        for keyword, full_name in company_mappings.items():
            if keyword in affiliation_lower:
                companies.append(full_name)
        
        # Extract universities using keyword matching
        for keyword, full_name in university_mappings.items():
            if keyword in affiliation_lower:
                universities.append(full_name)
        
        # Try to extract more specific patterns
        import re
        
        # Look for patterns like "Professor at X" or "VP at Y"
        professor_patterns = [
            r'professor\s+(?:of\s+)?[^,]*\s+(?:at\s+)?([^,\n]+)',
            r'prof\s+(?:of\s+)?[^,]*\s+(?:at\s+)?([^,\n]+)',
            r'vp\s+(?:of\s+)?[^,]*\s+(?:at\s+)?([^,\n]+)',
            r'head\s+(?:of\s+)?[^,]*\s+(?:at\s+)?([^,\n]+)',
            r'co-founder\s+(?:of\s+)?[^,]*\s+(?:at\s+)?([^,\n]+)',
            r'chief\s+[^,]*\s+(?:at\s+)?([^,\n]+)'
        ]
        
        for pattern in professor_patterns:
            matches = re.findall(pattern, affiliation, re.IGNORECASE)
            for match in matches:
                match = match.strip()
                if match and len(match) > 2:  # Avoid very short matches
                    # Check if it looks like a university
                    if any(uni_keyword in match.lower() for uni_keyword in ['university', 'college', 'institute', 'school']):
                        if match not in universities:
                            universities.append(match)
                    # Check if it looks like a company
                    elif any(comp_keyword in match.lower() for comp_keyword in ['inc', 'corp', 'llc', 'labs', 'research']):
                        if match not in companies:
                            companies.append(match)
        
        # Determine primary affiliation
        primary_affiliation = affiliation
        if companies and universities:
            # If both present, use the first one mentioned
            primary_affiliation = companies[0] if affiliation_lower.find(companies[0].lower()) < affiliation_lower.find(universities[0].lower()) else universities[0]
        elif companies:
            primary_affiliation = companies[0]
        elif universities:
            primary_affiliation = universities[0]
        
        return AffiliationExtraction(
            companies=list(set(companies)),
            universities=list(set(universities)),
            primary_affiliation=primary_affiliation,
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