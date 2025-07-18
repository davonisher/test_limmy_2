import asyncio
import httpx
import json
from extract_companies_universities import OllamaClient, AffiliationExtraction

async def debug_extraction():
    """Debug the extraction process with a few sample cases"""
    
    # Test cases that were failing
    test_cases = [
        ("Jitendra MALIK", "Professor of EECS, UC Berkeley"),
        ("Thomas Brox", "University of Freiburg"),
        ("Sergey Levine", "UC Berkeley, Physical Intelligence"),
        ("Sebastian Thrun", "Professor of Computer Science, Stanford"),
        ("Peter E. Hart", "Consultant"),
        ("Dragomir Anguelov", "VP and Head of Research, Waymo"),
        ("Yoshua Bengio", "Professor of computer science, University of Montreal, Mila, IVADO, CIFAR"),
        ("Geoffrey Hinton", "Emeritus Prof. Computer Science, University of Toronto"),
        ("Ilya Sutskever", "Co-Founder and Chief Scientist at Safe Superintelligence Inc"),
        ("Ross Girshick", "Co-Founder at Vercept"),
        ("Robert Tibshirani", "Professor of Biomedical Data Sciences, and of Statistics, Stanford University"),
        ("Yann LeCun", "Chief AI Scientist at Facebook & JT Schwarz Professor at the Courant Institute, New York University"),
        ("Trevor Hastie", "Professor of Statistics, Stanford University")
    ]
    
    ollama = OllamaClient()
    
    async with httpx.AsyncClient(timeout=30) as client:
        for name, affiliation in test_cases:
            print(f"\n{'='*60}")
            print(f"Testing: {name}")
            print(f"Affiliation: {affiliation}")
            print(f"{'='*60}")
            
            try:
                result = await ollama.extract_affiliation_info(name, affiliation, client=client)
                print(f"SUCCESS:")
                print(f"  Companies: {result.companies}")
                print(f"  Universities: {result.universities}")
                print(f"  Primary: {result.primary_affiliation}")
                print(f"  Reasoning: {result.reasoning}")
            except Exception as e:
                print(f"FAILED: {e}")
                print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    asyncio.run(debug_extraction()) 