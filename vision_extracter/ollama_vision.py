import asyncio
import httpx
import logging
import base64
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebsiteDescription(BaseModel):
    """Structured output for website description"""
    description: str
    key_features: list[str]
    target_audience: str
    technology_type: str

class OllamaVisionClient:
    """
    Ollama client for vision-based analysis of AI tool websites.
    Uses vision models to analyze screenshots and generate detailed descriptions.
    """
    def __init__(self, model="llava:7b"):
        self.base_url = "http://localhost:11434"
        self.model = model

    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string for Ollama API.
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise

    async def describe_ai_tool_website(self, image_path: str, client: Optional[httpx.AsyncClient] = None) -> str:
        """
        Generate a detailed 300-500 word description of an AI tool website from a screenshot.
        
        Args:
            image_path: Path to the website screenshot
            client: Optional HTTP client for connection reuse
            
        Returns:
            Detailed description of the AI tool website
        """
        # Verify image exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Encode image
        try:
            image_base64 = self.encode_image(image_path)
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return f"Error: Could not process image at {image_path}"

        # Create detailed prompt for AI tool website analysis
        prompt = """Analyze this screenshot of an AI tool website and provide a comprehensive description of 300-500 words in English. Include the following aspects:

1. **MAIN PURPOSE & FUNCTIONALITY**: What does this AI tool do? What problem does it solve?

2. **KEY FEATURES & CAPABILITIES**: What are the main features visible on the page? What services or functionalities are offered?

3. **USER INTERFACE & DESIGN**: Describe the layout, design elements, color scheme, and overall user experience.

4. **TARGET AUDIENCE**: Who is this tool designed for? (developers, businesses, consumers, etc.)

5. **PRICING & ACCESSIBILITY**: Any visible pricing information, free tiers, or accessibility options.

6. **TECHNOLOGY INDICATORS**: What type of AI technology appears to be used? (computer vision, NLP, generative AI, etc.)

7. **COMPETITIVE POSITIONING**: How does this tool position itself in the market?

Please write in a professional, informative tone suitable for a business analysis or research report. Focus on concrete details visible in the screenshot rather than assumptions."""

        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_base64]
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.3,  # Moderate creativity for descriptive text
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        try:
            if client is None:
                async with httpx.AsyncClient(timeout=60) as ac:  # Longer timeout for vision
                    response = await ac.post(f"{self.base_url}/api/chat", json=data)
            else:
                response = await client.post(f"{self.base_url}/api/chat", json=data)
            
            response.raise_for_status()
            result = response.json()
            
            content = result.get("message", {}).get("content", "")
            if content:
                return content.strip()
            else:
                return "Error: No content received from Ollama"
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Ollama: {e.response.status_code} - {e.response.text}")
            return f"Error: HTTP {e.response.status_code} from Ollama API"
        except httpx.RequestError as e:
            logger.error(f"Request error to Ollama: {e}")
            return "Error: Could not connect to Ollama. Make sure it's running on localhost:11434"
        except Exception as e:
            logger.error(f"Unexpected error during vision analysis: {e}")
            return f"Error: {str(e)}"

    async def analyze_website_batch(self, image_paths: list[str]) -> dict[str, str]:
        """
        Analyze multiple website screenshots in batch.
        
        Args:
            image_paths: List of paths to website screenshots
            
        Returns:
            Dictionary mapping image paths to descriptions
        """
        results = {}
        
        async with httpx.AsyncClient(timeout=60) as client:
            for image_path in image_paths:
                logger.info(f"Analyzing website: {Path(image_path).name}")
                try:
                    description = await self.describe_ai_tool_website(image_path, client)
                    results[image_path] = description
                    # Small delay between requests to avoid overwhelming Ollama
                    await asyncio.sleep(2)
                except Exception as e:
                    logger.error(f"Failed to analyze {image_path}: {e}")
                    results[image_path] = f"Error: {str(e)}"
        
        return results

async def main():
    """
    Main function to analyze the specified AI tool website screenshot.
    """
    image_path = "1photoai.com__complete_page.png"
    
    # Initialize vision client
    vision_client = OllamaVisionClient(model="llava:7b")
    
    logger.info(f"Starting analysis of: {Path(image_path).name}")
    logger.info("Make sure Ollama is running with a vision model (e.g., 'ollama run llava:7b')")
    
    try:
        # Generate description
        description = await vision_client.describe_ai_tool_website(image_path)
        
        # Output results
        print("\n" + "="*80)
        print(f"AI TOOL WEBSITE ANALYSIS: {Path(image_path).name}")
        print("="*80)
        print(description)
        print("="*80)
        
        # Save to file
        output_file = Path(image_path).parent / f"{Path(image_path).stem}_description.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"AI Tool Website Analysis: {Path(image_path).name}\n")
            f.write("="*80 + "\n\n")
            f.write(description)
        
        logger.info(f"Description saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
