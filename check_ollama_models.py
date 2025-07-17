import asyncio
import httpx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def check_ollama_models():
    """Check what Ollama models are available"""
    base_url = "http://localhost:11434"
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Check if Ollama is running
            response = await client.get(f"{base_url}/api/tags")
            response.raise_for_status()
            models = response.json()
            
            logger.info("Available Ollama models:")
            for model in models.get("models", []):
                logger.info(f"  - {model.get('name', 'Unknown')} (Size: {model.get('size', 'Unknown')})")
            
            return models.get("models", [])
            
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        return []

async def test_model_availability(model_name):
    """Test if a specific model is available and working"""
    base_url = "http://localhost:11434"
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # Test with a simple prompt
            data = {
                "model": model_name,
                "prompt": "Hello, are you working?",
                "stream": False
            }
            
            response = await client.post(f"{base_url}/api/generate", json=data)
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"Model {model_name} is working! Response: {result.get('response', 'No response')[:100]}...")
            return True
            
    except Exception as e:
        logger.error(f"Model {model_name} is not available or not working: {e}")
        return False

async def main():
    """Main function to check Ollama models"""
    logger.info("Checking Ollama models...")
    
    models = await check_ollama_models()
    
    if models:
        # Test specific models
        test_models = ["gemma3n", "llama3", "llama3.2"]
        for model in test_models:
            await test_model_availability(model)
    else:
        logger.error("No models found or Ollama is not running")

if __name__ == "__main__":
    asyncio.run(main()) 