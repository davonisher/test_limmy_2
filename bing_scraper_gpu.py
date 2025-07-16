import asyncio
import aiohttp
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import csv
import pandas as pd
import numpy as np
import torch
from asyncio_throttle import Throttler
import time
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of top AI tool companies for use in queries
AI_TOOL_COMPANIES = [
    "OpenAI",
    "Google DeepMind",
    "Anthropic",
    "Microsoft",
    "NVIDIA",
    "Meta AI",
    "Cohere",
    "Hugging Face",
    "Stability AI",
    "Databricks",
    "UiPath",
    "DataRobot",
    "Scale AI",
    "Adept AI",
    "Runway",
    "Perplexity AI",
    "Mistral AI",
    "Reka AI",
    "Inflection AI",
    "Abacus AI",
    "SambaNova Systems",
    "Snorkel AI",
    "Pinecone",
    "Weights & Biases",
    "LangChain",
    "Taktile"
]

class GPUAcceleratedScraper:
    def __init__(self, max_concurrent=10, gpu_batch_size=1000):
        self.max_concurrent = max_concurrent
        self.gpu_batch_size = gpu_batch_size
        self.throttler = Throttler(rate_limit=5, period=1)  # 5 requests per second
        self.all_data = []
        
        # Initialize GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
        else:
            logger.warning("GPU not available, using CPU.")
    
    async def scrape_company_news(self, session, company, semaphore):
        """Scrape news for a single company with rate limiting"""
        async with semaphore:
            async with self.throttler:
                try:
                    query = f'"{company}"'
                    url = f'https://www.bing.com/news/search?q={query}'
                    
                    logger.info(f"Scraping news for: {company}")
                    
                    # Use aiohttp for faster requests
                    async with session.get(url) as response:
                        if response.status == 200:
                            content = await response.text()
                            articles = self.parse_articles(content, company)
                            return articles
                        else:
                            logger.warning(f"Failed to fetch {company}: {response.status}")
                            return []
                            
                except Exception as e:
                    logger.error(f"Error scraping {company}: {e}")
                    return []
    
    def parse_articles(self, content, company):
        """Parse articles from HTML content"""
        soup = BeautifulSoup(content, 'html.parser')
        articles = []
        
        # Try both possible Bing News article containers
        article_elements = soup.find_all('div', class_='news-card-body')
        if not article_elements:
            article_elements = soup.find_all('div', class_='t_s')
        
        for article in article_elements:
            # Extract article data
            title_tag = article.find('a', class_='title') or article.find('a')
            link = title_tag['href'] if title_tag and title_tag.has_attr('href') else 'No Link'
            title = title_tag.text.strip() if title_tag else 'No Title'
            
            image_tag = article.find('img')
            image_url = image_tag['src'] if image_tag and image_tag.has_attr('src') else 'No Image'
            
            source_tag = article.find('div', class_='source') or article.find('div', class_='source-card')
            source = source_tag.text.strip() if source_tag else 'No Source'
            
            snippet_tag = article.find('div', class_='snippet') or article.find('div', class_='snippet-card')
            snippet = snippet_tag.text.strip() if snippet_tag else 'No Snippet'
            
            articles.append({
                'title': title,
                'link': link,
                'image_url': image_url,
                'source': source,
                'snippet': snippet,
                'company': company
            })
        
        return articles
    
    def process_data_gpu(self, data):
        """Process scraped data using GPU acceleration (PyTorch)"""
        if not data:
            return data
        try:
            df = pd.DataFrame(data)
            if len(df) == 0:
                return data
            # Move string lengths to GPU for batch processing
            titles = df['title'].fillna('').astype(str).values
            snippets = df['snippet'].fillna('').astype(str).values
            # Convert to list of lengths
            title_lengths = torch.tensor([len(t) for t in titles], device=self.device)
            snippet_lengths = torch.tensor([len(s) for s in snippets], device=self.device)
            # Add processed features back to DataFrame
            df['title_length'] = title_lengths.cpu().numpy()
            df['snippet_length'] = snippet_lengths.cpu().numpy()
            df['has_image'] = df['image_url'] != 'No Image'
            # GPU-accelerated duplicate detection (hashing on GPU)
            title_hashes = torch.tensor([hash(t) for t in titles], device=self.device)
            _, unique_indices = torch.unique(title_hashes, return_inverse=False, return_counts=False, sorted=False, return_index=True)
            df = df.iloc[unique_indices.cpu().numpy()]
            logger.info(f"GPU processed {len(df)} articles (PyTorch)")
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"GPU processing error: {e}")
        return data
    
    async def scrape_all_companies(self):
        """Scrape all companies concurrently"""
        start_time = time.time()
        semaphore = asyncio.Semaphore(self.max_concurrent)
        connector = aiohttp.TCPConnector(limit=self.max_concurrent, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [
                self.scrape_company_news(session, company, semaphore)
                for company in AI_TOOL_COMPANIES
            ]
            logger.info(f"Starting concurrent scraping of {len(AI_TOOL_COMPANIES)} companies...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    self.all_data.extend(result)
                else:
                    logger.error(f"Task failed: {result}")
        logger.info(f"Processing {len(self.all_data)} articles with GPU acceleration (PyTorch)...")
        processed_data = self.process_data_gpu(self.all_data)
        self.save_to_csv(processed_data)
        end_time = time.time()
        logger.info(f"Scraping completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Total articles scraped: {len(processed_data)}")
        return processed_data
    
    def save_to_csv(self, data):
        if not data:
            logger.warning("No data to save")
            return
        filename = 'bing_news_articles_gpu.csv'
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['company', 'title', 'link', 'image_url', 'source', 'snippet', 'title_length', 'snippet_length', 'has_image']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        logger.info(f"Data saved to {filename}")

async def main():
    scraper = GPUAcceleratedScraper(max_concurrent=15, gpu_batch_size=1000)
    await scraper.scrape_all_companies()

if __name__ == "__main__":
    asyncio.run(main())