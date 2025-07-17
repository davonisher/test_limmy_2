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
from datetime import datetime
import os
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load companies from CSV file
TOOLS_CSV_PATH = 'top_1000_tools.csv'

def load_companies_from_csv():
    """Load AI tool companies from CSV file"""
    try:
        if os.path.exists(TOOLS_CSV_PATH):
            df_tools = pd.read_csv(TOOLS_CSV_PATH)
            companies = df_tools['tool_name'].dropna().unique().tolist()
            companies = companies[:100]  # Limit to first 100
            logger.info(f"Loaded {len(companies)} companies from CSV")
            return companies, df_tools
        else:
            logger.warning(f"CSV file not found: {TOOLS_CSV_PATH}")
            # Fallback to hardcoded list
            return [
                "OpenAI", "Google DeepMind", "Anthropic", "Microsoft", "NVIDIA",
                "Meta AI", "Cohere", "Hugging Face", "Stability AI", "Databricks"
            ], None
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return ["OpenAI", "Google DeepMind", "Anthropic"], None

def chunked(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

class OllamaClient:
    """
    Simple Ollama client for local LLM inference.
    Assumes Ollama is running locally on port 11434.
    """
    def __init__(self, model="llama3.2"):
        self.base_url = "http://localhost:11434"
        self.model = model

    async def classify_title(self, title, client=None):
        """
        Classify the article title as 'feature update', 'funding', or 'news article'.
        Returns one of: 'feature update', 'funding', 'news article'
        """
        # Skip very short titles
        if len(title.strip()) < 10:
            return "news article"
            
        prompt = (
            f"You are a news classifier. Classify this article title into exactly one of these three categories:\n"
            f"- 'feature update': for new features, product releases, updates, launches\n"
            f"- 'funding': for funding rounds, investments, fundraising, venture capital\n"
            f"- 'news article': for general news, partnerships, research, analysis\n\n"
            f"Title: \"{title}\"\n"
            f"Category (respond with only the category name):"
        )
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for more consistent classification
                "top_p": 0.9
            }
        }
        try:
            if client is None:
                async with httpx.AsyncClient(timeout=30) as ac:
                    response = await ac.post(f"{self.base_url}/api/generate", json=data)
            else:
                response = await client.post(f"{self.base_url}/api/generate", json=data)
            response.raise_for_status()
            result = response.json()
            # The response is expected to have a 'response' field with the model's answer
            answer = result.get("response", "").strip()
            
            # Debug: log some responses to see what the model is returning
            if len(answer) < 50:  # Only log short responses to avoid spam
                logger.debug(f"Model response for '{title[:50]}...': '{answer}'")
            
            # More robust answer parsing - be more specific
            answer_lower = answer.lower().strip()
            
            # Check for exact matches first
            if answer_lower in ["feature update", "feature", "update"]:
                return "feature update"
            elif answer_lower in ["funding", "fund"]:
                return "funding"
            elif answer_lower in ["news article", "news", "article"]:
                return "news article"
            
            # Check for funding-related keywords (most specific)
            funding_keywords = ["funding", "fundraise", "fund raise", "investment", "raise", "funded", "series", "round", "venture", "capital"]
            if any(keyword in answer_lower for keyword in funding_keywords):
                return "funding"
            
            # Check for feature update keywords (more specific)
            feature_keywords = ["feature", "update", "release", "launch", "new feature", "announces", "introduces"]
            if any(keyword in answer_lower for keyword in feature_keywords):
                return "feature update"
            
            # Default to news article
            return "news article"
        except Exception as e:
            logger.warning(f"Ollama classification failed for title '{title}': {e}")
            return "news article"

class GPUAcceleratedScraper:
    def __init__(self, max_concurrent=20, gpu_batch_size=1000):
        self.max_concurrent = max_concurrent
        self.gpu_batch_size = gpu_batch_size
        self.throttler = Throttler(rate_limit=10, period=1)  # 10 requests per second
        self.all_data = []
        
        # Load companies and tool info
        self.companies, self.df_tools = load_companies_from_csv()
        if self.df_tools is not None:
            self.tool_info = {row['tool_name']: (row['tool_id'], row['tool_url']) 
                             for _, row in self.df_tools.iterrows()}
        else:
            self.tool_info = {}
        
        # Initialize GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
        else:
            logger.warning("GPU not available, using CPU.")

        # Initialize Ollama client
        self.ollama = OllamaClient(model="llama3.2")
    
    async def scrape_company_batch(self, browser, company_batch):
        """Scrape a batch of companies using Playwright with parallel tabs"""
        batch_data = []
        
        # Open tabs in parallel
        pages = [await browser.new_page() for _ in company_batch]
        
        try:
            # Navigate all tabs to their respective Bing News search
            tasks = []
            for page, company in zip(pages, company_batch):
                query = f'"{company}" AND (AI OR "AI tool")'
                url = f'https://www.bing.com/news/search?q={query}&cc=us&setlang=en-us&qft=sortbydate%3d%221%22&form=YFNR'
                tasks.append(page.goto(url))
            await asyncio.gather(*tasks)
            
            # Scroll all tabs in parallel, 4 times max
            for scroll_round in range(4):
                scroll_tasks = []
                for page in pages:
                    scroll_tasks.append(page.evaluate('window.scrollTo(0, document.body.scrollHeight)'))
                await asyncio.gather(*scroll_tasks)
                await asyncio.gather(*[page.wait_for_timeout(2000) for page in pages])
                logger.info(f"Completed scroll round {scroll_round + 1}/4 for batch")
            
            # Extract content for each tab
            for page, company in zip(pages, company_batch):
                tool_id, tool_url = self.tool_info.get(company, (None, None))
                content = await page.content()
                articles = self.parse_articles_advanced(content, company, tool_id, tool_url)
                batch_data.extend(articles)
                logger.info(f"Extracted {len(articles)} articles for {company}")
                
        finally:
            # Close all pages
            await asyncio.gather(*[page.close() for page in pages])
        
        return batch_data
    
    def parse_articles_advanced(self, content, company, tool_id, tool_url):
        """Parse articles from HTML content with advanced selectors"""
        soup = BeautifulSoup(content, 'html.parser')
        articles = []
        
        # Multiple possible selectors for Bing News
        selectors = [
            'div.news-card-body',
            'div.t_s',
            'div.news-card',
            'article.news-card',
            'div[data-testid="news-card"]'
        ]
        
        article_elements = []
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                article_elements = elements
                break
        
        for article in article_elements:
            try:
                # Extract title and link
                title_tag = (
                    article.select_one('a.title') or 
                    article.select_one('a[href*="news"]') or 
                    article.select_one('a')
                )
                
                link = title_tag['href'] if title_tag and title_tag.has_attr('href') else 'No Link'
                title = title_tag.text.strip() if title_tag else 'No Title'
                
                # Clean and validate title
                if not title or title == 'No Title' or len(title) < 5:
                    continue
                
                # Extract image
                image_tag = article.select_one('img')
                image_url = image_tag['src'] if image_tag and image_tag.has_attr('src') else 'No Image'
                
                # Extract source
                source_selectors = [
                    'div.source',
                    'div.source-card',
                    'span.source',
                    'div[data-testid="source"]'
                ]
                source = 'No Source'
                for selector in source_selectors:
                    source_tag = article.select_one(selector)
                    if source_tag:
                        source = source_tag.text.strip()
                        break
                
                # Extract snippet
                snippet_selectors = [
                    'div.snippet',
                    'div.snippet-card',
                    'p.snippet',
                    'div[data-testid="snippet"]'
                ]
                snippet = 'No Snippet'
                for selector in snippet_selectors:
                    snippet_tag = article.select_one(selector)
                    if snippet_tag:
                        snippet = snippet_tag.text.strip()
                        break
                
                articles.append({
                    'tool_id': tool_id,
                    'company': company,
                    'tool_url': tool_url,
                    'title': title,
                    'link': link,
                    'image_url': image_url,
                    'source': source,
                    'snippet': snippet
                })
                
            except Exception as e:
                logger.debug(f"Error parsing article: {e}")
                continue
        
        return articles

    async def classify_titles_with_ollama(self, df):
        """
        Classify all article titles in the DataFrame using Ollama.
        Adds a new column 'ollama_category' with values: 'feature update', 'funding', or 'news article'.
        """
        titles = df['title'].fillna('').astype(str).tolist()
        categories = []
        
        logger.info(f"Starting Ollama classification for {len(titles)} titles using model: {self.ollama.model}")

        # Use a single httpx.AsyncClient for efficiency
        async with httpx.AsyncClient(timeout=30) as client:
            # Process in smaller batches to avoid overwhelming Ollama
            batch_size = 50
            for i in range(0, len(titles), batch_size):
                batch_titles = titles[i:i+batch_size]
                logger.info(f"Classifying batch {i//batch_size + 1}/{(len(titles)-1)//batch_size + 1} ({len(batch_titles)} titles)")
                
                tasks = [self.ollama.classify_title(title, client=client) for title in batch_titles]
                batch_categories = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle any exceptions in the batch
                for j, result in enumerate(batch_categories):
                    if isinstance(result, Exception):
                        logger.warning(f"Classification failed for title '{batch_titles[j]}': {result}")
                        categories.append("news article")  # Default fallback
                    else:
                        categories.append(result)
                
                # Small delay between batches
                await asyncio.sleep(1)

        df['ollama_category'] = categories
        logger.info(f"Completed Ollama classification. Categories: {dict(pd.Series(categories).value_counts())}")
        return df

    def process_data_gpu(self, data):
        """Process scraped data using GPU acceleration (PyTorch)"""
        if not data:
            return data
        try:
            df = pd.DataFrame(data)
            if len(df) == 0:
                return data
            
            logger.info(f"Starting GPU processing of {len(df)} articles...")
            
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
            
            # Fixed torch.unique() call for newer PyTorch versions
            try:
                # Try newer API first
                unique_hashes, inverse_indices, counts = torch.unique(title_hashes, return_inverse=True, return_counts=True, sorted=False)
                # Get indices of first occurrence of each unique hash
                unique_indices = torch.zeros_like(unique_hashes, dtype=torch.long)
                for i, hash_val in enumerate(unique_hashes):
                    unique_indices[i] = torch.where(title_hashes == hash_val)[0][0]
            except:
                # Fallback to older API
                unique_hashes, unique_indices = torch.unique(title_hashes, return_index=True)
            
            # Remove duplicates from DataFrame
            df = df.iloc[unique_indices.cpu().numpy()].reset_index(drop=True)
            
            # Now apply sentiment analysis to the deduplicated data
            titles_dedup = df['title'].fillna('').astype(str).values
            title_lower = [t.lower() for t in titles_dedup]
            
            positive_words = ['ai', 'breakthrough', 'innovation', 'success', 'launch', 'release', 'update']
            negative_words = ['bug', 'issue', 'problem', 'failure', 'down', 'error']
            
            positive_count = torch.tensor([sum(1 for word in positive_words if word in title) for title in title_lower], device=self.device)
            negative_count = torch.tensor([sum(1 for word in negative_words if word in title) for title in title_lower], device=self.device)
            
            df['positive_words'] = positive_count.cpu().numpy()
            df['negative_words'] = negative_count.cpu().numpy()
            df['sentiment_score'] = (positive_count - negative_count).cpu().numpy()
            
            # Log GPU utilization
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                logger.info(f"GPU memory used: {gpu_memory_used:.2f} GB / {gpu_memory_total:.2f} GB")
            
            logger.info(f"GPU processed {len(df)} articles (PyTorch) - Removed {len(data) - len(df)} duplicates")
            return df
        except Exception as e:
            logger.error(f"GPU processing error: {e}")
            logger.info("Falling back to CPU processing...")
            # Fallback to basic processing without GPU
            df = pd.DataFrame(data)
            df['title_length'] = df['title'].str.len()
            df['snippet_length'] = df['snippet'].str.len()
            df['has_image'] = df['image_url'] != 'No Image'
            return df

    async def scrape_all_companies(self):
        """Scrape all companies using batch processing with Playwright"""
        start_time = time.time()
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            
            try:
                # Process companies in batches
                for i, company_batch in enumerate(chunked(self.companies, 20)):
                    logger.info(f"Processing batch {i+1}/{(len(self.companies)-1)//20+1} with {len(company_batch)} companies")
                    
                    batch_data = await self.scrape_company_batch(browser, company_batch)
                    self.all_data.extend(batch_data)
                    
                    # Add delay between batches to avoid rate limiting
                    if i < (len(self.companies)-1)//20:
                        await asyncio.sleep(5)
                
            finally:
                await browser.close()
        
        # Process data with GPU acceleration
        logger.info(f"Processing {len(self.all_data)} articles with GPU acceleration (PyTorch)...")
        processed_df = self.process_data_gpu(self.all_data)

        # Classify with Ollama
        logger.info("Classifying article titles with Ollama...")
        processed_df = await self.classify_titles_with_ollama(processed_df)
        processed_data = processed_df.to_dict('records')
        
        # Save to CSV with timestamp
        self.save_to_csv(processed_data)
        
        end_time = time.time()
        logger.info(f"Scraping completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Total articles scraped: {len(processed_data)}")
        
        return processed_data
    
    def save_to_csv(self, data):
        """Save data to CSV file with timestamp"""
        if not data:
            logger.warning("No data to save")
            return
        
        today_str = datetime.now().strftime('%Y-%m-%d')
        filename = f'bing_news_articles_gpu_{today_str}.csv'
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'tool_id', 'company', 'tool_url', 'title', 'link', 'image_url', 
                'source', 'snippet', 'title_length', 'snippet_length', 'has_image',
                'positive_words', 'negative_words', 'sentiment_score', 'ollama_category'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        
        logger.info(f"Data saved to {filename}")

async def main():
    scraper = GPUAcceleratedScraper(max_concurrent=20, gpu_batch_size=1000)
    await scraper.scrape_all_companies()

if __name__ == "__main__":
    asyncio.run(main())