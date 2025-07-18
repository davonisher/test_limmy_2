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
from pydantic import BaseModel
from typing import Literal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load companies from CSV file
TOOLS_CSV_PATH = 'top_1000_tools.csv'

# Structured output schema for classification
class ArticleClassification(BaseModel):
    category: Literal["feature update", "funding", "news article", "irrelevant"]
    company_relevance: bool
    reasoning: str

def load_companies_from_csv():
    """Load AI tool companies from CSV file"""
    try:
        if os.path.exists(TOOLS_CSV_PATH):
            df_tools = pd.read_csv(TOOLS_CSV_PATH)
            tool_names = df_tools['tool_name'].dropna().unique().tolist()
            tool_names = tool_names[:100]  # Limit to first 100
            logger.info(f"Loaded {len(tool_names)} tool names from CSV")
            return tool_names, df_tools
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
    Simple Ollama client for local LLM inference using structured outputs.
    Assumes Ollama is running locally on port 11434.
    """
    def __init__(self, model="llama3.3:latest"):
        self.base_url = "http://localhost:11434"
        self.model = model

    async def classify_title(self, title, tool_name, client=None):
        """
        Classify the article title using structured outputs.
        Returns an ArticleClassification object.
        """
        # Skip very short titles
        if len(title.strip()) < 10:
            return ArticleClassification(
                category="news article",
                company_relevance=False,
                reasoning="Title too short"
            )
            
        # Create a detailed prompt
        prompt = f"""You are an expert news classifier. Analyze this article title and determine:

1. CATEGORY: Classify into exactly one category:
   - "feature update": ONLY if the article is about the tool's new features, product releases, updates, launches, or announcements
   - "funding": ONLY if the article is about the tool's funding rounds, investments, fundraising, venture capital, or financial news
   - "news article": for general news, partnerships, research, analysis, or other tool-related news
   - "irrelevant": if the article is NOT about the tool at all

2. COMPANY RELEVANCE: Determine if the article is actually about the tool being searched for.

TOOL BEING SEARCHED: {tool_name}
ARTICLE TITLE: "{title}"

IMPORTANT RULES:
- Only classify as "feature update" or "funding" if the article is DIRECTLY about {tool_name}
- If the article mentions {tool_name} but is not about their features/funding, use "news article"
- If the article doesn't mention {tool_name} at all, use "irrelevant"
- Be very strict about tool relevance

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
            "format": ArticleClassification.model_json_schema(),
            "options": {
                "temperature": 0.1,  # Low temperature for more consistent classification
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
                classification = ArticleClassification.model_validate_json(content)
                return classification
            else:
                raise ValueError("No content in response")
                
        except Exception as e:
            logger.warning(f"Ollama classification failed for title '{title}': {e}")
            # Fallback to basic classification
            tool_name_lower = tool_name.lower()
            title_lower = title.lower()
            company_relevance = tool_name_lower in title_lower
            
            if not company_relevance:
                return ArticleClassification(
                    category="irrelevant",
                    company_relevance=False,
                    reasoning="Tool name not found in title"
                )
            elif any(word in title_lower for word in ["funding", "fund", "investment", "raise", "series", "round"]):
                return ArticleClassification(
                    category="funding",
                    company_relevance=True,
                    reasoning="Funding-related keywords found"
                )
            elif any(word in title_lower for word in ["feature", "update", "release", "launch", "announce"]):
                return ArticleClassification(
                    category="feature update",
                    company_relevance=True,
                    reasoning="Feature update keywords found"
                )
            else:
                return ArticleClassification(
                    category="news article",
                    company_relevance=True,
                    reasoning="General tool news"
                )

class GPUAcceleratedScraper:
    def __init__(self, max_concurrent=20, gpu_batch_size=1000):
        self.max_concurrent = max_concurrent
        self.gpu_batch_size = gpu_batch_size
        self.throttler = Throttler(rate_limit=10, period=1)  # 10 requests per second
        self.all_data = []
        
        # Load tool names and tool info
        self.tool_names, self.df_tools = load_companies_from_csv()
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
        self.ollama = OllamaClient(model="llama3.3:latest")
    
    async def scrape_tool_batch(self, browser, tool_batch):
        """Scrape a batch of tools using Playwright with parallel tabs"""
        batch_data = []
        
        # Open tabs in parallel
        pages = [await browser.new_page() for _ in tool_batch]
        
        try:
            # Navigate all tabs to their respective Bing News search
            tasks = []
            for page, tool_name in zip(pages, tool_batch):
                # More specific search query to reduce irrelevant results
                query = f'"{tool_name}" AND (AI OR "artificial intelligence" OR "machine learning" OR "AI tool" OR "software" OR "technology")'
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
            for page, tool_name in zip(pages, tool_batch):
                tool_id, tool_url = self.tool_info.get(tool_name, (None, None))
                content = await page.content()
                articles = self.parse_articles_advanced(content, tool_name, tool_id, tool_url)
                batch_data.extend(articles)
                logger.info(f"Extracted {len(articles)} articles for {tool_name}")
                
        finally:
            # Close all pages
            await asyncio.gather(*[page.close() for page in pages])
        
        return batch_data
    
    def parse_articles_advanced(self, content, tool_name, tool_id, tool_url):
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
                    'tool_name': tool_name,
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
        Adds new columns 'ollama_category' and 'company_relevance'.
        """
        # Pre-filter obviously irrelevant articles
        logger.info(f"Pre-filtering {len(df)} articles for obvious irrelevance...")
        
        # Create a copy for filtering
        filtered_df = df.copy()
        
        # Remove articles where tool name is not in title (case insensitive)
        tool_name_in_title = []
        for _, row in filtered_df.iterrows():
            tool_name = str(row['tool_name']).lower()
            title = str(row['title']).lower()
            tool_name_in_title.append(tool_name in title)
        
        filtered_df['tool_name_in_title'] = tool_name_in_title
        pre_filtered_df = filtered_df[filtered_df['tool_name_in_title'] == True].copy()
        
        pre_filtered_count = len(df) - len(pre_filtered_df)
        logger.info(f"Pre-filtered out {pre_filtered_count} articles where tool name not in title")
        logger.info(f"Remaining articles for LLM classification: {len(pre_filtered_df)}")
        
        if len(pre_filtered_df) == 0:
            logger.warning("No articles remain after pre-filtering")
            df['ollama_category'] = 'irrelevant'
            df['company_relevance'] = False
            return df
        
        titles = pre_filtered_df['title'].fillna('').astype(str).tolist()
        tool_names = pre_filtered_df['tool_name'].fillna('').astype(str).tolist()
        categories = []
        relevances = []
        
        logger.info(f"Starting Ollama classification for {len(titles)} titles using model: {self.ollama.model}")

        # Use a single httpx.AsyncClient for efficiency
        async with httpx.AsyncClient(timeout=30) as client:
            # Process in smaller batches to avoid overwhelming Ollama
            batch_size = 50
            for i in range(0, len(titles), batch_size):
                batch_titles = titles[i:i+batch_size]
                batch_tool_names = tool_names[i:i+batch_size]
                logger.info(f"Classifying batch {i//batch_size + 1}/{(len(titles)-1)//batch_size + 1} ({len(batch_titles)} titles)")
                
                tasks = [self.ollama.classify_title(title, tool_name, client=client) 
                        for title, tool_name in zip(batch_titles, batch_tool_names)]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle any exceptions in the batch
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.warning(f"Classification failed for title '{batch_titles[j]}': {result}")
                        categories.append("news article")  # Default fallback
                        relevances.append(False)
                    else:
                        # Handle ArticleClassification object
                        categories.append(result.category)
                        relevances.append(result.company_relevance)
                
                # Small delay between batches
                await asyncio.sleep(1)

        pre_filtered_df['ollama_category'] = categories
        pre_filtered_df['company_relevance'] = relevances
        
        # Filter out irrelevant articles
        relevant_df = pre_filtered_df[pre_filtered_df['company_relevance'] == True].copy()
        irrelevant_count = len(pre_filtered_df) - len(relevant_df)
        
        logger.info(f"Completed Ollama classification.")
        logger.info(f"Categories: {dict(pd.Series(categories).value_counts())}")
        logger.info(f"Tool relevance: {sum(relevances)}/{len(relevances)} articles relevant")
        logger.info(f"Filtered out {irrelevant_count} irrelevant articles")
        
        return relevant_df

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

    async def scrape_all_tools(self):
        """Scrape all tools using batch processing with Playwright"""
        start_time = time.time()
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            
            try:
                # Process tools in batches
                for i, tool_batch in enumerate(chunked(self.tool_names, 20)):
                    logger.info(f"Processing batch {i+1}/{(len(self.tool_names)-1)//20+1} with {len(tool_batch)} tools")
                    
                    batch_data = await self.scrape_tool_batch(browser, tool_batch)
                    self.all_data.extend(batch_data)
                    
                    # Add delay between batches to avoid rate limiting
                    if i < (len(self.tool_names)-1)//20:
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
                'tool_id', 'tool_name', 'tool_url', 'title', 'link', 'image_url', 
                'source', 'snippet', 'title_length', 'snippet_length', 'has_image',
                'positive_words', 'negative_words', 'sentiment_score', 'ollama_category', 'company_relevance'
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