import asyncio
import aiohttp
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import csv
import pandas as pd
import numpy as np
import cupy as cp
from asyncio_throttle import Throttler
import time
from concurrent.futures import ThreadPoolExecutor
import logging
import re
from urllib.parse import quote_plus
import json

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

class AdvancedGPUAcceleratedScraper:
    def __init__(self, max_concurrent=20, pages_per_company=5, gpu_batch_size=1000):
        self.max_concurrent = max_concurrent
        self.pages_per_company = pages_per_company
        self.gpu_batch_size = gpu_batch_size
        self.throttler = Throttler(rate_limit=10, period=1)  # 10 requests per second
        self.all_data = []
        
        # Initialize GPU if available
        try:
            self.gpu_available = True
            self.gpu_count = cp.cuda.runtime.getDeviceCount()
            logger.info(f"GPU detected: {cp.cuda.Device(0).name}")
            logger.info(f"GPU memory: {cp.cuda.Device(0).mem_info[1] / 1024**3:.2f} GB")
            logger.info(f"Number of GPUs: {self.gpu_count}")
        except Exception as e:
            self.gpu_available = False
            logger.warning(f"GPU not available: {e}")
    
    async def scrape_company_pages(self, session, company, semaphore):
        """Scrape multiple pages for a single company"""
        async with semaphore:
            all_articles = []
            
            for page in range(1, self.pages_per_company + 1):
                async with self.throttler:
                    try:
                        query = f'"{company}"'
                        offset = (page - 1) * 10  # Bing shows 10 results per page
                        url = f'https://www.bing.com/news/search?q={quote_plus(query)}&first={offset}'
                        
                        logger.info(f"Scraping {company} - page {page}")
                        
                        async with session.get(url) as response:
                            if response.status == 200:
                                content = await response.text()
                                articles = self.parse_articles_advanced(content, company, page)
                                all_articles.extend(articles)
                                
                                # If no articles found, stop pagination
                                if not articles:
                                    logger.info(f"No more articles for {company} at page {page}")
                                    break
                            else:
                                logger.warning(f"Failed to fetch {company} page {page}: {response.status}")
                                break
                                
                    except Exception as e:
                        logger.error(f"Error scraping {company} page {page}: {e}")
                        break
            
            return all_articles
    
    def parse_articles_advanced(self, content, company, page):
        """Advanced article parsing with multiple selectors"""
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
                
                # Extract timestamp if available
                timestamp_selectors = [
                    'div.timestamp',
                    'span.timestamp',
                    'time',
                    'div[data-testid="timestamp"]'
                ]
                timestamp = 'No Timestamp'
                for selector in timestamp_selectors:
                    timestamp_tag = article.select_one(selector)
                    if timestamp_tag:
                        timestamp = timestamp_tag.text.strip()
                        break
                
                articles.append({
                    'title': title,
                    'link': link,
                    'image_url': image_url,
                    'source': source,
                    'snippet': snippet,
                    'timestamp': timestamp,
                    'company': company,
                    'page': page
                })
                
            except Exception as e:
                logger.debug(f"Error parsing article: {e}")
                continue
        
        return articles
    
    def process_data_gpu_advanced(self, data):
        """Advanced GPU-accelerated data processing"""
        if not self.gpu_available or not data:
            return data
        
        try:
            # Convert to pandas DataFrame
            df = pd.DataFrame(data)
            
            if len(df) == 0:
                return data
            
            # GPU-accelerated text processing
            titles = cp.array(df['title'].fillna('').astype(str).values)
            snippets = cp.array(df['snippet'].fillna('').astype(str).values)
            
            # Text length calculations on GPU
            title_lengths = cp.char.str_len(titles)
            snippet_lengths = cp.char.str_len(snippets)
            
            # GPU-accelerated text cleaning
            # Remove special characters and normalize
            cleaned_titles = cp.char.replace(titles, cp.array(['&amp;', '&lt;', '&gt;', '&quot;', '&#39;']), 
                                           cp.array(['&', '<', '>', '"', "'"]))
            
            # GPU-accelerated feature extraction
            has_image = cp.array([url != 'No Image' for url in df['image_url']])
            has_timestamp = cp.array([ts != 'No Timestamp' for ts in df['timestamp']])
            
            # GPU-accelerated duplicate detection using title similarity
            title_hashes = cp.array([hash(title.lower().strip()) for title in titles])
            unique_indices = cp.unique(title_hashes, return_index=True)[1]
            
            # Apply GPU processing results
            df['title_length'] = cp.asnumpy(title_lengths)
            df['snippet_length'] = cp.asnumpy(snippet_lengths)
            df['has_image'] = cp.asnumpy(has_image)
            df['has_timestamp'] = cp.asnumpy(has_timestamp)
            df['cleaned_title'] = cp.asnumpy(cleaned_titles)
            
            # Remove duplicates
            df = df.iloc[cp.asnumpy(unique_indices)]
            
            # GPU-accelerated sentiment analysis (simplified)
            # Count positive/negative words in titles
            positive_words = cp.array(['ai', 'breakthrough', 'innovation', 'success', 'launch', 'release', 'update'])
            negative_words = cp.array(['bug', 'issue', 'problem', 'failure', 'down', 'error'])
            
            title_lower = cp.char.lower(titles)
            positive_count = cp.sum(cp.char.count(title_lower, positive_words), axis=1)
            negative_count = cp.sum(cp.char.count(title_lower, negative_words), axis=1)
            
            df['positive_words'] = cp.asnumpy(positive_count)
            df['negative_words'] = cp.asnumpy(negative_count)
            df['sentiment_score'] = cp.asnumpy(positive_count - negative_count)
            
            logger.info(f"GPU processed {len(df)} articles with advanced features")
            
            return df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Advanced GPU processing error: {e}")
            return data
    
    async def scrape_all_companies_advanced(self):
        """Scrape all companies with multiple pages concurrently"""
        start_time = time.time()
        
        # Create semaphore for concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Create aiohttp session with optimized settings
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=10,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        timeout = aiohttp.ClientTimeout(total=60, connect=30)
        
        async with aiohttp.ClientSession(
            connector=connector, 
            timeout=timeout,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        ) as session:
            # Create tasks for all companies
            tasks = [
                self.scrape_company_pages(session, company, semaphore)
                for company in AI_TOOL_COMPANIES
            ]
            
            # Execute all tasks concurrently
            logger.info(f"Starting advanced concurrent scraping of {len(AI_TOOL_COMPANIES)} companies...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect all results
            for i, result in enumerate(results):
                if isinstance(result, list):
                    self.all_data.extend(result)
                    logger.info(f"Collected {len(result)} articles from {AI_TOOL_COMPANIES[i]}")
                else:
                    logger.error(f"Task failed for {AI_TOOL_COMPANIES[i]}: {result}")
        
        # Process data with advanced GPU acceleration
        logger.info(f"Processing {len(self.all_data)} articles with advanced GPU acceleration...")
        processed_data = self.process_data_gpu_advanced(self.all_data)
        
        # Save to CSV and JSON
        self.save_data_advanced(processed_data)
        
        end_time = time.time()
        logger.info(f"Advanced scraping completed in {end_time - start_time:.2f} seconds")
        logger.info(f"Total articles scraped: {len(processed_data)}")
        
        return processed_data
    
    def save_data_advanced(self, data):
        """Save data to both CSV and JSON formats"""
        if not data:
            logger.warning("No data to save")
            return
        
        # Save to CSV
        csv_filename = 'bing_news_articles_gpu_advanced.csv'
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'company', 'title', 'link', 'image_url', 'source', 'snippet', 
                'timestamp', 'page', 'title_length', 'snippet_length', 'has_image',
                'has_timestamp', 'cleaned_title', 'positive_words', 'negative_words', 'sentiment_score'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        
        # Save to JSON for easier processing
        json_filename = 'bing_news_articles_gpu_advanced.json'
        with open(json_filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(data, jsonfile, indent=2, ensure_ascii=False)
        
        # Create summary statistics
        df = pd.DataFrame(data)
        summary = {
            'total_articles': len(data),
            'companies_covered': df['company'].nunique(),
            'articles_with_images': df['has_image'].sum(),
            'articles_with_timestamps': df['has_timestamp'].sum(),
            'avg_title_length': df['title_length'].mean(),
            'avg_snippet_length': df['snippet_length'].mean(),
            'avg_sentiment_score': df['sentiment_score'].mean(),
            'top_companies': df['company'].value_counts().head(5).to_dict()
        }
        
        summary_filename = 'scraping_summary.json'
        with open(summary_filename, 'w', encoding='utf-8') as summaryfile:
            json.dump(summary, summaryfile, indent=2)
        
        logger.info(f"Data saved to {csv_filename}, {json_filename}, and {summary_filename}")
        logger.info(f"Summary: {summary}")

async def main():
    """Main function to run the advanced GPU-accelerated scraper"""
    scraper = AdvancedGPUAcceleratedScraper(
        max_concurrent=25,  # Higher concurrency for faster scraping
        pages_per_company=5,  # Scrape 5 pages per company
        gpu_batch_size=2000
    )
    await scraper.scrape_all_companies_advanced()

if __name__ == "__main__":
    asyncio.run(main()) 