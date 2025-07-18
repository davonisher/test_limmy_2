#!/usr/bin/env python3
"""
Enhanced Product Hunt scraper to extract:
- Page title
- Page URL
- Visit website button URL
- Product name

This version is optimized for server deployment with headless browser and no proxy dependencies.
Includes cron job functionality to run every 4 hours.
"""

import asyncio
import pandas as pd
import csv
from playwright.async_api import async_playwright
from urllib.parse import urljoin
import random
import os
import logging
from datetime import datetime, timedelta
import time
import signal
import sys

# --- Stealth async patch ---
# You need to install playwright-stealth: pip install playwright-stealth
try:
    from playwright_stealth import stealth_async
    STEALTH_AVAILABLE = True
except ImportError:
    print("Warning: playwright-stealth not installed. Stealth mode will not be used.")
    STEALTH_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper_cron.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
TOOLS_CSV_PATH = 'PH_check1.csv'
CRON_INTERVAL_HOURS = 4  # Run every 4 hours
SAMPLE_SIZE = 500

# Global flag for graceful shutdown
shutdown_flag = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_flag
    logger.info(f"Received signal {signum}. Starting graceful shutdown...")
    shutdown_flag = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def get_timestamped_filename(base_name):
    """Generate a filename with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(base_name)
    return f"{name}_{timestamp}{ext}"

async def scrape_product_info(url):
    """
    Extract product information from a Product Hunt page.
    
    Args:
        url (str): The Product Hunt product URL
        
    Returns:
        dict: Dictionary containing extracted information
    """
    async with async_playwright() as p:
        # Launch browser in headless mode for server deployment
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        # Apply stealth async if available
        if STEALTH_AVAILABLE:
            await stealth_async(page)
        
        try:
            print(f"Scraping: {url}")
            
            # Navigate to the page with timeout
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            
            # Wait for content to load
            await asyncio.sleep(5)
            
            # Extract page title
            page_title = await page.title()
            
            # Extract product name/title from the page
            product_name = None
            product_selectors = [
                "h1",
                "h1.color-darker-grey",
                "[data-test='product-name']",
                ".product-name"
            ]
            
            for selector in product_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        product_name = await element.inner_text()
                        product_name = product_name.strip()
                        if product_name:
                            break
                except Exception:
                    continue
            
            # Extract visit website button URL
            visit_url = None
            visit_selectors = [
                '[data-test="visit-website-button"]',
                'a[href*="ref=producthunt"]',
                'a:has-text("Visit website")',
                '.visit-website',
                'a[target="_blank"]:has-text("Visit")'
            ]
            
            for selector in visit_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        visit_url = await element.get_attribute("href")
                        if visit_url:
                            # Make sure it's a full URL
                            if visit_url.startswith('http'):
                                break
                            else:
                                visit_url = urljoin(url, visit_url)
                                break
                except Exception:
                    continue
            
            # Extract additional URL from span element
            additional_url = None
            try:
                # Try multiple selectors to find the span with the URL
                span_selectors = [
                    'span[class*="max-w-[260px]"][class*="truncate"][class*="font-semibold"]',
                    'span.max-w-\\[260px\\].truncate.font-semibold',
                    'span[class="max-w-[260px] truncate font-semibold"]',
                    'span.font-semibold.truncate'
                ]
                
                for selector in span_selectors:
                    try:
                        span_element = await page.query_selector(selector)
                        if span_element:
                            text_content = await span_element.inner_text()
                            text_content = text_content.strip()
                            # Check if it looks like a URL
                            if '.' in text_content and '/' in text_content:
                                additional_url = text_content
                                print(f"Found additional URL: {additional_url}")
                                break
                    except Exception:
                        continue
                        
            except Exception as e:
                print(f"Error extracting additional URL: {e}")
            
            result = {
                'page_url': url,
                'page_title': page_title,
                'product_name': product_name,
                'visit_website_url': visit_url,
                'additional_url': additional_url
            }
            
            print(f"✅ Successfully scraped: {product_name}")
            return result
            
        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")
            return {
                'page_url': url,
                'page_title': None,
                'product_name': None,
                'visit_website_url': None,
                'additional_url': None
            }
        finally:
            await browser.close()

async def scrape_csv_file(csv_file_path, output_file_path, sample_size=100):
    """
    Scrape a sample of URLs from a CSV file and save results.
    
    Args:
        csv_file_path (str): Path to input CSV file
        output_file_path (str): Path to output CSV file
        sample_size (int): Number of URLs to sample
    """
    # Read URLs from CSV
    try:
        df = pd.read_csv(csv_file_path)
        # Check what columns are available
        print(f"Available columns: {df.columns.tolist()}")
        
        # Try to find the URL column - it might be 'urls' or 'url' or something else
        url_column = None
        for col in df.columns:
            if 'url' in col.lower():
                url_column = col
                break
        
        if url_column is None:
            # If no URL column found, use the first column
            url_column = df.columns[0]
            print(f"No URL column found, using first column: {url_column}")
        
        urls = df[url_column].dropna().tolist()
        print(f"Found {len(urls)} URLs in column '{url_column}'")
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    print(f"Found {len(urls)} URLs in CSV")
    
    # Sample URLs (or less if not enough)
    if len(urls) > sample_size:
        sampled_urls = urls[:sample_size]
        print(f"Sampling first {sample_size} URLs for scraping")
    else:
        sampled_urls = urls
        print(f"Sampling all {len(sampled_urls)} URLs for scraping")
    
    results = []
    
    # Process URLs in smaller batches to avoid overwhelming the server
    batch_size = 3
    for i in range(0, len(sampled_urls), batch_size):
        batch_urls = sampled_urls[i:i+batch_size]
        
        # Create tasks for the batch
        tasks = [scrape_product_info(url) for url in batch_urls]
        
        # Execute batch
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in the batch
        for j, result in enumerate(batch_results):
            if isinstance(result, Exception):
                print(f"❌ Exception for {batch_urls[j]}: {result}")
                results.append({
                    'page_url': batch_urls[j],
                    'page_title': None,
                    'product_name': None,
                    'visit_website_url': None,
                    'additional_url': None
                })
            else:
                results.append(result)
        
        print(f"Completed batch {i//batch_size + 1}/{(len(sampled_urls) + batch_size - 1)//batch_size}")
        
        # Wait between batches to be respectful
        if i + batch_size < len(sampled_urls):
            await asyncio.sleep(3)
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file_path, index=False)
    print(f"\n✅ Results saved to: {output_file_path}")
    
    # Print summary
    successful_scrapes = sum(1 for r in results if r['product_name'] is not None)
    print(f"Successfully scraped: {successful_scrapes}/{len(sampled_urls)} products")
    
    return results

async def run_single_scrape():
    """Run a single scraping session"""
    # Define input and output file paths
    csv_file_path = TOOLS_CSV_PATH
    output_file_path = get_timestamped_filename("scraped_results.csv")
    
    logger.info("Starting Product Hunt scraper (server optimized)...")
    logger.info(f"Input CSV: {csv_file_path}")
    logger.info(f"Output CSV: {output_file_path}")
    
    # Sample websites - adjust sample_size as needed
    results = await scrape_csv_file(csv_file_path, output_file_path, sample_size=SAMPLE_SIZE)
    
    # Display first few results
    if results:
        logger.info("\nFirst 3 results:")
        for i, result in enumerate(results[:3]):
            logger.info(f"{i+1}. {result['product_name']} - {result['visit_website_url']}")
    
    return results

async def cron_job():
    """Main cron job function that runs continuously"""
    logger.info(f"Starting cron job - will run every {CRON_INTERVAL_HOURS} hours")
    logger.info("Press Ctrl+C to stop the cron job")
    
    while not shutdown_flag:
        try:
            start_time = datetime.now()
            logger.info(f"=== Starting scraping session at {start_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
            
            # Run the scraper
            results = await run_single_scrape()
            
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"=== Scraping session completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
            logger.info(f"Session duration: {duration}")
            
            if results:
                successful_scrapes = sum(1 for r in results if r['product_name'] is not None)
                logger.info(f"Session summary: {successful_scrapes}/{len(results)} products scraped successfully")
            
            # Calculate next run time
            next_run = start_time + timedelta(hours=CRON_INTERVAL_HOURS)
            logger.info(f"Next scraping session scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Wait until next scheduled time or until shutdown is requested
            wait_seconds = CRON_INTERVAL_HOURS * 3600
            while wait_seconds > 0 and not shutdown_flag:
                # Check every minute for shutdown signal
                sleep_interval = min(60, wait_seconds)
                await asyncio.sleep(sleep_interval)
                wait_seconds -= sleep_interval
                
                if shutdown_flag:
                    logger.info("Shutdown requested, stopping cron job")
                    break
            
        except Exception as e:
            logger.error(f"Error in cron job: {e}")
            logger.info("Waiting 1 hour before retrying...")
            await asyncio.sleep(3600)  # Wait 1 hour before retrying
    
    logger.info("Cron job stopped")

async def main():
    """Main function - can run single scrape or cron job"""
    if len(sys.argv) > 1 and sys.argv[1] == "--cron":
        # Run as cron job
        await cron_job()
    else:
        # Run single scrape
        await run_single_scrape()

if __name__ == "__main__":
    asyncio.run(main())