#!/usr/bin/env python3
"""
Enhanced Product Hunt scraper to extract:
- Page title
- Page URL
- Visit website button URL
- Product name

This version is optimized for server deployment with headless browser and no proxy dependencies.
"""

import asyncio
import pandas as pd
import csv
from playwright.async_api import async_playwright
from urllib.parse import urljoin
import random

# --- Stealth async patch ---
# You need to install playwright-stealth: pip install playwright-stealth
try:
    from playwright_stealth import stealth_async
    STEALTH_AVAILABLE = True
except ImportError:
    print("Warning: playwright-stealth not installed. Stealth mode will not be used.")
    STEALTH_AVAILABLE = False

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
import os
import pandas as pd
import asyncio
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Load companies from CSV file
TOOLS_CSV_PATH = 'PH_check1.csv'

def load_companies_from_csv():
    """Load AI tool companies from CSV file"""
    if os.path.exists(TOOLS_CSV_PATH):
        df_tools = pd.read_csv(TOOLS_CSV_PATH)
        tool_names = df_tools['urls'].dropna().unique().tolist()
        tool_names = tool_names[:100]  # Limit to first 100
        logger.info(f"Loaded {len(tool_names)} tool names from CSV")
        return tool_names, df_tools
    else:
        logger.warning(f"CSV file not found: {TOOLS_CSV_PATH}")
        return [], None

async def main():
    """Main function to run the scraper."""
    # Define input and output file paths
    csv_file_path = TOOLS_CSV_PATH  # Use the CSV path defined at the top
    output_file_path = "product_hunt/scraped_results4.csv"
    
    print("Starting Product Hunt scraper (server optimized)...")
    print(f"Input CSV: {csv_file_path}")
    print(f"Output CSV: {output_file_path}")
    
    # Sample websites - adjust sample_size as needed
    results = await scrape_csv_file(csv_file_path, output_file_path, sample_size=10)
    
    # Display first few results
    if results:
        print("\nFirst 3 results:")
        for i, result in enumerate(results[:3]):
            print(f"{i+1}. {result['product_name']} - {result['visit_website_url']}")

if __name__ == "__main__":
    asyncio.run(main())