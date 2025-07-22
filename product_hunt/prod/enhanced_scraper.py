#!/usr/bin/env python3
"""
High-Performance Product Hunt scraper for 24,000 URLs in 24 hours
- Parallel browser instances (5-10 concurrent browsers)
- Optimized human behavior mimicking 
- Progress tracking and resume capability
- Error handling and retry logic
"""

import asyncio
import pandas as pd
import csv
from playwright.async_api import async_playwright
from urllib.parse import urljoin
import random
import time
import json
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Stealth async patch ---
# You need to install playwright-stealth: pip install playwright-stealth
try:
    from playwright_stealth import stealth_async
    STEALTH_AVAILABLE = True
except ImportError:
    print("Warning: playwright-stealth not installed. Stealth mode will not be used.")
    STEALTH_AVAILABLE = False

# High-performance human behavior simulation
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
]

VIEWPORTS = [
    {"width": 1920, "height": 1080},
    {"width": 1366, "height": 768},
    {"width": 1440, "height": 900},
    {"width": 1536, "height": 864},
    {"width": 1280, "height": 720}
]

# Performance-optimized delays for 24-hour target
async def fast_human_delay(min_ms=50, max_ms=300):
    """Very fast but still human-like delays"""
    delay = random.uniform(min_ms, max_ms) / 1000
    await asyncio.sleep(delay)

async def quick_read_simulation(text_length):
    """Quick reading simulation - 0.5-2 seconds max"""
    if text_length:
        read_time = min(2, max(0.2, text_length / 1000))
        await asyncio.sleep(read_time + random.uniform(0.1, 0.3))

async def minimal_mouse_movement(page):
    """Minimal mouse movement for performance"""
    try:
        if random.random() < 0.3:  # Only 30% of the time
            viewport = await page.viewport_size()
            if viewport:
                x = random.randint(200, viewport['width'] - 200)
                y = random.randint(200, viewport['height'] - 200)
                await page.mouse.move(x, y)
                await fast_human_delay(50, 200)
    except Exception:
        pass

async def quick_scroll(page):
    """Quick scroll simulation"""
    try:
        if random.random() < 0.4:  # Only 40% of the time
            scroll_amount = random.randint(100, 400)
            await page.mouse.wheel(0, scroll_amount)
            await fast_human_delay(100, 300)
    except Exception:
        pass

async def minimal_human_behavior(page):
    """Minimal human behavior for maximum speed"""
    if random.random() < 0.4:  # Only 40% chance to do any behavior
        behavior = random.choice([minimal_mouse_movement, quick_scroll])
        await behavior(page)

class ProgressTracker:
    """Track progress and save state for resumption"""
    
    def __init__(self, progress_file="scraper_progress.json"):
        self.progress_file = progress_file
        self.completed_urls = set()
        self.failed_urls = set()
        self.results = []
        self.save_counter = 0
        self.load_progress()
    
    def load_progress(self):
        """Load existing progress if available"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.completed_urls = set(data.get('completed_urls', []))
                    self.failed_urls = set(data.get('failed_urls', []))
                    self.save_counter = data.get('save_counter', 0)
                    logger.info(f"Loaded progress: {len(self.completed_urls)} completed, {len(self.failed_urls)} failed, save #{self.save_counter}")
            except Exception as e:
                logger.error(f"Error loading progress: {e}")
    
    def save_progress(self):
        """Save current progress every 20 URLs"""
        try:
            self.save_counter += 1
            data = {
                'completed_urls': list(self.completed_urls),
                'failed_urls': list(self.failed_urls),
                'timestamp': datetime.now().isoformat(),
                'save_counter': self.save_counter,
                'total_completed': len(self.completed_urls),
                'total_failed': len(self.failed_urls)
            }
            with open(self.progress_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"📊 Progress saved #{self.save_counter}: {len(self.completed_urls)} completed, {len(self.failed_urls)} failed")
            
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    def is_completed(self, url):
        """Check if URL is already completed"""
        return url in self.completed_urls
    
    def mark_completed(self, url):
        """Mark URL as completed"""
        self.completed_urls.add(url)
    
    def mark_failed(self, url):
        """Mark URL as failed"""
        self.failed_urls.add(url)
    
    def get_stats(self):
        """Get current progress statistics"""
        return {
            'completed': len(self.completed_urls),
            'failed': len(self.failed_urls),
            'total_processed': len(self.completed_urls) + len(self.failed_urls),
            'save_counter': self.save_counter
        }

async def scrape_single_product_fast(url, browser_id):
    """
    Ultra-fast product scraping with minimal human behavior
    Target: 3-8 seconds per page
    """
    async with async_playwright() as p:
        user_agent = random.choice(USER_AGENTS)
        viewport = random.choice(VIEWPORTS)
        
        browser = await p.chromium.launch(
            headless=True,  # Headless for maximum performance
            args=[
                '--no-first-run',
                '--disable-blink-features=AutomationControlled',
                '--disable-features=VizDisplayCompositor',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--disable-ipc-flooding-protection',
            ]
        )
        
        context = await browser.new_context(
            user_agent=user_agent,
            viewport=viewport,
            locale='en-US',
            timezone_id='America/New_York',
        )
        
        page = await context.new_page()

        if STEALTH_AVAILABLE:
            await stealth_async(page)
        
        try:
            start_time = time.time()
            
            # Minimal pre-navigation delay
            await fast_human_delay(100, 500)
            
            # Navigate with shorter timeout
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            
            # Quick initial wait
            await fast_human_delay(300, 800)
            
            # Optional minimal behavior
            await minimal_human_behavior(page)
            
            # Quick content load wait
            await asyncio.sleep(random.uniform(1, 3))
            
            # Extract page title quickly
            page_title = await page.title()
            
            # Quick title reading simulation
            if page_title:
                await quick_read_simulation(len(page_title))
            
            # Extract product name with minimal delays
            product_name = None
            product_selectors = [
                "h1",
                "h1.color-darker-grey", 
                "[data-test='product-name']",
                ".product-name"
            ]
            
            for selector in product_selectors:
                try:
                    await fast_human_delay(50, 200)
                    element = await page.query_selector(selector)
                    if element:
                        if random.random() < 0.3:  # Only hover 30% of the time
                            await element.hover()
                            await fast_human_delay(100, 300)
                        
                        product_name = await element.inner_text()
                        product_name = product_name.strip()
                        if product_name:
                            await quick_read_simulation(len(product_name))
                            break
                except Exception:
                    continue
            
            # Quick behavior between extractions
            if random.random() < 0.2:  # Only 20% chance
                await minimal_human_behavior(page)
            
            # Extract visit website URL quickly
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
                    await fast_human_delay(50, 200)
                    element = await page.query_selector(selector)
                    if element:
                        if random.random() < 0.3:  # Only hover 30% of the time
                            await element.hover()
                            await fast_human_delay(100, 300)
                        
                        visit_url = await element.get_attribute("href")
                        if visit_url:
                            if visit_url.startswith('http'):
                                break
                            else:
                                visit_url = urljoin(url, visit_url)
                                break
                except Exception:
                    continue
            
            # Extract additional URL quickly
            additional_url = None
            try:
                span_selectors = [
                    'span[class*="max-w-[260px]"][class*="truncate"][class*="font-semibold"]',
                    'span.max-w-\\[260px\\].truncate.font-semibold',
                    'span[class="max-w-[260px] truncate font-semibold"]',
                    'span.font-semibold.truncate'
                ]
                
                for selector in span_selectors:
                    try:
                        await fast_human_delay(50, 150)
                        span_element = await page.query_selector(selector)
                        if span_element:
                            if random.random() < 0.2:  # Only hover 20% of the time
                                await span_element.hover()
                                await fast_human_delay(100, 250)
                            
                            text_content = await span_element.inner_text()
                            text_content = text_content.strip()
                            if '.' in text_content and '/' in text_content:
                                additional_url = text_content
                                break
                    except Exception:
                        continue
                        
            except Exception:
                pass
            
            # Minimal final behavior
            if random.random() < 0.2:  # Only 20% chance
                await minimal_human_behavior(page)
            
            # Quick final delay
            await fast_human_delay(200, 600)
            
            elapsed_time = time.time() - start_time
            
            result = {
                'page_url': url,
                'page_title': page_title,
                'product_name': product_name,
                'visit_website_url': visit_url,
                'additional_url': additional_url,
                'scrape_time': elapsed_time,
                'browser_id': browser_id
            }
            
            logger.info(f"✅ Browser {browser_id}: {product_name} ({elapsed_time:.2f}s)")
            return result
            
        except Exception as e:
            logger.error(f"❌ Browser {browser_id} error for {url}: {e}")
            return {
                'page_url': url,
                'page_title': None,
                'product_name': None,
                'visit_website_url': None,
                'additional_url': None,
                'scrape_time': None,
                'browser_id': browser_id,
                'error': str(e)
            }
        finally:
            await fast_human_delay(100, 400)
            await browser.close()

async def process_url_batch(urls, browser_id, progress_tracker, semaphore, output_file_path):
    """Process a batch of URLs with one browser instance"""
    results = []
    
    for i, url in enumerate(urls):
        async with semaphore:  # Limit concurrent requests
            if progress_tracker.is_completed(url):
                logger.info(f"Browser {browser_id}: Skipping already completed URL: {url}")
                continue
            
            try:
                result = await scrape_single_product_fast(url, browser_id)
                results.append(result)
                progress_tracker.mark_completed(url)
                
                # Quick delay between URLs (1-3 seconds)
                if i < len(urls) - 1:
                    delay = random.uniform(1, 3)
                    await asyncio.sleep(delay)
                
                # Save progress and results every 20 URLs
                if (i + 1) % 20 == 0:
                    # Save JSON progress
                    progress_tracker.save_progress()
                    
                    # Save CSV results (append mode for incremental saving)
                    try:
                        # Create backup filename for this browser batch
                        backup_filename = f"backup_browser_{browser_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        results_df = pd.DataFrame(results)
                        results_df.to_csv(backup_filename, index=False)
                        logger.info(f"Browser {browser_id}: Saved backup after {i + 1} URLs to {backup_filename}")
                        
                        # Also append to main results file if it exists
                        if os.path.exists(output_file_path):
                            results_df.to_csv(output_file_path, mode='a', header=False, index=False)
                        else:
                            results_df.to_csv(output_file_path, index=False)
                        
                    except Exception as save_error:
                        logger.error(f"Browser {browser_id}: Error saving backup: {save_error}")
                    
                    logger.info(f"Browser {browser_id}: Completed {i + 1}/{len(urls)} URLs - Progress and results saved!")
                
            except Exception as e:
                logger.error(f"Browser {browser_id}: Failed to process {url}: {e}")
                progress_tracker.mark_failed(url)
                results.append({
                    'page_url': url,
                    'page_title': None,
                    'product_name': None,
                    'visit_website_url': None,
                    'additional_url': None,
                    'error': str(e),
                    'browser_id': browser_id
                })
    
    return results

async def scrape_24k_urls_parallel(csv_file_path, output_file_path, num_browsers=8, max_concurrent=4):
    """
    Scrape 24,000 URLs in 24 hours using parallel browsers
    
    Args:
        csv_file_path: Path to CSV with URLs
        output_file_path: Output CSV file
        num_browsers: Number of parallel browser instances (8 recommended)
        max_concurrent: Maximum concurrent requests per browser (4 recommended)
    """
    progress_tracker = ProgressTracker()
    
    # Load URLs
    try:
        df = pd.read_csv(csv_file_path)
        url_column = None
        for col in df.columns:
            if 'url' in col.lower():
                url_column = col
                break
        
        if url_column is None:
            url_column = df.columns[0]
        
        all_urls = df[url_column].dropna().tolist()
        logger.info(f"Loaded {len(all_urls)} URLs from CSV")
        
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return
    
    # Filter out already completed URLs
    remaining_urls = [url for url in all_urls if not progress_tracker.is_completed(url)]
    logger.info(f"Remaining URLs to process: {len(remaining_urls)}")
    
    if not remaining_urls:
        logger.info("All URLs already completed!")
        return
    
    # Split URLs among browsers
    urls_per_browser = len(remaining_urls) // num_browsers
    url_batches = []
    
    for i in range(num_browsers):
        start_idx = i * urls_per_browser
        if i == num_browsers - 1:  # Last browser gets remaining URLs
            end_idx = len(remaining_urls)
        else:
            end_idx = (i + 1) * urls_per_browser
        
        batch = remaining_urls[start_idx:end_idx]
        if batch:
            url_batches.append(batch)
    
    logger.info(f"Split URLs into {len(url_batches)} batches for {num_browsers} browsers")
    for i, batch in enumerate(url_batches):
        logger.info(f"Browser {i}: {len(batch)} URLs")
    
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Start parallel processing
    start_time = time.time()
    logger.info(f"Starting parallel scraping with {num_browsers} browsers at {datetime.now()}")
    
    # Create tasks for each browser
    tasks = []
    for i, url_batch in enumerate(url_batches):
        task = process_url_batch(url_batch, i, progress_tracker, semaphore, output_file_path)
        tasks.append(task)
    
    # Run all browsers in parallel
    all_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Flatten results
    final_results = []
    for browser_results in all_results:
        if isinstance(browser_results, Exception):
            logger.error(f"Browser failed with exception: {browser_results}")
        else:
            final_results.extend(browser_results)
    
    # Save final results
    try:
        # Merge all backup files into the final result
        all_backup_files = [f for f in os.listdir('.') if f.startswith('backup_browser_') and f.endswith('.csv')]
        
        if all_backup_files:
            logger.info(f"Found {len(all_backup_files)} backup files to merge")
            
            # Combine all backup files
            combined_data = []
            for backup_file in all_backup_files:
                try:
                    backup_df = pd.read_csv(backup_file)
                    combined_data.append(backup_df)
                    logger.info(f"Merged {len(backup_df)} rows from {backup_file}")
                except Exception as e:
                    logger.error(f"Error reading backup file {backup_file}: {e}")
            
            if combined_data:
                # Combine with any existing final results
                if final_results:
                    final_df = pd.DataFrame(final_results)
                    combined_data.append(final_df)
                
                # Create master results file
                master_df = pd.concat(combined_data, ignore_index=True)
                
                # Remove duplicates based on page_url
                master_df = master_df.drop_duplicates(subset=['page_url'], keep='first')
                
                # Save final combined results
                master_df.to_csv(output_file_path, index=False)
                logger.info(f"✅ Master results saved to: {output_file_path} ({len(master_df)} unique URLs)")
                
                # Create timestamped backup of final results
                timestamp_backup = f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                master_df.to_csv(timestamp_backup, index=False)
                logger.info(f"📋 Timestamped backup: {timestamp_backup}")
                
                final_results = master_df.to_dict('records')
        
        else:
            # No backup files, save final results normally
            if final_results:
                results_df = pd.DataFrame(final_results)
                results_df.to_csv(output_file_path, index=False)
                logger.info(f"✅ Results saved to: {output_file_path}")
        
        # Save final progress
        progress_tracker.save_progress()
        
        # Print comprehensive summary
        elapsed_time = time.time() - start_time
        successful_scrapes = sum(1 for r in final_results if r.get('product_name') is not None)
        stats = progress_tracker.get_stats()
        
        logger.info(f"""
        🎉 SCRAPING COMPLETE! 🎉
        ⏱️  Total time: {elapsed_time/3600:.2f} hours
        ✅ Successfully scraped: {successful_scrapes}/{len(final_results)} products
        📈 Average time per URL: {elapsed_time/len(final_results) if final_results else 0:.2f} seconds
        🚀 URLs per hour: {len(final_results)/(elapsed_time/3600) if elapsed_time > 0 else 0:.0f}
        💾 Progress saves: {stats['save_counter']} (every 20 URLs)
        📂 Backup files created: {len(all_backup_files)}
        🎯 Success rate: {(successful_scrapes/len(final_results)*100) if final_results else 0:.1f}%
        """)
        
        # Clean up backup files (optional)
        cleanup_backups = input("Delete backup files? (y/N): ").lower().strip() == 'y'
        if cleanup_backups:
            for backup_file in all_backup_files:
                try:
                    os.remove(backup_file)
                    logger.info(f"🗑️  Deleted backup: {backup_file}")
                except Exception as e:
                    logger.error(f"Error deleting {backup_file}: {e}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")

# Load companies from CSV file
TOOLS_CSV_PATH = '/Users/macbook/Library/CloudStorage/OneDrive-HvA/rmai/jobs/prod/test_limmy/product_hunt/prod/PH_check1.csv'

def load_companies_from_csv():
    """Load AI tool companies from CSV file"""
    if os.path.exists(TOOLS_CSV_PATH):
        df_tools = pd.read_csv(TOOLS_CSV_PATH)
        tool_names = df_tools['urls'].dropna().unique().tolist()
        # Use all URLs for 24k target
        logger.info(f"Loaded {len(tool_names)} tool names from CSV")
        return tool_names, df_tools
    else:
        logger.warning(f"CSV file not found: {TOOLS_CSV_PATH}")
        return [], None

async def main():
    """Main function for 24k URLs in 24 hours"""
    csv_file_path = TOOLS_CSV_PATH
    output_file_path = f"scraped_results_24k_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    logger.info("🚀 Starting 24K URL scraper for 24-hour target!")
    logger.info(f"📁 Input CSV: {csv_file_path}")
    logger.info(f"📁 Output CSV: {output_file_path}")
    
    # Start parallel scraping
    # 8 browsers, 4 concurrent per browser = effectively 32 concurrent requests
    await scrape_24k_urls_parallel(
        csv_file_path, 
        output_file_path, 
        num_browsers=8,  # 8 parallel browser instances
        max_concurrent=4  # 4 concurrent requests per browser
    )

if __name__ == "__main__":
    asyncio.run(main())