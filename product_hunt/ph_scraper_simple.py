import asyncio
import pandas as pd
import random
import logging
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from playwright_stealth import stealth_async

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Random user agents
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.48',
    'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0'
]

# Function to block non-document resources
async def block_aggressively(route, request):
    if request.resource_type not in ["document"]:
        await route.abort()
    else:
        await route.continue_()

# Function to extract only the specific data we need
async def extract_specific_data(page):
    data = {}
    
    try:
        # Wait for the page to load
        await page.wait_for_load_state('networkidle', timeout=15000)
        
        # Get page title and URL
        data['page_title'] = await page.title()
        data['page_url'] = await page.url()
        
        # Check if we're blocked
        if "Product Hunt – The best new products in tech." in data['page_title']:
            logging.warning("Page blocked/redirected")
            data['tool_name'] = 'BLOCKED'
            return data
        
        # Extract tool name using the specific selector
        tool_name_selector = 'h1[data-sentry-component="LegacyText"]'
        tool_name_element = await page.query_selector(tool_name_selector)
        if tool_name_element:
            data['tool_name'] = await tool_name_element.inner_text()
            logging.info(f"Found tool name: {data['tool_name']}")
        else:
            # Try alternative selector with classes
            tool_name_selector_alt = 'h1.text-16.font-normal.text-dark-gray.text-24.font-semibold.text-gray-900'
            tool_name_element = await page.query_selector(tool_name_selector_alt)
            if tool_name_element:
                data['tool_name'] = await tool_name_element.inner_text()
                logging.info(f"Found tool name (alt): {data['tool_name']}")
            else:
                data['tool_name'] = 'N/A'
                logging.warning("Could not find tool name with any selector")
        
    except Exception as e:
        logging.error(f"Error extracting data: {e}")
        data = {
            'tool_name': 'N/A',
            'page_title': 'N/A',
            'page_url': 'N/A'
        }
    
    return data

# Function to scrape a single Product Hunt page
async def scrape_page(context, url):
    page = await context.new_page()
    await stealth_async(page)
    
    # Set random user agent
    user_agent = random.choice(USER_AGENTS)
    await page.set_extra_http_headers({'User-Agent': user_agent})
    await page.evaluate("() => { Object.defineProperty(navigator, 'webdriver', { get: () => undefined }); }")
    
    # Block non-document resources
    await page.route("**/*", block_aggressively)
    
    try:
        logging.info(f"Navigating to: {url}")
        await page.goto(url, timeout=30000)
        
        # Wait a bit
        await asyncio.sleep(random.uniform(2, 4))
        
        # Extract the specific data
        data = await extract_specific_data(page)
        data['url'] = url
        
        await page.close()
        return data
        
    except PlaywrightTimeoutError:
        logging.error(f"Timeout for {url}")
        await page.close()
        return {
            'url': url,
            'tool_name': 'TIMEOUT',
            'page_title': 'TIMEOUT',
            'page_url': 'TIMEOUT'
        }
    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")
        await page.close()
        return {
            'url': url,
            'tool_name': 'ERROR',
            'page_title': 'ERROR',
            'page_url': 'ERROR'
        }

# Function to process tasks
async def process_tasks(pending_tasks, scraped_data, batch_number):
    results = await asyncio.gather(*(t[0] for t in pending_tasks), return_exceptions=True)
    
    for i, result in enumerate(results):
        task, full_url_scraped = pending_tasks[i]
        if isinstance(result, Exception):
            logging.error(f"Exception for {full_url_scraped}: {result}")
            scraped_data.append({
                'url': full_url_scraped,
                'tool_name': 'EXCEPTION',
                'page_title': 'EXCEPTION',
                'page_url': 'EXCEPTION'
            })
        else:
            scraped_data.append(result)
    
    # Save data
    logging.info(f"Saving batch {batch_number}...")
    scraped_df = pd.DataFrame(scraped_data)
    scraped_df.to_csv(f"ph_simple_data_batch_{batch_number}.csv", index=False)
    logging.info(f"Batch {batch_number} saved.")

# Main function
async def main():
    # Load the dataset
    
    dataset_path = "/Users/macbook/Library/CloudStorage/OneDrive-HvA/rmai/website/extract/product_hunt/PH_check1.csv"
    try:
        df = pd.read_csv(dataset_path)
        print(f"Dataset loaded with {len(df)} rows.")
        
        # Take first 5 URLs for testing
        test_urls = df['urls'].head(5).tolist()
        print(f"Testing with {len(test_urls)} URLs: {test_urls}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    scraped_data = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        
        pending_tasks = []
        batch_number = 1
        
        for url in test_urls:
            print(f"Adding to queue: {url}")
            task = asyncio.create_task(scrape_page(context, url))
            pending_tasks.append((task, url))
        
        print(f"Processing batch {batch_number} with {len(pending_tasks)} tasks...")
        await process_tasks(pending_tasks, scraped_data, batch_number)
        
        print("Scraping completed.")
        await context.close()
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main()) 