import asyncio
import pandas as pd
import re
import random
import platform
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup
from playwright_stealth import stealth_async
import time
from openpyxl.utils.exceptions import IllegalCharacterError

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.48",
]

# Remove illegal characters for Excel (openpyxl)
def remove_illegal_chars(text):
    if not isinstance(text, str):
        return text
    # Remove illegal unicode characters for openpyxl
    # openpyxl uses the following regex for illegal chars: [\x00-\x08\x0b\x0c\x0e-\x1f]
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

# Block non-document resources
async def block_aggressively(route, request):
    if request.resource_type not in ["document"]:
        await route.abort()
    else:
        await route.continue_()

# Extract scientist data from Google Scholar page
async def extract_scientist_data(page):
    html_content = await page.content()
    soup = BeautifulSoup(html_content, 'html.parser')
    
    scientists = []
    
    # Debug: Print page title and some content
    print(f"Page title: {soup.title.string if soup.title else 'No title'}")
    
    # Find all scientist cards
    scientist_cards = soup.find_all('div', class_='gsc_1usr')
    print(f"Found {len(scientist_cards)} scientist cards")
    
    # If no cards found, try alternative selectors
    if len(scientist_cards) == 0:
        print("No gsc_1usr cards found, trying alternative selectors...")
        # Try different possible selectors
        alternative_selectors = [
            'div[class*="gsc_1usr"]',
            'div[class*="gs_ai"]',
            '.gs_ai_t',
            'div[class*="scholar"]'
        ]
        
        for selector in alternative_selectors:
            cards = soup.select(selector)
            print(f"Selector '{selector}' found {len(cards)} elements")
            if len(cards) > 0:
                scientist_cards = cards
                break
    
    # Debug: Print some HTML content to understand structure
    if len(scientist_cards) == 0:
        print("Still no cards found. Printing first 1000 chars of HTML for debugging:")
        print(html_content[:1000])
        return scientists
    
    for i, card in enumerate(scientist_cards):
        try:
            print(f"Processing card {i+1}:")
            
            # Extract name
            name_element = card.find('h3', class_='gs_ai_name')
            if not name_element:
                name_element = card.find('h3')  # Try without class
            if not name_element:
                name_element = card.find('a')  # Try finding any link
            
            name = name_element.get_text(strip=True) if name_element else "N/A"
            print(f"  Name: {name}")
            
            # Extract profile URL
            profile_link = name_element.find('a') if name_element else None
            if not profile_link and name_element and name_element.name == 'a':
                profile_link = name_element
            profile_url = "https://scholar.google.com" + profile_link['href'] if profile_link and profile_link.get('href') else "N/A"
            print(f"  Profile URL: {profile_url}")
            
            # Extract affiliation
            affiliation_element = card.find('div', class_='gs_ai_aff')
            if not affiliation_element:
                affiliation_element = card.find('div', string=re.compile(r'.*'))
            affiliation = affiliation_element.get_text(strip=True) if affiliation_element else "N/A"
            print(f"  Affiliation: {affiliation}")
            
            # Extract email
            email_element = card.find('div', class_='gs_ai_eml')
            email = email_element.get_text(strip=True) if email_element else "N/A"
            print(f"  Email: {email}")
            
            # Extract citation count
            citation_element = card.find('div', class_='gs_ai_cby')
            citations = citation_element.get_text(strip=True) if citation_element else "N/A"
            print(f"  Citations: {citations}")
            
            # Extract research interests/keywords
            interests_element = card.find('div', class_='gs_ai_int')
            interests = []
            if interests_element:
                interest_links = interests_element.find_all('a', class_='gs_ai_one_int')
                interests = [link.get_text(strip=True) for link in interest_links]
            interests_text = ", ".join(interests) if interests else "N/A"
            print(f"  Research interests: {interests_text}")
            
            # Extract profile image URL
            img_element = card.find('img')
            image_url = img_element['src'] if img_element and img_element.get('src') else "N/A"
            print(f"  Image URL: {image_url}")
            
            scientist_data = {
                'name': remove_illegal_chars(name),
                'profile_url': profile_url,
                'affiliation': remove_illegal_chars(affiliation),
                'email': remove_illegal_chars(email),
                'citations': remove_illegal_chars(citations),
                'research_interests': remove_illegal_chars(interests_text),
                'image_url': image_url
            }
            
            scientists.append(scientist_data)
            print(f"  Added scientist: {name}")
            
        except Exception as e:
            print(f"Error extracting scientist data from card {i+1}: {e}")
            continue
    
    return scientists

# Visit a single Google Scholar URL and return scientist data
async def scrape_scholar_page(context, url):
    page = await context.new_page()
    await stealth_async(page)

    user_agent = random.choice(USER_AGENTS)
    await page.set_extra_http_headers({'User-Agent': user_agent})
    await page.evaluate("() => { Object.defineProperty(navigator, 'webdriver', { get: () => undefined }); }")

    # Don't block resources for Google Scholar to avoid issues
    # await page.route("**/*", block_aggressively)

    try:
        print(f"Navigating to: {url}")
        
        # Try to navigate with different wait strategies
        try:
            await page.goto(url, timeout=45000, wait_until='domcontentloaded')
        except Exception as e:
            print(f"First navigation attempt failed: {e}")
            # Try again with networkidle
            await page.goto(url, timeout=45000, wait_until='networkidle')
        
        await asyncio.sleep(5)  # Increased wait time
        
        # Check if we got blocked or redirected
        current_url = page.url
        print(f"Current URL after navigation: {current_url}")
        
        # Check if redirected to sign-in page
        if "accounts.google.com" in current_url or "signin" in current_url:
            print("Redirected to sign-in page. Waiting for manual login...")
            print("Please log in to your Google account in the browser window.")
            print("The script will continue automatically after successful login.")
            
            # Wait for user to complete login (up to 5 minutes)
            max_wait_time = 300  # 5 minutes
            wait_interval = 5    # Check every 5 seconds
            waited_time = 0
            
            while waited_time < max_wait_time:
                await asyncio.sleep(wait_interval)
                waited_time += wait_interval
                
                current_url = page.url
                if "scholar.google.com" in current_url and "signin" not in current_url:
                    print("Successfully logged in! Continuing with scraping...")
                    break
                elif waited_time % 30 == 0:  # Remind every 30 seconds
                    print(f"Still waiting for login... ({waited_time}s elapsed)")
            
            if waited_time >= max_wait_time:
                print("Timeout waiting for login")
                await page.close()
                return {'url': url, 'scientists': [], 'count': 0, 'error': 'Login timeout', 'status': 'error'}
        
        # Check for captcha or blocking
        page_content = await page.content()
        if "captcha" in page_content.lower() or "robot" in page_content.lower():
            print("Detected captcha or robot check")
            await page.close()
            return {'url': url, 'scientists': [], 'count': 0, 'error': 'Captcha detected', 'status': 'error'}
        
        scientists = await extract_scientist_data(page)
        await page.close()
        
        return {
            'url': url, 
            'scientists': scientists, 
            'count': len(scientists),
            'status': 'success'
        }
        
    except PlaywrightTimeoutError:
        print(f"Timeout occurred while scraping {url}")
        await page.close()
        return {'url': url, 'scientists': [], 'count': 0, 'error': 'TimeoutError', 'status': 'error'}
    except Exception as e:
        print(f"Error occurred while scraping {url}: {e}")
        await page.close()
        return {'url': url, 'scientists': [], 'count': 0, 'error': str(e), 'status': 'error'}

# Process a batch of scraping tasks and save progress
async def process_tasks(pending_tasks, all_scientists, df, output_path, scrape_counter, batch_number):
    results = await asyncio.gather(*(t[0] for t in pending_tasks), return_exceptions=True)

    for i, result in enumerate(results):
        task, url, idx = pending_tasks[i]

        if isinstance(result, Exception):
            print(f"Error occurred while scraping {url}: {result}")
            continue
        else:
            if result.get('status') == 'error':
                print(f"Error scraping {result['url']}: {result.get('error', 'Unknown error')}")
                continue
            else:
                scientists = result.get('scientists', [])
                all_scientists.extend(scientists)
                scrape_counter += 1
                print(f"Successfully scraped {len(scientists)} scientists from {url}")

        # Save progress every 10 pages
        if scrape_counter % 10 == 0:
            try:
                # Convert to DataFrame and save
                temp_df = pd.DataFrame(all_scientists)
                temp_df.to_excel(output_path, index=False)
                print(f"Progress saved after {scrape_counter} pages to {output_path}")
            except IllegalCharacterError as e:
                print(f"Skipped saving at {scrape_counter} due to IllegalCharacterError: {e}")
            except Exception as e:
                print(f"Skipped saving at {scrape_counter} due to error: {e}")

    print(f"Batch {batch_number} processed.")
    return scrape_counter

async def main(use_gpu=True, start_page=0, num_pages=1, headless=True):
    output_path = "/Users/macbook/Library/CloudStorage/OneDrive-HvA/rmai/scientist/scientists_data.xlsx"
    
    # Base URL for Google Scholar AI scientists search
    base_url = "https://scholar.google.com/citations?view_op=search_authors&hl=en&mauthors=label:artificial_intelligence"
    
    all_scientists = []
    scrape_counter = 0

    async with async_playwright() as p:
        launch_options = {
            "headless": headless,
            "args": [
                '--no-sandbox',
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--disable-extensions',
                '--disable-plugins',
                '--disable-images',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
                '--disable-features=TranslateUI',
                '--disable-ipc-flooding-protection',
                '--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ]
        }
        
        if use_gpu and platform.system() == "Darwin":
            print("Running with GPU acceleration enabled")
            launch_options["args"].extend([
                '--use-gl=angle',
                '--use-angle=metal',
                '--enable-gpu-rasterization',
                '--enable-gpu',
                '--ignore-gpu-blocklist',
                '--enable-accelerated-2d-canvas',
                '--disable-gpu-sandbox',
                '--disable-features=UseOzonePlatform',
                '--enable-features=Metal'
            ])
        else:
            print("Running without GPU acceleration")

        browser_start_time = time.time()
        # Try using Firefox instead of Chromium to avoid detection
        try:
            browser = await p.chromium.launch(headless=headless, channel="chrome")
            print("Using Chromium browser")
        except Exception as e:
            print(f"Chromium launch failed: {e}")
            print("Falling back to Chromium with additional stealth settings")
            browser = await p.chromium.launch(**launch_options)
        browser_launch_time = time.time() - browser_start_time
        print(f"Browser launch time: {browser_launch_time:.2f} seconds")

        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            locale='en-US',
            timezone_id='America/New_York',
            extra_http_headers={
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
        )
        pending_tasks = []
        batch_number = 0

        # Generate URLs for the specified number of pages
        for page_num in range(start_page, start_page + num_pages):
            if page_num == 0:
                url = base_url
            else:
                astart = page_num * 10
                url = f"{base_url}&astart={astart}"
            
            print(f"Adding to scrape queue: Page {page_num + 1} - {url}")
            task = asyncio.create_task(scrape_scholar_page(context, url))
            pending_tasks.append((task, url, page_num))

            # Process batches of 5 tasks (to be conservative with Google Scholar)
            if len(pending_tasks) >= 5:
                batch_number += 1
                print(f"Processing batch {batch_number} with {len(pending_tasks)} tasks...")
                scrape_counter = await process_tasks(pending_tasks, all_scientists, None, output_path, scrape_counter, batch_number)
                pending_tasks = []
                
                # Add delay between batches to be respectful to Google Scholar
                await asyncio.sleep(5)

        # Process any remaining tasks
        if pending_tasks:
            batch_number += 1
            print(f"Processing the remaining {len(pending_tasks)} tasks in batch {batch_number}...")
            scrape_counter = await process_tasks(pending_tasks, all_scientists, None, output_path, scrape_counter, batch_number)

        print("Scraping completed.")
        await context.close()
        await browser.close()

    # Final save
    try:
        if all_scientists:
            df = pd.DataFrame(all_scientists)
            df.to_excel(output_path, index=False)
            print(f"Final DataFrame saved to {output_path} with {len(all_scientists)} scientists")
        else:
            print("No scientists data to save")
    except IllegalCharacterError as e:
        print(f"Skipped final save due to IllegalCharacterError: {e}")
    except Exception as e:
        print(f"Skipped final save due to error: {e}")

if __name__ == "__main__":
    use_gpu = True if platform.system() == "Darwin" else True
    # Test with just one page first, and set headless=False for debugging
    asyncio.run(main(use_gpu=use_gpu, start_page=0, num_pages=1, headless=False))
