#!/usr/bin/env python3
"""
Simple test scraper to extract the title from a Product Hunt page.
This is a basic implementation for testing purposes.
"""

import asyncio
import sys
from playwright.async_api import async_playwright

async def get_product_title(url):
    """
    Extract the title from a Product Hunt product page.
    
    Args:
        url (str): The Product Hunt product URL
        
    Returns:
        str: The product title or None if not found
    """
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            # Navigate to the page
            await page.goto(url, wait_until="domcontentloaded")
            
            # Wait a bit for content to load
            await asyncio.sleep(2)
            
            # Try multiple possible selectors for the title
            title_selectors = [
                "h1",  # Most common
                "h1.color-darker-grey",  # From the existing scraper
                "[data-test='product-name']",  # Product Hunt specific
                ".product-name",
                "h1[data-test='product-name']"
            ]
            
            title = None
            for selector in title_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        title = await element.inner_text()
                        title = title.strip()
                        if title:
                            break
                except:
                    continue
            
            return title
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
        finally:
            await browser.close()

async def test_title_scraper():
    """Test the title scraper with a sample Product Hunt URL."""
    # Test with a well-known Product Hunt product
    test_url = "https://www.producthunt.com/products/figma"
    
    print(f"Testing title extraction from: {test_url}")
    title = await get_product_title(test_url)
    
    if title:
        print(f"✅ Success! Title extracted: '{title}'")
    else:
        print("❌ Failed to extract title")
    
    return title

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If URL provided as argument, use it
        url = sys.argv[1]
        print(f"Extracting title from: {url}")
        title = asyncio.run(get_product_title(url))
        if title:
            print(f"Title: {title}")
        else:
            print("Could not extract title")
    else:
        # Run test with default URL
        asyncio.run(test_title_scraper())