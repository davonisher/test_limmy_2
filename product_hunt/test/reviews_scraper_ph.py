# imoporting libraries
import re
import random
import logging
import pandas as pd
import asyncio
from random import randint
from playwright.async_api import async_playwright

# function to extract reviews from the page
async def extract_reviews_ph(page):
    comments_data_ph = []
    
    # Selector for name of the tool
    tool_name_selector = "h1.color-darker-grey"
    tool_name_element = await page.query_selector(tool_name_selector)
    tool_name = await tool_name_element.inner_text() if tool_name_element else 'N/A'

    # tool link selector
    tool_link_selector = "a.styles_format__k3_8m"
    tool_link_element = await page.query_selector(tool_link_selector)
    tool_link = await tool_link_element.get_attribute("href") if tool_link_element else 'N/A'

    # score selector
    score_selector = "div.styles_reviewPositive__JY_9N"
    score_element = await page.query_selector(score_selector)
    score_text = await score_element.inner_text() if score_element else 'N/A'

    # Extract just the score number (bijv. "3.9")
    #score = re.search(r'(\d+\\d+)/', score_text)
    #score = score.group(1) if score else 'N/A'
    
    # Extract amount of reviews
    all_time_reviews_amount = re.search(r'All time \((\d+) reviews\)', score_text)
    all_time_reviews_amount = all_time_reviews_amount.group(1) if all_time_reviews_amount else 'N/A' 
    
    # Extract the review score
    recent_tool_score = "div.styles_recentReviewRating__n9hRY"
    tool_element = await page.query_selector(recent_tool_score)
    recent_review_score = await tool_element.inner_text() if tool_element else 'N/A'
    
    # Extract the amount of recent reviews
    recent_reviews_selector = "div.flex.direction-row > span.color-lighter-grey.fontSize-14.fontWeight-400"
    recent_reviews_element = await page.query_selector(recent_reviews_selector)
    recent_reviews_text = await recent_reviews_element.inner_text() if recent_reviews_element else 'N/A'

    # using regex to extract the amount of recent reviews
    recent_reviews_amount = re.search(r'Recently \((\d+) reviews\)', recent_reviews_text)
    recent_reviews_amount = recent_reviews_amount.group(1) if recent_reviews_amount else 'N/A'

    # Get all review elements by their common starting ID
    comment_elements = await page.query_selector_all('div[id^="review-"]')
    
    # If there are no reviews, return the empty list
    if not comment_elements:
        return comments_data_ph
    # Loop through each review element to extract the data
    for comment_element in comment_elements:
        # Get the ID from the review element
        review_id = await comment_element.get_attribute("id")
        
        # Extract the review text
        review_text_selector = "div.styles_htmlText__iftLe"
        review_text_element = await comment_element.query_selector(review_text_selector)
        review_text = await review_text_element.inner_text() if review_text_element else 'N/A'
        
        # Extract user information, including the href of the user
        user_info_selector = "a.color-darker-grey.fontSize-16.fontWeight-600"
        user_info_element = await comment_element.query_selector(user_info_selector)
        user_info_text = await user_info_element.inner_text() if user_info_element else 'N/A'
        user_info_href = await user_info_element.get_attribute("href") if user_info_element else 'N/A'

        # Extract 'Helpful' count
        helpful_selector = "div.color-lighter-grey.fontSize-12.fontWeight-600"
        helpful_element = await comment_element.query_selector(helpful_selector)
        helpful_text = await helpful_element.inner_text() if helpful_element else 'N/A'

        # Extract just the number from the text 'Helpful (5)'
        helpful_count = re.search(r'Helpful \((\d+)\)', helpful_text)
        helpful_count = helpful_count.group(1) if helpful_count else '0'

        # Extract the date of the review
        date_selector = "time"
        date_element = await comment_element.query_selector(date_selector)
        review_date = await date_element.get_attribute("datetime") if date_element else 'N/A'

        # Append the extracted data to the list
        comments_data_ph.append([tool_name, review_id, review_text, review_date, user_info_text, user_info_href, helpful_count, score_text, all_time_reviews_amount, tool_link, recent_review_score, recent_reviews_amount ])
   
    # Return the list of extracted data
    return comments_data_ph

# function to click the 'Show more' button
async def click_show_more(page):
    # maximum number of times to click 'Show more'
    max_clicks = 150  
    # Selector for the 'Show more' button
    #show_more_button_selector = ".styles_reset__1_PU9.styles_button__7X8Df.styles_full__wgSm4.mb-8" 
    show_more_button_selector = ".styles_reset__0clCw.styles_button__BmLM4.styles_full__j4aVK.mb-8"
    # Loop through a maximum number of times to click the 'Show more' button
    for _ in range(max_clicks):
        try:
            # waiting for the 'Show more' button to appear, to ensure it's ready to be clicked
            await page.wait_for_selector(show_more_button_selector, state='visible', timeout=8000)
            # Click the 'Show more' button
            show_more_button = await page.query_selector(show_more_button_selector)
            #
            if show_more_button:
                await show_more_button.click()
                # wait for new content to load
                await asyncio.sleep(3)  
            else:
                # if the button is not found, end the loop
                break
            # if the button is clicked successfully, continue to the next iteration
        except Exception as e:
            # if error occurs, log the error and end the loop
            #print(f"Could not click 'Show more': {e}")
            break

# random user agents
USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.48',
        'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0']

# function to scrape the page
async def scrape_page_ph(context, full_url):
    # Create a new page        
    page = await context.new_page()
    
    # Randomly select a user agent from the list
    user_agent = random.choice(USER_AGENTS)

    # Set the 'User-Agent' header using the randomly selected value and add proxy
    await page.set_extra_http_headers({'User-Agent': user_agent})

    #if previous_url:
      #  headers['Referer'] = previous_url

    # Set HTTP headers
    #await page.set_extra_http_headers(headers)
    #await page.set_default_navigation_timeout(60000)  # Set a reasonable timeout

    # Navigate to the page and wait for the initial content to load
    await page.goto(full_url, wait_until="domcontentloaded")
    #await page.wait_for_load_state(state="networkidle")

    # make a screenshot of the page for debugging purposes
    #await page.screenshot(path='test.png', full_page=True)

    # check if the page is blocked
    #if response.status == 404 or "Product not found" in await page.text_content('body'):
        #logging.info(f"Product not found: {full_url}")
        #await page.close()
        #return None, None  # Return None to indicate that the product was not found
    
    #click the 'Show more' button
    await click_show_more(page)

    #general_info = await extract_general_info_ph(page)
    reviews_ph = await extract_reviews_ph(page)

    return reviews_ph

# proxy server
args = [   '--proxy-server=http://127.0.0.1:24000']

# function to scrape a single URL
async def scrape_single_url(context, base_url, product_url, final_reviews_list_ph):
    # Construct the full URL
    full_url = f"{base_url}{product_url}/reviews"
    
    success = False
    # Retry up to 3 times to load the page and extract the data
    for retry in range(2):  
        try:
            # Log the URL being scraped
            logging.info(f"Scraping: {full_url}")
            # Scrape the page
            reviews_ph = await scrape_page_ph(context, full_url)
            # Append the product URL to each item in the list
            final_reviews_list_ph.extend(reviews_ph)
            # Set the success flag to True
            success = True
            print(f"Scraping van URL {full_url} voltooid.")
            # Exit the retry loop if successful 
            break  
        # If an error occurs, log the error and wait for 2 seconds before retrying
        except Exception as e:
            logging.error(f"Failed attempt {retry + 1} for {full_url}: {str(e)}")
            await asyncio.sleep(2)  # Wacht even voordat je het opnieuw probeert

    # close the context if the scraping was not successful
    if not success:
        await context.close()

# main function
async def main():
    # Read the list of URLs from a CSV file
    base_url = "https://producthunt.com/"
    urls_df = pd.read_csv("new/code/data/processed_data/product_hunt_general_info.csv")
    product_urls = urls_df['product_url'].tolist()[900:905]
    
    #define the batch size and the sub-batch size
    batch_size = 500
    sub_batch_size = 10
    # Calculate the number of batches
    num_batches = len(product_urls) // batch_size + (1 if len(product_urls) % batch_size else 0)

# Launch the browser and create a new context
    async with async_playwright() as p:
        # Loop through each batch of URLs
        for batch in range(num_batches):
            # Launch the browser and create a new context
            browser = await p.chromium.launch(headless=False) #, args = args)
            # Create a new context
            context = await browser.new_context()
            # Calculate the start and end index for the current batch
            start_index = batch * batch_size
            # Calculate the end index for the current batch
            end_index = start_index + batch_size
            # Get the URLs for the current batch
            batch_urls = product_urls[start_index:end_index]
            # Create an empty list to store the extracted data
            final_reviews_list_ph = []

            # Loop through each sub-batch of URLs
            for i in range(0, len(batch_urls), sub_batch_size):

                # Get the URLs for the current sub-batch
                sub_batch_urls = batch_urls[i:i+sub_batch_size]
                # Scrape the URLs in the current sub-batch
                tasks = [scrape_single_url(context, base_url, url, final_reviews_list_ph) for url in sub_batch_urls]

                # Wait for all tasks to complete
                await asyncio.gather(*tasks)

                # Log the completion of the sub-batch
                print(f"Sub-batch {i // sub_batch_size + 1} of batch {batch + 1} is completed.")
                
                #await context.close()
                #browser = await p.chromium.launch(headless=False)
                #context = await browser.new_context()

                # add the product URL to each item in the list
                for review in final_reviews_list_ph:
                    # Add the product URL to the review
                    review.append(sub_batch_urls[i // sub_batch_size])

            # add to dataframe 
            comments_df = pd.DataFrame(final_reviews_list_ph, columns=['tool_name', 'Review ID', 'Review Text', 'Review Date', 'User Info name', 'User Info', 'Helpful Count', 'all_time_score', 'amount_of_reviews', 'Tool Link', 'recent_review_score', 'recent_reviews_amount', 'Product URL'])
            # Save the reviews to a CSV file
            comments_df.to_csv(f'ai_tools_comments_ph_{batch+1}_2.csv', index=False)

            # close context at the end of each batch
            await context.close()

# Run the main function
asyncio.run(main())


