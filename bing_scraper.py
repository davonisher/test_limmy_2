import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import csv

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

async def scrape_bing_news():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)  # Set headless=False if you want to see the browser
        page = await browser.new_page()

        all_data = []

        for company in AI_TOOL_COMPANIES:
            query = f'"{company}"'
            url = f'https://www.bing.com/news/search?q={query}'
            await page.goto(url)

            # Scroll to the bottom of the page to load more content
            previous_height = await page.evaluate('document.body.scrollHeight')
            while True:
                await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                await page.wait_for_timeout(2000)
                new_height = await page.evaluate('document.body.scrollHeight')
                if new_height == previous_height:
                    break
                previous_height = new_height

            # Extract content
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            # Try both possible Bing News article containers
            articles = soup.find_all('div', class_='news-card-body')
            if not articles:
                articles = soup.find_all('div', class_='t_s')

            for article in articles:
                # Try to extract title and link
                title_tag = article.find('a', class_='title') or article.find('a')
                link = title_tag['href'] if title_tag and title_tag.has_attr('href') else 'No Link'
                title = title_tag.text.strip() if title_tag else 'No Title'

                # Try to extract image
                image_tag = article.find('img')
                image_url = image_tag['src'] if image_tag and image_tag.has_attr('src') else 'No Image'

                # Try to extract source and snippet
                source_tag = article.find('div', class_='source') or article.find('div', class_='source-card')
                source = source_tag.text.strip() if source_tag else 'No Source'

                snippet_tag = article.find('div', class_='snippet') or article.find('div', class_='snippet-card')
                snippet = snippet_tag.text.strip() if snippet_tag else 'No Snippet'

                all_data.append({
                    'title': title,
                    'link': link,
                    'image_url': image_url,
                    'source': source,
                    'snippet': snippet,
                    'company': company
                })

        # Save data to CSV
        with open('bing_news_articles.csv', 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['company', 'title', 'link', 'image_url', 'source', 'snippet']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_data:
                writer.writerow(row)

        print(f'Scraped {len(all_data)} articles. Data saved to bing_news_articles.csv.')

        await browser.close()

# Run the scrape function
if __name__ == "__main__":
    asyncio.run(scrape_bing_news())