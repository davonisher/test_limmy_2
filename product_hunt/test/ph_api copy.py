import pandas as pd
import aiohttp
import asyncio
import json
import time

# Laad een sample van 5 uit de opgegeven CSV
import csv

csv_path = '/Users/macbook/Library/CloudStorage/OneDrive-HvA/rmai/website/extract/product_hunt/urls/PH_check1.csv'

product_hunt_urls = []
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        url = row.get('urls')
        if url:
            product_hunt_urls.append(url)
        if len(product_hunt_urls) >= 5:
            break

async def fetch_product_hunt_reviews(session, product_hunt_url, api_key):
    # Definieer de API-URL en API-sleutel
    url = "https://api.app.outscraper.com/producthunt"
    
    # Definieer de parameters voor de API-oproep
    params = {
        "query": product_hunt_url,
        "limit": 1,
        "async": "true"
    }

    # Definieer de headers, inclusief de API-sleutel
    headers = {
        "X-API-KEY": api_key
    }

    try:
        async with session.get(url, headers=headers, params=params) as response:
            response.raise_for_status()
            initial_response = await response.json()
            
            # Haal de results_location URL op
            results_url = initial_response["results_location"]
            
            # Wacht en controleer periodiek of de resultaten beschikbaar zijn
            while True:
                async with session.get(results_url, headers=headers) as results_response:
                    results_response.raise_for_status()
                    results = await results_response.json()
                    
                    if results and "data" in results:
                        return results["data"]
                    else:
                        # Resultaten zijn nog niet beschikbaar, wacht en controleer opnieuw
                        print("Resultaten nog niet beschikbaar, wacht 30 seconden...")
                        await asyncio.sleep(30)  
    
    except aiohttp.ClientResponseError as e:
        print(f"Error: {e}")
        return None

async def main():
    # API-sleutel
    api_key = "YXV0aDB8NjJkMTExMjllZGI2Mzg4ODQ2YTg1YTgxfDE0OTQ4MjQyOTU"

    # Resultaten opslaan in een lijst
    all_reviews = []

    # Asynchrone sessie openen
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in product_hunt_urls:  
            task = asyncio.ensure_future(fetch_product_hunt_reviews(session, url, api_key))
            tasks.append(task)

        # Wacht tot alle taken zijn voltooid
        reviews_list = await asyncio.gather(*tasks)

        for reviews in reviews_list[:5]:
            if reviews:
                all_reviews.extend(reviews)

    # Sla alle resultaten op in een JSON-bestand
    with open('product_hunt_reviews_all.json', 'w') as json_file:
        json.dump(all_reviews, json_file, indent=4)

    print("Alle reviews opgeslagen in product_hunt_reviews_all.json")

# Start het asynchrone proces
if __name__ == '__main__':
    start_time = time.time()
    asyncio.get_event_loop().run_until_complete(main())
    print(f"Process completed in {time.time() - start_time:.2f} seconds.")
