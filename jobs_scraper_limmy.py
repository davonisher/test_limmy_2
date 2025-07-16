import requests
from bs4 import BeautifulSoup

query = 'taktile'
url = f'https://jobs.ashbyhq.com/{query}'
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)

# Parse all HTML of the website using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

print(soup.prettify())
