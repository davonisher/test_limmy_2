# Jobs Scraper

Een eenvoudige web scraper voor het ophalen van job listings van Ashby job boards.

## Installatie

1. Zorg ervoor dat Python 3.7+ is geïnstalleerd
2. Installeer de dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Gebruik

Run de scraper:
```bash
python jobs_scraper_limmy.py
```

## Dependencies

- `requests`: Voor HTTP requests
- `beautifulsoup4`: Voor HTML parsing
- `lxml`: XML/HTML parser backend

## Deployment op Ubuntu Server

1. Clone de repository
2. Navigeer naar de project directory
3. Installeer dependencies: `pip install -r requirements.txt`
4. Run de script: `python jobs_scraper_limmy.py` 